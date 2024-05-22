from functools import partial
from typing import Any
import torch
from torch import vmap
from torch.func import jvp
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MeanMetric, MinMetric, MaxMetric
from torch_ema import ExponentialMovingAverage
from torch_geometric.data import Batch
from torch_geometric.utils import (
    to_dense_batch, to_dense_adj, from_dgl
)
from tqdm import tqdm
import schedulefree
from dgl import DGLGraph


from src.sfm import (
    OTSampler,
    compute_exact_loglikelihood,
    estimate_categorical_kl,
    eval_gpt_nll,
    manifold_from_name,
    ot_train_step,
    metropolis_sphere_perturbation,
    default_perturbation_schedule,
)
from src.models.net import TangentWrapper
from src.data.components.promoter_eval import SeiEval
from src.data.components.fbd import FBD
from src.data.components.qm_utils import *
from src.data.components.molecule_prior import align_prior
from src.data.components.molecule_builder import SampledMolecule
from src.data.components.sample_analyzer import SampleAnalyzer
from src.data import RetroBridgeDatasetInfos, PlaceHolder, retrobridge_utils
from src.experiments.retrosynthesis import (
    DummyExtraFeatures,
    ExtraFeatures,
    ExtraMolecularFeatures,
    SamplingMolecularMetrics,
    compute_retrosynthesis_metrics,
)


def geodesic(manifold, start_point, end_point):
    # https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/manifolds/utils.py#L6
    shooting_tangent_vec = manifold.logmap(start_point, end_point)

    def path(t):
        """Generate parameterized function for geodesic curve.
        Parameters
        ----------
        t : array-like, shape=[n_points,]
            Times at which to compute points of the geodesics.
        """
        tangent_vecs = torch.einsum("i,...k->...ik", t, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)
        return points_at_time_t

    return path


def perturbed_geodesic(manifold, start_point, end_point):
    shooting_tangent_vec = manifold.logmap(start_point, end_point)
    def path_perturbed(t):
        """Generate parameterized function for geodesic curve.
        Parameters
        ----------
        t : array-like, shape=[n_points,]
            Times at which to compute points of the geodesics.
        """
        tangent_vecs = torch.einsum("i,...k->...ik", t, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)
        return metropolis_sphere_perturbation(points_at_time_t, default_perturbation_schedule(t))
    return path_perturbed


def build_n_atoms_dist(n_atoms_hist_file: str):
    """Builds the distribution of the number of atoms in a ligand."""
    n_atoms, n_atom_counts = torch.load(n_atoms_hist_file)
    n_atoms_prob = n_atom_counts / n_atom_counts.sum()
    n_atoms_dist = torch.distributions.Categorical(probs=n_atoms_prob)
    return n_atoms_dist


class SFMModule(LightningModule):
    """
    Module for the Toy DFM dataset.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        manifold: str = "sphere",
        ot_method: str = "exact",
        promoter_eval: bool = False,
        kl_eval: bool = False,
        kl_samples: int = 512_000,
        label_smoothing: float | None = None,
        ema: bool = False,
        ema_decay: float = 0.99,
        tangent_euler: bool = True,
        closed_form_drv: bool = False,
        debug_grads: bool = False,
        inference_steps: int = 100,
        tangent_wrapper: bool = True,
        datamodule: Any = None,  # in retrobridge
        # enhancer
        eval_fbd: bool = False,
        fbd_every: int = 10,
        mel_or_dna: bool = True,  # if True, then MEL; if not Fly Brain DNA
        fbd_classifier_path: str | None = None,
        # retobridge parameters:
        input_dims: dict = {'X': 40, 'E': 10, 'y': 12},
        output_dims: dict = {'X': 17, 'E': 5, 'y': 0},
        transition: str = None,
        extra_features: str = "all",
        extra_molecular_features: bool = False,
        use_context: bool = True,
        number_chain_steps_to_save: int = 50,
        chains_to_save: int = 5,
        fix_product_nodes: bool = True,
        samples_to_generate: int = 128,
        samples_per_input: int = 5,
        retrobridge_eval_every: int = 10,
        # ppl
        eval_ppl: bool = False,
        eval_ppl_every: int = 10,
        normalize_loglikelihood: bool = False,
        # GPT NLL?
        gpt_nll_eval: bool = False,
        eval_gpt_nll_every: int = 10,
        gpt_nll_samples: int = 512,
        # qm
        eval_unconditional_mols: bool = False,
        eval_n_mols: int = 0,
        eval_unconditional_mols_every: int = 10,
        predict_mol: bool = False,  # instead of tangent vector
        # misc
        fast_matmul: bool = False,
    ):
        """
        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net", "fbd", "n_atoms_dist"], logger=False)
        # if basically zero or zero
        self.smoothing = label_smoothing if label_smoothing and label_smoothing > 1e-6 else None
        self.tangent_euler = tangent_euler
        self.closed_form_drv = closed_form_drv
        self.promoter_eval = promoter_eval
        self.manifold = manifold_from_name(manifold)
        # don't wrap for retrobridge!
        self.net = net if not tangent_wrapper or datamodule is not None else TangentWrapper(self.manifold, net)
        if ema:
            self.ema = ExponentialMovingAverage(self.net.parameters(), decay=ema_decay).to(self.device)
        else:
            self.ema = None
        self.sampler = OTSampler(self.manifold, ot_method) if ot_method != "None" else None
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_ppl = MeanMetric()
        self.test_ppl = MeanMetric()
        self.sp_mse = MeanMetric()
        self.test_sp_mse = MeanMetric()
        self.min_grad = MinMetric()
        self.max_grad = MaxMetric()
        self.mean_grad = MeanMetric()
        self.kl_eval = kl_eval
        self.kl_samples = kl_samples
        self.debug_grads = debug_grads
        self.inference_steps = inference_steps
        # PPL
        self.eval_ppl = eval_ppl
        self.eval_ppl_every = eval_ppl_every
        self.normalize_loglikelihood = normalize_loglikelihood
        # GPT NLL
        self.eval_gpt_nll = gpt_nll_eval
        self.gpt_nll_every = eval_gpt_nll_every
        self.gpt_nll_samples = gpt_nll_samples
        # QM
        self.eval_unconditional_mols = eval_unconditional_mols
        self.eval_n_mols = eval_n_mols
        self.eval_unconditional_mols_every = eval_unconditional_mols_every
        self.predict_mol = predict_mol
        if self.eval_unconditional_mols:
            self.n_atoms_dist = build_n_atoms_dist("./data/qm9/train_data_n_atoms_histogram.pt")
            self.mol_features = ["a", "x", "e", "c"]
            for ft in self.mol_features:
                for tp in ["train", "val", "test"]:
                    setattr(self, f"{tp}_{ft}_loss", MeanMetric())
        # retrobridge:
        self.retrobridge = datamodule is not None
        self.retrobridge_eval_every = retrobridge_eval_every

        # enhancer
        self.eval_fbd = eval_fbd
        self.fbd_every = fbd_every
        if eval_fbd:
            self.fbd = FBD(
                dim=4,
                k=500,
                num_cls=47 if mel_or_dna else 81,
                hidden=128,  # read config
                depth=4 if mel_or_dna else 1,
                ckpt_path=fbd_classifier_path,
            )
            self.val_fbd = MeanMetric()
            self.test_fbd = MeanMetric()

        # retrobridge variables
        if datamodule is not None:
            self.input_dims = input_dims
            self.output_dims = output_dims
            self.transition = transition
            self.extra_features = extra_features
            self.extra_molecular_features = extra_molecular_features
            self.use_context = use_context
            self.number_chain_steps_to_save = number_chain_steps_to_save
            self.chains_to_save = chains_to_save
            self.fix_product_nodes = fix_product_nodes
            self.samples_to_generate = samples_to_generate
            self.samples_per_input = samples_per_input
            self.dataset_infos = RetroBridgeDatasetInfos(datamodule)
            self.extra_features = (
                ExtraFeatures(extra_features, dataset_info=self.dataset_infos)
                if extra_features is not None
                else DummyExtraFeatures()
            )
            self.domain_features = (
                ExtraMolecularFeatures(dataset_infos=self.dataset_infos)
                if extra_molecular_features
                else DummyExtraFeatures()
            )
            self.dataset_infos.compute_input_output_dims(
                datamodule=datamodule,
                extra_features=self.extra_features,
                domain_features=self.domain_features,
                use_context=use_context,
            )
            self.val_molecular_metrics = SamplingMolecularMetrics(
                self.dataset_infos,
                datamodule.train_smiles,
            )
        else:
            self.dataset_infos = None
        if fast_matmul:
            torch.set_float32_matmul_precision("high")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(x, t)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_ppl.reset()
        self.sp_mse.reset()
        if hasattr(self, "val_fbd"):
            self.val_fbd.reset()

    def on_validation_epoch_start(self):
        for optim in self.trainer.optimizers:
            # schedule free needs to set to eval
            if isinstance(optim, schedulefree.AdamWScheduleFree):
                optim.eval()

    def on_after_backward(self):
        for optim in self.trainer.optimizers:
            # schedule free needs to set to train
            if isinstance(optim, schedulefree.AdamWScheduleFree):
                optim.train()

    def on_test_epoch_start(self):
        for optim in self.trainer.optimizers:
            # schedule free needs to set to eval
            if isinstance(optim, schedulefree.AdamWScheduleFree):
                optim.eval()

    def model_step(
        self, x_1: torch.Tensor, extra_args: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Perform a single model step on a batch of data.
        """
        # points are on the simplex
        x_1 = self.manifold.project(x_1)
        return ot_train_step(
            self.manifold.smooth_labels(x_1, mx=self.smoothing) if self.smoothing else x_1,
            self.manifold,
            self.net,
            self.sampler,
            closed_form_drv=self.closed_form_drv,
            extra_args=extra_args,
        )[0]

    def retrobridge_step(
        self, data: Batch,
    ) -> torch.Tensor:
        """
        Performs a step on retrobridge data, returns loss.
        """
        # Getting graphs of reactants (target) and product (context)
        reactants, r_node_mask = retrobridge_utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        reactants = reactants.mask(r_node_mask)

        product, p_node_mask = retrobridge_utils.to_dense(
            data.p_x, data.p_edge_index, data.p_edge_attr, data.batch,
        )
        product = product.mask(p_node_mask)

        t = torch.rand(data.batch_size, 1, device=data.x.device)

        # product is prior, reactants is targret
        # now, need to train over product of these buggers

        target_edge_shape = (product.E.size(0), -1, product.E.size(-1))
        edges_t, edges_target = self.compute_target(
            product.E.reshape(target_edge_shape),
            reactants.E.reshape(target_edge_shape),
            t,
        )
        edges_t = edges_t.reshape_as(product.E)
        edges_target = edges_target.reshape_as(product.E)
        # symmetrise edges_t, edges_target
        edges_t = self.symmetrise_edges(edges_t)
        edges_target = self.symmetrise_edges(edges_target)
        feats_t, feats_target = self.compute_target(
            product.X,
            reactants.X,
            t,
        )
        # mask things not included
        feats_t, edges_t = self.mask_like_placeholder(p_node_mask, feats_t, edges_t)
        feats_target, edges_target = self.mask_like_placeholder(
            p_node_mask,
            feats_target,
            edges_target,
        )


        # checks
        # assert (torch.isclose(feats_t.square().sum(dim=-1), torch.tensor(1.0)) | torch.isclose(feats_t.square().sum(dim=-1), torch.tensor(0.0))).all()
        # assert (torch.isclose(edges_t.square().sum(dim=-1), torch.tensor(1.0)) | torch.isclose(edges_t.square().sum(dim=-1), torch.tensor(0.0))).all()
        # f_mask = torch.isclose(feats_t.square().sum(dim=-1), torch.tensor(1.0))
        # assert self.manifold.all_belong_tangent(feats_t[f_mask], feats_target[f_mask])
        # flat_e = edges_t.reshape(edges_t.size(0), -1, edges_t.size(-1))
        # e_mask = torch.isclose(flat_e.square().sum(dim=(-1)), torch.tensor(1.0))
        # assert self.manifold.all_belong_tangent(flat_e[e_mask], edges_target.reshape_as(flat_e)[e_mask])


        # define data
        noisy_data = {
            "t": t,
            "E_t": edges_t,
            "X_t": feats_t,
            "y": product.y,
            "y_t": product.y,
            "node_mask": r_node_mask,
        }

        # Computing extra features + context and making predictions
        context = product.clone() if self.use_context else None
        extra_data = self.compute_extra_data(noisy_data, context=context)

        pred = self.retrobridge_forward(noisy_data, extra_data, r_node_mask)
        # have two targets, need two projections
        modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
        fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
        assert torch.all(modifiable_nodes | fixed_nodes)
        loss_x_raw = (
            self.manifold.masked_tangent_projection(feats_t, pred.X) - feats_target
        ) * modifiable_nodes
        # reshape for B, K, D shape
        edges_t = edges_t.reshape(edges_t.size(0), -1, edges_t.size(-1))
        pred_reshaped = pred.E.reshape(pred.E.size(0), -1, pred.E.size(-1))
        loss_edges_raw = (
            self.manifold.masked_tangent_projection(edges_t, pred_reshaped) - edges_target.reshape_as(edges_t)
        )
        # loss_edges_raw = loss_edges_raw.reshape_as(product.E)

        # final_X, final_E = self.mask_like_placeholder(p_node_mask, loss_x_raw, loss_edges_raw)
        loss = (
            loss_x_raw.square().sum(dim=(-1, -2))
            + 5.0 * loss_edges_raw.square().sum(dim=(-1, -2, -3))  # 5.0* done in retrobridge
        ).mean()
        return loss

    def retrobridge_forward(self, noisy_data: dict[str, Any], extra_data: PlaceHolder, node_mask: torch.Tensor) -> torch.Tensor:
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.net(X, E, y, node_mask)

    def encode_empty_cat(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes no category as a feature in the last coordinate.
        """
        no_cat_pad = torch.zeros(*tensor.shape[:-1], 1).to(tensor)
        no_cat_pad[tensor.sum(dim=-1) == 0] = 1
        return torch.cat([no_cat_pad, tensor], dim=-1)

    def symmetrise_edges(self, E: torch.Tensor) -> torch.Tensor:
        """
        Symmetrises the edges tensor.
        """
        i, j = torch.triu_indices(E.size(1), E.size(2))
        E[:, i, j, :] = E[:, j, i, :]
        return E

    def compute_extra_data(self, noisy_data, context=None, condition_on_t=True):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        if context is not None:
            extra_X = torch.cat((extra_X, context.X), dim=-1)
            extra_E = torch.cat((extra_E, context.E), dim=-1)

        if condition_on_t:
            t = noisy_data['t']
            extra_y = torch.cat((extra_y, t), dim=1)
        return PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def set_zero_diag(self, E: torch.Tensor) -> torch.Tensor:
        """
        Sets the diagonal of `E` to all zeros. Taken from `retrobridge_utils`.
        """
        diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
        E[diag] = 0
        return E

    def mask_like_placeholder(self, node_mask: torch.Tensor, X: torch.Tensor, E: torch.Tensor, collapse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)

        if collapse:
            X = torch.argmax(X, dim=-1)
            E = torch.argmax(E, dim=-1)

            X[node_mask == 0] = - 1
            E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            X[node_mask == 0, :] = 0
            E[~((e_mask1 * e_mask2).squeeze(-1)), :] = 0
            E = self.set_zero_diag(E)
            # assert torch.allclose(E, torch.transpose(E, 1, 2))
        return X, E

    def compute_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes flow-matching target; returns point and target itself.
        """
        if not self.closed_form_drv:
            with torch.inference_mode(False):
                def cond_u(x0, x1, t):
                    path = geodesic(self.manifold.sphere, x0, x1)
                    x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
                    return x_t, u_t
                x_t, target = vmap(cond_u)(x_0, x_1, time)
            x_t = x_t.squeeze()
            target = target.squeeze()
            if x_0.size(0) == 1:
                # squeezing will remove the batch
                x_t = x_t.unsqueeze(0)
                target = target.unsqueeze(0)
        else:
            mask = torch.isclose(x_0.square().sum(dim=-1), torch.tensor(1.0))
            x_t = torch.zeros_like(x_0)
            x_t[mask] = self.manifold.geodesic_interpolant(x_0, x_1, time)[mask]
            x_t[mask] = metropolis_sphere_perturbation(
                x_t, default_perturbation_schedule(time)
            )[mask]
            target = torch.zeros_like(x_t)
            target[mask] = self.manifold.log_map(x_0, x_1)[mask]
            target[mask] = self.manifold.parallel_transport(x_0, x_t, target)[mask]
        # assert self.manifold.all_belong_tangent(x_t, target)
        return x_t, target

    def retrobridge_eval(self):
        """Evaluation metrics for retrobridge."""
        samples_left_to_generate = self.samples_to_generate
        samples_left_to_save = 0 # self.samples_to_save
        chains_left_to_save = self.chains_to_save

        samples = []
        grouped_samples = []
        # grouped_scores = []
        ground_truth = []

        ident = 0
        print(f'Sampling epoch={self.current_epoch}')

        dataloader = self.trainer.datamodule.val_dataloader()
        for data in tqdm(dataloader, total=samples_left_to_generate // dataloader.batch_size):
            if samples_left_to_generate <= 0:
                break

            data = data.to(self.device)
            bs = len(data.batch.unique())
            to_generate = bs
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            batch_groups = []
            # batch_scores = []
            for sample_idx in range(self.samples_per_input):
                mol_sample = self.sample_molecule(data)
                molecule_list = retrobridge_utils.create_pred_reactant_molecules(mol_sample.X, mol_sample.E, data.batch, to_generate)
                samples.extend(molecule_list)
                batch_groups.append(molecule_list)
                # batch_scores.append(scores)
                if sample_idx == 0:
                    ground_truth.extend(
                        retrobridge_utils.create_true_reactant_molecules(data, to_generate)
                    )

            ident += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

            # Regrouping sampled reactants for computing top-N accuracy
            for mol_idx_in_batch in range(bs):
                mol_samples_group = []
                mol_scores_group = []
                # for batch_group, scores_group in zip(batch_groups, batch_scores):
                for batch_group in batch_groups:
                    mol_samples_group.append(batch_group[mol_idx_in_batch])
                    # mol_scores_group.append(scores_group[mol_idx_in_batch])

                assert len(mol_samples_group) == self.samples_per_input
                grouped_samples.append(mol_samples_group)
                # grouped_scores.append(mol_scores_group)

        to_log = compute_retrosynthesis_metrics(
            grouped_samples=grouped_samples,
            ground_truth=ground_truth,
            atom_decoder=self.dataset_infos.atom_decoder,
            grouped_scores=None,
        )
        for metric_name, metric in to_log.items():
            self.log(metric_name, metric)

        to_log = self.val_molecular_metrics(samples)
        for metric_name, metric in to_log.items():
            self.log(metric_name, metric)

        self.val_molecular_metrics.reset()

    @torch.no_grad()
    def sample_molecule(
        self,
        data: Batch,
    ) -> PlaceHolder:
        """
        Samples reactants given a product contained in `data`.
        """
        # generate molecules
        product, node_mask = retrobridge_utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(node_mask)
        X, E, y = (
            product.X,
            product.E,
            torch.empty((node_mask.shape[0], 0), device=self.device),
        )
        # do joint tangent Euler method
        dt = torch.tensor(1.0 / self.inference_steps, device=data.x.device)
        t = torch.zeros(data.batch_size, 1, device=data.x.device)
        context = product.clone() if self.use_context else None
        orig_edge_shape = E.shape
        # Masks for fixed and modifiable nodes  | from Retrobridge
        fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)
        # start!
        for _ in range(self.inference_steps):
            noisy_data = {
                "t": t,
                "E_t": E,
                "X_t": X,
                "y_t": y,
                "node_mask": node_mask,
            }
            extra_data = self.compute_extra_data(noisy_data, context=context)
            pred = self.retrobridge_forward(noisy_data, extra_data, node_mask)
            # prep
            target_edge_shape = (E.size(0), -1, E.size(-1))
            # make a step
            X = self.manifold.exp_map(
                X, self.manifold.make_tangent(X, pred.X) * dt,
            )
            X = self.manifold.project(X)
            X = X * modifiable_nodes + product.X * fixed_nodes
            E = E.reshape(target_edge_shape)
            E = self.manifold.exp_map(
                E,
                self.manifold.masked_tangent_projection(
                    E,
                    pred.E.reshape((pred.E.size(0), -1, pred.E.size(-1))),
                ) * dt,
            )
            E = self.manifold.masked_projection(E)
            y = pred.y
            t += dt
            E = E.reshape(orig_edge_shape)
            X, E = self.mask_like_placeholder(node_mask, X, E)
            E = self.symmetrise_edges(E)

        ret = PlaceHolder(
            X=X,
            E=E,
            y=y,
        ).mask(node_mask, collapse=True)

        # E = self.symmetrise_edges(E)
        return ret

    def produce_text_samples(self, n: int) -> list[str]:
        """
        Produces `n` text samples.
        """
        samples = self.manifold.tangent_euler(
            self.manifold.uniform_prior(n, 256, 28).to(self.device),
            self.net,
            self.inference_steps,
        )
        chars = samples.argmax(dim=-1).cpu()
        rets = []
        for sample in chars:
            rets += ["".join([self.trainer.datamodule.itos[c.item()] for c in sample])]
        return rets

    def densify_dgl_edges(self, graph: DGLGraph, edges_attr: torch.Tensor, node_batch_idx: torch.Tensor) -> torch.Tensor:
        graph.edata["edge_attr"] = edges_attr
        tg = from_dgl(graph)
        return to_dense_adj(tg.edge_index, node_batch_idx, tg.edge_attr)

    def qm_step(self, graph: DGLGraph) -> torch.Tensor:
        """
        Perform a single step on a QM9 graph.
        """
        node_batch_idx, edge_batch_idx = get_batch_idxs(graph)
        edge_upper_index = get_upper_edge_mask(graph)
        # random positions
        # compute rest
        if not self.predict_mol:
            t = torch.rand(graph.batch_size, 1, device=graph.device)
            # prepare
            e_1 = self.densify_dgl_edges(graph, graph.edata["e_1_true"], node_batch_idx)
            x_1, x_mask = to_dense_batch(graph.ndata["x_1_true"], node_batch_idx)
            a_1, a_mask = to_dense_batch(graph.ndata["a_1_true"], node_batch_idx)
            x_0 = torch.randn_like(x_1)  # positions!
            x_0 = align_prior(x_0, x_1, permutation=True, rigid_body=True)  # like OT
            a_0 = self.manifold.uniform_prior(*a_1.shape).to(graph.device)
            c_0 = self.manifold.uniform_prior(*c_1.shape).to(graph.device)
            c_1, c_mask = to_dense_batch(graph.ndata["c_1_true"], node_batch_idx)
            edges_flat = e_1.reshape(e_1.size(0), -1, e_1.size(-1))
            e_0 = self.manifold.uniform_prior(*edges_flat.shape).to(graph.device)

            # sample points
            # to_dense_adj(graph.adj().to_dense(), node_batch_idx, graph.edata["e_1_true"])
            # need to compute target wrt Euclidean flow matching
            # x_t, x_target = ...
            a_t, a_target = self.compute_target(a_0, a_1, t)
            c_t, c_target = self.compute_target(c_0, c_1, t)
            e_t, e_target = self.compute_target(e_0, edges_flat, t)
            e_t = e_t.reshape_as(e_1)
            e_1_mask = torch.isclose(e_1.sum(dim=-1), torch.tensor(1.0))
            e_t = e_t[e_1_mask]
            e_t[~edge_upper_index] = e_t[edge_upper_index]

            graph.ndata["a_t"] = a_t[a_mask]
            graph.ndata["c_t"] = c_t[c_mask]
            graph.edata["e_t"] = e_t
            graph.ndata["x_t"] = (x_0 * t.unsqueeze(-1) + x_1 * (1 - t.unsqueeze(-1)))[x_mask]
            ret_dict = self.net(graph, t.squeeze(), node_batch_idx, edge_upper_index)
            a_pred = self.manifold.make_tangent(a_t[a_mask], ret_dict["a"])
            c_pred = self.manifold.make_tangent(c_t[c_mask], ret_dict["c"])
            e_pred = self.manifold.make_tangent(e_t[edge_upper_index], ret_dict["e"])
            x_pred = ret_dict["x"]  # euclid, always tangent

            # losses
            a_loss = (a_pred - a_target[a_mask]).square().sum()
            x_loss = (x_pred - (x_1 - x_0)[x_mask]).square().sum()
            c_loss = (c_pred - c_target[c_mask]).square().sum()
            e_loss = (e_pred - e_target[e_1_mask.reshape(e_1_mask.size(0), -1)][edge_upper_index]).square().sum()
            return (a_loss + x_loss + c_loss + e_loss) / graph.batch_size, a_loss, x_loss, c_loss, e_loss
        else:
            # predict molecule directly!
            t = torch.rand(graph.batch_size, 1, device=graph.device)
            # easier here
            n_nodes = graph.num_nodes()
            n_edges = graph.num_edges()
            graph.ndata["x_0"] = torch.randn_like(graph.ndata["x_1_true"])
            graph.ndata["a_0"] = self.manifold.uniform_prior(n_nodes, 1, graph.ndata["a_1_true"].size(-1)).to(graph.device).squeeze()
            graph.ndata["c_0"] = self.manifold.uniform_prior(n_nodes, 1, graph.ndata["c_1_true"].size(-1)).to(graph.device).squeeze()
            graph.edata["e_0"] = self.manifold.uniform_prior(n_edges, 1, graph.edata["e_1_true"].size(-1)).to(graph.device).squeeze()
            graph.edata["e_0"][~edge_upper_index] = graph.edata["e_0"][edge_upper_index]

            # linear bugger
            graph.ndata["x_t"] = (
                t[node_batch_idx] * graph.ndata["x_0"] 
                + (1 - t)[node_batch_idx] * graph.ndata["x_1_true"]
            ).squeeze()
            # rest is on manifold
            graph.ndata["a_t"] = self.manifold.geodesic_interpolant(
                graph.ndata["a_0"].unsqueeze(1), graph.ndata["a_1_true"].unsqueeze(1), t[node_batch_idx]
            ).squeeze()
            graph.ndata["c_t"] = self.manifold.geodesic_interpolant(
                graph.ndata["c_0"].unsqueeze(1), graph.ndata["c_1_true"].unsqueeze(1), t[node_batch_idx]
            ).squeeze()
            graph.edata["e_t"] = self.manifold.geodesic_interpolant(
                graph.edata["e_0"].unsqueeze(1), graph.edata["e_1_true"].unsqueeze(1), t[edge_batch_idx]
            ).squeeze()

            # remove softmax, because cross entropy loss
            pred_dict = self.net(graph, t.squeeze(), node_batch_idx, edge_upper_index, apply_softmax=False, remove_com=True)

            # loss is from the paper
            x_loss = 3.0 * (pred_dict["x"] - graph.ndata["x_1_true"]).square().sum() / graph.batch_size
            a_loss = 0.4 * F.cross_entropy(pred_dict["a"], graph.ndata["a_1_true"], reduction="sum") / graph.batch_size
            c_loss = 1.0 * F.cross_entropy(pred_dict["c"], graph.ndata["c_1_true"], reduction="sum") / graph.batch_size
            e_loss = 2.0 * F.cross_entropy(pred_dict["e"], graph.edata["e_1_true"][edge_upper_index], reduction="sum") / graph.batch_size
            return x_loss + a_loss + c_loss + e_loss, a_loss, x_loss, c_loss, e_loss

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantizes to one-hot encoding.
        """
        return F.one_hot(tensor.argmax(dim=-1), num_classes=tensor.size(-1))

    @torch.no_grad()
    def sample_unconditional_molecule(self, n_atoms: torch.Tensor) -> list[SampledMolecule]:
        batch_size = n_atoms.size(0)
        # get the edge indicies for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=self.device)
            g_i.ndata["c_t"] = self.manifold.uniform_prior(n_atoms_i, 1, 6).to(self.device).squeeze()
            g_i.ndata["x_t"] = torch.randn(n_atoms_i, 1, 3).to(self.device).squeeze()
            g_i.ndata["a_t"] = self.manifold.uniform_prior(n_atoms_i, 1, 5).to(self.device).squeeze()
            g_i.edata["e_t"] = self.manifold.uniform_prior(len(edge_idxs[0]), 1, 5).squeeze().to(self.device)
            g.append(g_i)

        g = dgl.batch(g)

        upper_edge_mask = get_upper_edge_mask(g)
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # inference
        dt = torch.tensor(1.0 / self.inference_steps, device=self.device)
        t = torch.zeros(batch_size, device=self.device)
        if not self.predict_mol:
            for _ in range(self.inference_steps):
                dict_ret = self.net(g, t, node_batch_idx, upper_edge_mask)
                a = self.manifold.make_tangent(g.ndata["a_t"], dict_ret["a"])
                c = self.manifold.make_tangent(g.ndata["c_t"], dict_ret["c"])
                e = self.manifold.make_tangent(g.edata["e_t"][upper_edge_mask], dict_ret["e"])
                # update
                g.ndata["x_t"] = g.ndata["x_t"] + dict_ret["x"] * dt
                g.ndata["a_t"] = self.manifold.exp_map(g.ndata["a_t"], a * dt)
                g.ndata["c_t"] = self.manifold.exp_map(g.ndata["c_t"], c * dt)
                g.edata["e_t"][upper_edge_mask] = self.manifold.exp_map(g.edata["e_t"][upper_edge_mask], e * dt)
                g.edata["e_t"][~upper_edge_mask] = self.manifold.exp_map(g.edata["e_t"][~upper_edge_mask], e * dt)

            g.ndata["x_1"] = g.ndata["x_t"]
            g.ndata["a_1"] = self.quantize(g.ndata["a_t"])
            g.ndata["c_1"] = self.quantize(g.ndata["c_t"])
            g.edata["e_1"] = self.quantize(g.edata["e_t"])
        else:
            # predict mol, need integration
            for _ in range(self.inference_steps):
                dict_ret = self.net(g, t, node_batch_idx, upper_edge_mask, apply_softmax=True, remove_com=True)
                x_1_weight = dt / (1.0 - t)
                g.ndata["x_t"] = g.ndata["x_t"] * (1.0 - x_1_weight)[node_batch_idx, None] + dict_ret["x"] * x_1_weight[node_batch_idx, None]
                # things must be on sphere, hence squares
                g.ndata["c_t"] = self.manifold.geodesic_interpolant(g.ndata["c_t"], dict_ret["c"].square(), t[node_batch_idx])
                g.ndata["a_t"] = self.manifold.geodesic_interpolant(g.ndata["a_t"], dict_ret["a"].square(), t[node_batch_idx])
                e_t = self.manifold.geodesic_interpolant(g.edata["e_t"][upper_edge_mask], dict_ret["e"].square(), t[edge_batch_idx][upper_edge_mask])
                e_t_set = torch.zeros_like(g.edata["e_t"])
                e_t_set[upper_edge_mask] = e_t
                e_t_set[~upper_edge_mask] = e_t
                g.edata["e_t"] = e_t_set
            g.ndata["x_1"] = g.ndata["x_t"]
            g.ndata["a_1"] = g.ndata["a_t"]
            g.ndata["c_1"] = g.ndata["c_t"]
            g.edata["e_1"] = g.edata["e_t"]

        g.edata["ue_mask"] = upper_edge_mask
        g = g.to("cpu")

        molecules = []
        for _, g_i in enumerate(dgl.unbatch(g)):
            args = [g_i, ['C', 'H', 'N', 'O', 'F',]]
            molecules.append(SampledMolecule(*args, exclude_charges=False))

        return molecules

    def training_step(
        self, x_1: torch.Tensor | list[torch.Tensor] | Batch, batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        if isinstance(x_1, list):
            x_1, signal = x_1
            # Only one of the two signal inputs is used (the first one)
            if len(signal.shape) == 2:
                signal = signal[:, :, 0].unsqueeze(-1)
                loss = self.model_step(x_1, {"signal": signal})
            else:
                loss = self.model_step(x_1, {"cls": signal})
        elif isinstance(x_1, Batch):
            loss = self.retrobridge_step(x_1)
        elif isinstance(x_1, DGLGraph):
            loss, a_loss, x_loss, c_loss, e_loss  = self.qm_step(x_1)
            self.train_a_loss(a_loss)
            self.train_x_loss(x_loss)
            self.train_c_loss(c_loss)
            self.train_e_loss(e_loss)
            for ft in self.mol_features:
                self.log(f"train/{ft}-loss", getattr(self, f"train_{ft}_loss"), on_step=False, on_epoch=True, prog_bar=True)
        else:
            loss = self.model_step(x_1)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Nothing to do."""

    def validation_step(self, x_1: torch.Tensor | list[torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if isinstance(x_1, list):
            x_1, signal = x_1
            # Only one of the two signal inputs is used (the first one)
            if len(signal.shape) == 2:
                signal = signal[:, :, 0].unsqueeze(-1)
                loss = self.model_step(x_1, {"signal": signal})
            else:
                loss = self.model_step(x_1, {"cls": signal})
        elif isinstance(x_1, Batch):
            loss = self.retrobridge_step(x_1)
        elif isinstance(x_1, DGLGraph):
            loss, a_loss, x_loss, c_loss, e_loss  = self.qm_step(x_1)
            self.val_a_loss(a_loss)
            self.val_x_loss(x_loss)
            self.val_c_loss(c_loss)
            self.val_e_loss(e_loss)
            for ft in self.mol_features:
                self.log(f"val/{ft}-loss", getattr(self, f"val_{ft}_loss"), on_step=False, on_epoch=True, prog_bar=False)
        else:
            loss = self.model_step(x_1)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.promoter_eval:
            mse = self.compute_sp_mse(x_1, signal, batch_idx)
            self.sp_mse(mse)
            self.log("val/sp-mse", self.sp_mse, on_step=False, on_epoch=True, prog_bar=True)
        if self.eval_ppl and (self.trainer.current_epoch + 1) % self.eval_ppl_every == 0:
            net = self.net if signal is None else (
                partial(self.net, signal=signal) if len(signal.shape) != 1 else
                partial(self.net, cls=signal)
            )
            ppl = compute_exact_loglikelihood(
                net, x_1, self.manifold.sphere, normalize_loglikelihood=self.normalize_loglikelihood,
                num_steps=self.inference_steps,
            ).mean()
            self.val_ppl(ppl)
            self.log("val/ppl", self.val_ppl, on_step=False, on_epoch=True, prog_bar=True)
        if self.eval_fbd and (self.trainer.current_epoch + 1) % self.fbd_every == 0:
            self.val_fbd(self.compute_fbd(x_1, signal, self.inference_steps // 4, batch_idx))
            self.log("val/fbd", self.val_fbd, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        if self.kl_eval:
            # evaluate KL
            real_probs = self.trainer.val_dataloaders.dataset.probs.to(self.device)
            kl = estimate_categorical_kl(
                self.net,
                self.manifold,
                real_probs,
                self.kl_samples // 10,
                batch=self.hparams.get("kl_batch", 2048),
                silent=True,
                tangent=self.tangent_euler,
                inference_steps=self.inference_steps,
            )
            self.log("val/kl", kl, on_step=False, on_epoch=True, prog_bar=True)
        if self.dataset_infos is not None and (self.retrobridge_eval_every == 1 or (self.trainer.current_epoch + 1) % self.retrobridge_eval_every == 0):
            # evaluate retrobridge
            self.retrobridge_eval()
        if self.eval_gpt_nll and (self.trainer.current_epoch + 1) % self.gpt_nll_every == 0:
            nll = eval_gpt_nll(
                self.produce_text_samples(self.gpt_nll_samples),
                self.device,
            )
            self.log("val/gpt-nll", nll, on_step=False, on_epoch=True, prog_bar=True)
        if self.eval_unconditional_mols and (self.trainer.current_epoch + 1) % self.eval_unconditional_mols_every == 0:
            molecules = self.sample_unconditional_molecule(self.n_atoms_dist.sample((self.eval_n_mols // 4,)))
            analyzer = SampleAnalyzer()
            stats = analyzer.analyze(molecules)
            for key, value in stats.items():
                self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, x_1: torch.Tensor | list[torch.Tensor], batch_idx: int):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if isinstance(x_1, list):
            x_1, signal = x_1
            # Only one of the two signal inputs is used (the first one)
            if len(signal.shape) == 2:
                signal = signal[:, :, 0].unsqueeze(-1)
                loss = self.model_step(x_1, {"signal": signal})
            else:
                loss = self.model_step(x_1, {"cls": signal})
        elif isinstance(x_1, Batch):
            loss = self.retrobridge_step(x_1)
        elif isinstance(x_1, DGLGraph):
            loss, a_loss, x_loss, c_loss, e_loss  = self.qm_step(x_1)
            self.test_a_loss(a_loss)
            self.test_x_loss(x_loss)
            self.test_c_loss(c_loss)
            self.test_e_loss(e_loss)
            for ft in self.mol_features:
                self.log(f"test/{ft}-loss", getattr(self, f"test_{ft}_loss"), on_step=False, on_epoch=True, prog_bar=True)
        else:
            loss = self.model_step(x_1)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.promoter_eval:
            mse = self.compute_sp_mse(x_1, signal)
            self.test_sp_mse(mse)
            self.log("test/sp-mse", self.test_sp_mse, on_step=False, on_epoch=True, prog_bar=True)
        if self.eval_fbd:
            self.test_fbd(self.compute_fbd(x_1, signal, self.inference_steps, None))
            self.log("test/fbd", self.test_fbd, on_step=False, on_epoch=True, prog_bar=True)
        if self.eval_ppl:
            net = self.net if signal is None else (
                partial(self.net, signal=signal) if len(signal.shape) != 1 else
                partial(self.net, cls=signal)
            )
            ppl = compute_exact_loglikelihood(
                net, x_1, self.manifold.sphere, normalize_loglikelihood=self.normalize_loglikelihood,
                num_steps=self.inference_steps,
            ).mean()
            print(ppl)
            self.test_ppl(ppl)
            self.log("test/ppl", self.test_ppl, on_step=False, on_epoch=True, prog_bar=True)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.debug_grads:
            norms = grad_norm(self.net, norm_type=2).values()
            self.min_grad(min(norms))
            self.max_grad(max(norms))
            self.mean_grad(sum(norms) / len(norms))
            self.log("train/min_grad", self.min_grad, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/max_grad", self.max_grad, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/mean_grad", self.mean_grad, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        """Evaluates KL if required."""
        if self.kl_eval:
            # evaluate KL
            real_probs = self.trainer.test_dataloaders.dataset.probs.to(self.device)
            kl = estimate_categorical_kl(
                self.net,
                self.manifold,
                real_probs,
                self.kl_samples,
                batch=self.hparams.get("kl_batch", 2048),
                tangent=self.tangent_euler,
            )
            self.log("test/kl", kl, on_step=False, on_epoch=True, prog_bar=False)
        if self.eval_unconditional_mols:
            molecules = self.sample_unconditional_molecule(self.n_atoms_dist.sample((self.eval_n_mols,)))
            analyzer = SampleAnalyzer()
            stats = analyzer.analyze(molecules)
            for key, value in stats.items():
                self.log(f"test/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

    def compute_sp_mse(
        self,
        x_1: torch.Tensor,
        signal: torch.Tensor,
        batch_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Computes the model's SP MSE.
        """
        eval_model = partial(self.net, signal=signal)
        pred = self.manifold.tangent_euler(
            self.manifold.uniform_prior(*x_1.shape[:-1], 4).to(x_1.device),
            eval_model,
            steps=self.inference_steps,
            tangent=self.tangent_euler,
        )
        mx = torch.argmax(pred, dim=-1)
        one_hot = F.one_hot(mx, num_classes=4)
        return SeiEval().eval_sp_mse(seq_one_hot=one_hot, target=x_1, b_index=batch_idx)

    def compute_fbd(
        self,
        x_1: torch.Tensor,
        signal: torch.Tensor,
        steps: int,
        batch_idx: int | None,
    ):
        """
        Computes the FBD.
        """
        eval_model = partial(self.net, cls=signal)
        pred = self.manifold.tangent_euler(
            self.manifold.uniform_prior(*x_1.shape).to(self.device),
            eval_model,
            steps=steps,
            tangent=self.tangent_euler,
        )
        return self.fbd(pred.argmax(dim=-1), x_1.argmax(dim=-1), batch_idx)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update()

    def setup(self, stage: str):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    SFMModule(None, None, None, False)
