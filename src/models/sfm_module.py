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
        self.save_hyperparameters(ignore=["net", "fbd"], logger=False)
        # if basically zero or zero
        self.smoothing = label_smoothing if label_smoothing and label_smoothing > 1e-6 else None
        self.tangent_euler = tangent_euler
        self.closed_form_drv = closed_form_drv
        self.promoter_eval = promoter_eval
        self.manifold = manifold_from_name(manifold)
        # don't wrap for retrobridge!
        self.net = TangentWrapper(self.manifold, net).to(self.device) if not datamodule else net
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

    def on_train_epoch_start(self):
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
            x_t[mask] = self.manifold.geodesic_interpolant(x_0, x_1, t)[mask]
            x_t[mask] = metropolis_sphere_perturbation(
                x_t, default_perturbation_schedule(t)
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
                """molecule_list, true_molecule_list, products_list, scores, _, _ = self.sample_batch(
                    data=data,
                    batch_id=ident,
                    batch_size=to_generate,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps_to_save=self.number_chain_steps_to_save,
                    sample_idx=sample_idx,
                )"""
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

    def qm_step(self, graph: DGLGraph) -> torch.Tensor:
        """
        Perform a single step on a QM9 graph.
        """
        t = torch.rand(graph.batch_size, 1, device=graph.device)
        node_batch_idx = get_node_batch_idxs(graph)
        edge_upper_index = get_upper_edge_mask(graph)
        # random positions
        # x_0 = self.manifold.uniform_prior(*graph.ndata["x_1_true"].shape).to(graph.device)
        a_1 = graph.ndata["a_1_true"].unsqueeze(0)
        c_1 = graph.ndata["c_1_true"].unsqueeze(0)
        a_0 = self.manifold.uniform_prior(*a_1.shape).to(graph.device)
        c_0 = self.manifold.uniform_prior(*c_1.shape).to(graph.device)
        edges = graph.edata["e_1_true"]
        edges_flat = edges.reshape(edges.size(0), -1, edges.size(-1))
        e_0 = self.manifold.uniform_prior(*edges_flat.shape).to(graph.device)

        # sample points
        print(a_0.shape, a_1.shape)
        a_t, a_target = self.compute_target(a_0, a_1, t)
        e_t, e_target = self.compute_target(e_0, edges_flat, t)
        c_t, c_target = self.compute_target(c_0, c_1, t)

        graph.ndata["a_t"] = a_t
        graph.ndata["c_t"] = c_t
        graph.edata["e_t"] = e_t.reshape_as(edges)
        ret_dict = self.model(graph, t, node_batch_idx, edge_upper_index)
        a_pred = ret_dict["a"]
        c_pred = ret_dict["c"]
        e_pred = ret_dict["e"].reshape_as(e_target)
        return (
            (a_pred - a_target).square().sum(dim=(-1, -2))
            + (c_pred - c_target).square().sum(dim=(-1, -2))
            + (e_pred - e_target).square().sum(dim=(-1, -2))
        ).mean()


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
            loss = self.qm_step(x_1)
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
        else:
            loss = self.model_step(x_1)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.promoter_eval:
            mse = self.compute_sp_mse(x_1, signal)
            self.test_sp_mse(mse)
            self.log("test/sp-mse", self.test_sp_mse, on_step=False, on_epoch=True, prog_bar=True)
        if False and self.eval_fbd:
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
