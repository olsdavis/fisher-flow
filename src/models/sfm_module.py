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


from src.sfm import (
    OTSampler,
    estimate_categorical_kl,
    manifold_from_name,
    ot_train_step,
)
from src.models.net import TangentWrapper
from src.data.components.promoter_eval import SeiEval
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
        debug_grads: bool = False,
        inference_steps: int = 100,
        datamodule: Any = None,  # in retrobridge
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
        lambda_train: list[int] = [5, 0],
        samples_to_generate: int = 128,
        samples_per_input: int = 5,
    ) -> None:
        """
        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # if basically zero or zero
        self.smoothing = label_smoothing if label_smoothing and label_smoothing > 1e-6 else None
        self.tangent_euler = tangent_euler
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
        self.sp_mse = MeanMetric()
        self.test_sp_mse = MeanMetric()
        self.min_grad = MinMetric()
        self.max_grad = MaxMetric()
        self.mean_grad = MeanMetric()
        self.kl_eval = kl_eval
        self.kl_samples = kl_samples
        self.debug_grads = debug_grads
        self.inference_steps = inference_steps
        self.retrobridge = datamodule is not None

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
            self.lambda_train = lambda_train
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
            # self.val_molecular_metrics = SamplingMolecularMetrics(self.dataset_infos, datamodule.train_smiles,)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(x, t)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.sp_mse.reset()

    def model_step(
        self, x_1: torch.Tensor, signal: torch.Tensor | None = None,
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
            signal=signal,
            closed_form_drv=False,
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
        #retrobridge_utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)[0].E.sum(dim=-1)
        product = product.mask(p_node_mask)

        t = torch.rand(data.batch_size, 1, device=data.x.device)

        # product is prior, reactants is targret
        # now, need to train over product of these buggers

        # encode no edges as 1 in last dimension
        start_e = self.encode_empty_cat(product.E)
        end_e = self.encode_empty_cat(reactants.E)
        edges_t, edges_target = self.compute_target(
            start_e,
            end_e,
            t,
        )
        # same for features
        start_f = self.encode_empty_cat(product.X)
        end_f = self.encode_empty_cat(reactants.X)
        feats_t, feats_target = self.compute_target(
            start_f,
            end_f,
            t,
        )
        noisy_data = {
            "t": t,
            "E_t": edges_t[:, :, :, :-1],  # remove last item
            "X_t": feats_t[:, :, :-1],  # remove last item
            "y": product.y,
            "y_t": product.y,
            "node_mask": r_node_mask,
        }

        # Computing extra features + context and making predictions
        context = product.clone() if self.use_context else None
        extra_data = self.compute_extra_data(noisy_data, context=context)

        pred = self.retrobridge_forward(noisy_data, extra_data, r_node_mask)
        # have two targets, need two projections
        loss_x = (self.manifold.make_tangent(feats_t, pred.X, missing_coordinate=True) - feats_target).square().sum(dim=(-1, -2))
        # reshape for B, K, D shape
        edges_t = edges_t.reshape(edges_t.size(0), -1, edges_t.size(-1))
        pred_reshaped = pred.E.reshape(pred.E.size(0), -1, pred.E.size(-1))
        loss_edges = (
            self.manifold.make_tangent(edges_t, pred_reshaped, missing_coordinate=True) - edges_target.reshape_as(edges_t)
        ).square().sum(dim=(-1, -2))
        loss = (loss_x + loss_edges).mean()
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

    def compute_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes flow-matching target; returns point and target itself.
        """
        with torch.inference_mode(False):
            # https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/model_pl.py
            def cond_u(x0, x1, t):
                path = geodesic(self.manifold.sphere, x0, x1)
                x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
                return x_t, u_t
            x_t, target = vmap(cond_u)(x_0, x_1, t)
        x_t = x_t.squeeze()
        target = target.squeeze()
        # assert self.manifold.all_belong_tangent(x_t, target)
        return x_t, target
    
    def retrobridge_eval(self):
        """Evaluation metrics for retrobridge."""
        samples_left_to_generate = self.samples_to_generate

        samples = []
        grouped_samples = []
        ground_truth = []

        ident = 0

        dataloader = self.trainer.datamodule.val_dataloader()
        for data in tqdm(dataloader, total=samples_left_to_generate // dataloader.batch_size):
            if samples_left_to_generate <= 0:
                break

            data = data.to(self.device)
            bs = len(data.batch.unique())
            to_generate = bs
            batch_groups = []
            ground_truth.extend(
                retrobridge_utils.create_true_reactant_molecules(data, batch_size=bs)
            )
            for _ in range(self.samples_per_input):
                mol_batch = self.sample_molecule(
                    data=data,
                )
                molecule_list = retrobridge_utils.create_pred_reactant_molecules(
                    mol_batch.X, mol_batch.E, data.batch, batch_size=bs,
                )
                samples.extend(molecule_list)
                batch_groups.append(molecule_list)

            ident += to_generate
            samples_left_to_generate -= to_generate

            # Regrouping sampled reactants for computing top-N accuracy
            for mol_idx_in_batch in range(bs):
                mol_samples_group = []
                for batch_group in zip(batch_groups):
                    mol_samples_group.append(batch_group[mol_idx_in_batch])

                assert len(mol_samples_group) == self.samples_per_input
                grouped_samples.append(mol_samples_group)

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

    @torch.inference_mode()
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
        # start!
        for _ in range(self.inference_steps):
            noisy_data = {
                "t": t,
                "E_t": E,
                "X_t": X,
                "y": y,
                "y_t": product.y,
                "node_mask": node_mask,
            }
            extra_data = self.compute_extra_data(noisy_data, context=context)
            pred = self.retrobridge_forward(noisy_data, extra_data, node_mask)
            # put on simplex
            E = self.encode_empty_cat(pred.E)
            X = self.encode_empty_cat(pred.X)
            # prep
            target_edge_shape = (E.size(0), -1, E.size(-1))
            # make a step
            X = self.manifold.exp_map(
                X, self.manifold.make_tangent(X, pred.X, missing_coordinate=True) * dt,
            )
            X = self.manifold.project(X)[:, :, :-1]
            E = E.reshape(target_edge_shape)
            E = self.manifold.exp_map(
                E,
                self.manifold.make_tangent(
                    E,
                    pred.E.reshape((pred.E.size(0), -1, pred.E.size(-1))),
                    missing_coordinate=True,
                ) * dt,
            )
            E = self.manifold.project(E)
            E = E.reshape(*orig_edge_shape[:-1], orig_edge_shape[-1] + 1)[:, :, :, :-1]
            y = pred.y
            t += dt
        # X and E, flattened, are on the sphere; so we can determine the
        # last coordinate that we removed
        # that is useful to determine whether the edge/node is present or
        # not at all
        def to_one_hot(tensor: torch.Tensor):
            orig_shape = tensor.shape
            tensor = tensor.reshape(orig_shape[0], -1, orig_shape[-1])
            remaining = tensor.square().sum(dim=-1, keepdim=True)
            combined = torch.cat([tensor, (1.0 - remaining).sqrt()], dim=-1)
            argmax = combined.argmax(dim=-1)
            ret = F.one_hot(argmax, num_classes=combined.shape[-1])
            ret[argmax == combined.shape[-1] - 1, :] = 0
            return ret[:, :, :-1].reshape(orig_shape)
        return PlaceHolder(
            X=to_one_hot(X), E=to_one_hot(E), y=y,
        ).mask(node_mask)

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
            signal = signal[:, :, 0].unsqueeze(-1)
            loss = self.model_step(x_1, signal)
        elif isinstance(x_1, Batch):
            loss = self.retrobridge_step(x_1)
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
            signal = signal[:, :, 0].unsqueeze(-1)
            loss = self.model_step(x_1, signal)
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
        if self.dataset_infos is not None:
            # evaluate retrobridge
            self.retrobridge_eval()

    def test_step(self, x_1: torch.Tensor | list[torch.Tensor], batch_idx: int):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if isinstance(x_1, list):
            x_1, signal = x_1
            # Only one of the two signal inputs is used (the first one)
            signal = signal[:, :, 0].unsqueeze(-1)
            loss = self.model_step(x_1, signal)
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
