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
import schedulefree


from src.sfm import (
    OTSampler,
    compute_exact_loglikelihood,
    estimate_categorical_kl,
    manifold_from_name,
    ot_train_step,
)
from src.models.net import TangentWrapper
from src.data.components.promoter_eval import SeiEval
from src.data.components.fbd import FBD


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
        closed_form_drv: bool = False,
        debug_grads: bool = False,
        inference_steps: int = 100,
        tangent_wrapper: bool = True,
        # enhancer
        eval_fbd: bool = False,
        fbd_every: int = 10,
        mel_or_dna: bool = True,  # if True, then MEL; if not Fly Brain DNA
        fbd_classifier_path: str | None = None,
        # ppl
        eval_ppl: bool = False,
        eval_ppl_every: int = 10,
        normalize_loglikelihood: bool = False,
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
        self.save_hyperparameters(ignore=["fbd", "n_atoms_dist"], logger=False)
        # if basically zero or zero
        self.smoothing = label_smoothing if label_smoothing and label_smoothing > 1e-6 else None
        self.tangent_euler = tangent_euler
        self.closed_form_drv = closed_form_drv
        self.promoter_eval = promoter_eval
        self.manifold = manifold_from_name(manifold)
        self.net = net if not tangent_wrapper else TangentWrapper(self.manifold, net)
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
        if hasattr(self, "val_x_loss"):
            self.val_x_loss.reset()
            self.val_e_loss.reset()

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
            target = torch.zeros_like(x_t)
            target[mask] = self.manifold.log_map(x_0, x_1)[mask]
            target[mask] = self.manifold.parallel_transport(x_0, x_t, target)[mask]
        return x_t, target

    def training_step(
        self, x_1: torch.Tensor | list[torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        signal = None
        if isinstance(x_1, list):
            x_1, signal = x_1
            # Only one of the two signal inputs is used (the first one)
            if len(signal.shape) == 3:
                signal = signal[:, :, 0].unsqueeze(-1)
                loss = self.model_step(x_1, {"signal": signal})
            else:
                loss = self.model_step(x_1, {"cls": signal})
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
        signal = None
        if isinstance(x_1, list):
            x_1, signal = x_1
            # Only one of the two signal inputs is used (the first one)
            if len(signal.shape) == 3:
                signal = signal[:, :, 0].unsqueeze(-1)
                loss = self.model_step(x_1, {"signal": signal})
            else:
                loss = self.model_step(x_1, {"cls": signal})
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

    def test_step(self, x_1: torch.Tensor | list[torch.Tensor], batch_idx: int):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        signal = None
        if isinstance(x_1, list):
            x_1, signal = x_1
            # Only one of the two signal inputs is used (the first one)
            if len(signal.shape) == 3:
                signal = signal[:, :, 0].unsqueeze(-1)
                loss = self.model_step(x_1, {"signal": signal})
            else:
                loss = self.model_step(x_1, {"cls": signal})
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
            self.log("test/ppl", self.test_ppl, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
