"""Copyright (c) Meta Platforms, Inc. and affiliates."""
from functools import partial
import math
from typing import Any, List
import traceback
import numpy as np

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.func import vjp, jvp, vmap, jacrev
from torch_ema import ExponentialMovingAverage
from torchmetrics import MeanMetric
from lightning import LightningModule
from torchdiffeq import odeint
from geoopt import (
    Sphere as geoopt_Sphere,
    Euclidean,
    ProductManifold,
)
from src.data.components.promoter_eval import SeiEval


class ProjectToTangent(torch.nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield, manifold, metric_normalize):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold
        self.metric_normalize = metric_normalize

    def forward(self, t, x, signal: torch.Tensor | None = None):
        x = self.manifold.projx(x)
        v = self.vecfield(t, x, signal=signal)
        v = self.manifold.proju(x, v.reshape(x.size(0), -1))

        if self.metric_normalize and hasattr(self.manifold, "metric_normalized"):
            v = self.manifold.metric_normalized(x, v)

        return v


def geodesic(manifold, start_point, end_point):
    if start_point.dim() == 2:
        start_point = start_point.unsqueeze(0)
        end_point = end_point.unsqueeze(0)
    start_point = start_point.view(start_point.shape[0], -1)
    end_point = end_point.view(end_point.shape[0], -1)
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


@torch.no_grad()
def projx_integrator(
    manifold, odefunc, x0, t, method="euler", projx=True, local_coords=False, pbar=False
):
    step_fn = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
    }[method]

    xts = [x0]
    vts = []

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    xt = x0
    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t0, xt)
        xt = step_fn(
            odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None
        )
        if projx:
            xt = manifold.projx(xt)
        vts.append(vt)
        xts.append(xt)
    vts.append(odefunc(t1, xt))
    return torch.stack(xts), torch.stack(vts)


@torch.no_grad()
def projx_integrator_return_last(
    manifold, odefunc, x0, t, method="euler", projx=True, local_coords=False, pbar=False
):
    """Has a lower memory cost since this doesn't store intermediate values."""

    step_fn = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
    }[method]

    xt = x0

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t0, xt)
        xt = step_fn(
            odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None
        )
        if projx:
            xt = manifold.projx(xt)
    return xt


def euler_step(odefunc, xt, vt, t0, dt, manifold=None):
    if manifold is not None:
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt


def midpoint_step(odefunc, xt, vt, t0, dt, manifold=None):
    half_dt = 0.5 * dt
    if manifold is not None:
        x_mid = xt + half_dt * vt
        v_mid = odefunc(t0 + half_dt, x_mid)
        v_mid = manifold.transp(x_mid, xt, v_mid)
        return manifold.expmap(xt, dt * v_mid)
    else:
        x_mid = xt + half_dt * vt
        return xt + dt * odefunc(t0 + half_dt, x_mid)


def rk4_step(odefunc, xt, vt, t0, dt, manifold=None):
    k1 = vt
    if manifold is not None:
        raise NotImplementedError
    else:
        k2 = odefunc(t0 + dt / 3, xt + dt * k1 / 3)
        k3 = odefunc(t0 + dt * 2 / 3, xt + dt * (k2 - k1 / 3))
        k4 = odefunc(t0 + dt, xt + dt * (k1 - k2 + k3))
        return xt + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


class Sphere(geoopt_Sphere):
    def transp(self, x, y, v):
        denom = 1 + self.inner(x, x, y, keepdim=True)
        res = v - self.inner(x, y, v, keepdim=True) / denom * (x + y)
        cond = denom.gt(1e-3)
        return torch.where(cond, res, -v)

    def uniform_logprob(self, x):
        dim = x.shape[-1]
        return torch.full_like(
            x[..., 0],
            math.lgamma(dim / 2) - (math.log(2) + (dim / 2) * math.log(math.pi)),
        )

    def random_base(self, *args, **kwargs):
        return self.random_uniform(*args, **kwargs)

    def base_logprob(self, *args, **kwargs):
        return self.uniform_logprob(*args, **kwargs)


def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def output_and_div(vecfield, x, v=None, div_mode="exact"):
    if div_mode == "exact":
        dx = vecfield(x)
        div = vmap(div_fn(vecfield))(x)
    else:
        dx, vjpfunc = vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class FlipWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, x, signal=None):
        return self.net(x, t, signal=signal)


class ManifoldFMLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        # TODO: or not? manifold: str = "sphere",
        # TODO: ot_method: str = "exact",
        kl_eval: bool = False,
        promoter_eval: bool = False,
        kl_samples: int = 512_000,
        ema: bool = False,
        ema_decay: float = 0.9,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        local_coords: bool = False,
        eval_projx: bool = False,
        div_mode: str = "exact",
        normalize_ll: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.manifold = ProductManifold(*[(Sphere(), net.dim) for _ in range(net.k)])
        self.model = ProjectToTangent(FlipWrapper(net), self.manifold, False).to(self.device)  # TODO: add option for norm
        self.dim = net.dim * net.k
        self.atol = atol
        self.rtol = rtol
        self.local_coords = local_coords
        self.eval_projx = eval_projx
        self.div_mode = div_mode
        self.normalize_ll = normalize_ll
        self.promoter_eval = promoter_eval
        self.kl_eval = kl_eval
        self.kl_samples = kl_samples
        self.promoter_eval = promoter_eval

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metric = MeanMetric()
        self.val_metric = MeanMetric()
        self.test_metric = MeanMetric()
        self.sp_mse = MeanMetric()
        if ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay).to(self.device)
        else:
            self.ema = None

    def setup(self, stage: str):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.get("compile", False) and stage == "fit":
            self.model = torch.compile(self.model)

    @property
    def vecfield(self):
        return self.model

    @torch.no_grad()
    def compute_cost(self, batch):
        if isinstance(batch, dict):
            x0 = batch["x0"]
        else:
            x0 = (
                self.manifold.random_combined(batch.shape[0], np.prod(batch.shape[1:]))
                # .reshape(batch.shape[0], self.dim)
                .to(batch.device)
            )

        # Solve ODE.
        x1 = odeint(
            self.vecfield,
            x0,
            t=torch.linspace(0, 1, 2).to(x0.device),
            atol=self.atol,
            rtol=self.rtol,
        )[-1]

        x1 = self.manifold.projx(x1)

        return self.manifold.dist(x0, x1)

    @torch.no_grad()
    def sample(self, n_samples: int, x0: torch.Tensor | None = None, signal: torch.Tensor | None = None):
        """
        Samples `n_samples` points from the model with starting point `x0`.
        """
        if x0 is None:
            # Sample from base distribution.
            x0 = (
                self.manifold.random_combined(n_samples, self.dim)
                .reshape(n_samples, self.dim)
                .to(self.device)
            )

        vecfield = (
            partial(self.vecfield, signal=signal) if signal is not None else self.vecfield
        )

        # Solve ODE.
        if not self.eval_projx and not self.local_coords:
            # If no projection, use adaptive step solver.
            x1 = odeint(
                vecfield,
                x0,
                t=torch.linspace(0, 1, 2).to(self.device),
                atol=self.atol,
                rtol=self.rtol,
                options={"min_step": 1e-5}
            )[-1]
        else:
            # If projection, use 1000 steps.
            x1 = projx_integrator_return_last(
                self.manifold,
                vecfield,
                x0,
                t=torch.linspace(0, 1, 1001).to(self.device),
                method="euler",
                projx=self.eval_projx,
                local_coords=self.local_coords,
                pbar=True,
            )
        # comment existing in old code base:
        # x1 = self.manifold.projx(x1)
        return x1

    @torch.no_grad()
    def sample_all(self, n_samples, device, x0=None):
        if x0 is None:
            # Sample from base distribution.
            x0 = (
                self.manifold.random_base(n_samples, self.dim)
                .reshape(n_samples, self.dim)
                .to(device)
            )

        # Solve ODE.
        xs, _ = projx_integrator(
            self.manifold,
            self.vecfield,
            x0,
            t=torch.linspace(0, 1, 1001).to(device),
            method="euler",
            projx=True,
            pbar=True,
        )
        return xs

    @torch.no_grad()
    def compute_exact_loglikelihood(
        self,
        batch: torch.Tensor,
        t1: float = 1.0,
        return_projx_error: bool = False,
        num_steps=1000,
    ):
        """Computes the negative log-likelihood of a batch of data."""

        try:
            nfe = [0]

            with torch.inference_mode(mode=False):
                v = None
                if self.div_mode == "rademacher":
                    v = torch.randint(low=0, high=2, size=batch.shape).to(batch) * 2 - 1

                def odefunc(t, tensor):
                    nfe[0] += 1
                    t = t.to(tensor)
                    x = tensor[..., : self.dim]
                    vecfield = lambda x: self.vecfield(t, x)
                    dx, div = output_and_div(vecfield, x, v=v, div_mode=self.div_mode)

                    if hasattr(self.manifold, "logdetG"):

                        def _jvp(x, v):
                            return jvp(self.manifold.logdetG, (x,), (v,))[1]

                        corr = vmap(_jvp)(x, dx)
                        div = div + 0.5 * corr.to(div)

                    div = div.reshape(-1, 1)
                    del t, x
                    return torch.cat([dx, div], dim=-1)

                # Solve ODE on the product manifold of data manifold x euclidean.
                product_man = ProductManifold(
                    (self.manifold, self.dim), (Euclidean(), 1)
                )
                state1 = torch.cat([batch, torch.zeros_like(batch[..., :1])], dim=-1)

                with torch.no_grad():
                    if not self.eval_projx and not self.local_coords:
                        # If no projection, use adaptive step solver.
                        state0 = odeint(
                            odefunc,
                            state1,
                            t=torch.linspace(t1, 0, 2).to(batch),
                            atol=self.atol,
                            rtol=self.rtol,
                            method="dopri5",
                            options={"min_step": 1e-5},
                        )[-1]
                    else:
                        # If projection, use 1000 steps.
                        state0 = projx_integrator_return_last(
                            product_man,
                            odefunc,
                            state1,
                            t=torch.linspace(t1, 0, num_steps + 1).to(batch),
                            method="euler",
                            projx=self.eval_projx,
                            local_coords=self.local_coords,
                            pbar=True,
                        )

                # log number of function evaluations
                self.log("nfe", nfe[0], prog_bar=True, logger=True)

                x0, logdetjac = state0[..., : self.dim], state0[..., -1]
                x0_ = x0
                x0 = self.manifold.projx(x0)

                # log how close the final solution is to the manifold.
                integ_error = (x0[..., : self.dim] - x0_[..., : self.dim]).abs().max()
                self.log("integ_error", integ_error)

                logp0 = self.manifold.base_logprob(x0)
                logp1 = logp0 + logdetjac

                if self.normalize_ll:
                    logp1 = logp1 / self.dim

                # Mask out those that left the manifold
                masked_logp1 = logp1

                if return_projx_error:
                    return logp1, integ_error
                else:
                    return masked_logp1
        except:
            traceback.print_exc()
            return torch.zeros(batch.shape[0]).to(batch)

    def loss_fn(self, batch: torch.Tensor | list[torch.Tensor]):
        return self.rfm_loss_fn(batch)

    def rfm_loss_fn(self, batch: torch.Tensor | list[torch.Tensor]):
        if isinstance(batch, list):
            x1, signal = batch
            vecfield = partial(self.vecfield, signal=signal)
        else:
            x1 = batch
            vecfield = self.vecfield
        x0 = self.manifold.random_combined(
            x1.shape[0], np.prod(x1.shape[1:])
        ).to(x1).reshape(x1.shape)

        N = x1.shape[0]

        t = torch.rand(N).reshape(-1, 1).to(x1)

        with torch.inference_mode(False):
            def cond_u(x0, x1, t):
                path = geodesic(self.manifold, x0, x1)
                x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
                return x_t, u_t
            x_t, u_t = vmap(cond_u)(x0, x1, t)
        x_t = x_t.reshape(N, self.dim)
        u_t = u_t.reshape(N, self.dim)

        v = vecfield(t, x_t)
        diff = v.view(N, self.dim) - u_t
        return self.manifold.inner(x_t, diff, diff).mean() / self.dim

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.loss_fn(batch)

        if torch.isfinite(loss):
            # log train metrics
            self.train_metric.update(loss)
            self.log("train/loss", self.train_metric, on_step=False, on_epoch=True)
            return loss
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss.item()}.")

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metric.reset()
        self.sp_mse.reset()

    def on_train_epoch_end(self, *args):
        pass

    def validation_step(self, batch, batch_idx: int):
        loss = self.loss_fn(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.val_metric.update(torch.tensor(0.0))
        if self.promoter_eval:
            self.estimate_sp_mse(*batch, batch_idx=batch_idx)

    def on_validation_epoch_end(self, *args):
        self.val_metric.reset()
        if self.kl_eval:
            kl = self.estimate_categorical_kl(
                self.trainer.val_dataloaders.dataset.probs.to(self.device),
                self.kl_samples // 10,
                self.hparams.get("kl_batch", 2048),
            )
            self.log("val/kl", kl, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        # logprob = self.compute_exact_loglikelihood(batch)
        # loss = -logprob.mean()
        # batch_size = batch.shape[0]
        self.test_metric.update(self.loss_fn(batch))
        self.log("test/loss", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs: List[Any]):
        self.test_metric.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.model.parameters())
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

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update(self.model.parameters())

    @torch.inference_mode()
    def estimate_categorical_kl(
        self,
        real_dist: torch.Tensor,
        n: int,
        batch: int = 512,
        sampling_mode: str = "max",
        silent: bool = True,
    ) -> float:
        """
        Estimates the categorical KL divergence between points produced by the
        model `model` and `real_dist`. Done by sampling `n` points and estimating
        thus the different probabilities.

        Parameters:
            - `model`: the model;
            - `manifold`: manifold over which the model was trained;
            - `real_dist`: the real distribution tensor of shape `(k, d)`;
            - `n`: the number of points over which the estimate should be done;
            - `batch`: the number of points to draw per batch;
            - `inference_steps`: the number of steps to take for inference;
            - `sampling_mode`: how to sample points; if "sample", then samples
                from the distribution produced by the model; if "max" then takes
                the argmax of the distribution.

        Returns:
            An estimate of the KL divergence of the model's distribution from
            the real distribution, i.e., "KL(model || real_dist)".
        """
        assert sampling_mode in ["sample", "max"], "not a valid sampling mode"
        # init acc
        acc = torch.zeros_like(real_dist, device=real_dist.device).float()
        self.eval()
        to_sample = [batch] * (n // batch)
        if n % batch != 0:
            to_sample += [n % batch]
        for draw in to_sample:
            # x_0 = manifold.uniform_prior(
            #     draw, real_dist.size(0), real_dist.size(1),
            # ).to(real_dist.device)
            # x_1 = manifold.tangent_euler(x_0, model, inference_steps, tangent=tangent)
            x_1 = self.sample(draw).reshape(-1, *real_dist.shape)
            # x_1 = manifold.send_to(x_1, NSimplex)
            if sampling_mode == "sample":
                # TODO: remove or fix for Categorical
                raise NotImplementedError("Sampling from Dirichlet not implemented")
                # dist = Dirichlet(x_1)
                # samples = dist.sample()
                # acc += samples.sum(dim=0)
            else:
                samples = F.one_hot(
                    x_1.argmax(dim=-1),
                    real_dist.size(-1),
                ).float()
                acc += samples.sum(dim=0)

        acc /= acc.sum(dim=-1, keepdim=True)
        if not silent:
            print(acc)
        ret = (acc * (acc.log() - real_dist.log())).sum(dim=-1).mean().item()
        return ret

    @torch.inference_mode()
    def estimate_sp_mse(self, x_1: torch.Tensor, signal: torch.Tensor, batch_idx: int):
        pred = self.sample(x_1.shape[0], signal=signal).reshape(*x_1.shape)
        mx = torch.argmax(x_1, dim=-1)
        one_hot = F.one_hot(mx, num_classes=4)
        mse = SeiEval().eval_sp_mse(pred, one_hot, batch_idx)
        self.sp_mse(mse)
        self.log("val/sp-mse", self.sp_mse, on_step=False, on_epoch=True, prog_bar=True)
