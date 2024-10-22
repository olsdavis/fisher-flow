"""Distributional utils."""
import math
import random
import traceback
import torch
from torch import Tensor, nn
import numpy as np
from scipy.linalg import sqrtm
import tqdm
from torchdiffeq import odeint
from geoopt import Euclidean, Manifold as GManifold, ProductManifold
from src.sfm import Manifold, NSimplex


def _output_and_div(vecfield, x, v=None, div_mode="exact"):
    # From: https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/model_pl.py#L45
    def div_fn(u):
        """Accepts a function u:R^D -> R^D."""
        J = torch.func.jacrev(u)
        return lambda x: torch.trace(J(x))
    if div_mode == "exact":
        dx = vecfield(x)
        div = torch.vmap(div_fn(vecfield))(x)
    else:
        dx, vjpfunc = torch.func.vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


def set_seeds(seed: int = 0):
    """
    Sets the seeds for torch, numpy and random to `seed`.
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def uniform_logprob(x):
    dim = x.shape[-1]
    return torch.full_like(
        x[..., 0],
        math.lgamma(dim / 2) - (math.log(2) + (dim / 2) * math.log(math.pi)) + dim * math.log(2),
    ).sum(dim=-1)  # sum logs over product space


@torch.inference_mode()
def estimate_categorical_kl(
    model: nn.Module,
    manifold: Manifold,
    real_dist: Tensor,
    n: int,
    batch: int = 512,
    inference_steps: int = 100,
    sampling_mode: str = "max",
    silent: bool = False,
    tangent: bool = True,
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
    acc = torch.zeros_like(real_dist, device=real_dist.device).int()

    model.eval()
    to_sample = [batch] * (n // batch)
    if n % batch != 0:
        to_sample += [n % batch]
    for draw in (tqdm.tqdm(to_sample) if not silent else to_sample):
        x_0 = manifold.uniform_prior(
            draw, real_dist.size(0), real_dist.size(1),
        ).to(real_dist.device)
        x_1 = manifold.tangent_euler(x_0, model, inference_steps, tangent=tangent)
        x_1 = manifold.send_to(x_1, NSimplex)
        if sampling_mode == "sample":
            # TODO: remove or fix for Categorical
            raise NotImplementedError("Sampling from Dirichlet not implemented")
            # dist = Dirichlet(x_1)
            # samples = dist.sample()
            # acc += samples.sum(dim=0)
        else:
            samples = nn.functional.one_hot(
                x_1.argmax(dim=-1),
                real_dist.size(-1),
            )
            acc += samples.sum(dim=0)
    acc = acc.float()
    acc /= n
    # acc.clamp_min_(1e-12)
    if not silent:
        print(acc)
    ret = (acc * (acc.log() - real_dist.log())).sum(dim=-1).mean().item()
    return ret


# The following is adapted from: https://github.com/facebookresearch/riemannian-fm
def _euler_step(odefunc, xt, vt, t0, dt, manifold=None):
    if manifold is not None:
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt


@torch.no_grad()
def _projx_integrator_return_last(
    manifold, odefunc, x0, t, method="euler", projx=True, local_coords=False, pbar=False
):
    """Has a lower memory cost since this doesn't store intermediate values."""

    step_fn = {
        "euler": _euler_step,
    }[method]

    xt = x0

    t0s = t[:-1]
    if pbar:
        t0s = tqdm.tqdm(t0s)

    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t0, xt)
        xt = step_fn(
            odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None
        )
        if projx:
            xt = manifold.projx(xt)
    return xt


@torch.no_grad()
def compute_exact_loglikelihood(
    model: nn.Module,
    batch: Tensor,
    manifold: GManifold,
    t1: float = 1.0,
    num_steps: int = 1000,
    div_mode: str = "rademacher",
    local_coords: bool = False,
    eval_projx: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    normalize_loglikelihood: bool = False,
) -> Tensor:
    """Computes the negative log-likelihood of a batch of data."""
    # Based on https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/model_pl.py#L449

    try:
        with torch.inference_mode(mode=False):
            v = None
            if div_mode == "rademacher":
                v = torch.randint(low=0, high=2, size=batch.shape).to(batch) * 2 - 1

            def odefunc(t, tensor):
                # t = t.to(tensor)
                x = tensor[..., : batch.size(-1)]
                vecfield = lambda x: model(x, t)
                dx, div = _output_and_div(vecfield, x, v=v, div_mode=div_mode)

                if hasattr(manifold, "logdetG"):

                    def _jvp(x, v):
                        return torch.func.jvp(manifold.logdetG, (x,), (v,))[1]

                    corr = torch.func.vmap(_jvp)(x, dx)
                    div = div + 0.5 * corr#.to(div)

                # div = div.view(-1, 1)
                div = div[..., None]
                del t, x
                return torch.cat([dx, div], dim=-1)

            # Solve ODE on the product manifold of data manifold x euclidean.
            product_man = ProductManifold(
                (manifold, batch.size(-1)), (Euclidean(), 1)
            )
            state1 = torch.cat([batch, torch.zeros_like(batch[..., :1])], dim=-1)

            with torch.no_grad():
                if not eval_projx and not local_coords:
                    # If no projection, use adaptive step solver.
                    state0 = odeint(
                        odefunc,
                        state1,
                        t=torch.linspace(0, t1, 2).to(batch),
                        atol=atol,
                        rtol=rtol,
                        method="dopri5",
                        options={"min_step": 1e-5},
                    )[-1]
                else:
                    # If projection, use 1000 steps.
                    state0 = _projx_integrator_return_last(
                        product_man,
                        odefunc,
                        state1,
                        t=torch.linspace(0, t1, num_steps + 1).to(batch),
                        method="euler",
                        projx=eval_projx,
                        local_coords=local_coords,
                        pbar=True,
                    )

            x0, logdetjac = state0[..., : batch.size(-1)], state0[..., -1]
            # x0_ = x0
            x0 = manifold.projx(x0).abs()

            # log how close the final solution is to the manifold.
            # integ_error = (x0[..., : self.dim] - x0_[..., : self.dim]).abs().max()
            # self.log("integ_error", integ_error)

            # logp0 = manifold.base_logprob(x0)
            logp0 = uniform_logprob(x0)
            logp1 = logp0 + logdetjac.sum(dim=-1)

            if normalize_loglikelihood:
                logp1 = logp1 / np.prod(batch.shape[1:])

            # Mask out those that left the manifold
            masked_logp1 = logp1
            return masked_logp1
    except:
        traceback.print_exc()
        return torch.zeros(batch.shape[0]).to(batch)


def get_wasserstein_dist(embeds1, embeds2):
    # Taken from: https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/flow_utils.py#L38
    if np.isnan(embeds2).any() or np.isnan(embeds1).any() or len(embeds1) == 0 or len(embeds2) == 0:
        return float('nan')
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist
