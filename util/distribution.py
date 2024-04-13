"""Distributional utils."""
import random
import torch
from torch import Tensor, nn
from torch.distributions.dirichlet import Dirichlet
import numpy as np
from util import Manifold, NSimplex, NSphere


def set_seeds(seed: int = 0):
    """
    Sets the seeds for torch, numpy and random to `seed`.
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def estimate_categorical_kl(
    model: nn.Module,
    manifold: Manifold,
    real_dist: Tensor,
    n: int,
    batch: int = 512,
    inference_steps: int = 100,
    sampling_mode: str = "sample",
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
    acc = torch.zeros_like(real_dist, device=real_dist.device)

    model.eval()
    to_draw = n
    while to_draw > 0:
        draw = min(batch, to_draw)
        x_0 = manifold.uniform_prior(
            draw, real_dist.size(0), real_dist.size(1),
        ).to(real_dist.device)
        x_1 = manifold.tangent_euler(x_0, model, inference_steps)
        if isinstance(manifold, NSphere):
            x_1 = NSimplex().inv_sphere_map(x_1)
        if sampling_mode == "sample":
            dist = Dirichlet(x_1)
            samples = dist.sample()
            acc += samples.sum(dim=0)
        else:
            samples = torch.nn.functional.one_hot(
                x_1.argmax(dim=-1),
                real_dist.size(-1),
            )
            acc += samples.sum(dim=0)
        to_draw -= draw
        del x_0
        del x_1
        del samples

    acc /= float(n)
    ret = (acc * (acc.log() - real_dist.log())).sum().item()
    del acc
    return ret
