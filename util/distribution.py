"""Distributional utils."""
import random
import torch
from torch import Tensor
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
import numpy as np


def set_seeds(seed: int = 0):
    """
    Sets the seeds for torch, numpy and random to `seed`.
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def generate_dirichlet_product(n: int, k: int, d: int) -> Tensor:
    """
    Generates `n` points from a product of `k` `d`-simplex Dirichlet uniform
    distributions.
    """
    alphas = [[1 for _ in range(d)] for _ in range(k)]
    return Dirichlet(torch.Tensor(alphas)).sample((n,))


def generate_dirichlet(n: int, alpha: list[float]) -> Tensor:
    """
    Generates `n` points generated from a Dirichlet distribution of
    parameter `alpha`.
    """
    return Dirichlet(torch.Tensor(alpha)).sample((n,))


def generate_dirichlet_mixture(n: int, *alphas: list[float]) -> Tensor:
    """
    Generates `n` points from the even mixture of the Dirichlet distributions
    of parameter `alpha` for each `alpha` in `alphas`.
    """
    assert alphas, "there must be at least one set of alpha parameters"
    dist = torch.stack([torch.Tensor(alpha) for alpha in alphas])
    even = torch.ones((dist.size(0),)) / dist.size(0)
    return MixtureSameFamily(
        component_distribution=Dirichlet(dist),
        mixture_distribution=Categorical(even),
    ).sample((n,))


def estimate_categorical_kl(points: Tensor, real_dist: Tensor) -> float:
    """
    Returns an estimate of the KL divergence of the empirical distribution
    of `points` from `real_dist`, i.e., "KL(points || real_dist)".

    Parameters:
        - `points`: the points from which to estimate the Dirichlet distribution;
            the Tensor is of shape `(B, K, D)` defining probability distributions
            on each of the `BK` `D`-simplices;
        - `real_dist`: the real probability distribution probabilities, a Tensor
            of dimension `(K, D)`.

    Using the same method as https://github.com/HannesStark/dirichlet-flow-matching.
    """
    cat = Categorical(points)
    samples = cat.sample()
    samples = torch.nn.functional.one_hot(samples, num_classes=points.size(-1))
    emp_dist = samples.mean(dim=0)
    return emp_dist * (emp_dist.log() - real_dist.log()).sum()
