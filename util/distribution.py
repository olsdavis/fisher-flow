"""Distributional utils."""
import random
import torch
from torch import Tensor
import numpy as np


def generate_dirichlet(n: int, alpha: list[float]) -> Tensor:
    """
    Generates `n` points generated from a Dirichlet distribution of
    parameter `alpha`.
    """
    return torch.stack([np.random.dirichlet(alpha) for _ in range(n)])


def generate_dirichlet_mixture(n: int, *alphas: list[float]) -> Tensor:
    """
    Generates `n` points from the even mixture of the Dirichlet distributions
    of parameter `alpha` for each `alpha` in `alphas`.
    """
    assert alphas, "there must be at least one set of alpha parameters"
    return torch.stack([
        np.random.dirichlet(random.choice(alphas)) for _ in range(n)
    ])
