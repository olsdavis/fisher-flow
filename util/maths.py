"""Some maths utils."""
import torch
from torch import Tensor


def usinc(theta: Tensor, eps: float = 1e-7) -> Tensor:
    """Unnormalized sinc."""

    return torch.where(
        torch.abs(theta) < eps,
        1.0,  # sinc(0) = 1, by definition
        torch.where(
            torch.isinf(theta),
            0.0,  # 0 at infinity
            theta.sin() / theta,
        )
    )


def safe_arccos(x: Tensor, eps: float = 1e-6) -> Tensor:
    """A safe version of `x.arccos()`."""
    return torch.where(
        torch.abs(x - 1.0) < eps,
        torch.zeros_like(x, device=x.device),
        torch.where(
            torch.abs(x + 1.0) < eps,
            torch.ones_like(x, device=x.device) * torch.pi,
            x.arccos(),
        )
    )
