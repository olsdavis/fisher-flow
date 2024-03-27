"""Some maths utils."""
import torch
from torch import Tensor


def usinc(theta: Tensor) -> Tensor:
    """Unnormalized sinc."""
    return torch.sin(theta) / theta
