"""Some maths utils."""
import torch
from torch import Tensor


@torch.jit.script
def usinc(theta: Tensor) -> Tensor:
    """Unnormalized sinc."""
    return torch.sinc(theta / torch.pi)


def safe_arccos(x: Tensor) -> Tensor:
    """A safe version of `x.arccos()`."""
    return x.clamp(-1.0, 1.0).acos()


__f_dot = torch.vmap(torch.vmap(torch.dot))


def fast_dot(u: Tensor, v: Tensor, keepdim: bool = True) -> Tensor:
    """A faster and unified version of dot products."""
    # ret = __f_dot(p, q)
    # if keepdim:
    #     ret = ret.unsqueeze(-1)
    # return ret
    ret = torch.einsum("bnd,bnd->bn", u, v)
    if keepdim:
        ret = ret.unsqueeze(-1)
    return ret
