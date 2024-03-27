"""Some utils for manifolds."""
from abc import ABC, abstractmethod


import torch
from torch import Tensor, nn


class Manifold(ABC):
    """
    Defines a few essential functions for manifolds.
    """

    @abstractmethod
    def exp_map(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Defines the exponential map at `p` in the direction `v`.

        Parameters:
            - `p`: the point on the manifold at which the map should be taken,
                of dimensions `(B, ..., D)`.
            - `v`: the direction of the map, same dimensions as `p`.

        Returns:
            The exponential map.
        """
    
    @abstractmethod
    def log_map(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Defines the logarithmic map from `p` to `q`.

        Parameters:
            - `p`, `q`: two points on the manifold of dimensions 
                `(B, ..., D)`.
        
        Returns:
            The logarithmic map.
        """

    def geodesic_interpolant(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Returns the geodesic interpolant at time `t`, i.e.,
        `exp_{x_0}(t log_{x_0}(x_1))`.

        Parameters:
            - `x_0`, `x_1`: two points on the manifold of dimensions
                `(B, ..., D)`.
            - `t`: the time tensor of dimensions `(B)` or `(B, 1)`.
        
        Returns:
            The geodesic interpolant at time `t`.
        """
        if len(t.shape) < 2:
            t = t.unsqueeze(-1)
        return self.exp_map(x_0, t * self.log_map(x_0, x_1))
    
    @torch.no_grad()
    def tangent_euler(
        self,
        x_0: Tensor,
        model: nn.Module,
        steps: int,
    ) -> list[Tensor]:
        """
        Applies Euler integration on the manifold for the field defined
        by `model`.
        """
        raise NotImplementedError("TODO")


def usinc(theta: Tensor) -> Tensor:
    """Unnormalized sinc."""
    return torch.sin(theta) / theta


class NSimplex(Manifold):
    """
    Defines an n-simplex (representable in n - 1 dimensions).
    """

    def exp_map(self, p: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.exp_map`.
        """
        s = torch.sqrt(p)
        xs = v / s / 2.0
        theta = xs.norm(dim=-1, keepdim=True)
        return (torch.cos(theta) * s + usinc(theta) * xs).square()
    
    def log_map(self, p: Tensor, q: Tensor) -> Tensor:
        """
        See `Manifold.log_map`.
        """
        rt_prod = (p * q).sqrt()
        dot = rt_prod.sum(dim=-1, keepdim=True)
        dist = 2.0 * torch.arccos(dot)
        denom = (1.0 - dot ** 2).sqrt()
        fact = rt_prod - dot * p
        return (dist / denom) * fact
