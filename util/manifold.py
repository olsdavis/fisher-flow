"""Some utils for manifolds."""
from abc import ABC, abstractmethod


import torch
from torch import Tensor


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


class NSimplex(Manifold):
    """
    Defines an n-simplex (representable in n - 1 dimensions).
    """

    def exp_map(self, p: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.exp_map`.
        """
        vp = v / p.sqrt()
        vpn = vp.norm(dim=-1, keepdim=True)

        # calculate terms
        cst = 0.5 * (p + vp.square() / vpn.square())
        cos_term = 0.5 * (p - vp.square() / vpn.square()) * torch.cos(vpn)
        sin_term = (vp / vpn) * p.sqrt() * torch.sin(vpn)
        return cst + cos_term + sin_term
    
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
