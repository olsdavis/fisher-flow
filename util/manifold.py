"""Some utils for manifolds."""
from abc import ABC, abstractmethod
from functools import partial
import math


import torch
from torch import Tensor, nn
import ot
from einops import rearrange


from util import usinc


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
    
    @abstractmethod
    def geodesic_distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Returns the geodesic distance of points `p` and `q` on the manifold.

        Parameters:
            - `p`, `q`: two points on the manifold of dimensions
                `(B, ..., D)`.
        
        Returns:
            The geodesic distance.
        """

    @torch.no_grad()
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
        dt = 1.0 / steps
        x = x_0
        for i in range(steps):
            t = torch.ones((x.size(0),) + (1,) * (len(x_0.shape) - 1)) * dt * (i + 1)
            x = self.exp_map(x, model(x, t) * dt)
        return x

    def pairwise_geodesic_distance(
        self,
        x_0: Tensor,
        x_1: Tensor,
    ) -> Tensor:
        """
        Computes the pairwise distances between `x_0` and `x_1`.
        Based on: `https://github.com/DreamFold/FoldFlow/blob/main/FoldFlow/utils/optimal_transport.py`.
        """
        n = x_0.size(0)
        x0 = rearrange(x_0, 'b c d -> b (c d)', c=3, d=3)
        x1 = rearrange(x_1, 'b c d -> b (c d)', c=3, d=3)
        mega_batch_x0 = rearrange(x0.repeat_interleave(n, dim=0), 'b (c d) -> b c d', c=3, d=3)
        mega_batch_x1 = rearrange(x1.repeat(n, 1), 'b (c d) -> b c d', c=3, d=3)
        distances = self.geodesic_distance(mega_batch_x0, mega_batch_x1)**2
        return distances.reshape(n, n)

    def wassertstein_dist(
        self,
        x_0: Tensor,
        x_1: Tensor,
        method: str = "exact",
        reg: float = 0.05,
        power: int = 2,
    ) -> float:
        """
        Estimates the `power`-Wassertstein distance between the two distributions
        the samples of which are in `x_0` and `x_1`.

        Based on: `https://github.com/DreamFold/FoldFlow/blob/main/FoldFlow/utils/optimal_transport.py`.
        """
        assert power in [1, 2], "power must be either 1 or 2"
        if method == "exact":
            ot_fn = ot.emd2
        elif method == "sinkhorn":
            ot_fn = partial(ot.sinkhorn2, reg=reg)
        else:
            raise NotImplementedError(f"not implemented method: {method}")
        a, b = ot.unif(x_0.shape[0]), ot.unif(x_1.shape[0])
        m = self.pairwise_geodesic_distance(x_0, x_1)
        if power == 2:
            m = m ** 2
        ret = ot_fn(a, b, m.detach().cpu().numpy(), numItermax=1e7)
        if power == 2:
            ret = math.sqrt(ret)
        return ret


class NSimplex(Manifold):
    """
    Defines an n-simplex (representable in n - 1 dimensions).

    Based on `https://juliamanifolds.github.io`.
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
    
    def geodesic_distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        See `Manifold.geodesic_distance`.
        """
        return 2.0 * torch.arccos((p * q).sqrt().sum(dim=-1, keepdim=True))
