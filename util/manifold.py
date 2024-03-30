"""Some utils for manifolds."""
from abc import ABC, abstractmethod
from functools import partial
import math


import torch
from torch import Tensor, nn
import ot
from einops import rearrange


from util import safe_arccos, usinc


def str_to_ot_method(method: str, reg: float = 0.05, reg_m: float = 1.0, loss: bool = False):
    """
    Returns the `OT` method corresponding to `method`.
    """
    if method == "exact":
        return ot.emd if not loss else ot.emd2
    elif method == "sinkhorn":
        return partial(ot.sinkhorn if not loss else ot.sinkhorn2, reg=reg)
    elif method == "unbalanced":
        assert not loss, "no loss method available"
        return partial(ot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
    elif method == "partial":
        assert not loss, "no loss method available"
        return partial(ot.partial.entropic_partial_wasserstein, reg=reg)
    raise ValueError(f"Unknown method: {method}")


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
            - `t`: the time tensor of dimensions `(B, 1)`.
        
        Returns:
            The geodesic interpolant at time `t`.
        """
        t = t.unsqueeze(-1)
        t = t.repeat(1, x_0.size(1), 1)
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
            t = torch.full((x.size(0), 1), dt * (i + 1), device=x_0.device)
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
        # points
        n = x_0.size(0)
        # if on product space
        prods = 0 if len(x_0.shape) == 2 else x_0.size(1)
        # dimension on product space
        d = x_0.size(-1)

        if prods > 0:
            x_0 = rearrange(x_0, 'b c d -> b (c d)', c=prods, d=d)
            x_1 = rearrange(x_1, 'b c d -> b (c d)', c=prods, d=d)

        x_0 = x_0.repeat_interleave(n, dim=0)
        x_1 = x_1.repeat(n, 1)

        if prods > 0:
            x_0 = rearrange(x_0, 'b (c d) -> b c d', c=prods, d=d)
            x_1 = rearrange(x_1, 'b (c d) -> b c d', c=prods, d=d)
        distances = self.geodesic_distance(x_0, x_1) ** 2
        return distances.reshape(n, n)

    def wasserstein_dist(
        self,
        x_0: Tensor,
        x_1: Tensor,
        method: str = "exact",
        reg: float = 0.05,
        power: int = 2,
    ) -> float:
        """
        Estimates the `power`-Wasserstein distance between the two distributions
        the samples of which are in `x_0` and `x_1`.

        Based on: `https://github.com/DreamFold/FoldFlow/blob/main/FoldFlow/utils/optimal_transport.py`.
        """
        assert power in [1, 2], "power must be either 1 or 2"
        ot_fn = str_to_ot_method(method, reg=reg, loss=True)
        a, b = ot.unif(x_0.shape[0]), ot.unif(x_1.shape[0])
        m = self.pairwise_geodesic_distance(x_0, x_1)
        if power == 2:
            m = m ** 2
        ret = ot_fn(a, b, m.detach().cpu().numpy(), numItermax=1e7)
        if power == 2:
            # for slighlty negative values
            ret = ret if ret > 0.0 else 0.0
            ret = math.sqrt(ret)
        return ret

    @abstractmethod
    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Calculates the Riemannian metric at point `x` between
        `u` and `v`.
        """

    def square_norm_at(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Calculates the square of the norm of `v` at the tangent space of `x`.
        """
        return self.metric(x, v, v)

    @abstractmethod
    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        Calculates the parallel transport of `v` in the tangent plane of `p`
        to that of `q`.

        Parameters:
            - `p`: starting point;
            - `q`: end point;
            - `v`: the vector to transport.
        """


class NSimplex(Manifold):
    """
    Defines an n-simplex (representable in n - 1 dimensions).

    Based on `https://juliamanifolds.github.io`.
    """

    def exp_map(self, p: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.exp_map`.
        """
        s = p.sqrt()
        xs = v / (s + 1e-7) / 2.0
        theta = xs.norm(dim=-1, keepdim=True)
        return (theta.cos() * s + usinc(theta) * xs).square()

    def log_map(self, p: Tensor, q: Tensor) -> Tensor:
        """
        See `Manifold.log_map`.
        """
        rt_prod = (p * q).sqrt()
        dot = rt_prod.sum(dim=-1, keepdim=True)
        dist = 2.0 * torch.arccos(dot)
        denom = (1.0 - dot ** 2).sqrt()
        fact = rt_prod - dot * p
        return (dist / (denom + 1e-7)) * fact

    def geodesic_distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        See `Manifold.geodesic_distance`.
        """
        d = (p * q).sqrt().sum(dim=-1, keepdim=True)
        return 2.0 * safe_arccos(d).sum(dim=1)

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.metric`.
        """
        eps = 1e-7
        div = x.sqrt() + eps
        return ((u / div).abs().clamp(eps, 1e8) * (v / div).abs().clamp(eps, 1e8)).sum(dim=-1, keepdim=True)

    def _sphere_map(self, p: Tensor):
        """
        Maps `p` to the positive orthant of the sphere.
        """
        return p.sqrt()

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.parallel_transport`. Based on the parallel transport of
        `NSphere`.
        """
        sphere = NSphere()
        q_s = self._sphere_map(q)
        y_s = sphere.parallel_transport(
            self._sphere_map(p),
            q_s,
            v / (p.sqrt() + 1e-7),
        )
        return y_s * q_s


class NSphere(Manifold):
    """
    Defines an n-dimensional sphere.
    
    Based on: `https://juliamanifolds.github.io`.
    """

    def exp_map(self, p: Tensor, v: Tensor) -> Tensor:
        raise NotImplementedError()

    def log_map(self, p: Tensor, q: Tensor) -> Tensor:
        raise NotImplementedError()

    def geodesic_distance(self, p: Tensor, q: Tensor) -> Tensor:
        raise NotImplementedError()

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        raise NotImplementedError()

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.parallel_transport`.
        """
        m = p + q
        mnorm2 = m.square().sum(dim=-1, keepdim=True)
        factor = 2.0 * (v * q).sum(dim=-1, keepdim=True) / (mnorm2 + 1e-7)
        return v - m * factor
