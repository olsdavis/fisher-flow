"""Some utils for manifolds."""
from abc import ABC, abstractmethod
from functools import partial
import math
from typing import Type


import torch
from torch.distributions import Dirichlet
from torch import Tensor, nn
import ot
from geoopt import Sphere as GSphere
from einops import rearrange


from src.sfm import (
    PowerSpherical,
    fast_dot,
    safe_arccos,
    usinc,
)


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


def default_perturbation_schedule(t: Tensor) -> Tensor:
    """Default quadratic perturbation schedule."""
    return t * (1.0 - t)


def metropolis_sphere_perturbation(
    x: Tensor,
    scale: Tensor,
) -> Tensor:
    """
    Metropolis-Hastings for a perturbation on the positive orthant of the sphere.
    """
    # mask = torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=torch.bool)
    # Metroplis-Hastings
    # while not mask.all():
    #     import ipdb; ipdb.set_trace()
    #     select = ~mask.expand_as(x)
    #     x[select] = PowerSpherical(x, scale).sample()[select]
    #     mask = (x >= 0.0).all(dim=-1, keepdim=True)
    while len(scale.shape) < len(x.shape) - 1:
        scale = scale.unsqueeze(-1)
    return PowerSpherical(x, scale.expand(x.shape[:-1])).sample().abs()


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
        # assert self.all_belong(x_0)
        # assert self.all_belong(x_1)
        t = t.unsqueeze(-1)
        x_t = self.exp_map(x_0, t * self.log_map(x_0, x_1))
        return self.project(x_t)

    @torch.inference_mode()
    def tangent_euler(
        self,
        x_0: Tensor,
        model: nn.Module,
        steps: int,
        tangent: bool = True,
        stochastic: bool = False,
    ) -> Tensor:
        """
        Applies Euler integration on the manifold for the field defined
        by `model`.

        Parameters:
            - `x_0`: the starting point;
            - `model`: the field;
            - `steps`: the number of steps;
            - `tangent`: if `True`, performs tangent Euler integration;
                otherwise performs classical Euler integration.
        """
        dt = torch.tensor(1.0 / steps, device=x_0.device)
        x = x_0
        t = torch.zeros((x.size(0), 1), device=x_0.device, dtype=x_0.dtype)
        for _ in range(steps):
            if stochastic:
                x = metropolis_sphere_perturbation(x, default_perturbation_schedule(t))
            if tangent:
                x = self.exp_map(x, model(x=x, t=t) * dt)
            else:
                x = x + model(x=x, t=t) * dt
            x = self.project(x)
            t += dt
        return x

    @torch.no_grad()
    def pairwise_geodesic_distance(
        self,
        x_0: Tensor,
        x_1: Tensor,
    ) -> Tensor:
        """
        Computes the pairwise distances between `x_0` and `x_1`.
        Based on: `https://github.com/DreamFold/FoldFlow/blob/main/FoldFlow/utils/optimal_transport.py`.
        """
        n, prods, d = x_0.shape

        x_0 = rearrange(x_0, 'b c d -> b (c d)', c=prods, d=d)
        x_1 = rearrange(x_1, 'b c d -> b (c d)', c=prods, d=d)

        x_0 = x_0.repeat_interleave(n, dim=0)
        x_1 = x_1.repeat(n, 1)

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

    @abstractmethod
    def make_tangent(self, p: Tensor, v: Tensor, missing_coordinate: bool = False) -> Tensor:
        """
        Projects the vector `v` on the tangent space of `p`. If `missing_coordinate`, adds an
        extra entry for each product space that makes it tangent.
        """

    @abstractmethod
    def uniform_prior(self, n: int, k: int, d: int) -> Tensor:
        """
        Returns samples from a uniform prior on the manifold.
        """

    @abstractmethod
    def smooth_labels(self, labels: Tensor, mx: float = 0.98) -> Tensor:
        """
        Smoothes the labels on the manifold.
        """

    @abstractmethod
    def send_to(self, x: Tensor, m: Type["Manifold"]) -> Tensor:
        """
        Sends the points `x` to the manifold `m`.
        """

    @abstractmethod
    def all_belong(self, x: Tensor) -> bool:
        """
        Returns `True` iff all points belong to the manifold.
        """

    @abstractmethod
    def all_belong_tangent(self, x: Tensor, v: Tensor) -> bool:
        """
        Returns `True` iff all tangent vectors belong to the tangent space of the manifold
        at point `x`.
        """

    @abstractmethod
    def project(self, x: Tensor) -> Tensor:
        """
        Projects the points `x` to the manifold.
        """

    def masked_tangent_projection(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Projects the tangent vector `v` to the tangent space of `p` only for batch
        indices where `p` is on the manifold.
        """
        raise NotImplementedError()

    def masked_projection(self, p: Tensor) -> Tensor:
        """
        Projects `p` only where points are non-zero.
        """
        raise NotImplementedError()


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
        xs = v / s / 2.0
        theta = xs.norm(dim=-1, keepdim=True)
        return (theta.cos() * s + usinc(theta) * xs).square()

    def log_map(self, p: Tensor, q: Tensor) -> Tensor:
        """
        See `Manifold.log_map`.
        """
        ret = torch.zeros_like(p)
        z = (p * q).sqrt()
        s = z.sum(dim=-1, keepdim=True)
        close = ((s.square() - 1.0).abs() < 1e-7).expand_as(ret)
        ret[~close] = (2.0 * safe_arccos(s) / (1.0 - s.square()).sqrt() * (z - s * p))[~close]
        return ret
        """
        z = (p * q).sqrt()
        s = z.sum(dim=-1, keepdim=True)
        return 2.0 * safe_arccos(s) / (1.0 - s.square()).sqrt() * (z - s * p)"""

    def geodesic_distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        See `Manifold.geodesic_distance`.
        """
        d = (p * q).sqrt().sum(dim=-1)
        return (2.0 * safe_arccos(d)).square().sum(dim=-1).sqrt()

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.metric`.
        """
        # can just ignore points that have some zero coordinates
        # ie on the boundary; doesn't work with mask (changes shape)
        return ((u * v) / x.sqrt()).sum(dim=-1, keepdim=True)

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.parallel_transport`. Based on the parallel transport of
        `NSphere`.
        """
        sphere = NSphere()
        q_s = self.send_to(q, NSphere)
        y_s = sphere.parallel_transport(
            self.send_to(p, NSphere),
            q_s,
            v / p.sqrt(),
        )
        return y_s * q_s

    def make_tangent(self, p: Tensor, v: Tensor, missing_coordinate: bool = False) -> Tensor:
        """
        See `Manifold.make_tangent`.
        """
        if missing_coordinate:
            return torch.cat([v, -v.sum(dim=-1, keepdim=True)], dim=-1)
        return v - v.mean(dim=-1, keepdim=True)

    def uniform_prior(self, n: int, k: int, d: int) -> Tensor:
        """
        See `Manifold.uniform_prior`.
        """
        return Dirichlet(torch.ones((k, d))).sample((n,))

    @torch.no_grad()
    def smooth_labels(self, labels: Tensor, mx: float = 0.98) -> Tensor:
        """
        See `Manifold.smooth_labels`.
        """
        num_classes = labels.size(-1)

        # Value to be added to each non-target class
        increase = (1.0 - mx) / (num_classes - 1)

        # Create a tensor with all elements set to the increase value
        smooth_labels = torch.full_like(labels.float(), increase)

        # Set the target classes to the smoothing value
        smooth_labels[labels == 1] = mx

        return smooth_labels

    def send_to(self, x: Tensor, m: Type["Manifold"]) -> Tensor:
        """
        See `Manifold.send_to`.
        """
        if m == NSphere:
            return x.sqrt()
        elif m == NSimplex:
            return x
        raise NotImplementedError(f"unimplemented for {m}")

    def all_belong(self, x: Tensor) -> bool:
        """
        See `Manifold.all_belong`.
        """
        return torch.allclose(x.sum(dim=-1), torch.tensor(1.0))

    def all_belong_tangent(self, x: Tensor, v: Tensor) -> bool:
        """
        See `Manifold.all_belong_tangent`.
        """
        return torch.allclose(v.sum(dim=-1), torch.tensor(0.0), atol=1e-4)

    def project(self, x: Tensor) -> Tensor:
        """
        See `Manifold.project`.
        """
        return x / x.sum(dim=-1, keepdim=True)


class NSphere(Manifold):
    """
    Defines an n-dimensional sphere.

    Based on: `https://juliamanifolds.github.io`.
    """

    def exp_map(self, p: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.exp_map`.
        """
        theta = v.norm(dim=-1, keepdim=True, p=2)  # norm is independent of point for sphere
        ret = theta.cos() * p + usinc(theta) * v
        # TODO: remove?
        return ret.abs()

    def log_map(self, p: Tensor, q: Tensor) -> Tensor:
        """
        See `Manifold.log_map`.
        """
        cos = fast_dot(p, q).clamp(-1.0, 1.0)
        # otherwise
        theta = safe_arccos(cos)
        x = (q - cos * p) / usinc(theta)
        # X .- real(dot(p, X)) .* p
        return x - fast_dot(x, p) * p

    def geodesic_distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        See `Manifold.geodesic_distance`.
        """
        cos = fast_dot(p, q, keepdim=False)
        # sum across product space
        return safe_arccos(cos).square().sum(dim=-1).sqrt()

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.metric`.
        """
        return fast_dot(u, v)

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.parallel_transport`. Note: assumes this is on 1-sphere
        """
        m = p + q
        # mnorm2 = m.square().sum(dim=-1, keepdim=True)
        # mnorm2 = 2.0 + 2.0 * fast_dot(p, q)
        mnorm2 = 1.0 + fast_dot(p, q)
        # factor = 2.0 * fast_dot(v, q) / mnorm2
        factor = fast_dot(v / mnorm2, q)
        return v - m * factor

    def parallel_transport_alt(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.parallel_transport`. Based on geomstats.
        """
        direction = self.log_map(p, q)
        theta = direction.norm(dim=-1, keepdim=True)
        eps = torch.where(theta == 0.0, 1.0, theta)
        normalized_b = direction / eps
        pb = fast_dot(v, normalized_b)
        p_orth = v - pb * normalized_b
        return (
            - theta.sin() * pb * p
            + theta.cos() * pb * normalized_b
            + p_orth
        )

    def make_tangent_alt(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Alternative. See `Manifold.make_tangent`.
        """
        # alternative version:
        # p_unit = p / p.norm(dim=-1, keepdim=True)
        p_unit = torch.nn.functional.normalize(p, p=2, dim=-1)
        normalized = v / (p_unit * v).sum(dim=-1, keepdim=True)
        ret = normalized - p
        return ret

    def make_tangent(self, p: Tensor, v: Tensor, missing_coordinate: bool = False) -> Tensor:
        """
        See `Manifold.make_tangent`.
        """
        p = self.project(p)
        if missing_coordinate:
            return torch.cat([v, -(p[:, :, :-1] * v).sum(dim=-1, keepdim=True)], dim=-1)
        # return v - p * fast_dot(p, v)
        # keep the normalisation even if = 1: more precise!
        sq = p.square().sum(dim=-1, keepdim=True)
        inner = fast_dot(p / sq, v, keepdim=True)
        # coef = inner / sq
        ret = v - inner * p
        # dirty trick that makes it a tiny bit more precise
        # ret[:, :, 0] = ret[:, :, 0] - (p * ret).sum(dim=-1)
        return ret

    def uniform_prior(self, n: int, k: int, d: int) -> Tensor:
        """
        See `Manifold.uniform_prior`.
        """
        x_0 = torch.randn((n, k, d))
        x_0 = x_0 / x_0.norm(p=2, dim=-1, keepdim=True)
        return x_0.abs()

    def smooth_labels(self, labels: Tensor, mx: float = 0.9999) -> Tensor:
        """
        See `Manifold.smooth_labels`.
        """
        return NSimplex().send_to(NSimplex().smooth_labels(labels, mx), NSphere)

    def send_to(self, x: Tensor, m: type[Manifold]) -> Tensor:
        """
        See `Manifold.send_to`.
        """
        if m == NSphere:
            return x
        elif m == NSimplex:
            return x.square()
        raise NotImplementedError(f"unimplemented for {m}")

    def batch_square_norm(self, x: Tensor) -> Tensor:
        """
        Returns the square of the euclidean norm of `x` in the last coordinate.
        """
        return x.square().sum(dim=-1)

    def all_belong(self, x: Tensor) -> bool:
        """
        See `Manifold.all_belong`.
        """
        return (
            torch.allclose(self.batch_square_norm(x), torch.tensor(1.0))
        )

    def all_belong_tangent(self, x: Tensor, v: Tensor) -> bool:
        """
        See `Manifold.all_belong_tangent`.
        """
        return torch.allclose(fast_dot(x, v), torch.tensor(0.0), atol=1e-5)

    def project(self, x: Tensor) -> Tensor:
        """
        See `Manifold.project`.
        """
        return x / x.norm(dim=-1, keepdim=True)


class GeooptSphere(Manifold):
    """Wrapper for Geoopt Sphere."""
    def __init__(self):
        self.sphere = GSphere()

    def exp_map(self, p: Tensor, v: Tensor) -> Tensor:
        return self.sphere.expmap(p, v)

    def log_map(self, p: Tensor, q: Tensor) -> Tensor:
        return self.sphere.logmap(p, q)

    def geodesic_distance(self, p: Tensor, q: Tensor) -> Tensor:
        return self.sphere.dist2(p, q).sum(dim=-1).sqrt()

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return self.sphere.inner(x, u, v)

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        return self.sphere.transp(p, q, v)

    def parallel_transport_alt(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        denom = 1 + self.metric(p, p, q).unsqueeze(-1)
        res = v - self.metric(p, q, v).unsqueeze(-1) / denom * (p + q)
        cond = denom.gt(1e-3)
        return torch.where(cond, res, -v)

    def make_tangent(self, p: Tensor, v: Tensor, missing_coordinate: bool = False) -> Tensor:
        if missing_coordinate:
            return NSphere().make_tangent(p, v, missing_coordinate)
        p = self.project(p)
        return self.sphere.proju(p, v)

    def masked_tangent_projection(self, p: Tensor, v: Tensor) -> Tensor:
        mask = torch.isclose(p.square().sum(dim=-1), torch.tensor(1.0))
        p[mask, :] = self.project(p[mask, :])
        return self.sphere.proju(p, v)

    def masked_projection(self, p: Tensor) -> Tensor:
        mask = torch.isclose(p.square().sum(dim=-1), torch.tensor(0.0))
        p[~mask, :] = self.project(p[~mask, :])
        return p

    def uniform_prior(self, n: int, k: int, d: int) -> Tensor:
        ret = self.sphere.random_uniform((n, k, d)).abs()
        return ret

    def smooth_labels(self, labels: Tensor, mx: float = 0.98) -> Tensor:
        return NSimplex().send_to(NSimplex().smooth_labels(labels, mx), NSphere)

    def send_to(self, x: Tensor, m: type[Manifold]) -> Tensor:
        if m == NSphere:
            return x
        elif m == NSimplex:
            return x.square()
        raise NotImplementedError(f"unimplemented for {m}")

    def all_belong(self, x: Tensor) -> bool:
        return torch.allclose(x.norm(dim=-1), torch.tensor(1.0))

    def all_belong_tangent(self, x: Tensor, v: Tensor) -> bool:
        return torch.allclose(fast_dot(x, v), torch.tensor(0.0), atol=1e-5)

    def project(self, x: Tensor) -> Tensor:
        """
        See `Manifold.project`.
        """
        return self.sphere.projx(x).abs()


class LinearNSimplex(NSimplex):
    def exp_map(self, p: Tensor, v: Tensor) -> Tensor:
        return p + v

    def log_map(self, p: Tensor, q: Tensor) -> Tensor:
        return q - p

    def geodesic_distance(self, p: Tensor, q: Tensor) -> Tensor:
        return (p - q).square().sum(dim=(-1, -2)).sqrt()

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return fast_dot(u, v)

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        return v

    def uniform_prior(self, n: int, k: int, d: int) -> Tensor:
        return NSimplex().uniform_prior(n, k, d)

    def all_belong_tangent(self, x: Tensor, v: Tensor) -> bool:
        return torch.allclose(fast_dot(x, v), torch.tensor(0.0))


def manifold_from_name(name: str) -> Manifold:
    """
    Returns the manifold corresponding to `name`.
    """
    if name == "sphere":
        return GeooptSphere()
    elif name == "simplex":
        return NSimplex()
    elif name == "linear-simplex":
        return LinearNSimplex()
    raise ValueError(f"Unknown manifold: {name}")
