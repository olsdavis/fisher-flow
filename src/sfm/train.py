"""Loss utils."""
import torch
from torch import Tensor, nn, vmap
from torch.func import jvp
from src.sfm import Manifold, OTSampler


def print_stats(x: Tensor, name: str):
    print(f"{name}: {x.mean().item():.8f} ± {x.std().item():.8f}; min: {x.min().item():.8f}; max: {x.max().item():.8f}")


def geodesic(manifold, start_point, end_point):
    # https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/manifolds/utils.py#L6
    shooting_tangent_vec = manifold.logmap(start_point, end_point)

    def path(t):
        """Generate parameterized function for geodesic curve.
        Parameters
        ----------
        t : array-like, shape=[n_points,]
            Times at which to compute points of the geodesics.
        """
        tangent_vecs = torch.einsum("i,...k->...ik", t, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)
        return points_at_time_t

    return path


def ot_train_step(
    x_1: Tensor,
    m: Manifold,
    model: nn.Module,
    sampler: OTSampler | None,
    extra_args: dict[str, Tensor] | None = None,
    closed_form_drv: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns the loss for a single (OT-)CFT training step along with the
    model's output and the target vector.

    Parameters:
        - `x_1`: batch of data points;
        - `m`: manifold;
        - `model`: the model to apply;
        - `sampler` (optional): the sampler for the OT plan;
        - `time_eps`: "guard" for sampling the time;
        - `signal` (optional): extra signal for some datasets;
        - `closed_form_drv`: whether to use the closed-form derivative;
            if `False`, uses autograd.
    """
    b = x_1.size(0)
    k = x_1.size(1)
    d = x_1.size(-1)
    t = torch.rand((b, 1), device=x_1.device)
    x_0 = m.uniform_prior(b, k, d).to(x_1.device)
    return cft_loss_function(
        x_0, x_1, t, m, model, sampler, extra_args=extra_args, closed_form_drv=closed_form_drv,
    )


def cft_loss_function(
    x_0: Tensor,
    x_1: Tensor,
    t: Tensor,
    m: Manifold,
    model: nn.Module,
    sampler: OTSampler | None,
    extra_args: dict[str, Tensor] | None = None,
    closed_form_drv: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Our CFT loss function. If `sampler` is provided, OT-CFT loss is calculated.

    Parameters:
        - `x_0`: starting point (drawn from prior);
        - `x_1`: end point (drawn from data);
        - `t`: the times;
        - `m`: the manifold;
        - `model`: the model to apply;
        - `sampler` (optional): the sampler for the OT plan;
        - `signal` (optional): extra signal for some datasets;
        - `closed_form_drv`: whether to use the closed-form derivative;
            if `False`, uses autograd.

    Returns:
        The loss tensor, the model output, and the target vector.
    """
    if sampler:
        x_0, x_1 = sampler.sample_plan(x_0, x_1)
    if closed_form_drv:
        x_t = m.geodesic_interpolant(x_0, x_1, t)
        target = m.log_map(x_0, x_1)
        target = m.parallel_transport(x_0, x_t, target)
        # target = m.log_map(x_t, x_1) / (1.0 - t.unsqueeze(-1) + 1e-7)
    else:
        with torch.inference_mode(False):
            # https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/model_pl.py
            def cond_u(x0, x1, t):
                path = geodesic(m.sphere, x0, x1)
                x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
                return x_t, u_t
            x_t, target = vmap(cond_u)(x_0, x_1, t)
        x_t = x_t.squeeze()
        target = target.squeeze()
        assert m.all_belong_tangent(x_t, target)

    # now calculate diffs
    out = model(x=x_t, t=t, **(extra_args or {}))

    diff = out - target
    return diff.square().sum(dim=(-1, -2)).mean(), out, target
