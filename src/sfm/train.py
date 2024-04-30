"""Loss utils."""
import torch
from torch import vmap
from torch.func import jvp
from torch.distributions.dirichlet import Dirichlet
from torch import Tensor, nn
from src.sfm import Manifold, OTSampler


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


def dfm_train_step(
    x_1: Tensor,
    model: nn.Module,
) -> Tensor:
    """
    Returns the loss for a Dirichlet Flow Matching training step.

    Parameters:
        - `x_1`: the data tensor, must be one-hot encoding;
        - `model`: the model to train.
    """
    b = x_1.size(0)
    t = torch.rand((b, 1), device=x_1.device)
    alpha_t = torch.ones_like(x_1) + x_1 * t
    # iterate over
    x_t = Dirichlet(alpha_t).sample().to(x_1.device)
    p_hat = model(x_t, t)
    return nn.functional.cross_entropy(p_hat, x_1)


def ot_train_step(
    x_1: Tensor,
    m: Manifold,
    model: nn.Module,
    sampler: OTSampler | None,
    time_eps: float = 0.0,
    signal: Tensor | None = None,
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
    t = torch.rand((b, 1), device=x_1.device) * (1.0 - time_eps)
    x_0 = m.uniform_prior(b, k, d).to(x_1.device)
    return cft_loss_function(
        x_0, x_1, t, m, model, sampler, signal=signal, closed_form_drv=closed_form_drv,
    )


def cft_loss_function(
    x_0: Tensor,
    x_1: Tensor,
    t: Tensor,
    m: Manifold,
    model: nn.Module,
    sampler: OTSampler | None,
    signal: Tensor | None = None,
    eps: float = 1e-8,
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
    # TODO: Check this
    # print(out.norm(dim=-1).sum(dim=1).max(), target.norm(dim=-1).sum(dim=1).min())
    if closed_form_drv:
        x_t = m.geodesic_interpolant(x_0, x_1, t)
        target = m.log_map(x_0, x_1)
        target = m.parallel_transport(x_0, x_t, target)
    else:
        # https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/model_pl.py
        def cond_u(x0, x1, t):
            path = geodesic(m.sphere, x0, x1)
            x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
            return x_t, u_t
        x_t, target = vmap(cond_u)(x_0, x_1, t)
        x_t = x_t.squeeze()
        target = target.squeeze()
    # now calculate diffs
    if signal is not None:
        out = model(x_t, signal, t)
    else:
        out = model(x_t, t)
    out = m.make_tangent(x_t, out)
    diff = out - target
    loss = m.square_norm_at(x_t, diff)
    loss = loss.sum(dim=1)
    return loss.mean(), out, target
