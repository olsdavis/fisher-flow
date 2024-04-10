"""Loss utils."""
import torch
from torch.distributions.dirichlet import Dirichlet
from torch import Tensor, nn
from util import Manifold, OTSampler


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
    criterion = nn.CrossEntropyLoss()
    b = x_1.size(0)
    t = torch.rand((b, 1), device=x_1.device)
    alpha_t = torch.ones_like(x_1) + x_1 * t
    # iterate over
    x_t = Dirichlet(alpha_t).sample().to(x_1.device)
    p_hat = model(x_t, t)
    return criterion(p_hat, x_1)


def ot_train_step(
    x_1: Tensor,
    m: Manifold,
    model: nn.Module,
    sampler: OTSampler | None,
    time_eps: float = 0.0,
) -> Tensor:
    """
    Returns the loss for a single (OT-)CFT training step.

    Parameters:
        - `x_1`: batch of data points;
        - `m`: manifold;
        - `model`: the model to apply;
        - `sampler` (optional): the sampler for the OT plan;
        - `time_eps`: "guard" for sampling the time.
    """
    b = x_1.size(0)
    k = x_1.size(1)
    d = x_1.size(-1)
    t = torch.rand((b, 1), device=x_1.device) * (1.0 - time_eps)
    x_0 = m.uniform_prior(b, k, d).to(x_1.device)
    return cft_loss_function(x_0, x_1, t, m, model, sampler)


def cft_loss_function(
    x_0: Tensor,
    x_1: Tensor,
    t: Tensor,
    m: Manifold,
    model: nn.Module,
    sampler: OTSampler | None,
) -> Tensor:
    """
    Our CFT loss function. If `sampler` is provided, OT-CFT loss is calculated.

    Parameters:
        - `x_0`: starting point (drawn from prior);
        - `x_1`: end point (drawn from data);
        - `t`: the times;
        - `m`: the manifold;
        - `model`: the model to apply;
        - `sampler` (optional): the sampler for the OT plan.
    
    Returns:
        The loss tensor.
    """
    if sampler:
        x_0, x_1 = sampler.sample_plan(x_0, x_1)
    x_t = m.geodesic_interpolant(x_0, x_1, t)
    target = m.log_map(x_0, x_1)
    target = m.parallel_transport(x_0, x_t, target)
    out = model(x_t, t)
    out = m.make_tangent(x_t, out)
    # TODO: Check this
    # print(out.norm(dim=-1).sum(dim=1).max(), m.log_map(x_0, x_1).norm(dim=-1).sum(dim=1).min())
    diff = out - target
    loss = m.square_norm_at(x_t, diff)
    loss = loss.sum(dim=1)
    return loss.mean()
