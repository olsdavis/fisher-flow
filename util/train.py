"""Loss utils."""
import numpy as np
import torch
from torch.distributions.dirichlet import Dirichlet
from torch import Tensor, nn
from util import Manifold, OTSampler, generate_dirichlet_product


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
    d = x_1.size(1)
    t = torch.rand((b, 1, 1), device=x_1.device)
    t = t.repeat((1, d, 1))
    alpha_t = torch.ones_like(x_1) + x_1 * t
    # iterate over 
    samples = [torch.stack([Dirichlet(alpha_d).sample() for alpha_d in alpha_p]) for alpha_p in alpha_t]
    x_t = torch.stack(samples)
    p_hat = model(x_t, t)
    return criterion(p_hat, x_1)


def ot_train_step(
    x_1: Tensor,
    m: Manifold,
    model: nn.Module,
    sampler: OTSampler | None,
) -> Tensor:
    """
    Returns the loss for a single (OT-)CFT training step.

    Parameters:
        - `x_1`: batch of data points;
        - `m`: manifold;
        - `model`: the model to apply;
        - `sampler` (optional): the sampler for the OT plan.
    """
    b = x_1.size(0)
    d = x_1.size(1)
    t = torch.rand((b, 1, 1), device=x_1.device)
    t = t.repeat((1, d, 1))
    x_0 = generate_dirichlet_product(b, d, x_1.size(-1))
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
    diff = out - target
    loss = m.square_norm_at(x_t, diff)
    # if product space
    if len(loss.shape) == 3:
        loss = loss.sum(dim=1)
    return loss.mean()
