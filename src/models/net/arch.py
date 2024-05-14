"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
import torch.nn as nn

import src.models.net.diffeq_layers as diffeq_layers
from src.models.net.actfn import Sine, Softplus


ACTFNS = {
    "swish": diffeq_layers.TimeDependentSwish,
    "sine": Sine,
    "srelu": Softplus,
}


class TMLP(nn.Module):
    def __init__(
        self,
        k: int, dim: int, dim_out: int | None = None, hidden: int = 256, depth: int = 6, activation="swish", fourier=None, pre_flat: bool = True,
    ):
        super().__init__()
        self.pre_flat = pre_flat
        self.k = k
        self.dim = dim
        assert depth > 1, "No weak linear nets here"
        d_out = dim_out or k * dim
        actfn = ACTFNS[activation]
        if fourier:
            layers = [
                diffeq_layers.diffeq_wrapper(
                    PositionalEncoding(n_fourier_features=fourier)
                ),
                diffeq_layers.ConcatLinear_v2(k * dim * fourier * 2, hidden),
            ]
        else:
            layers = [diffeq_layers.ConcatLinear_v2(k * dim, hidden)]

        for _ in range(depth - 2):
            layers.append(actfn(hidden))
            layers.append(diffeq_layers.ConcatLinear_v2(hidden, hidden))
        layers.append(actfn(hidden))
        layers.append(diffeq_layers.ConcatLinear_v2(hidden, d_out))
        self.net = diffeq_layers.SequentialDiffEq(*layers)

    def forward(self, x, t):
        if self.pre_flat:
            x = x.view(-1, self.k * self.dim)
        ret = self.net(t, x)
        if self.pre_flat:
            return ret.view(-1, self.k, self.dim)
        return ret


class TMLPSignal(TMLP):
    def __init__(
        self,
        k: int, dim: int, hidden: int = 256, depth: int = 6, activation="swish", fourier=None,
    ):
        super().__init__(k, dim+2, k*dim, hidden, depth, activation, fourier, pre_flat=False)
        self.dim = dim

    def forward(self, x, t, signal):
        x = torch.cat([x.view(-1, self.k, self.dim), signal], dim=-1)
        return super().forward(x.view(x.size(0), -1), t).view(-1, self.k, self.dim)


class PositionalEncoding(nn.Module):
    """Assumes input is in [0, 2pi]."""

    def __init__(self, n_fourier_features):
        super().__init__()
        self.n_fourier_features = n_fourier_features

    def forward(self, x):
        feature_vector = [
            torch.sin((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        feature_vector += [
            torch.cos((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        return torch.cat(feature_vector, dim=-1)

