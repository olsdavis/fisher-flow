"""Defines models used in this code base."""
import torch
from torch import Tensor, nn


def str_to_activation(name: str) -> nn.Module:
    """
    Returns the activation function associated to the name `name`.
    """
    acts = {
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(0.01),
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
    }
    return acts[name]


class MLP(nn.Module):
    """
    Defines a simple MLP with time.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        hidden: int,
        simplex_tangent: bool = True,
        activation: str = "relu",
    ):
        """
        Parameters:
            - `dim`: the dimension of the input and output;
            - `depth`: the depth of the network;
            - `hidden`: the size of hidden features;
            - `simplex_tangent`: when `True` makes the point output constrained
            to the tangent space of the simplex, i.e., `x: 1^T x = 0`;
            - `activation`: the activation function to use.
        """
        super().__init__()
        act = str_to_activation(activation)
        self.simplex_tangent = simplex_tangent

        net = []
        for i in range(depth):
            out = hidden
            if i == depth - 1:
                if simplex_tangent:
                    out = dim - 1
                else:
                    out = dim
            net += [
                nn.Linear(
                    # +1 for time 
                    dim + 1 if i == 0 else hidden,
                    out,
                )
            ]
            if i < depth - 1:
                net += [act]
        self.net = nn.Sequential(*net)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Applies the MLP to the input `(x, t)`.
        """
        out = self.net(torch.cat([x, t], dim=-1))
        if self.simplex_tangent:
            x_3 = -out.sum(dim=-1, keepdim=True)
            return torch.cat([out, x_3], dim=-1)
        return out
