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


class ProductMLP(nn.Module):
    """
    An MLP that operates over all k d-simplices at the same time.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        hidden: int,
        depth: int,
        simplex_tangent: bool = True,
        activation: str = "relu",
    ):
        """
        Parameters:
            - `dim`: the dimension of each simplex;
            - `k`: the number of simplices in the product;
            - `hidden`: the hidden dimension;
            - `depth`: the depth of the network;
            - `simplex_tangent`: when `True` makes the point output constrained
            to the tangent space of the simplex, i.e., `x: 1^T x = 0`;
            - `activation`: the activation function.
        """
        super().__init__()
        self.simplex_tangent = simplex_tangent
        activation = str_to_activation(activation)
        net = []
        for i in range(depth):
            net += [
                nn.Linear(
                    k * dim + 1 if i == 0 else hidden,
                    hidden if i < depth - 1 else
                        k * (dim - 1 if simplex_tangent else dim),
                )
            ]
            if i < depth - 1:
                net += [activation]
        self.net = nn.Sequential(*net)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Applies the MLP to the input `(x, t)`.
        """
        final_shape = list(x.shape)
        # remove one dimension if tangent space
        if self.simplex_tangent:
            final_shape[-1] = final_shape[-1] - 1
        final_shape = tuple(final_shape)
        x = x.view((x.size(0), -1))
        # run
        out = self.net(torch.cat([x, t], dim=-1))
        out = out.reshape(final_shape)
        if self.simplex_tangent:
            x_3 = -out.sum(dim=-1, keepdim=True)
            return torch.cat([out, x_3], dim=-1)
        return out


# Old code:


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0]).to(x.device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).to(x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self) -> int:
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self) -> int:
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(-1)

    def __len__(self) -> int:
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class Block(nn.Module):
    def __init__(
        self, size: int, t_emb_size: int = 0, add_t_emb=False, concat_t_emb=False
    ):
        super().__init__()

        in_size = size + t_emb_size if concat_t_emb else size
        self.ff = nn.Linear(in_size, size)
        self.act = nn.GELU()

        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        in_arg = torch.cat([x, t_emb], dim=-1) if self.concat_t_emb else x
        out = x + self.act(self.ff(in_arg))

        if self.add_t_emb:
            out = out + t_emb

        return out


class TembMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        k: int,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
        add_t_emb: bool = False,
        concat_t_emb: bool = False,
        energy_function=None,
    ):
        super().__init__()

        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)

        positional_embeddings = []
        for i in range(k * dim):
            embedding = PositionalEmbedding(emb_size, input_emb, scale=25.0)

            self.add_module(f"input_mlp{i}", embedding)

            positional_embeddings.append(embedding)

        self.channels = 1
        self.self_condition = False
        concat_size = len(self.time_mlp.layer) + sum(
            map(lambda x: len(x.layer), positional_embeddings)
        )

        layers = [nn.Linear(concat_size, hidden_size)]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size, emb_size, add_t_emb, concat_t_emb))

        in_size = emb_size + hidden_size if concat_t_emb else emb_size
        layers.append(nn.Linear(in_size, k * (dim - 1)))

        self.layers = layers
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, x_self_cond=False):
        final_shape = list(x.shape)
        final_shape[-1] -= 1
        final_shape = tuple(final_shape)

        x = x.view((x.size(0), -1))
        positional_embs = [
            self.get_submodule(f"input_mlp{i}")(x[:, i]) for i in range(x.shape[-1])
        ]

        t_emb = self.time_mlp(t.squeeze())
        x = torch.cat((*positional_embs, t_emb), dim=-1)

        for i, layer in enumerate(self.layers):
            if i == 0:
                x = nn.GELU()(layer(x))
                if self.add_t_emb:
                    x = x + t_emb

            elif i == len(self.layers) - 1:
                if self.concat_t_emb:
                    x = torch.cat([x, t_emb], dim=-1)

                x = layer(x)

            else:
                x = layer(x, t_emb)
        x = x.view(final_shape)
        x_last = -x.sum(dim=-1, keepdim=True)
        return torch.cat([x, x_last], dim=-1)
