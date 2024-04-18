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
        "swish": nn.SiLU(),
    }
    return acts[name]


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
        activation: str = "relu",
        **_,
    ):
        """
        Parameters:
            - `dim`: the dimension of each simplex;
            - `k`: the number of simplices in the product;
            - `hidden`: the hidden dimension;
            - `depth`: the depth of the network;
            - `activation`: the activation function.

        Other arguments are ignored.
        """
        super().__init__()
        act = str_to_activation(activation)
        net: list[nn.Module] = []
        for i in range(depth):
            net += [
                nn.Linear(
                    k * dim + 1 if i == 0 else hidden,
                    hidden if i < depth - 1 else
                        k * dim,
                )
            ]
            if i < depth - 1:
                net += [act]
        self.net = nn.Sequential(*net)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Applies the MLP to the input `(x, t)`.
        """
        final_shape = x.shape
        x = x.view((x.size(0), -1))
        # run
        out = self.net(torch.cat([x, t], dim=-1))
        out = out.reshape(final_shape)
        return out


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding.
    """

    def __init__(self, size: int, scale: float = 1.0):
        """
        Parameters:
            - `size`: the size of the embedding;
            - `scale`: the scale of factor to increase initially frequency.
        """
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the sinusoidal embeddeing to `x`.
        """
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
    """
    Applies a linear scaling to the input.
    """

    def __init__(self, size: int, scale: float = 1.0):
        """
        Parameters:
            - `size`: the size of the input;
            - `scale`: the scale factor.
        """
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Scales `x` with an extra dimension at the end.
        """
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self) -> int:
        return 1


class LearnableEmbedding(nn.Module):
    """
    A learnable linear embedding.
    """

    def __init__(self, size: int):
        """
        Paramters:
            - `size`: the size of the learnt embedding.
        """
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the learnt embedding to `x`.
        """
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self) -> int:
        return self.size


class IdentityEmbedding(nn.Module):
    """
    Identity embedding.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns `x` with an extra dimension at the end.
        """
        return x.unsqueeze(-1)

    def __len__(self) -> int:
        return 1


class ZeroEmbedding(nn.Module):
    """
    Zero (trivial) embedding.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Returns zero."""
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    """
    Positional embedding for inputs.
    """

    def __init__(self, size: int, emb_type: str, **kwargs):
        """
        Parameters:
            - `size`: the size of the input;
            - `emb_type`: the type of embedding to use; either `sinusoidal`, `linear`,
                `learnable`, `zero`, or `identity`;
            - `**kwargs`: arguments for the specific embedding.
        """
        super().__init__()
        if emb_type == "sinusoidal":
            self.layer: nn.Module = SinusoidalEmbedding(size, **kwargs)
        elif emb_type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif emb_type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif emb_type == "zero":
            self.layer = ZeroEmbedding()
        elif emb_type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {emb_type}")

    def forward(self, x: torch.Tensor):
        """
        Applies the positional embedding to `x`.
        """
        return self.layer(x)


class TembBlock(nn.Module):
    """
    A basic block for the `TembMLP`.
    """
    def __init__(
        self,
        size: int,
        activation: str,
        t_emb_size: int = 0,
        add_t_emb: bool = False,
        concat_t_emb: bool = False,
    ):
        """
        Parameters:
            - `size`: the size of the input and output;
            - `activation`: the activation function to use;
            - `t_emb_size`: the size of the time embedding;
            - `add_t_emb`: whether the time embeddings should be added residually;
            - `concat_t_emb`: whether the time embeddings should be concatenated.
        """
        super().__init__()
        in_size = size + t_emb_size if concat_t_emb else size
        self.ff = nn.Linear(in_size, size)
        self.act = str_to_activation(activation)
        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        Applies the block to `(x, t)`.
        """
        in_arg = torch.cat([x, t_emb], dim=-1) if self.concat_t_emb else x
        out = x + self.act(self.ff(in_arg))
        if self.add_t_emb:
            out = out + t_emb
        return out


class TembMLP(nn.Module):
    """
    A more advanced MLP with time embeddings.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        hidden: int = 128,
        depth: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
        add_t_emb: bool = False,
        concat_t_emb: bool = False,
        activation: str = "gelu",
        **_,
    ):
        """
        Parameters:
            - `dim`: dimension per space;
            - `k`: spaces in product space;
            - `hidden_size`: hidden features;
            - `emb_size`: the size of the embedding;
            - `time_emb`: the type of time embedding;
            - `input_emb`: the type of input embedding;
            - `add_t_emb`: if the time embedding should be residually added;
            - `concat_t_emb`: if the time embedding should be concatenated;
            - `activation`: the activation function to use.

        Other arguments are ignored.
        """
        super().__init__()
        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb
        self.activation = str_to_activation(activation)
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        positional_embeddings = []
        for i in range(k * dim):
            embedding = PositionalEmbedding(emb_size, input_emb, scale=25.0)
            self.add_module(f"input_mlp{i}", embedding)
            positional_embeddings.append(embedding)

        concat_size = len(self.time_mlp.layer) + sum(
            map(lambda x: len(x.layer), positional_embeddings)
        )
        layers: list[nn.Module] = [nn.Linear(concat_size, hidden)]
        for _ in range(depth):
            layers.append(TembBlock(hidden, activation, emb_size, add_t_emb, concat_t_emb))

        in_size = emb_size + hidden if concat_t_emb else hidden
        layers.append(nn.Linear(in_size, k * dim))

        self.layers = layers
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Applies the model to input `(x, t)`.
        """
        final_shape = x.shape

        x = x.view((x.size(0), -1))
        positional_embs = [
            self.get_submodule(f"input_mlp{i}")(x[:, i]) for i in range(x.shape[-1])
        ]

        t_emb = self.time_mlp(t.squeeze())
        x = torch.cat((*positional_embs, t_emb), dim=-1)

        for i, layer in enumerate(self.layers):
            if i == 0:
                x = self.activation(layer(x))
                if self.add_t_emb:
                    x = x + t_emb

            elif i == len(self.layers) - 1:
                if self.concat_t_emb:
                    x = torch.cat([x, t_emb], dim=-1)
                x = layer(x)

            else:
                x = layer(x, t_emb)
        x = x.view(final_shape)
        return x


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.

    From `https://github.com/HannesStark/dirichlet-flow-matching/blob/main/model/promoter_model.py`.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        """Forward through."""
        x_proj = x[:, None] * self.W[None, :] * 2.0 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class BestBlock(nn.Module):
    """A block for BestMLP."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_dim: int,
        resid: bool = True,
        act: nn.Module | None = None,
        b_norm: bool | None = False,
    ):
        """
        Parameters:
            - `in_dim`: the input dimension, excluding time;
            - `out_dim`: the output dimension;
            - `time_dim`: the time dimension;
            - `resid`: whether to use residual connections;
            - `act`: the activation function to use.
        """
        super().__init__()
        assert not resid or in_dim == out_dim, "Residual connections require in_dim == out_dim"
        self.resid = resid
        self.act = act
        self.net = nn.Linear(in_dim + time_dim, out_dim)
        self.b_norm = nn.BatchNorm1d(out_dim) if b_norm else None

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass."""
        x_t = torch.cat([x, t], dim=-1)
        out = self.net(x_t)
        if self.act:
            out = self.act(out)
        if self.b_norm:
            out = self.b_norm(out)
        if self.resid:
            out = out + x
        return out


class BestMLP(nn.Module):
    """
    Defines an MLP with only time embeddings.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        hidden: int,
        depth: int,
        emb_size: int,
        activation: str = "lrelu",
        batch_norm: bool = False,
        **_,
    ):
        """
        Parameters:
            - `dim`: the dimension of each simplex;
            - `k`: the number of simplices in the product;
            - `hidden`: the hidden dimension;
            - `depth`: the depth of the network;
            - `emb_size`: the size of the embedding.

        Other arguments are ignored.
        """
        assert emb_size > 0, "emb_size must be positive"
        super().__init__()
        act = str_to_activation(activation)
        self.time_embedding = nn.Sequential(
            # SinusoidalEmbedding(emb_size, scale=25.0),
            nn.Linear(1, emb_size),
        )
        layers: list[nn.Module] = []
        fd = k * dim
        for i in range(depth):
            ind = fd if i == 0 else hidden
            out = hidden if i < depth - 1 else fd
            layers += [
                BestBlock(
                    ind,
                    out,
                    emb_size,
                    act=act if i < depth - 1 else None,
                    resid=ind == out,
                    b_norm=batch_norm and i < depth - 1,
                ),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass."""
        final_shape = x.shape
        x = x.view((x.size(0), -1))
        emb = self.time_embedding(t)
        for layer in self.net:
            x = layer(x, emb)
        x = x.reshape(final_shape)
        return x
