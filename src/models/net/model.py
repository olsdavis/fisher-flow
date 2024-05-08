"""Defines models used in this code base."""
import torch
from torch import Tensor, nn
import copy
import torch.nn.functional as F
import numpy as np
from src.sfm import Manifold


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


class TangentWrapper(nn.Module):
    """
    Wraps a model with a projection on a manifold's tangent space.
    """

    def __init__(self, manifold: Manifold, model: nn.Module):
        """
        Parameters:
            - `manifold`: the manifold to operate on;
            - `model`: the model to wrap.
        """
        super().__init__()
        self.manifold = manifold
        self.model = model
        self.missing_coordinate = (
            hasattr(model, "missing_coordinate") and model.missing_coordinate()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Forward pass with projection.
        """
        return self.manifold.make_tangent(
            x, self.model(x=x, t=t, **kwargs), missing_coordinate=self.missing_coordinate,
        )


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

    def __init__(self, size: int, emb_type: str, **kwself):
        """
        Parameters:
            - `size`: the size of the input;
            - `emb_type`: the type of embedding to use; either `sinusoidal`, `linear`,
                `learnable`, `zero`, or `identity`;
            - `**kwself`: arguments for the specific embedding.
        """
        super().__init__()
        if emb_type == "sinusoidal":
            self.layer: nn.Module = SinusoidalEmbedding(size, **kwself)
        elif emb_type == "linear":
            self.layer = LinearEmbedding(size, **kwself)
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
        b_norm: bool = False,
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
        missing_coordinate: bool = True,
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
        self._missing_coordinate = missing_coordinate
        act = str_to_activation(activation)
        self.time_embedding = nn.Sequential(
            # SinusoidalEmbedding(emb_size, scale=25.0),
            nn.Linear(1, emb_size),
        )
        layers: list[nn.Module] = []
        fd = k * dim
        for i in range(depth):
            ind = fd if i == 0 else hidden
            out = hidden if i < depth - 1 else (fd if not missing_coordinate else fd - k)
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
        prods = x.size(1)
        x = x.view((x.size(0), -1))
        emb = self.time_embedding(t)
        for layer in self.net:
            x = layer(x, emb)
        x = x.view(x.size(0), prods, -1)
        return x

    def missing_coordinate(self) -> bool:
        """
        Returns `True` iff requires missing coordinate completion.
        """
        return self._missing_coordinate


class BestSignalMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        k: int,
        hidden: int,
        depth: int,
        emb_size: int,
        activation: str = "lrelu",
        missing_coordinate: bool = True,
    ):
        super().__init__()
        self._missing_coordinate = missing_coordinate
        layers = []
        for i in range(depth):
            layers += [
                nn.Linear(
                    k * (dim + 2) + emb_size if i == 0 else hidden,
                    hidden if i < depth - 1 else (k * dim if not missing_coordinate else k * dim - k),
                ),
            ]
            if i < depth - 1:
                layers += [str_to_activation(activation)]
        self.mlp = nn.Sequential(*layers)
        self.temb = nn.Linear(1, emb_size)
        self.k = k
        self.dim = dim

    def forward(self, x: Tensor, signal: Tensor, t: Tensor) -> Tensor:
        """Forward pass."""
        temb = self.temb(t)
        original = x.shape
        x = x.view(x.size(0), -1)
        signal = signal.view(signal.size(0), -1)
        x = torch.cat([x, signal, temb], dim=-1)
        ret = self.mlp(x)
        return ret.view(original[0], self.k, -1)

    def missing_coordinate(self) -> bool:
        """
        Returns `True` iff requires missing coordinate completion.
        """
        return self._missing_coordinate



class BestEnhancerMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        k: int,
        hidden: int,
        depth: int,
        emb_size: int,
        activation: str = "lrelu",
        missing_coordinate: bool = True,
        signal_size: int = 1,
    ):
        super().__init__()
        self._missing_coordinate = missing_coordinate
        layers = []
        for i in range(depth):
            layers += [
                nn.Linear(
                    k * dim + signal_size + emb_size if i == 0 else hidden,
                    hidden if i < depth - 1 else (k * dim if not missing_coordinate else k * dim - k),
                ),
            ]
            if i < depth - 1:
                layers += [str_to_activation(activation)]
        self.mlp = nn.Sequential(*layers)
        self.temb = nn.Linear(1, emb_size)
        self.k = k
        self.dim = dim

    def forward(self, x: Tensor, signal: Tensor, t: Tensor) -> Tensor:
        """Forward pass."""
        temb = self.temb(t)
        original = x.shape
        x = x.view(x.size(0), -1)
        signal = signal.view(signal.size(0), -1)
        x = torch.cat([x, signal, temb], dim=-1)
        ret = self.mlp(x)
        return ret.view(original[0], self.k, -1)

    def missing_coordinate(self) -> bool:
        """
        Returns `True` iff requires missing coordinate completion.
        """
        return self._missing_coordinate


class UNet1DModel(nn.Module):
    """
    Adaptation of diffusers UNet1D.
    """
    from diffusers.models import UNet1DModel as DiffusersUNet

    def __init__(
        self,
        k: int,
        dim: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.diffusers_unet = self.DiffusersUNet(
            sample_size=dim,
            in_channels=k,
            out_channels=k,
            block_out_channels=(64, 64,),
            down_block_types=("DownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "UpBlock1D"),
            act_fn=activation,
            norm_num_groups=8,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.diffusers_unet(x, t.squeeze(), return_dict=False)[0]


class UNet1DSignal(nn.Module):
    """
    Adaptation of diffusers UNet1D.
    """
    from diffusers.models import UNet1DModel as DiffusersUNet

    def __init__(
        self,
        k: int,
        dim: int,
        activation: str = "swish",
    ):
        super().__init__()
        self.diffusers_unet = self.DiffusersUNet(
            sample_size=k,
            in_channels=dim+2,
            out_channels=dim,
            block_out_channels=(64, 128,),
            down_block_types=("DownBlock1D", "DownBlock1D"),
            up_block_types=("UpBlock1D", "UpBlock1D"),
            act_fn=activation,
            norm_num_groups=16,
        )

    def forward(self, x: Tensor, t: Tensor, signal: Tensor) -> Tensor:
        x = torch.cat([x, signal], dim=-1)
        x = x.transpose(1, 2)
        return self.diffusers_unet(x, t.squeeze(), return_dict=False)[0].transpose(1, 2)


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class CNNModel(nn.Module):
    def __init__(self,
            dim: int,
            k: int,
            hidden: int,
            mode: str,
            num_cls: int,
            depth: int,
            dropout: float,
            prior_pseudocount: float,
            cls_expanded_simplex: bool,
            clean_data: bool = False,
            classifier: bool = False,
            classifier_free_guidance: bool = False,
            **_,
        ):
        super().__init__()
        self.dim = dim
        self.k = k
        self.hidden = hidden
        self.mode = mode
        self.depth = depth
        self.dropout = dropout
        self.prior_pseudocount = prior_pseudocount
        self.cls_expanded_simplex = cls_expanded_simplex
        self.classifier = classifier
        self.cls_free_guidance = classifier_free_guidance
        self.clean_data = clean_data
        self.num_cls = num_cls

        if self.clean_data:
            self.linear = nn.Embedding(self.dim, embedding_dim=hidden)
        else:
            expanded_simplex_input = self.cls_expanded_simplex or not classifier and (self.mode == 'dirichlet' or self.mode == 'riemannian')
            inp_size = self.dim * (2 if expanded_simplex_input else 1)
            print("input size is %d" % (inp_size))
            if (self.mode == 'ardm' or self.mode == 'lrar') and not classifier:
                inp_size += 1 # plus one for the mask token of these models
            self.linear = nn.Conv1d(inp_size, self.hidden, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(
                                                GaussianFourierProjection(
                                                    embed_dim=self.hidden
                                                    ),
                                                nn.Linear(
                                                    self.hidden,
                                                    self.hidden
                                                    )
                                                )

        self.num_layers = 5 * self.depth
        self.convs = [nn.Conv1d(self.hidden, self.hidden, kernel_size=9, padding=4),
                                     nn.Conv1d(self.hidden, self.hidden, kernel_size=9, padding=4),
                                     nn.Conv1d(self.hidden, self.hidden, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(self.hidden, self.hidden, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(self.hidden, self.hidden, kernel_size=9, dilation=64, padding=256)]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(self.depth)])
        self.time_layers = nn.ModuleList([Dense(self.hidden, self.hidden) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(nn.Conv1d(self.hidden, self.hidden, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv1d(self.hidden, self.hidden
                                       if classifier else self.dim, kernel_size=1))
        self.dropout = nn.Dropout(self.dropout)
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(self.hidden, self.hidden),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden, self.num_cls))
        if self.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=self.hidden)
            self.cls_layers = nn.ModuleList([Dense(self.hidden, self.hidden) for _ in range(self.num_layers)])

    def forward(self, x, t: Tensor, cls = None, return_embedding=False):
        seq = x.view(-1, self.k, self.dim)
        if len(t.shape) == 0:
            # odeint is on
            t = t[None].expand(seq.size(0))
        if self.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            t = t.squeeze()
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat

    def missing_coordinate(self) -> bool:
        return False


def expand_simplex(xt, alphas, prior_pseudocount):
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:, None, None]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1), prior_weights
