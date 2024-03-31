"""Config files for models."""
from dataclasses import dataclass, asdict
from torch import nn
from util import ProductMLP, TembMLP
from .general import _load_config_raw


@dataclass
class ModelConfig:
    """
    Defines config files for ML models.
    """
    name: str
    depth: int
    hidden: int
    activation: str
    emb_size: int | None = 64
    time_emb: str | None = "sinusoidal"
    input_emb: str | None = "sinusoidal"
    add_t_emb: bool | None = False
    concat_t_emb: bool | None = False
    # simplex_tangent is not here, as it is defined by the training method
    # same for k, d


def load_model_config(path: str) -> ModelConfig:
    """
    Loads the config file at path `path`.
    """
    return ModelConfig(**_load_config_raw(path))


def model_from_config(
    k: int,
    dim: int,
    simplex_tangent: bool,
    config: ModelConfig,
) -> nn.Module:
    """
    Creates a model from the provided arguments.

    Parameters:
        - `k`: the number of spaces in the product;
        - `dim`: the dimension in each space;
        - `simplex_tangent`: whether the model should output points on the
            tangent space of the simplex;
        - `config`: the config itself.
    """
    models_available = {
        "ProductMLP": ProductMLP,
        "TembMLP": TembMLP,
    }
    return models_available[config.name](
        k=k,
        dim=dim,
        simplex_tangent=simplex_tangent,
        **asdict(config),
    )
