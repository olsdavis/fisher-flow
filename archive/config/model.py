"""Config files for models."""
from dataclasses import dataclass, asdict
from torch import nn
from util import BestMLP, ProductMLP, TembMLP, CNNModel
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
    batch_norm: bool | None = False
    clean_data: bool | None = False
    classifier: bool | None = False
    classifier_free_guidance: bool | None = False
    cls_expanded_simplex: bool | None = False
    num_cls: int | None = 3
    mode: str | None = "dirichlet"
    dropout: float | None = 0.0
    prior_pseudocount: float | None = 2.0

    # k, d defined by task


def load_model_config(path: str) -> ModelConfig:
    """
    Loads the config file at path `path`.
    """
    return ModelConfig(**_load_config_raw(path))


def model_from_config(
    k: int,
    dim: int,
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
        "BestMLP": BestMLP,
        "CNNModel": CNNModel,
    }
    return models_available[config.name](
        k=k,
        dim=dim,
        **asdict(config),
    )
