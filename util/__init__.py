"""Utils for the entire project."""
from .distribution import (
    estimate_categorical_kl,
    generate_dirichlet,
    generate_dirichlet_mixture,
    generate_dirichlet_product,
    set_seeds,
)
from .maths import (
    safe_arccos,
    usinc,
)
from .manifold import (
    Manifold,
    NSimplex,
    str_to_ot_method,
)
from .model import (
    MLP,
    ProductMLP,
    TembMLP,
    str_to_activation,
)
from .plot import (
    define_style,
    save_plot,
)
from .sampler import (
    OTSampler,
)
from .train import (
    cft_loss_function,
    dfm_train_step,
    ot_train_step,
)
