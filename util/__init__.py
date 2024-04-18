"""Utils for the entire project."""
from .maths import (
    fast_dot,
    safe_arccos,
    usinc,
)
from .manifold import (
    Manifold,
    NSimplex,
    NSphere,
    str_to_ot_method,
)
from .distribution import (
    estimate_categorical_kl,
    set_seeds,
)
from .misc import (
    reset_memory,
)
from .model import (
    BestMLP,
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
