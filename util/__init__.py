"""Utils for the entire project."""
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
from .maths import (
    fast_dot,
    safe_arccos,
    usinc,
)
from .manifold import (
    Manifold,
    NSimplex,
    NSphere,
    GeomNSphere,
    manifold_from_name,
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
    CNNModel,
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
