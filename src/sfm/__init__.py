"""Utils for the entire project."""
from .maths import (
    fast_dot,
    safe_arccos,
    usinc,
)
from .manifold import (
    GeooptSphere,
    Manifold,
    NSimplex,
    NSphere,
    manifold_from_name,
    str_to_ot_method,
)
from .distribution import (
    compute_exact_loglikelihood,
    estimate_categorical_kl,
    get_wasserstein_dist,
    set_seeds,
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
    ot_train_step,
)
