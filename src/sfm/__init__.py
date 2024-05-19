"""Utils for the entire project."""
from .maths import (
    fast_dot,
    safe_arccos,
    usinc,
)
from .powerspherical import PowerSpherical
from .manifold import (
    GeooptSphere,
    Manifold,
    NSimplex,
    NSphere,
    manifold_from_name,
    str_to_ot_method,
    default_perturbation_schedule,
    metropolis_sphere_perturbation,
)
from .distribution import (
    estimate_categorical_kl,
    set_seeds,
)
from .misc import (
    reset_memory,
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
