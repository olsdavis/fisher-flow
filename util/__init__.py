"""Utils for the entire project."""
from .distribution import (
    generate_dirichlet,
    generate_dirichlet_mixture,
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
    str_to_activation,
)
from .plot import (
    define_style,
    save_plot,
)
from .sampler import (
    OTSampler,
)
