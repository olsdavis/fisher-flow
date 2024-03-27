"""Utils for the entire project."""
from .distribution import (
    generate_dirichlet,
    generate_dirichlet_mixture,
)
from .manifold import (
    Manifold,
    NSimplex,
)
from .model import (
    MLP,
    str_to_activation,
)
from .maths import (
    usinc,
)
from .plot import (
    define_style,
)
