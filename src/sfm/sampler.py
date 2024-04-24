"""Defines sampling methods; useful for OT-sampling."""
import torch
from torch import Tensor


from src.sfm import Manifold, str_to_ot_method


class OTSampler:
    """
    Based on:
    `https://github.com/DreamFold/FoldFlow/blob/main/FoldFlow/utils/optimal_transport.py`.
    """

    def __init__(
        self,
        manifold: Manifold,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
    ):
        """
        Parameters:
            - `manifold`: the underlying manifold; useful for the geodesic
                distance;
            - `method`: the OT method;
            - `reg`: parameter for the OT method;
            - `reg_m`: parameter for the OT method, can be ignored depending on method.
        """
        self.manifold = manifold
        self.ot_fn = str_to_ot_method(method, reg, reg_m)
        self.normalize_cost = normalize_cost
 
    @torch.no_grad()
    def get_map(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        Compute the OT plan between a source and a target minibatch.
        """
        a, b = (
            torch.full((x0.shape[0],), 1.0 / x0.shape[0], device=x1.device),
            torch.full((x1.shape[0],), 1.0 / x1.shape[0], device=x1.device),
        )
        m = self.manifold.pairwise_geodesic_distance(x0, x1)
        if self.normalize_cost:
            m = m / m.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, m)
        # if not torch.all(torch.isfinite(p)):
        #     print("ERROR: p is not finite")
        #     print(p)
        #     print("Cost mean, max", m.mean(), m.max())
        #     print(x0, x1)
        #     raise ValueError("p is not finite")
        return p

    def sample_map(self, pi: Tensor, batch_size: int):
        """
        Draw source and target samples from `pi`, $(x,z) \sim \pi$.
        """
        p = pi.flatten()
        p = p / p.sum()
        # choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        # return np.divmod(choices, pi.shape[1])
        choices = torch.multinomial(
            p, num_samples=batch_size, replacement=True,
        ).long()
        return torch.floor_divide(choices, pi.shape[1]), torch.remainder(choices, pi.shape[1])

    def sample_plan(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from `pi` $(x,z) \sim \pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0])
        return x0[i], x1[j]
