"""Testing manifolds."""
import unittest
import torch
from util import NSimplex


class TestNSimplex(unittest.TestCase):
    """Tests n-simplices."""

    @torch.no_grad()
    def test_n_simplex_exp_log_basic(self):
        """
        Tests whether the log and exp map are defined correctly.
        """
        m = NSimplex()
        x_0 = torch.Tensor([0.25, 0.5, 0.25])
        x_1 = torch.Tensor([0.4, 0.3, 0.3])
        back = m.exp_map(x_0, m.log_map(x_0, x_1))
        self.assertTrue(
            torch.allclose(back, x_1),
            f"too large difference: {back}, {x_1}"
        )


if __name__ == "__main__":
    unittest.main()
