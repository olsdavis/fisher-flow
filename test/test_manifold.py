"""Testing manifolds."""
import unittest
import torch
from util import NSimplex, NSphere


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


class TestNSphere(unittest.TestCase):
    """Tests n-spheres."""

    @torch.no_grad()
    def test_n_sphere_exp_log_basic(self):
        """
        Tests whether the log and exp map are defined correctly.
        """
        m = NSphere()
        x_0 = torch.Tensor([0.5, 0.5, 0.5, 0.5])
        x_1 = torch.Tensor([0.20, 0.20, 0.50, 0.10]).sqrt()
        back = m.exp_map(x_0, m.log_map(x_0, x_1))
        self.assertTrue(
            torch.allclose(back, x_1),
            f"too large difference: {back}, {x_1}"
        )

    @torch.no_grad()
    def test_n_sphere_geodesic_distance(self):
        """
        Tests whether the geodesic distance is correct.
        """
        m = NSphere()
        x_0 = torch.Tensor([[[1.0, 0.0]]])
        x_1 = torch.Tensor([[[0.0, 1.0]]])
        dist = m.geodesic_distance(x_0, x_1)
        self.assertTrue(
            torch.allclose(dist, torch.tensor(torch.pi/2)),
            f"too large difference: {dist}"
        )

    @torch.no_grad()
    def test_n_sphere_make_tangent(self):
        """
        Tests whether the make_tangent function is correct.
        """
        m = NSphere()
        x = torch.Tensor([[[0.5, 0.5, 0.5, 0.5]]])
        v = torch.Tensor([[[0.1, 0.2, 0.3]]])
        tangent = m.make_tangent(x, v)
        self.assertTrue(
            torch.allclose((tangent * x).sum(), torch.zeros(1)),
            f"too large difference: {tangent}",
        )


if __name__ == "__main__":
    unittest.main()
