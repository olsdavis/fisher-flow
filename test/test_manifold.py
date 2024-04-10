"""Testing manifolds."""
import unittest
import torch
from util import NSimplex, NSphere, set_seeds


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

    @torch.no_grad()
    def test_log_map(self):
        """
        Tests the log map on close values.
        """
        # got this from an actual existing run
        x_0 = torch.Tensor([0.3978, 0.3383, 0.2639])
        x_1 = torch.Tensor([0.3975, 0.3384, 0.2641])
        self.assertTrue(
            not torch.any(torch.isnan(NSimplex().log_map(x_0, x_1))),
            "NaNs in log map",
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


class TestManifoldsGeneral(unittest.TestCase):
    """
    Testing general properties of manifolds.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.manifolds = [NSimplex(), NSphere()]

    @torch.no_grad()
    def test_symmetric_geodesic(self):
        """
        Tests whether the uniform prior is actually uniform.
        """
        for manifold in self.manifolds:
            set_seeds(0)
            x = manifold.uniform_prior(500, 3, 3)
            y = manifold.uniform_prior(500, 3, 3)
            self.assertTrue(
                torch.allclose(manifold.geodesic_distance(x, y), manifold.geodesic_distance(y, x)),
                f"geodesic not uniform for {manifold}",
            )

    @torch.no_grad()
    def test_logmap_expmap(self):
        """
        Tests whether the log and exp maps are inverses.
        """
        for manifold in self.manifolds:
            set_seeds(1)
            x = manifold.uniform_prior(500, 3, 3)
            y = manifold.uniform_prior(500, 3, 3)
            back = manifold.exp_map(x, manifold.log_map(x, y))
            self.assertTrue(
                torch.allclose(back, y, atol=1e-7),
                f"too large difference: {manifold} - {back}, {y}"
            )


if __name__ == "__main__":
    unittest.main()
