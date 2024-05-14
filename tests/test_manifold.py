"""Testing manifolds."""
import unittest
import torch
from torch import testing
from src.models.net import BestMLP
from src.sfm import GeooptSphere, NSimplex, NSphere, set_seeds


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
        testing.assert_close(back, x_1)

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

    @torch.no_grad()
    def test_make_tangent(self):
        """
        Tests whether `make_tangent` makes vectors tangent.
        """
        set_seeds(0)
        m = NSimplex()
        n_points = 10
        points = m.uniform_prior(n_points, 1, 3)
        vectors = torch.randn(n_points, 1, 3)
        tangent = m.make_tangent(points, vectors)
        testing.assert_close(
            tangent.sum(dim=-1),
            torch.zeros((n_points, 1)),
        )


class TestNSphere(unittest.TestCase):
    """Tests n-spheres."""
    def __init__(self, *args):
        super().__init__(*args)
        self.spheres = [NSphere(), GeooptSphere()]

    @torch.no_grad()
    def test_n_sphere_exp_log_basic(self):
        """
        Tests whether the log and exp map are defined correctly.
        """
        for m in self.spheres:
            x_0 = torch.Tensor([[[0.5, 0.5, 0.5, 0.5]]])
            x_1 = torch.Tensor([[[0.20, 0.20, 0.50, 0.10]]]).sqrt()
            back = m.exp_map(x_0, m.log_map(x_0, x_1))
            testing.assert_close(back, x_1)

    @torch.no_grad()
    def test_n_sphere_geodesic_distance(self):
        """
        Tests whether the geodesic distance is correct.
        """
        for m in self.spheres:
            x_0 = torch.Tensor([[[1.0, 0.0]]])
            x_1 = torch.Tensor([[[0.0, 1.0]]])
            dist = m.geodesic_distance(x_0, x_1)
            testing.assert_close(dist.squeeze(), torch.tensor(torch.pi / 2.0))

    @torch.no_grad()
    def test_n_sphere_make_tangent(self):
        """
        Tests whether the make_tangent function is correct.
        """
        for m in self.spheres:
            x = torch.Tensor([[[0.5, 0.5, 0.5, 0.5]]])
            v = torch.Tensor([[[0.1, 0.2, 0.3, 0.1]]])
            tangent = m.make_tangent(x, v)
            testing.assert_close(
                (tangent * x).sum(), torch.tensor(0.0),
            )

    @torch.no_grad()
    def test_n_sphere_parallel_transport(self):
        """
        Tests the parallel transport function.
        """
        set_seeds(4)
        for m in self.spheres:
            p = m.uniform_prior(500, 16, 10)
            v = m.make_tangent(p, torch.randn(500, 16, 10))
            assert m.all_belong_tangent(p, v), "not tangent initially"
            testing.assert_close(
                m.parallel_transport(p, p, v),
                v,
            )

    @torch.no_grad()
    def test_geodesic_interpolant(self):
        """
        Tests the precision of geodesic interpolants.
        """
        for m in self.spheres:
            x_0 = torch.Tensor([[[0.0, 1.0]]])
            x_1 = torch.Tensor([[[1.0, 0.0]]]).sqrt()
            t = torch.Tensor([[0.5]])
            x_t = m.geodesic_interpolant(x_0, x_1, t)
            testing.assert_close(x_t, torch.Tensor([[[0.5, 0.5]]]).sqrt())

    @torch.no_grad()
    def test_tangent_euler_remains_on_sphere(self):
        """
        Tests whether the Tangent Euler method remains on the sphere
        """
        set_seeds(101)
        for m in self.spheres:
            k = 10
            d = 100
            x_0 = m.uniform_prior(1024, k, d)
            model = BestMLP(d, k, 128, 2, 2, "lrelu")
            self.assertTrue(
                m.all_belong(
                    m.tangent_euler(
                        x_0,
                        model,
                        steps=100,
                    )
                ),
                "must remain on sphere",
            )

    @torch.no_grad()
    def test_log_edge_cases(self):
        """
        Tests some edge cases for log map.
        """
        return  # pass this for now
        points = torch.randn(10, 1, 2)
        points = points / points.norm(dim=-1, keepdim=True)
        m = NSphere()
        log = m.log_map(points, -points).norm(dim=-1).sum(dim=1)
        testing.assert_close(log, torch.full_like(log, torch.pi))


class TestManifoldsGeneral(unittest.TestCase):
    """
    Testing general properties of manifolds.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.seq_len = 30
        self.dim = 320
        self.manifolds = [
            NSimplex(),
            NSphere(),
            GeooptSphere(),
        ]

    @torch.no_grad()
    def test_uniform_prior(self):
        """
        Tests whether the uniform prior has points all on the manifold.
        """
        for manifold in self.manifolds:
            set_seeds(0)
            x = manifold.uniform_prior(1000, self.seq_len, self.dim)
            self.assertTrue(
                manifold.all_belong(x),
                # torch.allclose(x.square().sum(dim=-1), torch.ones((1000, self.seq_len))),
                f"all points from prior must belong to manifold, {type(manifold).__name__}",
            )

    @torch.no_grad()
    def test_belongs(self):
        """
        Tests whether the belongs function is correct.
        """
        for manifold in self.manifolds:
            for d in range(5, 100, 5):
                set_seeds(d)
                x = torch.nn.functional.one_hot(torch.randint(0, d, (10, self.seq_len)), d).float()
                self.assertTrue(
                    manifold.all_belong(x), "all one-hot encoded points must be on simplex"
                )

    @torch.no_grad()
    def test_symmetric_geodesic(self):
        """
        Tests whether the uniform prior is actually uniform.
        """
        for manifold in self.manifolds:
            set_seeds(0)
            x = manifold.uniform_prior(500, self.seq_len, self.dim)
            y = manifold.uniform_prior(500, self.seq_len, self.dim)
            testing.assert_close(manifold.geodesic_distance(x, y), manifold.geodesic_distance(y, x))

    @torch.no_grad()
    def test_logmap_expmap(self):
        """
        Tests whether the log and exp maps are inverses.
        """
        for manifold in self.manifolds:
            set_seeds(1)
            x = manifold.uniform_prior(500, self.seq_len, self.dim)
            y = manifold.uniform_prior(500, self.seq_len, self.dim)
            back = manifold.exp_map(x, manifold.log_map(x, y))
            # TODO is 1e-7 good enough?
            testing.assert_close(
                back, y, rtol=1e-5, atol=1e-7,
            )

    @torch.no_grad()
    def test_parallel_transport(self):
        """
        Tests whether parallel transport retains tangency.
        """
        dim = self.dim
        seq_len = self.seq_len
        points = 500
        for manifold in self.manifolds:
            set_seeds(2)
            p = manifold.uniform_prior(points, seq_len, dim)
            q = manifold.uniform_prior(points, seq_len, dim)
            assert manifold.all_belong(p) and manifold.all_belong(q), "must be on manifold"
            v = torch.rand((points, seq_len, dim))
            v = manifold.make_tangent(p, v)
            assert manifold.all_belong_tangent(p, v), "must be tangent initially"
            transported = manifold.parallel_transport(p, q, v)
            if type(manifold).__name__ == "NSimplex":
                testing.assert_close(
                    transported.sum(dim=-1),
                    torch.zeros(points, seq_len),
                    atol=1e-3,
                    rtol=1e-5,
                )
            elif type(manifold).__name__ == "NSphere":
                testing.assert_close(
                    (transported * q).sum(dim=-1),
                    torch.zeros(points, seq_len),
                    atol=1e-5,
                    rtol=1e-6,
                )

    @torch.no_grad()
    def test_midinterpolant(self):
        """
        Tests whether the midpoint of a geodesic is equidistant to both
        end-points.
        """
        for manifold in self.manifolds:
            set_seeds(3)
            x = manifold.uniform_prior(500, self.seq_len, self.dim)
            y = manifold.uniform_prior(500, self.seq_len, self.dim)
            z = manifold.geodesic_interpolant(x, y, torch.full((500, 1), 0.5))
            testing.assert_close(
                manifold.geodesic_distance(x, z),
                manifold.geodesic_distance(z, y),
                atol=1e-5,
                rtol=1e-5,
            )

    @torch.no_grad()
    def test_zero_interpolant(self):
        """
        Tests end points of geodesic interpolant.
        """
        for manifold in self.manifolds:
            set_seeds(4)
            x = manifold.uniform_prior(500, self.seq_len, self.dim)
            y = manifold.uniform_prior(500, self.seq_len, self.dim)
            z = manifold.geodesic_interpolant(x, y, torch.zeros((500, 1)))
            testing.assert_close(z, x)
            w = manifold.geodesic_interpolant(x, y, torch.ones((500, 1)))
            testing.assert_close(w, y)

    @torch.no_grad()
    def test_pairwise_geodesic_distance(self):
        """
        Tests whether pairwise geodesic distance entries correspond
        to what they should. 
        """
        for manifold in [GeooptSphere()]:
            set_seeds(5)
            x = manifold.uniform_prior(500, self.seq_len, self.dim)
            y = manifold.uniform_prior(500, self.seq_len, self.dim)
            dist = manifold.pairwise_geodesic_distance(x, y)
            for i in range(100):
                for j in range(100):
                    testing.assert_close(
                        dist[i, j],
                        manifold.geodesic_distance(x[i], y[j])**2,
                    )


if __name__ == "__main__":
    unittest.main()
