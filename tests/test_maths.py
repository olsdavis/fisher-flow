"""Testing maths functions."""
import unittest
import torch
from src.sfm import safe_arccos, usinc


class TestUSinc(unittest.TestCase):
    """Tests usinc."""

    @torch.no_grad()
    def test_usinc(self):
        """Tests whether the usinc function is not ill-defined."""
        self.assertTrue(
            torch.allclose(usinc(torch.Tensor([0.0])), torch.Tensor([1.0]), atol=1e-7),
            "sinc(0) = 1",
        )
        self.assertTrue(
            torch.allclose(usinc(torch.Tensor([torch.pi])), torch.Tensor([0.0]), atol=1e-7),
            "sinc(pi) = 0",
        )
        self.assertTrue(
            torch.allclose(usinc(torch.Tensor([torch.pi / 2.0])), torch.Tensor([2.0 / torch.pi]), atol=1e-7),
            "sinc(1) = 0",
        )


class TestSafeArccos(unittest.TestCase):
    """Tests arccos."""

    @torch.no_grad()
    def test_safe_arccos(self):
        """Tests whether the safe arccos function is not ill-defined."""
        self.assertTrue(
            torch.allclose(safe_arccos(torch.tensor(1.0)), torch.tensor(0.0)),
            "arccos(1) = 0",
        )
        self.assertTrue(
            torch.allclose(safe_arccos(torch.tensor(-1.0)), torch.tensor(torch.pi)),
            "arccos(-1) = pi",
        )
        self.assertTrue(
            torch.allclose(safe_arccos(torch.tensor(0.0)), torch.tensor(torch.pi / 2.0)),
            "arccos(0) = pi / 2",
        )


if __name__ == "__main__":
    unittest.main()
