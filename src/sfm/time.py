"""
Time schedules for flows.
"""
from abc import ABC, abstractmethod
import torch
from torch import Tensor


class TimeSchedule(ABC):
    """
    Abstract class for time schedulers.
    """

    @abstractmethod
    def alpha(self, t: Tensor) -> Tensor:
        """
        Returns the time schedule at time `t`.
        """

    @abstractmethod
    def alpha_prime(self, t: Tensor) -> Tensor:
        """
        Returns the derivative of the time schedule at time `t`.
        """


class LinearSchedule(TimeSchedule):
    """
    `alpha(t) = t` schedule.
    """

    def alpha(self, t: Tensor) -> Tensor:
        return t

    def alpha_prime(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)


class QuadraticSchedule(TimeSchedule):
    """
    `alpha(t) = t^2` schedule from Discrete Flow Matching by Gat, et al.
    """

    def alpha(self, t: Tensor) -> Tensor:
        return t.square()

    def alpha_prime(self, t: Tensor) -> Tensor:
        return 2.0 * t


class SqrtSchedule(TimeSchedule):
    """
    `alpha(t) = sqrt(t)` schedule.
    """

    def alpha(self, t: Tensor) -> Tensor:
        return t.sqrt()

    def alpha_prime(self, t: Tensor) -> Tensor:
        return 0.5 * (t + 1e-7).pow(-0.5)


class SineSchedule(TimeSchedule):
    """
    `alpha(t) = sin(pi * t / 2)` schedule.
    """

    def alpha(self, t: Tensor) -> Tensor:
        return (t * torch.pi / 2.0).sin()

    def alpha_prime(self, t: Tensor) -> Tensor:
        return ((t * torch.pi / 2.0).cos() * torch.pi / 2.0).clamp(min=1e-8)


class CubicSchedule(TimeSchedule):
    """
    `alpha(t) = 1 - (1 - t)^3` schedule.
    """

    def alpha(self, t: Tensor) -> Tensor:
        return 1.0 - (1.0 - t).pow(3)

    def alpha_prime(self, t: Tensor) -> Tensor:
        return 3.0 * (1.0 - t).pow(2)


def time_schedule_from_name(name: str) -> TimeSchedule:
    """
    Returns the time schedule from its name.
    """
    return {
        "linear": LinearSchedule(),
        "quadratic": QuadraticSchedule(),
        "sqrt": SqrtSchedule(),
        "sine": SineSchedule(),
        "cubic": CubicSchedule(),
    }[name]
