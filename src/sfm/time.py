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
        return t ** 2

    def alpha_prime(self, t: Tensor) -> Tensor:
        return 2 * t


def time_schedule_from_name(name: str) -> TimeSchedule:
    """
    Returns the time schedule from its name.
    """
    return {
        "linear": LinearSchedule(),
        "quadratic": QuadraticSchedule(),
    }[name]
