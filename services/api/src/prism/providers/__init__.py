"""Lane name in, guarded provider call out."""

from functools import lru_cache

from prism.providers.base import (
    BillingUnit,
    Lane,
    LaneBusy,
    LaneError,
    LaneNotImplemented,
    LaneRejected,
    LaneResult,
    LaneUnavailable,
    UnknownLane,
)
from prism.providers.breaker import BreakerState, CircuitBreaker
from prism.providers.lanes import LANE_NAMES, build_lanes
from prism.providers.registry import ProviderRegistry

__all__ = [
    "LANE_NAMES",
    "BillingUnit",
    "BreakerState",
    "CircuitBreaker",
    "Lane",
    "LaneBusy",
    "LaneError",
    "LaneNotImplemented",
    "LaneRejected",
    "LaneResult",
    "LaneUnavailable",
    "ProviderRegistry",
    "UnknownLane",
    "build_lanes",
    "get_registry",
]


@lru_cache
def get_registry() -> ProviderRegistry:
    """The process's registry. Its breakers' state is exactly this process's view."""
    return ProviderRegistry()
