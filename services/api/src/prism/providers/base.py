"""Lane definitions, and the errors the registry raises.

Four endings, deliberately not collapsed: `UnknownLane` (a typo),
`LaneNotImplemented` (defined lane, no provider), `LaneRejected` (never reached
the provider) and the provider's own error. Only a provider's own error is
evidence about it, and only those count toward a breaker (ADR 0014, amended by
0018 to include embeddings).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from prism.chat import ChatError, ChatProvider, Usage
from prism.embeddings import EmbeddingError

__all__ = [
    "PROVIDER_ERRORS",
    "BillingUnit",
    "Lane",
    "LaneBusy",
    "LaneError",
    "LaneNotImplemented",
    "LaneRejected",
    "LaneResult",
    "LaneUnavailable",
    "UnknownLane",
]

BillingUnit = Literal["tokens", "gpu_ms", "none"]

# A call that reached the provider and failed there (ADR 0018).
PROVIDER_ERRORS: tuple[type[Exception], ...] = (ChatError, EmbeddingError)


class LaneError(RuntimeError):
    """Anything the registry itself raises, as opposed to a provider."""


class UnknownLane(LaneError):
    def __init__(self, name: str, known: tuple[str, ...]) -> None:
        super().__init__(f"no lane named {name!r}; lanes are {', '.join(known)}")
        self.lane = name


class LaneNotImplemented(LaneError, NotImplementedError):
    """Defined — cap, unit, timeout — with nothing wired behind it."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"the {name!r} lane is defined but has no provider yet; "
            "it does not fall back to a cheaper lane"
        )
        self.lane = name


class LaneRejected(LaneError):
    """The provider was never called. Distinct from the provider failing."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.lane = name


class LaneUnavailable(LaneRejected):
    """The lane's circuit breaker is open."""

    def __init__(self, name: str, *, retry_after_s: float) -> None:
        super().__init__(name, f"the {name!r} lane is open; retry in {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s


class LaneBusy(LaneRejected):
    """No slot free within the wait. Our queue, not the provider — never a breaker failure."""

    def __init__(self, name: str, *, waited_s: float, concurrency: int) -> None:
        super().__init__(
            name,
            f"no slot on the {name!r} lane after {waited_s:.1f}s (concurrency {concurrency})",
        )
        self.waited_s = waited_s


@dataclass(frozen=True)
class Lane:
    """A lane's guards, and what is behind them.

    `timeout_s` bounds one call and is enforced by the provider; `queue_timeout_s`
    bounds the wait for a slot and is enforced here. `factory` is None until a
    lane is wired.
    """

    name: str
    billing_unit: BillingUnit
    concurrency: int
    queue_timeout_s: float
    timeout_s: float
    model: str | None = None
    factory: Callable[[], ChatProvider] | None = None

    @property
    def implemented(self) -> bool:
        return self.factory is not None


@dataclass(frozen=True)
class LaneResult[V]:
    """The provider's `Usage` unchanged, plus the lane that served it. Never priced."""

    lane: str
    value: V
    usage: Usage
