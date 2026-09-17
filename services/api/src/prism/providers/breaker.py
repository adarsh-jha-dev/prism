"""One circuit breaker per lane, in this process's memory (ADR 0014).

Consecutive failures, not an error rate over a window. Half-open admits exactly
one trial call and rejects every other caller until it resolves.
"""

import time
from collections.abc import Callable
from typing import Literal

import structlog

from prism.providers.base import LaneUnavailable

__all__ = ["BreakerState", "CircuitBreaker"]

log = structlog.get_logger(__name__)

BreakerState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    def __init__(
        self,
        lane: str,
        *,
        threshold: int,
        cooldown_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if threshold < 1:
            raise ValueError("a breaker that opens on zero failures is always open")
        self._lane = lane
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._state: BreakerState = "closed"
        self._failures = 0
        self._opened_at = 0.0
        self._probing = False

    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    def check(self) -> None:
        """Admit this caller, or raise. Never blocks."""
        if self._state == "open":
            waited = self._clock() - self._opened_at
            if waited < self._cooldown_s:
                raise LaneUnavailable(self._lane, retry_after_s=self._cooldown_s - waited)
            self._transition("half_open", reason="cooldown_elapsed")

        if self._state == "half_open":
            if self._probing:
                raise LaneUnavailable(self._lane, retry_after_s=self._cooldown_s)
            self._probing = True

    def record_success(self) -> None:
        self._probing = False
        self._failures = 0
        if self._state != "closed":
            self._transition("closed", reason="probe_succeeded")

    def record_failure(self, reason: str) -> None:
        """Count a failed provider call. Only a call that reached the provider gets here."""
        was_probing = self._probing
        self._probing = False
        self._failures += 1

        if was_probing or self._state == "half_open":
            self._open(reason=f"probe_failed: {reason}")
        elif self._state == "closed" and self._failures >= self._threshold:
            self._open(reason=f"{self._failures} consecutive failures: {reason}")

    def release_probe(self) -> None:
        """Give back a half-open slot for a call that never happened.

        A probe left in flight rejects every later caller with no failure to
        explain why.
        """
        self._probing = False

    def _open(self, *, reason: str) -> None:
        self._opened_at = self._clock()
        self._transition("open", reason=reason)

    def _transition(self, state: BreakerState, *, reason: str) -> None:
        previous, self._state = self._state, state
        log.warning(
            "lane_breaker",
            lane=self._lane,
            **{"from": previous},
            to=state,
            reason=reason,
            consecutive_failures=self._failures,
            cooldown_s=self._cooldown_s,
        )
