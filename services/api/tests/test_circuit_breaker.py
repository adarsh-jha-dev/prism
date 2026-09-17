"""Unit: the per-lane breaker's state machine, on a clock the test controls."""

import pytest
from structlog.testing import capture_logs

from prism.providers import CircuitBreaker, LaneUnavailable


class Clock:
    """A monotonic clock that only moves when a test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _breaker(*, threshold: int = 3, cooldown_s: float = 30.0) -> tuple[CircuitBreaker, Clock]:
    clock = Clock()
    return (
        CircuitBreaker("stub", threshold=threshold, cooldown_s=cooldown_s, clock=clock),
        clock,
    )


def test_starts_closed() -> None:
    breaker, _ = _breaker()
    assert breaker.state == "closed"
    breaker.check()


def test_opens_on_the_nth_consecutive_failure_and_not_before() -> None:
    breaker, _ = _breaker(threshold=3)

    for _ in range(2):
        breaker.record_failure("boom")
        breaker.check()  # still admitting callers

    assert breaker.state == "closed"
    breaker.record_failure("boom")
    assert breaker.state == "open"


def test_a_success_resets_the_count() -> None:
    breaker, _ = _breaker(threshold=3)

    breaker.record_failure("boom")
    breaker.record_failure("boom")
    breaker.record_success()
    breaker.record_failure("boom")
    breaker.record_failure("boom")

    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 2


def test_an_open_breaker_rejects_and_says_when_to_come_back() -> None:
    breaker, clock = _breaker(threshold=1, cooldown_s=30.0)
    breaker.record_failure("boom")
    clock.advance(10.0)

    with pytest.raises(LaneUnavailable) as caught:
        breaker.check()

    assert caught.value.lane == "stub"
    assert caught.value.retry_after_s == pytest.approx(20.0)


def test_cooldown_admits_exactly_one_probe() -> None:
    breaker, clock = _breaker(threshold=1, cooldown_s=30.0)
    breaker.record_failure("boom")
    clock.advance(30.0)

    breaker.check()
    assert breaker.state == "half_open"

    # Everyone else waits for the probe to resolve.
    with pytest.raises(LaneUnavailable):
        breaker.check()


def test_a_successful_probe_closes_the_breaker() -> None:
    breaker, clock = _breaker(threshold=1, cooldown_s=30.0)
    breaker.record_failure("boom")
    clock.advance(30.0)
    breaker.check()

    breaker.record_success()

    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 0
    breaker.check()


def test_a_failed_probe_reopens_and_restarts_the_cooldown() -> None:
    breaker, clock = _breaker(threshold=3, cooldown_s=30.0)
    for _ in range(3):
        breaker.record_failure("boom")
    clock.advance(30.0)
    breaker.check()

    # One failure is enough in half-open; the threshold does not apply again.
    breaker.record_failure("still down")

    assert breaker.state == "open"
    clock.advance(29.0)
    with pytest.raises(LaneUnavailable):
        breaker.check()
    clock.advance(1.0)
    breaker.check()


def test_releasing_a_probe_that_never_ran_frees_the_lane() -> None:
    breaker, clock = _breaker(threshold=1, cooldown_s=30.0)
    breaker.record_failure("boom")
    clock.advance(30.0)
    breaker.check()

    breaker.release_probe()

    breaker.check()  # the next caller gets the slot, not a permanent rejection
    assert breaker.state == "half_open"


def test_every_transition_logs() -> None:
    breaker, clock = _breaker(threshold=1, cooldown_s=30.0)

    with capture_logs() as logs:
        breaker.record_failure("boom")
        clock.advance(30.0)
        breaker.check()
        breaker.record_success()

    transitions = [
        (entry["from"], entry["to"]) for entry in logs if entry["event"] == "lane_breaker"
    ]
    assert transitions == [("closed", "open"), ("open", "half_open"), ("half_open", "closed")]
    assert all(entry["lane"] == "stub" and entry["reason"] for entry in logs)


def test_a_breaker_that_opens_on_zero_failures_is_refused() -> None:
    with pytest.raises(ValueError, match="always open"):
        CircuitBreaker("stub", threshold=0, cooldown_s=1.0)
