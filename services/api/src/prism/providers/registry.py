"""One interface in front of the lanes, with the guards around every call.

Guard order: resolve the lane and its provider, then the breaker, then the
semaphore under a bounded wait, then the call. An open lane raises before it
queues. Which lane to use is the Phase 3 router; cost is ADR 0013.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence

import structlog
from pydantic import BaseModel

from prism.chat import ChatError, ChatProvider, Message, Usage
from prism.config import Settings, get_settings
from prism.providers.base import (
    Lane,
    LaneBusy,
    LaneNotImplemented,
    LaneResult,
    LaneUnavailable,
    UnknownLane,
)
from prism.providers.breaker import CircuitBreaker
from prism.providers.lanes import build_lanes

__all__ = ["ProviderRegistry"]

log = structlog.get_logger(__name__)


class ProviderRegistry:
    def __init__(
        self,
        lanes: Mapping[str, Lane] | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._lanes = dict(lanes) if lanes is not None else build_lanes(resolved)
        self._slots = {
            name: asyncio.Semaphore(lane.concurrency) for name, lane in self._lanes.items()
        }
        self._breakers = {
            name: CircuitBreaker(
                name,
                threshold=resolved.breaker_failure_threshold,
                cooldown_s=resolved.breaker_cooldown_s,
            )
            for name in self._lanes
        }
        self._providers: dict[str, ChatProvider] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._lanes)

    def lane(self, name: str) -> Lane:
        try:
            return self._lanes[name]
        except KeyError:
            raise UnknownLane(name, self.names) from None

    def breaker(self, name: str) -> CircuitBreaker:
        self.lane(name)
        return self._breakers[name]

    def provider(self, name: str) -> ChatProvider:
        """The provider behind a lane, built once and kept.

        An unwired lane raises rather than returning the local one: a silent
        fallback would hide a routing bug behind a lane that costs nothing.
        """
        lane = self.lane(name)
        if lane.factory is None:
            raise LaneNotImplemented(name)
        if name not in self._providers:
            self._providers[name] = lane.factory()
        return self._providers[name]

    async def complete(
        self,
        lane: str,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LaneResult[str]:
        chosen = model or self.lane(lane).model

        async def run(provider: ChatProvider) -> tuple[str, Usage]:
            completion = await provider.complete(
                messages, model=chosen, temperature=temperature, max_tokens=max_tokens
            )
            return completion.text, completion.usage

        return await self._guarded(lane, run)

    async def structured[T: BaseModel](
        self,
        lane: str,
        messages: Sequence[Message],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LaneResult[T]:
        chosen = model or self.lane(lane).model

        async def run(provider: ChatProvider) -> tuple[T, Usage]:
            result = await provider.structured(
                messages, schema, model=chosen, temperature=temperature, max_tokens=max_tokens
            )
            return result.value, result.usage

        return await self._guarded(lane, run)

    async def _guarded[V](
        self,
        name: str,
        run: Callable[[ChatProvider], Awaitable[tuple[V, Usage]]],
    ) -> LaneResult[V]:
        lane = self.lane(name)
        provider = self.provider(name)
        breaker = self._breakers[name]

        try:
            breaker.check()
        except LaneUnavailable as exc:
            log.warning("lane_rejected", lane=name, reason="breaker_open", detail=str(exc))
            raise

        slots = self._slots[name]
        try:
            await asyncio.wait_for(slots.acquire(), lane.queue_timeout_s)
        except TimeoutError:
            # Not evidence about the provider: the queue is ours.
            breaker.release_probe()
            log.warning(
                "lane_rejected",
                lane=name,
                reason="queue_timeout",
                waited_s=lane.queue_timeout_s,
                concurrency=lane.concurrency,
            )
            raise LaneBusy(
                name, waited_s=lane.queue_timeout_s, concurrency=lane.concurrency
            ) from None

        try:
            try:
                value, usage = await run(provider)
            except ChatError as exc:
                detail = f"{type(exc).__name__}: {exc}"
                breaker.record_failure(detail)
                log.warning("lane_call_failed", lane=name, state=breaker.state, detail=detail)
                raise
            except BaseException:
                # A cancellation is not a verdict on the provider.
                breaker.release_probe()
                raise
        finally:
            slots.release()

        breaker.record_success()
        log.info(
            "lane_call",
            lane=name,
            provider=usage.provider,
            model=usage.model,
            billing_unit=usage.billing_unit,
            duration_ms=usage.duration_ms,
        )
        return LaneResult(lane=name, value=value, usage=usage)
