"""Node events, as `write_trace` commits them (ADR 0023).

An event carries the row's own fields, not a state update. Publishing cannot fail
a node, and a run with no subscriber allocates no channel.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

__all__ = ["EventChannel", "NodeEvent", "publish", "publishing"]


@dataclass(frozen=True)
class NodeEvent:
    """One execution of one node, as `query_traces` holds it."""

    query_id: UUID
    node: str
    sequence: int
    attempt: int
    status: str
    verdict: str | None
    started_at: datetime
    duration_ms: int
    error: str | None
    provider: str | None
    model: str | None
    cost_usd: Decimal | None


class EventChannel:
    """One run's events, in commit order.

    Unbounded: a run writes one event per node execution, capped by `max_attempts`.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[NodeEvent | None] = asyncio.Queue()
        self._closed = False

    def publish(self, event: NodeEvent) -> None:
        if not self._closed:
            self._queue.put_nowait(event)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[NodeEvent]:
        """Every event until the channel closes."""
        while (event := await self._queue.get()) is not None:
            yield event


_channel: ContextVar[EventChannel | None] = ContextVar("prism_node_events", default=None)


@contextmanager
def publishing(channel: EventChannel | None) -> Iterator[None]:
    """Subscribe `channel` to the nodes run inside this context."""
    token = _channel.set(channel)
    try:
        yield
    finally:
        _channel.reset(token)


def publish(event: NodeEvent) -> None:
    channel = _channel.get()
    if channel is not None:
        channel.publish(event)
