"""The trace writer: one row per node execution.

A node reports what is not state — its meter and its payloads — through the
TraceContext `traced` hands it (ADR 0016). The context outlives the node, so a
node that raises after a provider call still writes its meter on the error row.

A node that reports no usage writes billing_unit='none' with every meter NULL,
never zeros: migration 0009's meters_check refuses a meter the unit does not
own. A reported meter is priced in the same transaction as the insert, and
price_id, cost_usd and cost_basis are written together or not at all.
"""

import functools
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import text

from prism.chat.base import Usage
from prism.config import get_settings
from prism.core.ids import uuid7
from prism.db import get_engine
from prism.graph.state import GraphState
from prism.pricing import price

__all__ = [
    "AttemptCounter",
    "Node",
    "TraceContext",
    "TracedNode",
    "cap_payload",
    "link_checkpoints",
    "traced",
]

log = structlog.get_logger(__name__)


class TraceContext:
    """What a node reports about one execution of itself, beside its state update.

    Never state: nothing here reaches a checkpoint. One per execution, created by
    `traced` and discarded once the row is written.
    """

    __slots__ = ("_input", "_output", "_usage")

    def __init__(self) -> None:
        self._usage: Usage | None = None
        self._input: object = None
        self._output: object = None

    @property
    def usage(self) -> Usage | None:
        return self._usage

    @property
    def input(self) -> object:
        return self._input

    @property
    def output(self) -> object:
        return self._output

    def record_usage(self, usage: Usage) -> None:
        """At most once: a trace row has one provider and one meter."""
        if self._usage is not None:
            raise RuntimeError(
                f"usage already recorded for {self._usage.provider}/{self._usage.model}; "
                "one node execution is one provider call"
            )
        self._usage = usage

    def record_input(self, payload: object) -> None:
        """References, not copies (ADR 0012): chunk ids with scores, not chunk text."""
        self._input = payload

    def record_output(self, payload: object) -> None:
        self._output = payload


Node = Callable[[GraphState, TraceContext], Awaitable[dict[str, Any]]]
TracedNode = Callable[[GraphState], Awaitable[dict[str, Any]]]

# Which loop a node belongs to, declared at its decorator. `query_traces` has
# one `attempt` column and the node name says which loop it counts (ADR 0012);
# this is where the node says it, rather than the writer inferring it from a
# name it does not own.
AttemptCounter = Literal["retrieval_attempts", "grounding_attempts"]

_INSERT = text(
    """
    INSERT INTO query_traces (
        id, query_id, tenant_id, node_name, sequence, attempt,
        status, started_at, duration_ms, error,
        provider, model, billing_unit, input_tokens, output_tokens, gpu_ms,
        price_id, cost_usd, cost_basis,
        input_json, output_json, input_truncated, output_truncated
    ) VALUES (
        :id, :query_id, :tenant_id, :node_name, :sequence, :attempt,
        :status, :started_at, :duration_ms, :error,
        :provider, :model, :billing_unit, :input_tokens, :output_tokens, :gpu_ms,
        :price_id, :cost_usd, :cost_basis,
        CAST(:input_json AS jsonb), CAST(:output_json AS jsonb),
        :input_truncated, :output_truncated
    )
    """
)


def _json_default(value: object) -> str:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not a trace payload value")


def _dumps(value: object) -> str:
    # jsonb has no NaN; one here is a bug in the node, not something to store.
    return json.dumps(value, default=_json_default, allow_nan=False, separators=(",", ":"))


def cap_payload(payload: object, max_bytes: int) -> tuple[str | None, bool]:
    """The payload as JSON of at most max_bytes, and whether it was cut to get there.

    Cuts structurally, so the result always parses: the longest list is halved
    until it fits, then the longest string. Lists go first because a list of
    references loses entries, while a cut string may be a reference itself.
    """
    if payload is None:
        return None, False

    encoded = _dumps(payload)
    if len(encoded.encode()) <= max_bytes:
        return encoded, False

    original_bytes = len(encoded.encode())
    value: Any = json.loads(encoded)
    while len(encoded.encode()) > max_bytes:
        path = _longest(value, list) or _longest(value, str)
        if path is None:
            # Nothing left to cut: say how much there was instead.
            return _dumps({"omitted_bytes": original_bytes}), True
        value = _halve(value, path)
        encoded = _dumps(value)
    return encoded, True


def _longest(value: Any, kind: type[list[Any]] | type[str]) -> tuple[Any, ...] | None:
    """The path to the longest `kind` of two or more items, depth first."""
    best: tuple[int, tuple[Any, ...]] | None = None

    def walk(node: Any, path: tuple[Any, ...]) -> None:
        nonlocal best
        size = len(node) if isinstance(node, kind) else 0
        if size >= 2 and (best is None or size > best[0]):
            best = (size, path)
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, (*path, key))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, (*path, index))

    walk(value, ())
    return None if best is None else best[1]


def _halve(value: Any, path: tuple[Any, ...]) -> Any:
    if not path:
        return value[: len(value) // 2]
    head, *rest = path
    copy = dict(value) if isinstance(value, dict) else list(value)
    copy[head] = _halve(value[head], tuple(rest))
    return copy


async def write_trace(
    *,
    query_id: UUID,
    tenant_id: UUID,
    node_name: str,
    sequence: int,
    attempt: int,
    status: str,
    started_at: datetime,
    duration_ms: int,
    error: str | None,
    trace: TraceContext,
) -> None:
    max_bytes = get_settings().trace_payload_max_bytes
    input_json, input_truncated = cap_payload(trace.input, max_bytes)
    output_json, output_truncated = cap_payload(trace.output, max_bytes)
    usage = trace.usage

    async with get_engine().begin() as conn:
        priced = await price(conn, usage, at=started_at) if usage is not None else None
        if usage is not None and priced is None:
            # Written anyway: the row is the record that the call happened, and
            # its NULL price is what makes the query total NULL rather than low.
            log.warning(
                "trace.unpriced",
                node=node_name,
                provider=usage.provider,
                model=usage.model,
                billing_unit=usage.billing_unit,
                query_id=str(query_id),
            )
        await conn.execute(
            _INSERT,
            {
                "id": uuid7(),
                "query_id": query_id,
                "tenant_id": tenant_id,
                "node_name": node_name,
                "sequence": sequence,
                "attempt": attempt,
                "status": status,
                "started_at": started_at,
                "duration_ms": duration_ms,
                "error": error,
                "provider": usage.provider if usage else None,
                "model": usage.model if usage else None,
                "billing_unit": usage.billing_unit if usage else "none",
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "gpu_ms": usage.gpu_ms if usage else None,
                "price_id": priced.price_id if priced else None,
                "cost_usd": priced.cost_usd if priced else None,
                "cost_basis": priced.cost_basis if priced else None,
                "input_json": input_json,
                "output_json": output_json,
                "input_truncated": input_truncated,
                "output_truncated": output_truncated,
            },
        )


def traced(
    node_name: str, *, attempts: AttemptCounter | None = None
) -> Callable[[Node], TracedNode]:
    """Wrap a node so that running it writes exactly one trace row.

    LangGraph sees `(state) -> update`; the TraceContext never leaves the wrapper.

    `attempts` names the loop counter this node is inside. The row takes that
    counter plus one, read before the node body runs, so every node in one pass
    of a loop carries the same attempt — including the node that increments it,
    which does so on the way out. A node outside both loops declares nothing and
    writes attempt 1, which is the whole truth about it.
    """

    def decorate(fn: Node) -> TracedNode:
        @functools.wraps(fn)
        async def wrapper(state: GraphState) -> dict[str, Any]:
            sequence = state["sequence"] + 1
            attempt = 1 if attempts is None else state[attempts] + 1
            started_at = datetime.now(UTC)
            clock = time.perf_counter()
            trace = TraceContext()

            def elapsed_ms() -> int:
                return int((time.perf_counter() - clock) * 1000)

            try:
                update = await fn(state, trace)
            except Exception as exc:
                await write_trace(
                    query_id=state["query_id"],
                    tenant_id=state["tenant_id"],
                    node_name=node_name,
                    sequence=sequence,
                    attempt=attempt,
                    status="error",
                    started_at=started_at,
                    duration_ms=elapsed_ms(),
                    error=f"{type(exc).__name__}: {exc}",
                    trace=trace,
                )
                log.warning("node.error", node=node_name, query_id=str(state["query_id"]))
                raise

            await write_trace(
                query_id=state["query_id"],
                tenant_id=state["tenant_id"],
                node_name=node_name,
                sequence=sequence,
                attempt=attempt,
                status="ok",
                started_at=started_at,
                duration_ms=elapsed_ms(),
                error=None,
                trace=trace,
            )
            return {**update, "sequence": sequence}

        return wrapper

    return decorate


_LINK_CHECKPOINT = text(
    """
    UPDATE query_traces SET checkpoint_ref = :checkpoint_ref
     WHERE query_id = :query_id AND sequence = :sequence
    """
)


async def link_checkpoints(
    *,
    query_id: UUID,
    graph: CompiledStateGraph[GraphState],
    config: RunnableConfig,
) -> None:
    """Point each trace row at the checkpoint a fork of that node starts from.

    A node cannot record this while it runs: configurable['checkpoint_id'] is set
    only on a resume. Matched afterwards in run order; a row whose node name does
    not line up is left NULL rather than pointed at a wrong checkpoint.

    A record, not a control (ADR 0008's stance for metering). It also runs on the
    way out of a failed run, so a failure here is logged rather than raised over
    the one that failed the run.
    """
    try:
        starts: list[tuple[str, str]] = []
        async for snapshot in graph.aget_state_history(config):
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
            if checkpoint_id:
                # LangGraph's own tasks write no trace row; pairing skips them.
                starts.extend(
                    (node, str(checkpoint_id))
                    for node in snapshot.next
                    if not node.startswith("__")
                )
        starts.reverse()  # history is newest first; trace rows are oldest first

        async with get_engine().begin() as conn:
            rows = await conn.execute(
                text(
                    """
                    SELECT sequence, node_name FROM query_traces
                     WHERE query_id = :query_id ORDER BY sequence
                    """
                ),
                {"query_id": query_id},
            )
            updates = [
                {"query_id": query_id, "sequence": sequence, "checkpoint_ref": checkpoint_id}
                for (sequence, node_name), (start_node, checkpoint_id) in zip(
                    rows.all(), starts, strict=False
                )
                if node_name == start_node
            ]
            if updates:
                await conn.execute(_LINK_CHECKPOINT, updates)
    except Exception:
        log.warning("trace.checkpoint_link_failed", query_id=str(query_id), exc_info=True)
