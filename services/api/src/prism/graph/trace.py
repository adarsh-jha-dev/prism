"""The trace writer: one row per node execution.

A node that raises writes status='error' with the error text and re-raises.
Stubs call no model, so the provider, meter and price columns stay NULL and
billing_unit stays 'none' — migration 0009's meters_check and priced_check
enforce that pairing.
"""

import functools
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import text

from prism.core.ids import uuid7
from prism.db import get_engine
from prism.graph.state import GraphState

__all__ = ["Node", "link_checkpoints", "traced"]

log = structlog.get_logger(__name__)

Node = Callable[[GraphState], Awaitable[dict[str, Any]]]

_INSERT = text(
    """
    INSERT INTO query_traces (
        id, query_id, tenant_id, node_name, sequence, attempt,
        status, started_at, duration_ms, error
    ) VALUES (
        :id, :query_id, :tenant_id, :node_name, :sequence, :attempt,
        :status, :started_at, :duration_ms, :error
    )
    """
)


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
) -> None:
    async with get_engine().begin() as conn:
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
            },
        )


def traced(node_name: str) -> Callable[[Node], Node]:
    """Wrap a node so that running it writes exactly one trace row."""

    def decorate(fn: Node) -> Node:
        @functools.wraps(fn)
        async def wrapper(state: GraphState) -> dict[str, Any]:
            sequence = state["sequence"] + 1
            started_at = datetime.now(UTC)
            clock = time.perf_counter()

            def elapsed_ms() -> int:
                return int((time.perf_counter() - clock) * 1000)

            try:
                update = await fn(state)
            except Exception as exc:
                await write_trace(
                    query_id=state["query_id"],
                    tenant_id=state["tenant_id"],
                    node_name=node_name,
                    sequence=sequence,
                    # The loops assign this once they exist.
                    attempt=1,
                    status="error",
                    started_at=started_at,
                    duration_ms=elapsed_ms(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                log.warning("node.error", node=node_name, query_id=str(state["query_id"]))
                raise

            await write_trace(
                query_id=state["query_id"],
                tenant_id=state["tenant_id"],
                node_name=node_name,
                sequence=sequence,
                attempt=1,
                status="ok",
                started_at=started_at,
                duration_ms=elapsed_ms(),
                error=None,
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
