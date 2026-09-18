"""The entry point: mint a query, run the graph, finalize the row.

queries.status is checked against the three terminal states, so there is no
'running'. The row is inserted refused, with a reason, and a run that dies
halfway leaves it that way.
"""

import time
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import structlog
from langchain_core.runnables import RunnableConfig
from sqlalchemy import text

from prism.core.ids import uuid7
from prism.db import get_engine
from prism.graph.checkpointer import get_checkpointer
from prism.graph.graph import compile_graph
from prism.graph.state import GraphState, RefusalReason, TerminalStatus
from prism.graph.trace import link_checkpoints

__all__ = ["QueryRun", "mint_query", "run_query"]

log = structlog.get_logger(__name__)

_INSERT_QUERY = text(
    """
    INSERT INTO queries (
        id, tenant_id, collection_id, thread_id, question,
        status, refusal_reason, citation_count
    ) VALUES (
        :id, :tenant_id, :collection_id, :thread_id, :question,
        'refused', 'no_relevant_evidence', 0
    )
    """
)

_FINALIZE_QUERY = text(
    """
    UPDATE queries
       SET status = :status,
           refusal_reason = :refusal_reason,
           retrieval_attempts = :retrieval_attempts,
           grounding_attempts = :grounding_attempts,
           latency_ms = :latency_ms
     WHERE id = :id AND tenant_id = :tenant_id
    """
)


@dataclass(frozen=True)
class QueryRun:
    query_id: UUID
    thread_id: str
    status: TerminalStatus
    refusal_reason: RefusalReason | None
    latency_ms: int


async def mint_query(*, tenant_id: UUID, collection_id: UUID, question: str) -> tuple[UUID, str]:
    """Insert the row and its thread, one per query, before the graph starts."""
    query_id = uuid7()
    # Distinct from query_id: the checkpoint tables carry no tenant column and
    # cannot be under RLS (ADR 0015). Never accepted from a client — a resume
    # resolves it through queries, under the tenant predicate.
    thread_id = str(uuid7())

    async with get_engine().begin() as conn:
        await conn.execute(
            _INSERT_QUERY,
            {
                "id": query_id,
                "tenant_id": tenant_id,
                "collection_id": collection_id,
                "thread_id": thread_id,
                "question": question,
            },
        )
    return query_id, thread_id


async def run_query(*, tenant_id: UUID, collection_id: UUID, question: str) -> QueryRun:
    query_id, thread_id = await mint_query(
        tenant_id=tenant_id, collection_id=collection_id, question=question
    )

    state: GraphState = {
        "query_id": query_id,
        "tenant_id": tenant_id,
        "collection_id": collection_id,
        "thread_id": thread_id,
        "question": question,
        "retrieval_attempts": 0,
        "grounding_attempts": 0,
        "sequence": 0,
        "status": "refused",
        "refusal_reason": "no_relevant_evidence",
    }

    clock = time.perf_counter()
    graph = compile_graph(await get_checkpointer())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        # ainvoke returns the state dict untyped.
        final = cast(GraphState, await graph.ainvoke(state, config=config))
    finally:
        await link_checkpoints(query_id=query_id, graph=graph, config=config)
    latency_ms = int((time.perf_counter() - clock) * 1000)

    async with get_engine().begin() as conn:
        await conn.execute(
            _FINALIZE_QUERY,
            {
                "id": query_id,
                "tenant_id": tenant_id,
                "status": final["status"],
                "refusal_reason": final["refusal_reason"],
                "retrieval_attempts": final["retrieval_attempts"],
                "grounding_attempts": final["grounding_attempts"],
                "latency_ms": latency_ms,
            },
        )

    log.info(
        "query.finished",
        query_id=str(query_id),
        status=final["status"],
        refusal_reason=final["refusal_reason"],
        latency_ms=latency_ms,
    )
    return QueryRun(
        query_id=query_id,
        thread_id=thread_id,
        status=final["status"],
        refusal_reason=final["refusal_reason"],
        latency_ms=latency_ms,
    )
