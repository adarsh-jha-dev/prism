"""The entry point: mint a query, run the graph, finalize the row.

queries.status is checked against the three terminal states, so there is no
'running'. The row is inserted refused, with a reason, and a run that dies
halfway leaves it that way — total_cost_usd included, which stays NULL: an
incomplete total, not a zero one. That is also what a run dying between the gate
and this write leaves behind, which is the direction it is allowed to fail.

Finalization is one transaction: the outcome, the answer, the citation count and
the citation rows, which are held in state until here so a rejected generation
leaves none behind (ADR 0021).
"""

import time
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import structlog
from langchain_core.runnables import RunnableConfig
from sqlalchemy import text

from prism.config import get_settings
from prism.core.ids import uuid7
from prism.db import get_engine
from prism.graph.checkpointer import get_checkpointer
from prism.graph.events import EventChannel, publishing
from prism.graph.graph import compile_graph
from prism.graph.state import (
    CitationRef,
    GraphState,
    RefusalReason,
    TerminalStatus,
    search_params_from,
)
from prism.graph.trace import link_checkpoints

__all__ = [
    "Citation",
    "Outcome",
    "QueryRun",
    "finalize",
    "mint_query",
    "rank_citations",
    "run_query",
]

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
           final_answer = :final_answer,
           citation_count = :citation_count,
           retrieval_attempts = :retrieval_attempts,
           grounding_attempts = :grounding_attempts,
           latency_ms = :latency_ms,
           total_cost_usd = (
               -- One call we could not price makes the total unknown, not
               -- smaller (ADR 0013). A row with no provider made no call
               -- (ADR 0016), and a query with no calls costs exactly zero.
               SELECT CASE
                          WHEN bool_or(provider IS NOT NULL AND price_id IS NULL) THEN NULL
                          ELSE coalesce(sum(cost_usd), 0)
                      END
                 FROM query_traces
                WHERE query_id = :id AND tenant_id = :tenant_id
           )
     WHERE id = :id AND tenant_id = :tenant_id
    """
)


_INSERT_CITATION = text(
    """
    INSERT INTO query_citations (
        id, query_id, tenant_id, chunk_ref, chunk_id, document_id,
        page_number, chunk_index, rank, rerank_score, cited_content
    ) VALUES (
        :id, :query_id, :tenant_id, :chunk_ref, :chunk_id, :document_id,
        :page_number, :chunk_index, :rank, :rerank_score, :cited_content
    )
    """
)

# chunk_ref is what was cited and is never null; chunk_id says whether that
# chunk still exists (migration 0009). Resolved inside the finalizing
# transaction, because a chunk deleted between `generate` and here would
# otherwise fail the foreign key and lose a sound answer.
# filename comes back with it rather than entering state (ADR 0017).
_CITED_CHUNKS = text(
    """
    SELECT c.id, d.filename
      FROM chunks c
      JOIN documents d ON d.id = c.document_id
     WHERE c.tenant_id = :tenant_id
       AND c.id = ANY(CAST(:chunk_ids AS uuid[]))
    """
)


@dataclass(frozen=True)
class Citation:
    """One citation as finalization ranked it."""

    rank: int
    document_id: UUID
    filename: str | None
    page_number: int | None
    content: str


@dataclass(frozen=True)
class Outcome:
    """What finalization wrote."""

    status: TerminalStatus
    refusal_reason: RefusalReason | None
    answer: str | None
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class QueryRun:
    query_id: UUID
    thread_id: str
    status: TerminalStatus
    refusal_reason: RefusalReason | None
    latency_ms: int
    answer: str | None
    citations: tuple[Citation, ...]


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


async def run_query(
    *,
    tenant_id: UUID,
    collection_id: UUID,
    question: str,
    events: EventChannel | None = None,
) -> QueryRun:
    """Run one query to a terminal state. `events` subscribes to its node rows."""
    query_id, thread_id = await mint_query(
        tenant_id=tenant_id, collection_id=collection_id, question=question
    )

    state: GraphState = {
        "query_id": query_id,
        "tenant_id": tenant_id,
        "collection_id": collection_id,
        "thread_id": thread_id,
        "question": question,
        # Seeded, not absent: retrieval searches for the question until the
        # rewriter says otherwise, and `question` is never written again.
        "retrieval_query": question,
        "retrieval_attempts": 0,
        "grounding_attempts": 0,
        "sequence": 0,
        # Empty rather than absent: a channel with no value is indistinguishable
        # from one a node failed to write.
        "search_terms": [],
        "search_params": search_params_from(get_settings()),
        "query_embedding": None,
        "candidates": [],
        # The gate writes this; `generate` reads it on a retry (ADR 0022).
        "unsupported_spans": [],
        # `generate` writes these; nothing persists them until `verify_grounding`
        # passes an answer (ADR 0021). Seeded rather than absent, as above.
        "answer": None,
        "citations": [],
        "status": "refused",
        "refusal_reason": "no_relevant_evidence",
    }

    clock = time.perf_counter()
    graph = compile_graph(await get_checkpointer())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    with publishing(events):
        try:
            # ainvoke returns the state dict untyped.
            final = cast(GraphState, await graph.ainvoke(state, config=config))
        finally:
            await link_checkpoints(query_id=query_id, graph=graph, config=config)
    latency_ms = int((time.perf_counter() - clock) * 1000)

    outcome = await finalize(
        query_id=query_id, tenant_id=tenant_id, final=final, latency_ms=latency_ms
    )

    log.info(
        "query.finished",
        query_id=str(query_id),
        status=outcome.status,
        refusal_reason=outcome.refusal_reason,
        latency_ms=latency_ms,
    )
    return QueryRun(
        query_id=query_id,
        thread_id=thread_id,
        status=outcome.status,
        refusal_reason=outcome.refusal_reason,
        latency_ms=latency_ms,
        answer=outcome.answer,
        citations=outcome.citations,
    )


def rank_citations(citations: list[CitationRef]) -> list[CitationRef]:
    """Rank by rerank score, densely from 1, at write time.

    `query_citations.rank` is UNIQUE per query and checked >= 1, and it is the
    order the dashboard lists sources in — so it comes from a score we computed,
    never from the order the model happened to return its citations in.

    `rerank`'s fallback leaves a candidate unscored (ADR 0011). Unscored sorts
    last and keeps its relative order, rather than sorting as zero and ranking
    below a chunk that genuinely scored 0.05.
    """
    ordered = sorted(
        citations,
        key=lambda citation: (
            citation["rerank_score"] is None,
            -(citation["rerank_score"] or 0.0),
        ),
    )
    return [{**citation, "rank": rank} for rank, citation in enumerate(ordered, start=1)]


async def finalize(
    *,
    query_id: UUID,
    tenant_id: UUID,
    final: GraphState,
    latency_ms: int,
) -> Outcome:
    """Write the outcome, the answer and the citations in one transaction.

    `queries_citation_count_check` forbids an answered row carrying no
    citations, and the way to satisfy it is never to invent one: an answer with
    no evidence refuses `insufficient_evidence`, because evidence survived
    retrieval and it is generation that did not complete (ADR 0022).
    """
    status: TerminalStatus = final["status"]
    refusal_reason: RefusalReason | None = final["refusal_reason"]
    answer = final["answer"] if status == "answered" else None
    citations = rank_citations(final["citations"]) if status == "answered" else []

    if status == "answered" and not (answer and citations):
        # Unreachable through the graph: the gate fails an answer with no
        # evidence before it calls anything. A guard, because the alternatives
        # here are a constraint violation or a fabricated citation.
        log.error(
            "finalize.answered_without_evidence",
            query_id=str(query_id),
            citations=len(citations),
        )
        status, refusal_reason, answer, citations = (
            "refused",
            "insufficient_evidence",
            None,
            [],
        )

    live: dict[UUID, str] = {}
    async with get_engine().begin() as conn:
        if citations:
            rows = await conn.execute(
                _CITED_CHUNKS,
                {
                    "tenant_id": tenant_id,
                    "chunk_ids": [str(citation["chunk_id"]) for citation in citations],
                },
            )
            live = {row.id: row.filename for row in rows}

        await conn.execute(
            _FINALIZE_QUERY,
            {
                "id": query_id,
                "tenant_id": tenant_id,
                "status": status,
                "refusal_reason": refusal_reason,
                "final_answer": answer,
                "citation_count": len(citations),
                "retrieval_attempts": final["retrieval_attempts"],
                "grounding_attempts": final["grounding_attempts"],
                "latency_ms": latency_ms,
            },
        )
        if citations:
            await conn.execute(
                _INSERT_CITATION,
                [
                    {
                        "id": uuid7(),
                        "query_id": query_id,
                        "tenant_id": tenant_id,
                        "chunk_ref": citation["chunk_id"],
                        "chunk_id": (
                            citation["chunk_id"] if citation["chunk_id"] in live else None
                        ),
                        "document_id": citation["document_id"],
                        "page_number": citation["page_number"],
                        "chunk_index": citation["chunk_index"],
                        "rank": citation["rank"],
                        "rerank_score": citation["rerank_score"],
                        "cited_content": citation["content"],
                    }
                    for citation in citations
                ],
            )

    return Outcome(
        status=status,
        refusal_reason=refusal_reason,
        answer=answer,
        citations=tuple(
            Citation(
                rank=citation["rank"],
                document_id=citation["document_id"],
                filename=live.get(citation["chunk_id"]),
                page_number=citation["page_number"],
                content=citation["content"],
            )
            for citation in citations
        ),
    )
