"""Run the golden set through the whole graph, and read back what it wrote.

Calls `run_query`, so every number the report carries comes from `queries` and
`query_traces` — the rows the dashboard renders — rather than from anything the
harness observed on the way past.

A run that raises fails the eval run, as a RerankError already does in
`runner.py`: a query that did not finish is not a data point. A run that
*finished* while degrading is a different case — it is marked here and excluded
by the caller, never dropped (ADR 0011).

Questions are independent and run concurrently; results come back in golden-set
order regardless.
"""

import asyncio
from collections.abc import Sequence
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.collections import CollectionRef
from prism.config import Settings, get_settings
from prism.db import get_engine
from prism.eval.exclusions import degraded_query_ids
from prism.eval.golden import GoldenQuestion, GoldenSet
from prism.eval.metrics import AnswerResult
from prism.graph.run import QueryRun, run_query

__all__ = ["collect_results", "run_answer_set"]

log = structlog.get_logger(__name__)

_QUERIES = text(
    """
    SELECT id, status, refusal_reason, retrieval_attempts, grounding_attempts,
           latency_ms, total_cost_usd, citation_count
      FROM queries
     WHERE id = ANY(CAST(:query_ids AS uuid[]))
    """
)

# The attempt that decided the run, which is the last one to execute.
_GROUNDEDNESS = text(
    """
    SELECT DISTINCT ON (query_id)
           query_id, (output_json->>'groundedness')::float AS groundedness
      FROM query_traces
     WHERE query_id = ANY(CAST(:query_ids AS uuid[]))
       AND node_name = 'verify_grounding'
       AND output_json ? 'groundedness'
     ORDER BY query_id, sequence DESC
    """
)

# What `generate` was shown on that same attempt. Citation validity is checked
# across two writes — this row and the citation rows finalization wrote — so it
# measures the persistence path, not `_bind_citations` restating itself.
_PASSAGES = text(
    """
    SELECT DISTINCT ON (query_id) query_id, input_json->'candidates' AS candidates
      FROM query_traces
     WHERE query_id = ANY(CAST(:query_ids AS uuid[]))
       AND node_name = 'generate'
       AND input_json ? 'candidates'
     ORDER BY query_id, sequence DESC
    """
)

_METERS = text(
    """
    SELECT query_id,
           count(*) AS nodes,
           coalesce(sum(input_tokens), 0) AS input_tokens,
           coalesce(sum(output_tokens), 0) AS output_tokens,
           coalesce(
               sum(jsonb_array_length(output_json->'dropped_labels'))
                   FILTER (WHERE output_json ? 'dropped_labels'),
               0
           ) AS fabricated
      FROM query_traces
     WHERE query_id = ANY(CAST(:query_ids AS uuid[]))
     GROUP BY query_id
    """
)

_CITATIONS = text(
    "SELECT query_id, chunk_ref FROM query_citations "
    "WHERE query_id = ANY(CAST(:query_ids AS uuid[]))"
)


async def run_answer_set(
    golden: GoldenSet,
    *,
    collection: CollectionRef,
    concurrency: int | None = None,
    settings: Settings | None = None,
    engine: AsyncEngine | None = None,
) -> tuple[AnswerResult, ...]:
    """Run every question through the graph, in golden-set order."""
    settings = settings or get_settings()
    limit = asyncio.Semaphore(concurrency or settings.concurrency_ollama_local)

    async def one(question: GoldenQuestion) -> QueryRun:
        async with limit:
            log.info("eval.answer.start", question_id=question.id)
            return await run_query(
                tenant_id=collection.tenant_id,
                collection_id=collection.collection_id,
                question=question.question,
            )

    runs = await asyncio.gather(*(one(question) for question in golden.questions))
    return await collect_results(golden.questions, runs, engine=engine)


async def collect_results(
    questions: Sequence[GoldenQuestion],
    runs: Sequence[QueryRun],
    *,
    engine: AsyncEngine | None = None,
) -> tuple[AnswerResult, ...]:
    """Pair each question with the rows its run left behind."""
    if len(questions) != len(runs):
        raise ValueError("one run per question")
    query_ids = [run.query_id for run in runs]
    ids = [str(query_id) for query_id in query_ids]

    async with (engine or get_engine()).connect() as conn:
        rows = {r.id: r for r in await conn.execute(_QUERIES, {"query_ids": ids})}
        grounded = {
            r.query_id: r.groundedness
            for r in await conn.execute(_GROUNDEDNESS, {"query_ids": ids})
        }
        meters = {r.query_id: r for r in await conn.execute(_METERS, {"query_ids": ids})}
        passages: dict[UUID, set[UUID]] = {
            r.query_id: {UUID(c["chunk_id"]) for c in (r.candidates or [])}
            for r in await conn.execute(_PASSAGES, {"query_ids": ids})
        }
        cited: dict[UUID, list[UUID]] = {}
        for row in await conn.execute(_CITATIONS, {"query_ids": ids}):
            cited.setdefault(row.query_id, []).append(row.chunk_ref)

    degraded = await degraded_query_ids(query_ids, engine=engine)

    results: list[AnswerResult] = []
    for question, run in zip(questions, runs, strict=True):
        row = rows[run.query_id]
        meter = meters.get(run.query_id)
        shown = passages.get(run.query_id, set())
        refs = cited.get(run.query_id, [])
        results.append(
            AnswerResult(
                question_id=question.id,
                question=question.question,
                unanswerable=question.unanswerable,
                query_id=run.query_id,
                status=row.status,
                refusal_reason=row.refusal_reason,
                retrieval_attempts=row.retrieval_attempts,
                grounding_attempts=row.grounding_attempts,
                latency_ms=row.latency_ms,
                cost_usd=row.total_cost_usd,
                input_tokens=meter.input_tokens if meter else 0,
                output_tokens=meter.output_tokens if meter else 0,
                nodes=meter.nodes if meter else 0,
                groundedness=grounded.get(run.query_id),
                citations=len(refs),
                unresolved_citations=sum(1 for ref in refs if ref not in shown),
                fabricated_citations=meter.fabricated if meter else 0,
                degraded=run.query_id in degraded,
            )
        )
    return tuple(results)
