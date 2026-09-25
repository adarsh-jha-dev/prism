"""Run the golden set through the whole graph, and read back what it wrote.

Calls `run_query`, so every number the report carries comes from `queries` and
`query_traces` — the rows the dashboard renders — rather than from anything the
harness observed on the way past.

A run that fails costs one observation, not the run: it left a `queries` row and
an error trace row, which is the second shape of degradation `exclusions.py`
already recognises, so it comes back marked and the caller excludes it. Only a
failure before any node reported has nothing to record, and that one raises.

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
from prism.graph.events import EventChannel
from prism.graph.run import run_query

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

    async def one(question: GoldenQuestion) -> UUID:
        # The channel is here only for the id: a run that raises never returns
        # one, and without it there is no row to go and read.
        channel = EventChannel()
        async with limit:
            log.info("eval.answer.start", question_id=question.id)
            try:
                run = await run_query(
                    tenant_id=collection.tenant_id,
                    collection_id=collection.collection_id,
                    question=question.question,
                    events=channel,
                )
            except Exception:
                if channel.query_id is None:
                    raise
                log.warning(
                    "eval.answer.failed",
                    question_id=question.id,
                    query_id=str(channel.query_id),
                    exc_info=True,
                )
                return channel.query_id
            return run.query_id

    query_ids = await asyncio.gather(*(one(question) for question in golden.questions))
    return await collect_results(golden.questions, query_ids, engine=engine)


async def collect_results(
    questions: Sequence[GoldenQuestion],
    query_ids: Sequence[UUID],
    *,
    engine: AsyncEngine | None = None,
) -> tuple[AnswerResult, ...]:
    """Pair each question with the rows its run left behind."""
    if len(questions) != len(query_ids):
        raise ValueError("one run per question")
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
    for question, query_id in zip(questions, query_ids, strict=True):
        row = rows[query_id]
        meter = meters.get(query_id)
        shown = passages.get(query_id, set())
        refs = cited.get(query_id, [])
        results.append(
            AnswerResult(
                question_id=question.id,
                question=question.question,
                unanswerable=question.unanswerable,
                query_id=query_id,
                status=row.status,
                refusal_reason=row.refusal_reason,
                retrieval_attempts=row.retrieval_attempts,
                grounding_attempts=row.grounding_attempts,
                latency_ms=row.latency_ms,
                cost_usd=row.total_cost_usd,
                input_tokens=meter.input_tokens if meter else 0,
                output_tokens=meter.output_tokens if meter else 0,
                nodes=meter.nodes if meter else 0,
                groundedness=grounded.get(query_id),
                citations=len(refs),
                unresolved_citations=sum(1 for ref in refs if ref not in shown),
                fabricated_citations=meter.fabricated if meter else 0,
                degraded=query_id in degraded,
            )
        )
    return tuple(results)
