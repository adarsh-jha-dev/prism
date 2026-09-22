"""The `rerank` node: the floor as a loop gate, the fallback, and the trace row.

The reranker is scripted, because the subject is the node — what it scores
against, what it writes down, and what the graph does with an empty result —
not how a cross-encoder ranks. tests/test_rerank_live.py covers the real model.
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from sqlalchemy import text

from prism.config import get_settings
from prism.db import get_engine
from prism.eval.exclusions import degraded_query_ids
from prism.graph.checkpointer import get_checkpointer
from prism.graph.graph import compile_graph
from prism.graph.nodes import rerank as node_rerank
from prism.graph.run import run_query
from prism.graph.state import CandidateRef, GraphState, search_params_from
from prism.graph.trace import TraceContext
from prism.retrieval.hydrate import hydrate_chunks
from seeding import seed
from stub_reranker import StubReranker

if TYPE_CHECKING:
    from conftest import StubbedModels
    from prism.collections import CollectionRef

DIM = 768
QUESTION = "What is the chinchilla provisioning ratio?"
CHINCHILLA = "The chinchilla provisioning ratio is twenty tokens per parameter."
HARBOUR = "Unrelated material about harbour logistics and berth scheduling."
ALL_PASS = '{"verdicts": [{"label": 1, "score": 0.9}, {"label": 2, "score": 0.9}]}'


def _unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


async def _rows(query_id: UUID) -> list[dict[str, Any]]:
    async with get_engine().connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT node_name, sequence, attempt, status, provider, model,
                       billing_unit, cost_usd, price_id, input_json, output_json
                  FROM query_traces
                 WHERE query_id = :query_id
                 ORDER BY sequence
                """
            ),
            {"query_id": query_id},
        )
        return [dict(row) for row in result.mappings()]


def _named(rows: list[dict[str, Any]], node: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["node_name"] == node]


def _state(*, tenant_id: UUID, collection_id: UUID, candidates: list[CandidateRef]) -> GraphState:
    """A state good enough to call one node body with, outside the graph."""
    settings = get_settings()
    return {
        "query_id": UUID(int=1),
        "tenant_id": tenant_id,
        "collection_id": collection_id,
        "thread_id": "t",
        "question": QUESTION,
        "retrieval_query": QUESTION,
        "retrieval_attempts": 0,
        "grounding_attempts": 0,
        "sequence": 0,
        "search_terms": [],
        "search_params": search_params_from(settings),
        "query_embedding": None,
        "candidates": candidates,
        "status": "refused",
        "refusal_reason": "no_relevant_evidence",
    }


# ------------------------------------------------------------ the unit tier


def test_the_node_calls_the_function_the_eval_harness_calls() -> None:
    """One implementation, or a pinned baseline stops measuring this node.

    Identity, not behaviour: a second implementation that merely agreed today
    would drift, and the baseline would go on looking valid while it did.
    """
    from prism.eval import runner
    from prism.graph import nodes
    from prism.retrieval.rerank import rerank as canonical

    assert nodes.rerank_hits is canonical
    assert runner.rerank is canonical


# ----------------------------------------------------- the floor as a gate


@pytest.fixture
async def two_chunks(collection: "CollectionRef") -> "CollectionRef":
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0)), (HARBOUR, _unit(1))])
    return collection


@pytest.mark.integration
async def test_everything_below_the_floor_re_enters_the_loop_and_then_refuses(
    two_chunks: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The floor empties the set, and an empty set is a failed pass (ADR 0011).

    It refuses `no_relevant_evidence` — the same terminal as a grading failure,
    because it is the same finding arrived at more precisely. No third reason.
    """
    floor = get_settings().rerank_score_floor
    scorer = StubReranker(default=floor - 0.1)
    stubbed_models(grade=ALL_PASS, reranker=scorer)
    attempts = get_settings().max_attempts

    run = await run_query(
        tenant_id=two_chunks.tenant_id,
        collection_id=two_chunks.collection_id,
        question=QUESTION,
    )

    assert run.status == "refused"
    assert run.refusal_reason == "no_relevant_evidence"

    rows = await _rows(run.query_id)
    assert len(_named(rows, "rerank")) == attempts
    assert len(_named(rows, "rewrite_query")) == attempts - 1
    # The floor emptied the set, so the grader was never given anything to judge.
    for row in _named(rows, "grade_docs"):
        assert row["output_json"]["reason"] == "no_candidates"
    assert rows[-1]["node_name"] == "abstain"
    # Nothing on the pass path ran.
    assert _named(rows, "generate") == []


@pytest.mark.integration
async def test_a_floor_failure_does_not_count_the_attempt_twice(
    two_chunks: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """One pass of the loop is one attempt, whichever gate fails it.

    `rerank` declares the counter so its row carries the right attempt, and
    increments nothing: `grade_docs` still closes the pass.
    """
    floor = get_settings().rerank_score_floor
    stubbed_models(grade=ALL_PASS, reranker=StubReranker(default=floor - 0.1))
    attempts = get_settings().max_attempts

    run = await run_query(
        tenant_id=two_chunks.tenant_id,
        collection_id=two_chunks.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    assert [row["attempt"] for row in _named(rows, "rerank")] == list(range(1, attempts + 1))
    assert [row["attempt"] for row in _named(rows, "grade_docs")] == list(range(1, attempts + 1))

    async with get_engine().connect() as conn:
        counted = (
            await conn.execute(
                text("SELECT retrieval_attempts FROM queries WHERE id = :id"),
                {"id": run.query_id},
            )
        ).scalar_one()
    # Three passes, not six: the floor failing a pass is not a second attempt.
    assert counted == attempts


# --------------------------------------------------------- what it judges


@pytest.mark.integration
async def test_reranking_reads_the_question_and_never_the_retrieval_query(
    two_chunks: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Reranking is a relevance judgement, so it judges what was asked (ADR 0019)."""
    floor = get_settings().rerank_score_floor
    scorer = StubReranker(default=floor - 0.1)
    stubbed_models(grade=ALL_PASS, rewrite='{"query": "harbour berth scheduling"}', reranker=scorer)

    run = await run_query(
        tenant_id=two_chunks.tenant_id,
        collection_id=two_chunks.collection_id,
        question=QUESTION,
    )

    graph = compile_graph(await get_checkpointer())
    values = (await graph.aget_state({"configurable": {"thread_id": run.thread_id}})).values
    # The loop did move the search, so this is not vacuous.
    assert values["retrieval_query"] == "harbour berth scheduling"

    assert scorer.queries
    assert all(asked == QUESTION for asked in scorer.queries)


@pytest.mark.integration
async def test_the_trace_holds_the_whole_ranking_before_the_floor(
    two_chunks: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """`Reranked.ranked` exists so the viewer shows what the floor discarded."""
    floor = get_settings().rerank_score_floor
    scorer = StubReranker({CHINCHILLA: floor + 0.3, HARBOUR: floor - 0.3})
    stubbed_models(grade=ALL_PASS, reranker=scorer)

    run = await run_query(
        tenant_id=two_chunks.tenant_id,
        collection_id=two_chunks.collection_id,
        question=QUESTION,
    )

    row = _named(await _rows(run.query_id), "rerank")[0]
    assert row["input_json"]["floor"] == floor
    assert len(row["input_json"]["candidates"]) == 2

    output = row["output_json"]
    assert output["floor"] == floor
    assert output["fallback"] is False
    assert output["kept"] == 1
    # Both scored chunks are on the row, in reranked order, with their scores —
    # the discarded one included, marked rather than dropped.
    ranked = output["ranked"]
    assert [entry["rank"] for entry in ranked] == [1, 2]
    assert [entry["score"] for entry in ranked] == [floor + 0.3, floor - 0.3]
    assert [entry["kept"] for entry in ranked] == [True, False]
    assert all(entry["fused_rank"] for entry in ranked)

    # The score reaches state, where generation and a citation will read it.
    graph = compile_graph(await get_checkpointer())
    values = (await graph.aget_state({"configurable": {"thread_id": run.thread_id}})).values
    assert [c["rerank_score"] for c in values["candidates"]] == [floor + 0.3]


@pytest.mark.integration
async def test_the_row_names_the_reranker_and_prices_it_at_zero(
    two_chunks: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """No provider call, but a model ran, and $0 is a price (ADR 0013).

    Migration 0009's meters_check permits a model with billing_unit 'none' as
    long as no meter is set, so the name stays on the row.
    """
    stubbed_models(grade=ALL_PASS, reranker=StubReranker(default=0.9))

    run = await run_query(
        tenant_id=two_chunks.tenant_id,
        collection_id=two_chunks.collection_id,
        question=QUESTION,
    )

    row = _named(await _rows(run.query_id), "rerank")[0]
    assert row["status"] == "ok"
    assert row["model"] == get_settings().reranker_model
    assert row["provider"] == "in-process"
    assert row["billing_unit"] == "none"
    assert row["price_id"] is not None
    assert row["cost_usd"] == 0


@pytest.mark.integration
async def test_hydration_for_reranking_is_tenant_scoped(
    collection: "CollectionRef", other_collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """A chunk id is not an authorization, on this read as on every other.

    Handed the other tenant's chunk id directly, under our tenant: the node
    scores nothing and clears the set, rather than hydrating by id and passing
    their text to the reranker. Going through `run_query` could not show this —
    retrieval would never have produced the id in the first place.
    """
    await seed(other_collection.collection_id, [(CHINCHILLA, _unit(0))])
    async with get_engine().connect() as conn:
        theirs = (
            await conn.execute(
                text("SELECT id, document_id FROM chunks WHERE tenant_id = :tenant_id"),
                {"tenant_id": other_collection.tenant_id},
            )
        ).one()

    scorer = StubReranker(default=0.9)
    stubbed_models(reranker=scorer)
    state = _state(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        candidates=[
            {
                "chunk_id": theirs.id,
                "document_id": theirs.document_id,
                "rank": 1,
                "vector_rank": 1,
                "lexical_rank": None,
                "rerank_score": None,
            }
        ],
    )
    trace = TraceContext()

    update = await node_rerank.__wrapped__(state, trace)  # type: ignore[attr-defined]

    assert scorer.calls == []
    assert update == {"candidates": []}
    assert trace.output == {"ranked": [], "kept": 0, "reason": "no_hydrated_chunks"}

    # Their own scope reads it, so the empty above is the predicate working.
    assert await hydrate_chunks(
        [theirs.id],
        tenant_id=other_collection.tenant_id,
        collection_id=other_collection.collection_id,
    )


# ------------------------------------------------------------ the fallback


@pytest.mark.integration
async def test_a_rerank_error_keeps_fusion_order_and_marks_the_run_degraded(
    two_chunks: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Degradation is the node's decision, and it costs the measurement (ADR 0011).

    The query proceeds — rerank buys precision, not groundedness, and tau still
    decides at `verify_grounding`. What it must not do is count as a data point.
    """
    stubbed_models(grade=ALL_PASS, reranker=StubReranker(fail=True))

    run = await run_query(
        tenant_id=two_chunks.tenant_id,
        collection_id=two_chunks.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    row = _named(rows, "rerank")[0]
    # Not an error row: the node returned, and status='error' means it raised.
    assert row["status"] == "ok"
    assert row["output_json"]["fallback"] is True
    assert row["output_json"]["reason"] == "rerank_error"
    # No model ran, so no meter and no price.
    assert row["model"] is None and row["price_id"] is None

    # Fusion order kept, the floor unapplied, and no score invented.
    ranked = row["output_json"]["ranked"]
    assert [entry["rank"] for entry in ranked] == [1, 2]
    assert all(entry["score"] is None for entry in ranked)

    retrieved = _named(rows, "retrieve")[0]["output_json"]
    assert [entry["chunk_id"] for entry in ranked] == [
        str(candidate["chunk_id"]) for candidate in retrieved
    ]

    # The run went on rather than refusing for want of evidence.
    assert _named(rows, "generate")

    # And it is not a data point.
    assert await degraded_query_ids([run.query_id]) == {run.query_id}


@pytest.mark.integration
async def test_a_clean_run_is_not_excluded(
    two_chunks: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The exclusion has to be able to say no, or it excludes nothing usefully."""
    stubbed_models(grade=ALL_PASS, reranker=StubReranker(default=0.9))

    run = await run_query(
        tenant_id=two_chunks.tenant_id,
        collection_id=two_chunks.collection_id,
        question=QUESTION,
    )

    assert await degraded_query_ids([run.query_id]) == set()
