"""The retrieval loop: the refusal path first, then everything it must not do.

The models are scripted, because what is under test is the loop — how many times
it goes round, what it searches for on the way, what it records, and where it
stops — not what a grader writes. tests/test_graph_nodes.py covers the live path.
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError
from sqlalchemy import text

from prism.config import Settings, get_settings
from prism.db import get_engine
from prism.graph.checkpointer import get_checkpointer
from prism.graph.graph import compile_graph
from prism.graph.nodes import RelevanceVerdicts, grade_docs
from prism.graph.run import QueryRun, run_query
from prism.graph.state import GraphState
from prism.graph.trace import TraceContext
from prism.retrieval.hydrate import hydrate_chunks
from seeding import seed

if TYPE_CHECKING:
    from conftest import StubbedModels
    from prism.collections import CollectionRef

DIM = 768
QUESTION = "What is the chinchilla provisioning ratio?"
CHINCHILLA = "The chinchilla provisioning ratio is twenty tokens per parameter."
HARBOUR = "Unrelated material about harbour logistics and berth scheduling."

# Scored, not omitted: an empty list is no longer a grading the schema allows.
NOTHING_PASSES = '{"verdicts": [{"label": 1, "score": 0.02}]}'
FIRST_PASSES = '{"verdicts": [{"label": 1, "score": 0.91}]}'


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
                       billing_unit, input_json, output_json
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


# ------------------------------------------------------------ the unit tier


async def test_an_empty_candidate_set_fails_grading_without_a_model_call(
    stubbed_models: "StubbedModels",
) -> None:
    """A grader asked to judge nothing invents a verdict, so it is not asked.

    No database either: there is nothing to hydrate.
    """
    chat = stubbed_models(grade=FIRST_PASSES)
    state: GraphState = {
        "query_id": UUID(int=1),
        "tenant_id": UUID(int=2),
        "collection_id": UUID(int=3),
        "thread_id": "t",
        "question": QUESTION,
        "retrieval_query": QUESTION,
        "retrieval_attempts": 0,
        "grounding_attempts": 0,
        "sequence": 0,
        "search_terms": [],
        "search_params": {"k": 10, "candidate_k": 30, "rrf_k": 60, "rerank_score_floor": 0.44},
        "query_embedding": None,
        "candidates": [],
        "status": "refused",
        "refusal_reason": "no_relevant_evidence",
    }
    trace = TraceContext()

    # Called without `traced`, so the node body is the whole subject.
    update = await grade_docs.__wrapped__(state, trace)  # type: ignore[attr-defined]

    assert chat.calls == []
    assert trace.usage is None
    assert update == {"retrieval_attempts": 1}
    assert trace.output == {"verdicts": [], "kept": 0, "reason": "no_candidates"}


def test_the_grading_schema_forbids_an_empty_verdict_list() -> None:
    """The constraint reaches the provider, which is where it does its work.

    Without `minItems`, a local grader satisfies the schema with an empty array
    and every candidate fails for want of a verdict rather than on its merits.
    """
    schema = RelevanceVerdicts.model_json_schema()
    assert schema["properties"]["verdicts"]["minItems"] == 1
    assert "verdicts" in schema["required"]

    with pytest.raises(ValidationError):
        RelevanceVerdicts.model_validate({"verdicts": []})


def test_a_verdict_for_a_passage_that_was_not_sent_is_dropped() -> None:
    """Labels are matched, never positions: a shifted list says nothing itself."""
    from prism.graph.nodes import _scores_by_chunk
    from prism.retrieval.hydrate import HydratedChunk

    chunks = [
        HydratedChunk(
            chunk_id=UUID(int=index),
            document_id=UUID(int=99),
            filename="f.pdf",
            content="x",
            page_number=1,
            chunk_index=index,
        )
        for index in (1, 2)
    ]
    verdicts = RelevanceVerdicts.model_validate(
        {"verdicts": [{"label": 2, "score": 0.8}, {"label": 7, "score": 0.9}]}
    )

    scores = _scores_by_chunk(verdicts, chunks)
    # Label 2 is the second chunk, and label 7 was never sent.
    assert scores == {UUID(int=2): 0.8}


# ----------------------------------------------------- the refusal path


@pytest.fixture
async def unanswerable(collection: "CollectionRef") -> "CollectionRef":
    """A collection with chunks in it that no grader will pass."""
    await seed(collection.collection_id, [(HARBOUR, _unit(1))])
    return collection


@pytest.mark.integration
async def test_nothing_relevant_exhausts_retrieval_and_refuses_with_a_reason(
    unanswerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    stubbed_models(grade=NOTHING_PASSES)
    attempts = get_settings().max_attempts

    run = await run_query(
        tenant_id=unanswerable.tenant_id,
        collection_id=unanswerable.collection_id,
        question=QUESTION,
    )

    assert run.status == "refused"
    assert run.refusal_reason == "no_relevant_evidence"

    rows = await _rows(run.query_id)
    # Exactly the configured number of attempts, and one rewrite between each.
    assert len(_named(rows, "retrieve")) == attempts
    assert len(_named(rows, "grade_docs")) == attempts
    assert len(_named(rows, "rewrite_query")) == attempts - 1
    # The pass path is not on this route at all.
    assert _named(rows, "rerank") == []
    assert [row["node_name"] for row in rows][-1] == "abstain"

    async with get_engine().connect() as conn:
        final = (
            (
                await conn.execute(
                    text(
                        "SELECT status, refusal_reason, retrieval_attempts, grounding_attempts,"
                        " citation_count FROM queries WHERE id = :id"
                    ),
                    {"id": run.query_id},
                )
            )
            .mappings()
            .one()
        )
    assert final["status"] == "refused"
    assert final["refusal_reason"] == "no_relevant_evidence"
    assert final["retrieval_attempts"] == attempts
    assert final["grounding_attempts"] == 0
    assert final["citation_count"] == 0


@pytest.mark.integration
async def test_a_grader_that_never_passes_cannot_spin_past_the_maximum(
    unanswerable: "CollectionRef",
    stubbed_models: "StubbedModels",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound is the configured one, not the default that happens to be 3."""
    stubbed_models(grade=NOTHING_PASSES)
    settings = Settings(max_attempts=2)
    monkeypatch.setattr("prism.graph.graph.get_settings", lambda: settings)

    run = await run_query(
        tenant_id=unanswerable.tenant_id,
        collection_id=unanswerable.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    assert len(_named(rows, "retrieve")) == 2
    assert len(_named(rows, "rewrite_query")) == 1
    assert run.refusal_reason == "no_relevant_evidence"
    # The run ended rather than hitting LangGraph's recursion ceiling.
    assert rows[-1]["node_name"] == "abstain"


@pytest.mark.integration
async def test_the_submitted_question_survives_every_rewrite(
    unanswerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Grading and generation judge against what was asked, not what was searched."""
    chat = stubbed_models(grade=NOTHING_PASSES, rewrite='{"query": "harbour berth scheduling"}')

    run = await run_query(
        tenant_id=unanswerable.tenant_id,
        collection_id=unanswerable.collection_id,
        question=QUESTION,
    )

    graph = compile_graph(await get_checkpointer())
    values = (await graph.aget_state({"configurable": {"thread_id": run.thread_id}})).values
    assert values["question"] == QUESTION
    # The loop moved the search and left the question where it was.
    assert values["retrieval_query"] == "harbour berth scheduling"

    async with get_engine().connect() as conn:
        stored = (
            await conn.execute(
                text("SELECT question FROM queries WHERE id = :id"), {"id": run.query_id}
            )
        ).scalar_one()
    assert stored == QUESTION

    # Every grading call was handed the question, on the last pass as on the first.
    for messages in chat.calls_for("RelevanceVerdicts"):
        assert QUESTION in messages[-1].content
        assert "harbour berth scheduling" not in messages[-1].content


@pytest.mark.integration
async def test_a_rewrite_changes_what_the_next_retrieve_searches_for(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Not just the recorded text: the second pass retrieves a different chunk.

    Attempt 1 plans terms that only the chinchilla chunk matches; the rewrite
    plans terms that only the harbour chunk matches. The lexical half ANDs, so
    a term list that did not reach `retrieve` would leave the set unchanged.
    """
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0)), (HARBOUR, _unit(1))])
    chat = stubbed_models(
        plan=['{"terms": ["chinchilla"]}', '{"terms": ["berth"]}'],
        grade=NOTHING_PASSES,
        rewrite='{"query": "harbour berth scheduling"}',
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    first, second, *_ = _named(rows, "retrieve")

    assert first["input_json"]["retrieval_query"] == QUESTION
    assert second["input_json"]["retrieval_query"] == "harbour berth scheduling"
    assert first["input_json"]["terms"] == ["chinchilla"]
    assert second["input_json"]["terms"] == ["berth"]

    # The searches returned different evidence, which is the point of rewriting.
    assert first["output_json"] != second["output_json"]
    assert first["output_json"][0]["chunk_id"] != second["output_json"][0]["chunk_id"]

    # The planner was handed the rewritten query, and the grader the original.
    assert chat.calls_for("QueryPlan")[1][-1].content == "harbour berth scheduling"
    assert all(QUESTION in messages[-1].content for messages in chat.calls_for("RelevanceVerdicts"))


@pytest.mark.integration
async def test_trace_rows_carry_the_attempt_and_an_unbroken_sequence(
    unanswerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    stubbed_models(grade=NOTHING_PASSES)

    run = await run_query(
        tenant_id=unanswerable.tenant_id,
        collection_id=unanswerable.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))

    attempts = get_settings().max_attempts
    for node in ("plan_query", "embed_query", "retrieve", "grade_docs"):
        assert [row["attempt"] for row in _named(rows, node)] == list(range(1, attempts + 1))
    # A rewrite belongs to the attempt it produces, not the one that failed.
    assert [row["attempt"] for row in _named(rows, "rewrite_query")] == list(range(2, attempts + 1))
    # abstain is outside both loops, so its row says 1 and means it.
    assert [row["attempt"] for row in _named(rows, "abstain")] == [1]


@pytest.mark.integration
async def test_grade_docs_makes_exactly_one_provider_call_per_execution(
    unanswerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    chat = stubbed_models(grade=NOTHING_PASSES)

    run = await run_query(
        tenant_id=unanswerable.tenant_id,
        collection_id=unanswerable.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    graded = _named(rows, "grade_docs")
    assert len(chat.calls_for("RelevanceVerdicts")) == len(graded)
    # One provider call means one meter on the row (ADR 0016).
    assert all(row["provider"] == "ollama" for row in graded)
    assert all(row["billing_unit"] == "tokens" for row in graded)


@pytest.mark.integration
async def test_an_empty_collection_grades_without_a_call_and_writes_no_meter(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    chat = stubbed_models(grade=FIRST_PASSES)

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    assert chat.calls_for("RelevanceVerdicts") == []
    for row in _named(await _rows(run.query_id), "grade_docs"):
        assert row["status"] == "ok"
        assert row["provider"] is None and row["model"] is None
        assert row["billing_unit"] == "none"
        assert row["output_json"]["reason"] == "no_candidates"
    assert run.refusal_reason == "no_relevant_evidence"


@pytest.mark.integration
async def test_hydration_cannot_read_another_tenants_chunk_given_its_id(
    collection: "CollectionRef", other_collection: "CollectionRef"
) -> None:
    """The predicate is inside the query. A chunk id is not an authorization."""
    await seed(other_collection.collection_id, [(CHINCHILLA, _unit(0))])

    async with get_engine().connect() as conn:
        theirs = [
            row[0]
            for row in await conn.execute(
                text("SELECT id FROM chunks WHERE tenant_id = :tenant_id"),
                {"tenant_id": other_collection.tenant_id},
            )
        ]
    assert theirs

    # Their id, our tenant: nothing.
    assert (
        await hydrate_chunks(
            theirs, tenant_id=collection.tenant_id, collection_id=collection.collection_id
        )
        == []
    )
    # Their id and their tenant, but our collection: still nothing.
    assert (
        await hydrate_chunks(
            theirs, tenant_id=other_collection.tenant_id, collection_id=collection.collection_id
        )
        == []
    )
    # Their own scope reads it, so the empties above are the predicate working.
    found = await hydrate_chunks(
        theirs,
        tenant_id=other_collection.tenant_id,
        collection_id=other_collection.collection_id,
    )
    assert [chunk.chunk_id for chunk in found] == theirs


# -------------------------------------------------------- the pass path


@pytest.fixture
async def passed(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> tuple[QueryRun, "CollectionRef"]:
    """A run whose grading passes, so it takes the pass path through the stubs."""
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0))])
    stubbed_models(grade=FIRST_PASSES)
    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )
    return run, collection


@pytest.mark.integration
async def test_a_passing_grade_reaches_abstain_with_insufficient_evidence(
    passed: tuple[QueryRun, "CollectionRef"],
) -> None:
    """The stubs cite nothing, so the pass path refuses — and says which half failed.

    Never `answered`: an answer with no citations cannot be persisted (ADR 0012),
    and there is no path here that finalizes as one.
    """
    run, _ = passed

    assert run.status == "refused"
    assert run.refusal_reason == "insufficient_evidence"

    rows = await _rows(run.query_id)
    assert [row["node_name"] for row in rows] == [
        "plan_query",
        "embed_query",
        "retrieve",
        "grade_docs",
        "rerank",
        "generate",
        "verify_grounding",
        "abstain",
    ]
    abstained = _named(rows, "abstain")[0]
    assert abstained["output_json"]["refusal_reason"] == "insufficient_evidence"
    # It refused because evidence survived grading and nothing grounded an answer.
    assert abstained["input_json"]["candidates"] == 1

    async with get_engine().connect() as conn:
        final = (
            (
                await conn.execute(
                    text(
                        "SELECT status, refusal_reason, retrieval_attempts, citation_count"
                        " FROM queries WHERE id = :id"
                    ),
                    {"id": run.query_id},
                )
            )
            .mappings()
            .one()
        )
    assert final["status"] == "refused"
    assert final["refusal_reason"] == "insufficient_evidence"
    assert final["retrieval_attempts"] == 1
    assert final["citation_count"] == 0


@pytest.mark.integration
async def test_grade_docs_keeps_only_what_passed_the_threshold(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The threshold is doc_relevance_threshold, and it is the only decider."""
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0)), (HARBOUR, _unit(1))])
    floor = get_settings().doc_relevance_threshold
    stubbed_models(
        grade=(
            f'{{"verdicts": [{{"label": 1, "score": {floor + 0.2}}},'
            f' {{"label": 2, "score": {floor - 0.2}}}]}}'
        )
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    graded = _named(await _rows(run.query_id), "grade_docs")[0]
    verdicts = graded["output_json"]["verdicts"]
    assert [v["verdict"] for v in verdicts] == ["pass", "fail"]
    assert graded["output_json"]["kept"] == 1
    assert graded["output_json"]["threshold"] == floor
    # The rejected chunk is gone from state, so nothing downstream can cite it.
    graph = compile_graph(await get_checkpointer())
    values = (await graph.aget_state({"configurable": {"thread_id": run.thread_id}})).values
    assert [str(c["chunk_id"]) for c in values["candidates"]] == [verdicts[0]["chunk_id"]]


# ---------------------------------------------------------- replay


@pytest.mark.integration
async def test_a_fork_mid_loop_resumes_on_the_right_attempt(
    unanswerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The new field round-trips, and a checkpoint knows which pass it is in."""
    stubbed_models(grade=NOTHING_PASSES, rewrite='{"query": "harbour berth scheduling"}')

    run = await run_query(
        tenant_id=unanswerable.tenant_id,
        collection_id=unanswerable.collection_id,
        question=QUESTION,
    )

    graph = compile_graph(await get_checkpointer())
    config: RunnableConfig = {"configurable": {"thread_id": run.thread_id}}
    history = [snapshot async for snapshot in graph.aget_state_history(config)]

    # Oldest first, so the passes are in the order they ran.
    entries = [
        snapshot
        for snapshot in reversed(history)
        if snapshot.next and snapshot.next[0] == "retrieve"
    ]
    assert len(entries) == get_settings().max_attempts

    for index, snapshot in enumerate(entries):
        # `retrieve` runs before grade_docs closes the attempt, so the counter
        # is the number of passes already finished.
        assert snapshot.values["retrieval_attempts"] == index
        assert snapshot.values["question"] == QUESTION
        expected = QUESTION if index == 0 else "harbour berth scheduling"
        assert snapshot.values["retrieval_query"] == expected

    # A fork is a resume from that checkpoint's config, and it re-enters with
    # the attempt and the query that checkpoint held.
    forked = await graph.aget_state(entries[-1].config)
    assert forked.values["retrieval_attempts"] == get_settings().max_attempts - 1
    assert forked.values["retrieval_query"] == "harbour berth scheduling"
