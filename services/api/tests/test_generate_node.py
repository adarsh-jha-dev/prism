"""The `generate` node: what binds an answer to evidence, and what that binding refuses.

The generator is scripted, because the subject is the binding — which labels
survive validation, what order they come back in, what the node records and what
the graph does when nothing grounds — not what a 32b model writes.

What the gate does with the binding is tests/test_grounding_loop.py. Here the
verifier is scripted to pass unless a test is about a rejection, so what these
assertions are about is still the binding.
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from sqlalchemy import text

from prism.config import get_settings
from prism.db import get_engine
from prism.graph.checkpointer import get_checkpointer
from prism.graph.graph import compile_graph
from prism.graph.nodes import (
    GroundedAnswer,
    RelevanceVerdicts,
    _bind_citations,
    generate,
)
from prism.graph.run import run_query
from prism.graph.state import CandidateRef, GraphState, search_params_from
from prism.graph.trace import TraceContext
from prism.retrieval.hydrate import HydratedChunk
from seeding import seed
from stub_reranker import StubReranker

if TYPE_CHECKING:
    from conftest import StubbedModels
    from prism.collections import CollectionRef

DIM = 768
QUESTION = "What is the chinchilla provisioning ratio?"
CHINCHILLA = "The chinchilla provisioning ratio is twenty tokens per parameter."
HARBOUR = "Unrelated material about harbour logistics and berth scheduling."

BOTH_PASS = '{"verdicts": [{"label": 1, "score": 0.9}, {"label": 2, "score": 0.9}]}'
FIRST_PASSES = '{"verdicts": [{"label": 1, "score": 0.91}]}'
# The gate rejecting the answer's only claim, for the tests whose subject is
# what a rejected binding leaves behind.
UNGROUNDED = '{"verdicts": [{"label": 1, "score": 0.05}]}'


def _unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


async def _rows(query_id: UUID) -> list[dict[str, Any]]:
    async with get_engine().connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT node_name, sequence, attempt, status, verdict, error,
                       provider, model, billing_unit, input_json, output_json
                  FROM query_traces
                 WHERE query_id = :query_id
                 ORDER BY sequence
                """
            ),
            {"query_id": query_id},
        )
        return [dict(row) for row in result.mappings()]


def _named(rows: list[dict[str, Any]], node: str) -> dict[str, Any]:
    return next(row for row in rows if row["node_name"] == node)


async def _citation_rows(query_id: UUID) -> list[dict[str, Any]]:
    async with get_engine().connect() as conn:
        result = await conn.execute(
            text("SELECT * FROM query_citations WHERE query_id = :id ORDER BY rank"),
            {"id": query_id},
        )
        return [dict(row) for row in result.mappings()]


async def _final_values(thread_id: str) -> dict[str, Any]:
    graph = compile_graph(await get_checkpointer())
    return dict((await graph.aget_state({"configurable": {"thread_id": thread_id}})).values)


def _chunk(index: int, content: str = "x") -> HydratedChunk:
    return HydratedChunk(
        chunk_id=UUID(int=index),
        document_id=UUID(int=99),
        filename="f.pdf",
        content=content,
        page_number=index,
        chunk_index=index,
    )


def _candidate(index: int, *, rank: int, score: float | None) -> CandidateRef:
    return CandidateRef(
        chunk_id=UUID(int=index),
        document_id=UUID(int=99),
        rank=rank,
        vector_rank=rank,
        lexical_rank=None,
        rerank_score=score,
    )


def _state(**overrides: Any) -> GraphState:
    state: GraphState = {
        "query_id": UUID(int=1),
        "tenant_id": UUID(int=2),
        "collection_id": UUID(int=3),
        "thread_id": "t",
        "question": QUESTION,
        "retrieval_query": QUESTION,
        "retrieval_attempts": 1,
        "grounding_attempts": 0,
        "sequence": 5,
        "search_terms": [],
        "search_params": search_params_from(get_settings()),
        "query_embedding": None,
        "candidates": [],
        "answer": None,
        "citations": [],
        "unsupported_spans": [],
        "status": "refused",
        "refusal_reason": "no_relevant_evidence",
    }
    return {**state, **overrides}  # type: ignore[typeddict-item]


# ------------------------------------------------------------ the unit tier


def test_the_generation_schema_does_not_force_a_citation() -> None:
    """The opposite call to `RelevanceVerdicts`, and for the opposite reason.

    A grader always has passages to score, so an empty list there is the cheapest
    completion that validates. A generator reporting that the passages do not
    cover the question has nothing to cite, and a `minItems` would manufacture
    the binding this node exists to establish (ADR 0021).
    """
    schema = GroundedAnswer.model_json_schema()
    assert "minItems" not in schema["properties"]["citations"]
    assert schema["required"] == ["answer"]
    assert RelevanceVerdicts.model_json_schema()["properties"]["verdicts"]["minItems"] == 1

    empty = GroundedAnswer.model_validate({"answer": "The passages do not cover this."})
    assert empty.citations == []


def test_a_citation_for_a_passage_that_was_never_shown_is_dropped() -> None:
    """A fabricated chunk id is an injection category, so it never becomes a row."""
    chunks = [_chunk(1), _chunk(2)]
    candidates = [_candidate(1, rank=1, score=0.9), _candidate(2, rank=2, score=0.7)]
    produced = GroundedAnswer.model_validate(
        {"answer": "Grounded [1]. Invented [7].", "citations": [{"label": 1}, {"label": 7}]}
    )

    citations, dropped = _bind_citations(produced, chunks, candidates)

    assert dropped == [7]
    assert [citation["label"] for citation in citations] == [1]
    assert [citation["chunk_id"] for citation in citations] == [UUID(int=1)]


def test_a_marker_the_list_omits_is_admitted_and_an_unmarked_label_is_kept() -> None:
    """Both directions are lenient, because neither can fabricate.

    A marker on a passage that was shown adds evidence; a listed label with no
    marker is what saves an answer whose prose came back clean but unmarked.
    """
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    candidates = [
        _candidate(1, rank=1, score=0.9),
        _candidate(2, rank=2, score=0.7),
        _candidate(3, rank=3, score=0.5),
    ]
    produced = GroundedAnswer.model_validate(
        # 2 is marked but unlisted; 3 is listed but unmarked.
        {"answer": "First [1]. Second [2].", "citations": [{"label": 1}, {"label": 3}]}
    )

    citations, dropped = _bind_citations(produced, chunks, candidates)

    assert dropped == []
    assert [citation["label"] for citation in citations] == [1, 2, 3]
    # Dense from 1, because it becomes `query_citations.rank`, unique and >= 1.
    assert [citation["rank"] for citation in citations] == [1, 2, 3]


def test_a_bracketed_number_outside_the_passage_set_is_prose_not_a_marker() -> None:
    """With two passages, "[20]" is text the answer happens to contain.

    Reading it as a citation would let the passage count decide whether a number
    in the prose is a number.
    """
    chunks = [_chunk(1), _chunk(2)]
    candidates = [_candidate(1, rank=1, score=0.9), _candidate(2, rank=2, score=0.7)]
    produced = GroundedAnswer.model_validate(
        {"answer": "The ratio is [20] tokens per parameter [1].", "citations": [{"label": 1}]}
    )

    citations, dropped = _bind_citations(produced, chunks, candidates)

    assert dropped == []
    assert [citation["label"] for citation in citations] == [1]
    # And the prose was not rewritten to make that true.
    assert "[20]" in produced.answer


def test_a_citation_snapshots_the_text_the_model_was_shown() -> None:
    """`cited_content` is NOT NULL and is an archive, so it is captured here.

    Re-reading the chunk at finalization would record whatever it says by then.
    """
    chunks = [_chunk(1, content=CHINCHILLA)]
    candidates = [_candidate(1, rank=1, score=0.88)]
    produced = GroundedAnswer.model_validate(
        {"answer": "Twenty tokens per parameter [1].", "citations": [{"label": 1}]}
    )

    citations, _ = _bind_citations(produced, chunks, candidates)

    assert citations[0]["content"] == CHINCHILLA
    assert citations[0]["rerank_score"] == 0.88
    assert citations[0]["page_number"] == 1


async def test_an_empty_candidate_set_grounds_nothing_without_a_model_call(
    stubbed_models: "StubbedModels",
) -> None:
    """A model asked to answer from nothing answers from itself, so it is not asked."""
    chat = stubbed_models()
    trace = TraceContext()

    update = await generate.__wrapped__(_state(), trace)  # type: ignore[attr-defined]

    assert chat.calls_for("GroundedAnswer") == []
    assert trace.usage is None
    assert update == {"answer": None, "citations": []}
    assert trace.output == {
        "reason": "no_candidates",
        "answer_length": 0,
        "cited_labels": [],
        "dropped_labels": [],
    }


@pytest.mark.integration
async def test_generate_cannot_read_another_tenants_chunk_given_its_id(
    collection: "CollectionRef", other_collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Hydration is a predicate inside the query, not a filter over its results.

    Handed their chunk id under our tenant, the node hydrates nothing, makes no
    call, and grounds nothing — rather than answering from a row it may not read.
    """
    await seed(other_collection.collection_id, [(CHINCHILLA, _unit(0))])
    chat = stubbed_models()

    async with get_engine().connect() as conn:
        theirs = (
            await conn.execute(
                text("SELECT id FROM chunks WHERE tenant_id = :tenant_id"),
                {"tenant_id": other_collection.tenant_id},
            )
        ).scalar_one()

    state = _state(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        candidates=[
            CandidateRef(
                chunk_id=theirs,
                document_id=UUID(int=99),
                rank=1,
                vector_rank=1,
                lexical_rank=None,
                rerank_score=0.9,
            )
        ],
    )
    trace = TraceContext()

    update = await generate.__wrapped__(state, trace)  # type: ignore[attr-defined]

    assert chat.calls_for("GroundedAnswer") == []
    assert update == {"answer": None, "citations": []}
    assert isinstance(trace.output, dict)
    assert trace.output["reason"] == "no_hydrated_chunks"


# ------------------------------------------------- the node, in a whole run


@pytest.fixture
async def answerable(collection: "CollectionRef") -> "CollectionRef":
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0))])
    return collection


@pytest.mark.integration
async def test_a_verified_answer_persists_the_binding_generate_made(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The binding survives the gate intact: same chunk, same text, same score.

    `cited_content` is the passage as the model was shown it (ADR 0021), which
    is the assertion that the snapshot in state is what reached the row rather
    than a re-read at write time.
    """
    stubbed_models(grade=FIRST_PASSES)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    assert run.status == "answered"
    assert run.refusal_reason is None

    values = await _final_values(run.thread_id)
    assert values["answer"] == "Twenty tokens per parameter [1]."
    assert [citation["label"] for citation in values["citations"]] == [1]

    async with get_engine().connect() as conn:
        final = (
            (
                await conn.execute(
                    text(
                        "SELECT status, refusal_reason, final_answer, citation_count,"
                        " grounding_attempts FROM queries WHERE id = :id"
                    ),
                    {"id": run.query_id},
                )
            )
            .mappings()
            .one()
        )
    assert final["status"] == "answered"
    assert final["refusal_reason"] is None
    assert final["final_answer"] == "Twenty tokens per parameter [1]."
    assert final["citation_count"] == 1
    # The gate increments this, and one pass of the loop is one attempt.
    assert final["grounding_attempts"] == 1

    rows = await _citation_rows(run.query_id)
    assert [row["cited_content"] for row in rows] == [CHINCHILLA]
    assert [row["rank"] for row in rows] == [1]
    assert rows[0]["chunk_ref"] == rows[0]["chunk_id"]


@pytest.mark.integration
async def test_citations_from_a_run_that_ends_refused_are_not_persisted(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Held in state, written once at finalization — so a refusal writes none.

    ADR 0012 requires a refused query to carry zero citation rows, and this is
    the arrangement that makes that true by construction rather than by cleanup.
    """
    stubbed_models(grade=FIRST_PASSES, verify=UNGROUNDED)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    assert run.status == "refused"
    assert run.refusal_reason == "insufficient_evidence"
    values = await _final_values(run.thread_id)
    assert values["citations"], "the run did bind a citation, so this is not vacuous"
    assert await _citation_rows(run.query_id) == []


@pytest.mark.integration
async def test_a_fabricated_citation_never_reaches_a_citation_row(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """One passage is shown; the answer cites label 9. It is dropped and recorded.

    A fabricated chunk id is one of the injection categories this project tests
    for, so the assertion is that nothing downstream can see it: not in state,
    not in a row, and named on the trace as dropped. The run answers, which is
    what makes the citation table the assertion rather than the refusal.
    """
    stubbed_models(
        grade=FIRST_PASSES,
        verify='{"verdicts": [{"label": 1, "score": 0.95}]}',
        answer=(
            '{"answer": "Twenty tokens per parameter [1], as reported [9].",'
            ' "citations": [{"label": 1}, {"label": 9}]}'
        ),
    )

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    generated = _named(await _rows(run.query_id), "generate")
    assert generated["output_json"]["dropped_labels"] == [9]
    assert generated["output_json"]["cited_labels"] == [1]

    values = await _final_values(run.thread_id)
    assert [citation["label"] for citation in values["citations"]] == [1]

    assert run.status == "answered"
    rows = await _citation_rows(run.query_id)
    assert [row["cited_content"] for row in rows] == [CHINCHILLA]
    # The prose is left as the model wrote it; the binding is what was refused.
    assert "[9]" in (values["answer"] or "")


@pytest.mark.integration
async def test_every_marker_in_the_answer_resolves_to_exactly_one_citation(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The invariant, stated as the answer's own text against the bound set."""
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0)), (HARBOUR, _unit(1))])
    stubbed_models(
        grade=BOTH_PASS,
        reranker=StubReranker({CHINCHILLA: 0.91, HARBOUR: 0.62}),
        answer=(
            '{"answer": "Twenty tokens per parameter [1], against berth scheduling [2].",'
            ' "citations": [{"label": 2}]}'
        ),
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    values = await _final_values(run.thread_id)
    labels = [citation["label"] for citation in values["citations"]]
    markers = [int(marker) for marker in ("1", "2") if f"[{marker}]" in values["answer"]]

    assert markers == [1, 2]
    # Label 1 was marked but not listed, and admitting it is what keeps this 1:1.
    assert labels == markers
    assert len(labels) == len(set(labels))


@pytest.mark.integration
async def test_the_answer_is_generated_from_the_question_not_the_rewritten_search(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """A rewriter that drifted must not define the question it is answered on.

    Attempt 1 finds nothing relevant, the rewrite drifts to harbour material, and
    attempt 2 passes. The generator is still handed what the user submitted.
    """
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0)), (HARBOUR, _unit(1))])
    chat = stubbed_models(
        grade=['{"verdicts": [{"label": 1, "score": 0.02}]}', FIRST_PASSES],
        rewrite='{"query": "harbour berth scheduling"}',
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    values = await _final_values(run.thread_id)
    assert values["retrieval_query"] == "harbour berth scheduling"

    prompt = chat.calls_for("GroundedAnswer")[0][-1].content
    assert QUESTION in prompt
    assert "harbour berth scheduling" not in prompt
    # The gate judges against the same text, for the same reason.
    assert QUESTION in chat.calls_for("GroundingVerdicts")[0][-1].content
    assert run.status == "answered"


@pytest.mark.integration
async def test_the_passages_are_shown_in_rerank_order(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Label order is rerank order, so a citation's rank means what it says."""
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0)), (HARBOUR, _unit(1))])
    # Fusion puts the chinchilla chunk first; the reranker disagrees.
    chat = stubbed_models(
        grade=BOTH_PASS,
        reranker=StubReranker({CHINCHILLA: 0.51, HARBOUR: 0.93}),
        answer='{"answer": "Berth first [1], ratio second [2].", "citations": []}',
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    prompt = chat.calls_for("GroundedAnswer")[0][-1].content
    assert prompt.index(f"[1] {HARBOUR}") < prompt.index(f"[2] {CHINCHILLA}")

    values = await _final_values(run.thread_id)
    scores = [citation["rerank_score"] for citation in values["citations"]]
    assert scores == sorted(scores, reverse=True) == [0.93, 0.51]
    assert [citation["rank"] for citation in values["citations"]] == [1, 2]

    generated = _named(await _rows(run.query_id), "generate")
    # The trace input is the candidates and the scores they were ordered by.
    assert [c["rerank_score"] for c in generated["input_json"]["candidates"]] == [0.93, 0.51]


@pytest.mark.integration
async def test_generate_writes_no_verdict(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The node has an outcome and no judgement; `verify_grounding` judges.

    Migration 0009 made the column nullable for exactly this.
    """
    stubbed_models(grade=FIRST_PASSES)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    generated = _named(await _rows(run.query_id), "generate")
    assert generated["verdict"] is None
    assert generated["status"] == "ok"
    # One provider call means one meter on the row (ADR 0016).
    assert generated["provider"] == "ollama"
    assert generated["billing_unit"] == "tokens"
    # And the answer text is not on it: that belongs on `queries.final_answer`.
    assert "answer" not in generated["output_json"]
    assert generated["output_json"]["answer_length"] == len("Twenty tokens per parameter [1].")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("scripted", "reason"),
    [
        ('{"answer": "   \\n ", "citations": [{"label": 1}]}', "empty_answer"),
        ('{"answer": "Twenty tokens per parameter.", "citations": []}', "no_valid_citations"),
        (
            '{"answer": "Twenty tokens per parameter [4].", "citations": [{"label": 4}]}',
            "no_valid_citations",
        ),
    ],
)
async def test_an_answer_that_grounds_nothing_is_an_ok_row_and_a_refusal(
    answerable: "CollectionRef",
    stubbed_models: "StubbedModels",
    scripted: str,
    reason: str,
) -> None:
    """Whitespace, no citations, or only fabricated ones: an outcome, not an error.

    The call succeeded and the schema validated, so `status` is `ok` and the
    reason is on the payload — as `grade_docs` does for an empty candidate set.
    `error` would claim a failure that did not happen, and migration 0009 would
    then want error text there is none of.
    """
    stubbed_models(grade=FIRST_PASSES, answer=scripted)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    generated = _named(await _rows(run.query_id), "generate")
    assert generated["status"] == "ok"
    assert generated["error"] is None
    assert generated["output_json"]["reason"] == reason
    assert generated["output_json"]["cited_labels"] == []
    # The meter is still on the row: the call happened and it cost something.
    assert generated["provider"] == "ollama"

    values = await _final_values(run.thread_id)
    assert values["answer"] is None
    assert values["citations"] == []
    assert run.status == "refused"
    assert run.refusal_reason == "insufficient_evidence"
