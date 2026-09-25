"""The grounding loop: the rejection path first, then the answer it finally lets through.

The verifier is scripted, because the subject is the gate — what it compares,
where the threshold comes from, how many times the loop goes round and what one
transaction writes at the end of it — not what an 8b model judges.

This is the first file in which a query can finalize as `answered`, so most of
it is about the ways it must not.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from sqlalchemy import text

from prism.config import get_settings
from prism.db import get_engine
from prism.graph.checkpointer import get_checkpointer
from prism.graph.graph import compile_graph
from prism.graph.nodes import GroundingVerdicts, _spans, verify_grounding
from prism.graph.run import finalize, mint_query, rank_citations, run_query
from prism.graph.state import CitationRef, GraphState, search_params_from
from prism.graph.trace import TraceContext
from seeding import seed

if TYPE_CHECKING:
    from conftest import StubbedModels
    from prism.collections import CollectionRef

DIM = 768
QUESTION = "What is the chinchilla provisioning ratio?"
CHINCHILLA = "The chinchilla provisioning ratio is twenty tokens per parameter."
HARBOUR = "Unrelated material about harbour logistics and berth scheduling."

FIRST_PASSES = '{"verdicts": [{"label": 1, "score": 0.91}]}'
BOTH_PASS = '{"verdicts": [{"label": 1, "score": 0.9}, {"label": 2, "score": 0.9}]}'

GROUNDED = '{"verdicts": [{"label": 1, "score": 0.95}]}'
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
                       provider, billing_unit, cost_usd, price_id,
                       input_json, output_json
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


async def _query_row(query_id: UUID) -> dict[str, Any]:
    async with get_engine().connect() as conn:
        result = await conn.execute(text("SELECT * FROM queries WHERE id = :id"), {"id": query_id})
        return dict(result.mappings().one())


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


def _citation(index: int, *, rank: int, score: float | None, content: str = "x") -> CitationRef:
    return CitationRef(
        label=index,
        chunk_id=UUID(int=index),
        document_id=UUID(int=99),
        page_number=index,
        chunk_index=index,
        rank=rank,
        rerank_score=score,
        content=content,
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
        "sequence": 6,
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


@pytest.fixture
async def answerable(collection: "CollectionRef") -> "CollectionRef":
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0))])
    return collection


# ------------------------------------------------------------ the unit tier


def test_the_verification_schema_forbids_an_empty_verdict_list() -> None:
    """`RelevanceVerdicts`' reason, one node later.

    Without `minItems` the cheapest completion that validates is an empty array,
    and every span would then fail for want of a verdict — refusing in the right
    direction for entirely the wrong reason.
    """
    schema = GroundingVerdicts.model_json_schema()
    assert schema["properties"]["verdicts"]["minItems"] == 1

    with pytest.raises(ValueError, match="at least 1"):
        GroundingVerdicts.model_validate({"verdicts": []})


def test_a_decimal_point_is_not_a_claim_boundary() -> None:
    """The split never breaks before a digit.

    A system whose answers are mostly numbers cannot afford a splitter that
    turns "0.58" into two claims, because the fragment scores low and refuses a
    sound answer.
    """
    assert _spans("The threshold is 0.58 for this collection.") == [
        "The threshold is 0.58 for this collection."
    ]
    assert _spans("It holds approx. 20 tokens per parameter [1].") == [
        "It holds approx. 20 tokens per parameter [1]."
    ]


def test_sentences_split_and_fragments_merge_without_losing_text() -> None:
    """Every character of the answer is judged, in exactly one span."""
    answer = "Twenty tokens per parameter [1]. The ratio holds across model sizes [2]."
    spans = _spans(answer)
    assert spans == [
        "Twenty tokens per parameter [1].",
        "The ratio holds across model sizes [2].",
    ]

    # A short trailing fragment joins the claim before it rather than being
    # judged alone; nothing is dropped either way.
    merged = _spans("The ratio is twenty tokens per parameter. Roughly.")
    assert merged == ["The ratio is twenty tokens per parameter. Roughly."]
    assert "Roughly" in "".join(merged)


def test_a_leading_fragment_joins_the_claim_after_it() -> None:
    """It has no previous span to merge into, so it takes the next one."""
    assert _spans("Yes. The ratio is twenty tokens per parameter [1].") == [
        "Yes. The ratio is twenty tokens per parameter [1]."
    ]


async def test_an_answer_with_no_citations_fails_without_a_model_call(
    stubbed_models: "StubbedModels",
) -> None:
    """With the citations as the evidence, an answer that cited nothing has none.

    A verifier handed no passages invents a verdict, so it is not asked — and a
    fail with no meter is what sends the run back to `generate`.
    """
    chat = stubbed_models()
    trace = TraceContext()

    update = await verify_grounding.__wrapped__(  # type: ignore[attr-defined]
        _state(answer="Twenty tokens per parameter.", citations=[]), trace
    )

    assert chat.calls_for("GroundingVerdicts") == []
    assert trace.usage is None
    assert trace.verdict == "fail"
    assert update == {"grounding_attempts": 1, "unsupported_spans": []}
    assert isinstance(trace.output, dict)
    assert trace.output["reason"] == "no_citations"


async def test_a_cleared_generation_fails_the_gate_without_a_call(
    stubbed_models: "StubbedModels",
) -> None:
    """`generate` clears both channels when it binds nothing (ADR 0021).

    The edge from `generate` is unconditional, so this is the ordinary route for
    an empty or wholly fabricated generation — not a fork-only case.
    """
    chat = stubbed_models()
    trace = TraceContext()

    update = await verify_grounding.__wrapped__(_state(), trace)  # type: ignore[attr-defined]

    assert chat.calls_for("GroundingVerdicts") == []
    assert trace.verdict == "fail"
    assert update == {"grounding_attempts": 1, "unsupported_spans": []}
    assert isinstance(trace.output, dict)
    assert trace.output["reason"] == "no_answer"


async def test_one_unsupported_span_fails_an_otherwise_grounded_answer(
    stubbed_models: "StubbedModels",
) -> None:
    """The aggregate is the minimum, not the mean.

    Three claims at 0.95 and one at 0.05 average well above tau. Taking the mean
    would answer, and the fabricated claim is exactly the signature of the
    injections this project counts (ADR 0022).
    """
    stubbed_models(
        verify=(
            '{"verdicts": [{"label": 1, "score": 0.95}, {"label": 2, "score": 0.95},'
            ' {"label": 3, "score": 0.95}, {"label": 4, "score": 0.05}]}'
        )
    )
    answer = (
        "The ratio is twenty tokens per parameter [1]. "
        "It was measured on the chinchilla family [1]. "
        "The measurement is compute optimal [1]. "
        "It was independently confirmed by the harbour authority [1]."
    )
    trace = TraceContext()

    update = await verify_grounding.__wrapped__(  # type: ignore[attr-defined]
        _state(answer=answer, citations=[_citation(1, rank=1, score=0.9, content=CHINCHILLA)]),
        trace,
    )

    assert trace.verdict == "fail"
    assert "status" not in update
    assert update["unsupported_spans"] == [
        "It was independently confirmed by the harbour authority [1]."
    ]

    assert isinstance(trace.output, dict)
    assert trace.output["groundedness"] == 0.05
    assert trace.output["unsupported"] == 1
    assert trace.output["tau"] == get_settings().abstention_threshold
    # The mean would have passed, which is the whole point of this test.
    assert sum(span["score"] for span in trace.output["spans"]) / 4 > trace.output["tau"]


async def test_a_span_with_no_verdict_fails(stubbed_models: "StubbedModels") -> None:
    """Absence of a judgement is not evidence of groundedness.

    The verifier scores claim 1 and says nothing about claim 2, as a grader may
    return nine verdicts for ten passages. The unjudged claim fails.
    """
    stubbed_models(verify='{"verdicts": [{"label": 1, "score": 0.95}]}')
    answer = "The ratio is twenty tokens per parameter [1]. It was measured on chinchilla [1]."
    trace = TraceContext()

    update = await verify_grounding.__wrapped__(  # type: ignore[attr-defined]
        _state(answer=answer, citations=[_citation(1, rank=1, score=0.9, content=CHINCHILLA)]),
        trace,
    )

    assert trace.verdict == "fail"
    assert update["unsupported_spans"] == ["It was measured on chinchilla [1]."]
    assert isinstance(trace.output, dict)
    assert trace.output["spans"][1]["score"] is None


async def test_the_verifier_is_shown_the_cited_passages_under_their_own_labels(
    stubbed_models: "StubbedModels",
) -> None:
    """Claims are (n) and passages are [n], so a marker still resolves.

    The evidence is what the answer cited, not what survived grading: it is
    already in state as the model was shown it, so nothing can drift under the
    judgement (ADR 0021, ADR 0022).
    """
    chat = stubbed_models(verify=GROUNDED)
    trace = TraceContext()

    await verify_grounding.__wrapped__(  # type: ignore[attr-defined]
        _state(
            answer="Twenty tokens per parameter [1].",
            citations=[_citation(1, rank=1, score=0.9, content=CHINCHILLA)],
        ),
        trace,
    )

    prompt = chat.calls_for("GroundingVerdicts")[0][-1].content
    assert f"[1] {CHINCHILLA}" in prompt
    assert "(1) Twenty tokens per parameter [1]." in prompt
    assert QUESTION in prompt
    # Not the candidate pool: nothing uncited is put in front of the verifier.
    assert HARBOUR not in prompt


def test_rank_comes_from_the_score_not_the_models_ordering() -> None:
    """`query_citations.rank` is unique per query and starts at 1."""
    ranked = rank_citations(
        [
            _citation(1, rank=1, score=0.41),
            _citation(2, rank=2, score=0.93),
            _citation(3, rank=3, score=0.62),
        ]
    )

    assert [citation["rank"] for citation in ranked] == [1, 2, 3]
    assert [citation["label"] for citation in ranked] == [2, 3, 1]
    assert [citation["rerank_score"] for citation in ranked] == [0.93, 0.62, 0.41]


def test_an_unscored_citation_ranks_last_rather_than_as_zero() -> None:
    """`rerank`'s fallback leaves a candidate unscored (ADR 0011).

    Unscored is not a low score: sorting it as 0.0 would rank it below a chunk
    that genuinely scored 0.05, which is a claim nothing measured.
    """
    ranked = rank_citations(
        [
            _citation(1, rank=1, score=None),
            _citation(2, rank=2, score=0.05),
            _citation(3, rank=3, score=None),
        ]
    )

    assert [citation["label"] for citation in ranked] == [2, 1, 3]
    assert [citation["rank"] for citation in ranked] == [1, 2, 3]


# ------------------------------------------------------- the rejection path


@pytest.mark.integration
async def test_an_ungrounded_answer_regenerates_and_then_refuses(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The loop's whole course: generate, fail, generate, fail, generate, fail, refuse.

    Exactly `max_attempts` generations — the pinned total, not three on top of
    one — and the refusal is `insufficient_evidence`, because evidence survived
    retrieval and it is generation that did not complete.
    """
    attempts = get_settings().max_attempts
    stubbed_models(grade=FIRST_PASSES, verify=UNGROUNDED)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    assert run.status == "refused"
    assert run.refusal_reason == "insufficient_evidence"

    rows = await _rows(run.query_id)
    assert len(_named(rows, "generate")) == attempts
    assert len(_named(rows, "verify_grounding")) == attempts
    assert all(row["verdict"] == "fail" for row in _named(rows, "verify_grounding"))
    assert rows[-1]["node_name"] == "abstain"
    assert rows[-1]["output_json"]["refusal_reason"] == "insufficient_evidence"

    final = await _query_row(run.query_id)
    assert final["status"] == "refused"
    assert final["grounding_attempts"] == attempts
    # Retrieval passed on its first attempt and never re-entered its own loop.
    assert final["retrieval_attempts"] == 1
    assert final["citation_count"] == 0
    assert final["final_answer"] is None
    assert await _citation_rows(run.query_id) == []


@pytest.mark.integration
async def test_a_verifier_that_always_fails_cannot_spin(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Termination, asserted as a bound on the whole run rather than on a counter.

    A loop that read a counter it also wrote would satisfy an attempts assertion
    and still not terminate; this one counts the rows that actually got written.
    """
    stubbed_models(grade=FIRST_PASSES, verify=UNGROUNDED)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    assert run.status == "refused"
    # Retrieval ran once (5 nodes), generation three times (2 nodes each), then
    # abstain. Anything looser would pass for a loop that went round four times.
    assert len(rows) == 5 + 2 * get_settings().max_attempts + 1
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))


@pytest.mark.integration
async def test_a_regeneration_is_told_which_claims_were_unsupported(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """ "Stricter constraints" is the gate's own finding, not a re-roll (ADR 0022).

    The first attempt is rejected, the second is prompted with the rejected
    claim, and its trace row counts what it was told.
    """
    stubbed_models(grade=FIRST_PASSES, verify=[UNGROUNDED, GROUNDED])
    chat_answer = "Twenty tokens per parameter [1]."

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    generated = _named(await _rows(run.query_id), "generate")
    assert len(generated) == 2
    # The first attempt had nothing to be told; the second was told one claim.
    assert [row["input_json"]["unsupported_spans"] for row in generated] == [0, 1]
    assert [row["attempt"] for row in generated] == [1, 2]

    assert run.status == "answered"
    final = await _query_row(run.query_id)
    assert final["final_answer"] == chat_answer
    assert final["grounding_attempts"] == 2


@pytest.mark.integration
async def test_the_rejected_claim_reaches_the_next_generation_prompt(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The text itself, so the model knows what to drop rather than to try harder."""
    chat = stubbed_models(
        grade=FIRST_PASSES,
        answer=(
            '{"answer": "Twenty tokens per parameter [1]. '
            'Confirmed by the harbour authority [1].", "citations": [{"label": 1}]}'
        ),
        verify=(
            # Two claims, the second unsupported; then both supported, because
            # the scripted answer does not change between attempts.
            '{"verdicts": [{"label": 1, "score": 0.95}, {"label": 2, "score": 0.02}]}',
            '{"verdicts": [{"label": 1, "score": 0.95}, {"label": 2, "score": 0.95}]}',
        ),
    )

    await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    prompts = [messages[-1].content for messages in chat.calls_for("GroundedAnswer")]
    assert len(prompts) == 2
    assert "Confirmed by the harbour authority [1]." not in prompts[0]
    assert "Confirmed by the harbour authority [1]." in prompts[1]
    # And the claim that was supported is not fed back as a failure.
    assert prompts[1].count("Twenty tokens per parameter [1].") == 0


@pytest.mark.integration
async def test_a_run_that_spent_its_retrieval_budget_still_gets_its_whole_grounding_budget(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The two counters do not contaminate each other.

    Retrieval fails twice and passes on its last attempt; grounding then fails
    three times of its own. A shared counter would have refused immediately.
    """
    attempts = get_settings().max_attempts
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0))])
    stubbed_models(
        grade=['{"verdicts": [{"label": 1, "score": 0.02}]}'] * (attempts - 1) + [FIRST_PASSES],
        verify=UNGROUNDED,
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    assert len(_named(rows, "grade_docs")) == attempts
    assert len(_named(rows, "generate")) == attempts

    final = await _query_row(run.query_id)
    assert final["retrieval_attempts"] == attempts
    assert final["grounding_attempts"] == attempts
    assert final["status"] == "refused"
    assert final["refusal_reason"] == "insufficient_evidence"


@pytest.mark.integration
async def test_citations_from_a_rejected_attempt_never_reach_a_row(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Including when a later attempt succeeds citing something else.

    The first generation cites the harbour passage and is rejected; the second
    cites chinchilla and passes. Only the second is in the table, which is what
    "held in state until finalization" buys (ADR 0021).
    """
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0)), (HARBOUR, _unit(1))])
    from stub_reranker import StubReranker

    stubbed_models(
        grade=BOTH_PASS,
        reranker=StubReranker({CHINCHILLA: 0.91, HARBOUR: 0.62}),
        answer=[
            '{"answer": "Berth scheduling governs the ratio [2].", "citations": [{"label": 2}]}',
            '{"answer": "Twenty tokens per parameter [1].", "citations": [{"label": 1}]}',
        ],
        verify=[UNGROUNDED, GROUNDED],
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    assert run.status == "answered"
    rows = await _citation_rows(run.query_id)
    assert [row["cited_content"] for row in rows] == [CHINCHILLA]
    assert HARBOUR not in [row["cited_content"] for row in rows]
    # The rejected attempt is still inspectable, on the row that made it.
    generated = _named(await _rows(run.query_id), "generate")
    assert (
        generated[0]["output_json"]["cited_chunk_ids"]
        != (generated[1]["output_json"]["cited_chunk_ids"])
    )


@pytest.mark.integration
async def test_a_regenerations_trace_rows_carry_attempt_two(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Both nodes of the pass, and the sequence stays contiguous across it."""
    stubbed_models(grade=FIRST_PASSES, verify=[UNGROUNDED, GROUNDED])

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    assert [row["attempt"] for row in _named(rows, "generate")] == [1, 2]
    assert [row["attempt"] for row in _named(rows, "verify_grounding")] == [1, 2]
    # Retrieval's nodes belong to its own loop and stay on attempt 1.
    assert [row["attempt"] for row in _named(rows, "grade_docs")] == [1]
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))


@pytest.mark.integration
async def test_a_passing_verification_with_no_citations_refuses(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The guard, driven directly: the graph cannot produce this state.

    `queries_citation_count_check` forbids an answered row with no citations,
    and the way to satisfy it is to refuse rather than invent one. The reason is
    `insufficient_evidence`, because evidence survived retrieval (ADR 0022).
    """
    stubbed_models()
    query_id, _ = await mint_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    outcome = await finalize(
        query_id=query_id,
        tenant_id=answerable.tenant_id,
        final=_state(
            query_id=query_id,
            tenant_id=answerable.tenant_id,
            collection_id=answerable.collection_id,
            answer="Twenty tokens per parameter.",
            citations=[],
            grounding_attempts=1,
            status="answered",
            refusal_reason=None,
        ),
        latency_ms=12,
    )

    assert (outcome.status, outcome.refusal_reason) == ("refused", "insufficient_evidence")
    assert outcome.answer is None
    assert outcome.citations == ()
    final = await _query_row(query_id)
    assert final["status"] == "refused"
    assert final["final_answer"] is None
    assert final["citation_count"] == 0
    assert await _citation_rows(query_id) == []


# ----------------------------------------------------- the acceptance path


@pytest.mark.integration
async def test_a_passing_run_writes_one_answered_row_and_matching_citations(
    collection: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The counter, the rows and the ranks are one transaction's work.

    `citation_count` equals the rows written, and the ranks are contiguous from
    1 — which is what `query_citations_rank_key` and `rank >= 1` require of
    every writer, not only this one.
    """
    from stub_reranker import StubReranker

    await seed(collection.collection_id, [(CHINCHILLA, _unit(0)), (HARBOUR, _unit(1))])
    stubbed_models(
        grade=BOTH_PASS,
        reranker=StubReranker({CHINCHILLA: 0.91, HARBOUR: 0.62}),
        answer=(
            '{"answer": "Twenty tokens per parameter [1], against berth scheduling [2].",'
            ' "citations": [{"label": 1}, {"label": 2}]}'
        ),
        verify=GROUNDED,
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    assert run.status == "answered"
    assert run.refusal_reason is None

    async with get_engine().connect() as conn:
        answered = (
            await conn.execute(
                text("SELECT count(*) FROM queries WHERE id = :id AND status = 'answered'"),
                {"id": run.query_id},
            )
        ).scalar_one()
    assert answered == 1

    final = await _query_row(run.query_id)
    rows = await _citation_rows(run.query_id)
    assert final["citation_count"] == len(rows) == 2
    assert [row["rank"] for row in rows] == [1, 2]
    # Ranked by score at write time, so the top citation is the top chunk.
    assert [row["cited_content"] for row in rows] == [CHINCHILLA, HARBOUR]
    assert [row["rerank_score"] for row in rows] == [pytest.approx(0.91), pytest.approx(0.62)]
    assert all(row["chunk_ref"] == row["chunk_id"] for row in rows)
    assert all(row["tenant_id"] == collection.tenant_id for row in rows)


@pytest.mark.integration
async def test_total_cost_is_the_sum_of_the_priced_trace_rows(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The headline number, and it has to come from the rows rather than a tally."""
    stubbed_models(grade=FIRST_PASSES)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )
    assert run.status == "answered"

    rows = await _rows(run.query_id)
    assert all(row["price_id"] is not None for row in rows if row["provider"] is not None)
    expected = sum((row["cost_usd"] or Decimal(0)) for row in rows)

    final = await _query_row(run.query_id)
    assert final["total_cost_usd"] == expected
    assert final["latency_ms"] is not None and final["latency_ms"] >= 0


@pytest.mark.integration
async def test_an_unpriced_row_makes_the_total_null_rather_than_low(
    answerable: "CollectionRef", stubbed_models: "StubbedModels", monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unpriced is not free (ADR 0013), and an answered run is not exempt."""
    from prism.chat.base import Usage
    from prism.graph import trace as trace_module

    real_price = trace_module.price

    async def unpriceable(conn: Any, usage: Usage, *, at: Any) -> Any:
        # The generator's row alone, so the rest of the run stays priced and the
        # NULL is the one unpriced call rather than an empty table.
        unpriced = usage.model == get_settings().generator_model
        return None if unpriced else await real_price(conn, usage, at=at)

    monkeypatch.setattr(trace_module, "price", unpriceable)
    stubbed_models(grade=FIRST_PASSES)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    assert run.status == "answered"
    rows = await _rows(run.query_id)
    assert any(row["provider"] is not None and row["price_id"] is None for row in rows)
    assert (await _query_row(run.query_id))["total_cost_usd"] is None


@pytest.mark.integration
async def test_both_grading_nodes_write_a_verdict(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Migration 0009 made the column nullable for grading nodes; these are they.

    Every other node has an outcome and no judgement, and writes NULL.
    """
    stubbed_models(grade=FIRST_PASSES)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    rows = await _rows(run.query_id)
    assert _named(rows, "grade_docs")[0]["verdict"] == "pass"
    assert _named(rows, "verify_grounding")[0]["verdict"] == "pass"
    judging = {"grade_docs", "verify_grounding"}
    assert all(row["verdict"] is None for row in rows if row["node_name"] not in judging)


@pytest.mark.integration
async def test_the_gate_records_the_spans_the_aggregate_and_the_tau_it_used(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """A refusal must be as inspectable as an answer.

    A rejected attempt never reaches `queries.final_answer`, so this row is the
    only record of what the run refused to say — which is why the span text is
    on it (ADR 0022).
    """
    stubbed_models(grade=FIRST_PASSES, verify=UNGROUNDED)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    assert run.status == "refused"
    row = _named(await _rows(run.query_id), "verify_grounding")[0]
    output = row["output_json"]
    assert output["tau"] == get_settings().abstention_threshold
    assert output["groundedness"] == 0.05
    assert output["unsupported"] == 1
    assert [span["verdict"] for span in output["spans"]] == ["fail"]
    assert output["spans"][0]["span"] == "Twenty tokens per parameter [1]."
    # The evidence it judged against, by reference (ADR 0012).
    assert len(row["input_json"]["cited_chunk_ids"]) == 1
    # One provider call means one meter on the row (ADR 0016).
    assert row["provider"] == "ollama"
    assert row["billing_unit"] == "tokens"


# --------------------------------------------------------------- tau's source


@pytest.mark.integration
async def test_tau_is_the_collections_value_not_the_configured_default(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The first thing to read `collections.abstention_threshold` since 0001.

    The verifier scores 0.70. The collection's tau is 0.90, so the run refuses
    where the configured 0.58 would have answered.
    """
    async with get_engine().begin() as conn:
        await conn.execute(
            text("UPDATE collections SET abstention_threshold = 0.90 WHERE id = :id"),
            {"id": answerable.collection_id},
        )
    stubbed_models(grade=FIRST_PASSES, verify='{"verdicts": [{"label": 1, "score": 0.70}]}')

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    assert get_settings().abstention_threshold < 0.70 < 0.90
    assert run.status == "refused"
    assert run.refusal_reason == "insufficient_evidence"

    values = await _final_values(run.thread_id)
    assert values["search_params"]["abstention_threshold"] == pytest.approx(0.90)
    row = _named(await _rows(run.query_id), "verify_grounding")[0]
    assert row["output_json"]["tau"] == pytest.approx(0.90)


@pytest.mark.integration
async def test_a_fork_verifies_at_the_tau_the_original_run_pinned(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """Changing the collection's tau cannot retroactively re-judge a fork.

    The original ran at 0.50 and answered. The collection is then set to 0.95,
    and the fork — re-entered from the original's state — still answers, because
    the threshold it verifies at is the one in `search_params` (ADR 0022).
    """
    async with get_engine().begin() as conn:
        await conn.execute(
            text("UPDATE collections SET abstention_threshold = 0.50 WHERE id = :id"),
            {"id": answerable.collection_id},
        )
    stubbed_models(grade=FIRST_PASSES, verify='{"verdicts": [{"label": 1, "score": 0.60}]}')

    original = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )
    assert original.status == "answered"

    values = await _final_values(original.thread_id)
    assert values["search_params"]["abstention_threshold"] == pytest.approx(0.50)

    async with get_engine().begin() as conn:
        await conn.execute(
            text("UPDATE collections SET abstention_threshold = 0.95 WHERE id = :id"),
            {"id": answerable.collection_id},
        )

    # A fork re-enters at the gate with the state the original held, so nothing
    # re-pins: `plan_query` does not run.
    query_id, thread_id = await mint_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )
    graph = compile_graph(await get_checkpointer())
    forked_state = {
        **values,
        "query_id": query_id,
        "thread_id": thread_id,
        "sequence": 0,
        "grounding_attempts": 0,
        "status": "refused",
        "refusal_reason": "no_relevant_evidence",
    }
    final = await graph.ainvoke(forked_state, config={"configurable": {"thread_id": thread_id}})

    assert final["status"] == "answered"
    row = _named(await _rows(query_id), "verify_grounding")[0]
    assert row["output_json"]["tau"] == pytest.approx(0.50)
    assert row["verdict"] == "pass"
