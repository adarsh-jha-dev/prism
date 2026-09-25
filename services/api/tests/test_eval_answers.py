"""Reading a graph run back out of its rows.

The subject is `collect_results`: that the four reads line up on query_id, that
they address the attempt that decided the run, and that a degraded run comes back
marked rather than missing.
"""

from typing import TYPE_CHECKING

import pytest

from prism.eval.answers import collect_results
from prism.eval.golden import GoldenQuestion, PageRef
from prism.graph.run import run_query
from seeding import seed
from stub_reranker import StubReranker

if TYPE_CHECKING:
    from conftest import StubbedModels
    from prism.collections import CollectionRef

QUESTION = "What is the chinchilla provisioning ratio?"
CHINCHILLA = "The chinchilla provisioning ratio is twenty tokens per parameter."
HARBOUR = "Unrelated material about harbour logistics and berth scheduling."

NOTHING_PASSES = '{"verdicts": [{"label": 1, "score": 0.02}]}'
UNGROUNDED = '{"verdicts": [{"label": 1, "score": 0.05}]}'
DIM = 768


def _unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def question(unanswerable: bool = False) -> GoldenQuestion:
    return GoldenQuestion(
        id="gq-001",
        question=QUESTION,
        unanswerable=unanswerable,
        expected_answer=None,
        relevant=frozenset() if unanswerable else frozenset({PageRef(doc="seed.pdf", page=1)}),
        supporting_quote=None,
        tags=(),
    )


@pytest.fixture
async def answerable(collection: "CollectionRef") -> "CollectionRef":
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0))])
    return collection


@pytest.fixture
async def unanswerable(collection: "CollectionRef") -> "CollectionRef":
    await seed(collection.collection_id, [(HARBOUR, _unit(1))])
    return collection


@pytest.mark.integration
async def test_an_answered_run_carries_its_citations_groundedness_and_meters(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    stubbed_models(verify='{"verdicts": [{"label": 1, "score": 0.93}]}')

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )
    (result,) = await collect_results([question()], [run])

    assert result.status == "answered"
    assert result.refusal_reason is None
    assert result.groundedness == pytest.approx(0.93)
    assert result.citations == 1
    # The persisted citation resolves into the passage set generate was shown.
    assert result.unresolved_citations == 0
    assert result.fabricated_citations == 0
    assert (result.retrieval_attempts, result.grounding_attempts) == (1, 1)
    assert result.nodes == 7
    assert result.input_tokens > 0
    assert result.latency_ms >= 0
    assert result.cost_usd == 0
    assert not result.degraded


@pytest.mark.integration
async def test_a_retrieval_refusal_has_no_groundedness_and_no_citations(
    unanswerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The run never reached the gate, so the score is absent rather than zero."""
    stubbed_models(grade=NOTHING_PASSES)

    run = await run_query(
        tenant_id=unanswerable.tenant_id,
        collection_id=unanswerable.collection_id,
        question=QUESTION,
    )
    (result,) = await collect_results([question(unanswerable=True)], [run])

    assert (result.status, result.refusal_reason) == ("refused", "no_relevant_evidence")
    assert result.groundedness is None
    assert (result.citations, result.grounding_attempts) == (0, 0)
    assert result.retrieval_attempts == 3


@pytest.mark.integration
async def test_a_grounding_refusal_keeps_the_last_gate_score_and_persists_nothing(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """The deciding attempt is the last one, and a rejected answer leaves no citation."""
    stubbed_models(verify=UNGROUNDED)

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )
    (result,) = await collect_results([question()], [run])

    assert (result.status, result.refusal_reason) == ("refused", "insufficient_evidence")
    assert result.groundedness == pytest.approx(0.05)
    assert result.citations == 0
    assert result.grounding_attempts == 3


@pytest.mark.integration
async def test_a_rerank_fallback_comes_back_marked_degraded(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    """A run that fell back to fusion order is not measuring the optimized path (ADR 0011)."""
    stubbed_models(reranker=StubReranker(fail=True))

    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )
    (result,) = await collect_results([question()], [run])

    assert result.degraded


@pytest.mark.integration
async def test_collecting_needs_one_run_per_question(
    answerable: "CollectionRef", stubbed_models: "StubbedModels"
) -> None:
    stubbed_models()
    run = await run_query(
        tenant_id=answerable.tenant_id,
        collection_id=answerable.collection_id,
        question=QUESTION,
    )

    with pytest.raises(ValueError, match="one run per question"):
        await collect_results([question(), question()], [run])
