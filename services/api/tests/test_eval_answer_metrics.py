"""Answer-level eval metrics. Pure functions over AnswerResult — no database.

The aggregations, not the graph: what the report says about a set of runs, and
what it refuses to say when a run degraded or a total is unknown.
"""

from decimal import Decimal
from uuid import UUID

import pytest

from prism.config import Settings
from prism.eval.metrics import (
    AnswerResult,
    percentile,
    summarize_attempts,
    summarize_budget,
    summarize_citations,
    summarize_groundedness,
    summarize_refusals,
)
from prism.eval.report import AnswerBlock


def result(
    question_id: str,
    *,
    unanswerable: bool = False,
    status: str = "answered",
    refusal_reason: str | None = None,
    retrieval_attempts: int = 1,
    grounding_attempts: int = 1,
    latency_ms: int = 2000,
    cost_usd: Decimal | None = Decimal("0E-8"),
    groundedness: float | None = 0.9,
    citations: int = 1,
    unresolved_citations: int = 0,
    fabricated_citations: int = 0,
    degraded: bool = False,
) -> AnswerResult:
    return AnswerResult(
        question_id=question_id,
        question=f"question {question_id}",
        unanswerable=unanswerable,
        query_id=UUID(int=int(question_id[-3:])),
        status=status,  # type: ignore[arg-type]
        refusal_reason=refusal_reason,  # type: ignore[arg-type]
        retrieval_attempts=retrieval_attempts,
        grounding_attempts=grounding_attempts,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        input_tokens=1000,
        output_tokens=100,
        nodes=7,
        groundedness=groundedness,
        citations=citations,
        unresolved_citations=unresolved_citations,
        fabricated_citations=fabricated_citations,
        degraded=degraded,
    )


def refused(question_id: str, *, unanswerable: bool, reason: str) -> AnswerResult:
    return result(
        question_id,
        unanswerable=unanswerable,
        status="refused",
        refusal_reason=reason,
        citations=0,
        groundedness=0.1 if reason == "insufficient_evidence" else None,
    )


# ------------------------------------------------------------------ refusal


def test_correct_and_false_refusal_are_scored_over_their_own_class() -> None:
    """One rate per class. A refusal is right or wrong depending on the question."""
    results = [
        refused("001", unanswerable=True, reason="no_relevant_evidence"),
        refused("002", unanswerable=True, reason="insufficient_evidence"),
        result("003", unanswerable=True),
        result("004"),
        result("005"),
        refused("006", unanswerable=False, reason="no_relevant_evidence"),
    ]

    summary = summarize_refusals(results)

    assert (summary.unanswerable, summary.refused_correctly) == (3, 2)
    assert summary.correct_refusal_rate == pytest.approx(2 / 3)
    assert (summary.answerable, summary.refused_falsely) == (3, 1)
    assert summary.false_refusal_rate == pytest.approx(1 / 3)
    # Kept apart: a single breakdown reads as whichever rate it sits under.
    assert dict(summary.correct_reasons) == {
        "no_relevant_evidence": 1,
        "insufficient_evidence": 1,
    }
    assert dict(summary.false_reasons) == {"no_relevant_evidence": 1}


def test_a_class_with_no_questions_has_no_rate_rather_than_zero() -> None:
    summary = summarize_refusals([result("001")])

    assert summary.correct_refusal_rate is None
    assert summary.false_refusal_rate == 0.0


# ------------------------------------------------------------- groundedness


def test_groundedness_is_split_by_outcome() -> None:
    """tau separates them by construction; one mean over both would hide that."""
    results = [
        result("001", groundedness=0.9),
        result("002", groundedness=0.8),
        refused("003", unanswerable=False, reason="insufficient_evidence"),
    ]

    summary = summarize_groundedness(results, 0.58)

    assert summary.scored == 3
    assert summary.answered_mean == pytest.approx(0.85)
    assert summary.refused_mean == pytest.approx(0.1)
    assert (summary.min, summary.max) == (pytest.approx(0.1), pytest.approx(0.9))


def test_a_run_that_never_reached_the_gate_is_not_scored() -> None:
    """A retrieval refusal has no groundedness — it is absent, not zero."""
    summary = summarize_groundedness(
        [refused("001", unanswerable=True, reason="no_relevant_evidence")], 0.58
    )

    assert summary.scored == 0
    assert summary.mean is None


# ----------------------------------------------------------------- citations


def test_citation_validity_counts_every_citation_not_every_run() -> None:
    results = [
        result("001", citations=3, unresolved_citations=1),
        result("002", citations=1),
    ]

    summary = summarize_citations(results)

    assert (summary.citations, summary.unresolved) == (4, 1)
    assert summary.validity == pytest.approx(0.75)


def test_a_set_with_no_citations_has_no_validity_rather_than_zero() -> None:
    summary = summarize_citations([refused("001", unanswerable=True, reason="x")])

    assert summary.citations == 0
    assert summary.validity is None


# ------------------------------------------------------------------ attempts


def test_attempts_are_distributed_per_loop_and_counted_independently() -> None:
    results = [
        result("001", retrieval_attempts=1, grounding_attempts=1),
        result("002", retrieval_attempts=3, grounding_attempts=1),
        refused("003", unanswerable=True, reason="no_relevant_evidence"),
    ]
    results[2] = result(
        "003",
        unanswerable=True,
        status="refused",
        refusal_reason="no_relevant_evidence",
        retrieval_attempts=3,
        grounding_attempts=0,
    )

    summary = summarize_attempts(results, 3)

    assert dict(summary.retrieval) == {1: 1, 3: 2}
    assert dict(summary.grounding) == {1: 2, 0: 1}


# -------------------------------------------------------- cost and latency


def test_percentile_is_nearest_rank() -> None:
    values = [float(n) for n in range(1, 11)]

    assert percentile(values, 0.5) == 5.0
    assert percentile(values, 0.95) == 10.0
    assert percentile([], 0.5) is None
    with pytest.raises(ValueError, match="q must be in"):
        percentile(values, 0.0)


def test_an_unpriced_run_is_left_out_of_the_mean_not_counted_as_zero() -> None:
    """One call we could not price makes that total unknown, not smaller (ADR 0013)."""
    results = [
        result("001", cost_usd=Decimal("0.0020")),
        result("002", cost_usd=None),
    ]

    summary = summarize_budget(results, cost_budget_usd=0.005, latency_budget_s=6.0)

    assert summary.unpriced == 1
    assert summary.mean_cost_usd == Decimal("0.0020")
    assert summary.over_cost_budget == 0


def test_a_run_past_a_budget_is_counted_against_it() -> None:
    results = [
        result("001", cost_usd=Decimal("0.0060"), latency_ms=7000),
        result("002", cost_usd=Decimal("0.0010"), latency_ms=1000),
    ]

    summary = summarize_budget(results, cost_budget_usd=0.005, latency_budget_s=6.0)

    assert summary.over_cost_budget == 1
    assert summary.over_latency_budget == 1
    assert summary.p95_latency_ms == 7000.0


# ---------------------------------------------------------------- exclusions


def test_a_degraded_run_is_excluded_from_every_aggregate_and_counted() -> None:
    """A rerank fallback still taints a run (ADR 0011) — it is reported, not averaged in."""
    block = AnswerBlock(
        [
            refused("001", unanswerable=True, reason="no_relevant_evidence"),
            result("002", unanswerable=True, degraded=True),
            result("003"),
        ],
        Settings(),
    )

    assert [r.question_id for r in block.excluded] == ["002"]
    # The degraded run answered an unanswerable question; scoring it would have
    # halved the correct-refusal rate.
    assert block.refusal.unanswerable == 1
    assert block.refusal.correct_refusal_rate == 1.0
    assert block.budget.runs == 2
