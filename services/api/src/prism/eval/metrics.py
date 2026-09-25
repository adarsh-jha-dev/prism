"""Eval metrics. Pure functions over what one run produced.

Two halves. Retrieval metrics score ranked page references; answer metrics score
whole graph runs, from the rows `queries` and `query_traces` hold.

`recall@k` and `hit@k` are reported separately and are not the same number.
Recall is the fraction of a question's relevant pages found in the top k; hit is
whether *any* of them was. A question with four relevant pages cannot score
above 0.25 recall@1 however perfect the ranking, so recall alone reads as a
failure where hit reads as a success — both are true and the pair is the honest
report.

Ranks are chunk ranks, not page ranks: two chunks from one page occupy two
slots, because that is what they cost at retrieval time.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from statistics import fmean
from uuid import UUID

from prism.eval.golden import PageRef
from prism.graph.state import RefusalReason, TerminalStatus

__all__ = [
    "AnswerResult",
    "AnswerableSummary",
    "AttemptsSummary",
    "BudgetSummary",
    "CitationSummary",
    "FloorSummary",
    "GroundednessSummary",
    "QuestionResult",
    "RefusalSummary",
    "UnanswerableSummary",
    "above_floor",
    "hit_at_k",
    "percentile",
    "recall_at_k",
    "reciprocal_rank",
    "summarize_answerable",
    "summarize_attempts",
    "summarize_budget",
    "summarize_citations",
    "summarize_floor",
    "summarize_groundedness",
    "summarize_refusals",
    "summarize_unanswerable",
]


def recall_at_k(relevant: frozenset[PageRef], retrieved: Sequence[PageRef], k: int) -> float:
    """Fraction of `relevant` pages appearing in the first `k` retrieved chunks."""
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        raise ValueError("a question with no relevant pages has no recall to measure")
    found = {page for page in retrieved[:k] if page in relevant}
    return len(found) / len(relevant)


def hit_at_k(relevant: frozenset[PageRef], retrieved: Sequence[PageRef], k: int) -> bool:
    """Whether any relevant page appears in the first `k` retrieved chunks."""
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        raise ValueError("a question with no relevant pages has no hit to measure")
    return any(page in relevant for page in retrieved[:k])


def reciprocal_rank(relevant: frozenset[PageRef], retrieved: Sequence[PageRef]) -> float:
    """1/rank of the first relevant chunk, or 0.0 if none was retrieved."""
    if not relevant:
        raise ValueError("a question with no relevant pages has no rank to measure")
    for index, page in enumerate(retrieved, start=1):
        if page in relevant:
            return 1.0 / index
    return 0.0


@dataclass(frozen=True)
class QuestionResult:
    """What one golden question retrieved, with its page references in rank order."""

    question_id: str
    question: str
    unanswerable: bool
    relevant: frozenset[PageRef]
    retrieved: tuple[PageRef, ...]
    scores: tuple[float, ...]
    quote_found: bool | None
    unmappable_chunks: int

    @property
    def top_score(self) -> float | None:
        return self.scores[0] if self.scores else None


@dataclass(frozen=True)
class AnswerableSummary:
    k: int
    recall: float
    hit_rate: float
    mrr: float
    questions: int
    ceiling: float
    """Best recall@k reachable here: a question with more relevant pages than k
    caps below 1.0, so a low recall@k can mean a small k rather than a bad index."""


def summarize_answerable(results: Sequence[QuestionResult], k: int) -> AnswerableSummary:
    """Aggregate one k over the answerable questions. Ignores unanswerable ones."""
    scored = [r for r in results if not r.unanswerable]
    if not scored:
        raise ValueError("no answerable questions to summarize")

    return AnswerableSummary(
        k=k,
        recall=fmean(recall_at_k(r.relevant, r.retrieved, k) for r in scored),
        hit_rate=fmean(float(hit_at_k(r.relevant, r.retrieved, k)) for r in scored),
        mrr=fmean(reciprocal_rank(r.relevant, r.retrieved) for r in scored),
        questions=len(scored),
        ceiling=fmean(min(k, len(r.relevant)) / len(r.relevant) for r in scored),
    )


@dataclass(frozen=True)
class UnanswerableSummary:
    """Top-1 similarity on questions the corpus cannot answer.

    These score no recall. What they measure is the ceiling a refusal threshold
    has to clear: every one of them retrieves *something*, and `above_threshold`
    counts those whose best chunk looks, by score alone, as good as an answerable
    question's. That count is the floor on abstention error before any grader runs.
    """

    questions: int
    threshold: float
    above_threshold: int
    max_score: float | None
    mean_score: float | None
    min_score: float | None


def summarize_unanswerable(
    results: Sequence[QuestionResult], threshold: float
) -> UnanswerableSummary:
    scored = [r for r in results if r.unanswerable]
    tops = [r.top_score for r in scored if r.top_score is not None]

    return UnanswerableSummary(
        questions=len(scored),
        threshold=threshold,
        above_threshold=sum(1 for score in tops if score >= threshold),
        max_score=max(tops) if tops else None,
        mean_score=fmean(tops) if tops else None,
        min_score=min(tops) if tops else None,
    )


def above_floor(result: QuestionResult, floor: float) -> QuestionResult:
    """What a score floor keeps. Scores are best-first, so that is a prefix."""
    if len(result.scores) != len(result.retrieved):
        raise ValueError("a floor needs one score per retrieved chunk")
    kept = sum(1 for score in result.scores if score >= floor)
    return replace(result, retrieved=result.retrieved[:kept], scores=result.scores[:kept])


@dataclass(frozen=True)
class FloorSummary:
    """Questions whose every candidate fell below the floor, by class.

    An emptied answerable question re-enters the retrieval loop for nothing; an
    emptied unanswerable one is refused before any generation is paid for.
    """

    floor: float
    answerable: int
    answerable_emptied: int
    unanswerable: int
    unanswerable_emptied: int


def summarize_floor(results: Sequence[QuestionResult], floor: float) -> FloorSummary:
    def emptied(group: list[QuestionResult]) -> int:
        return sum(1 for r in group if r.top_score is None or r.top_score < floor)

    answerable = [r for r in results if not r.unanswerable]
    unanswerable = [r for r in results if r.unanswerable]
    return FloorSummary(
        floor=floor,
        answerable=len(answerable),
        answerable_emptied=emptied(answerable),
        unanswerable=len(unanswerable),
        unanswerable_emptied=emptied(unanswerable),
    )


# ---------------------------------------------------------- answer metrics


@dataclass(frozen=True)
class AnswerResult:
    """One golden question run through the whole graph.

    Every field comes from what the run wrote, not from the harness.
    """

    question_id: str
    question: str
    unanswerable: bool
    query_id: UUID
    status: TerminalStatus
    refusal_reason: RefusalReason | None
    retrieval_attempts: int
    grounding_attempts: int
    latency_ms: int | None
    """NULL on a run that never finalized — unknown, not zero."""
    cost_usd: Decimal | None
    input_tokens: int
    output_tokens: int
    nodes: int
    groundedness: float | None
    citations: int
    unresolved_citations: int
    """Citations whose chunk was not in the passage set `generate` was shown."""
    fabricated_citations: int
    """Labels the binder dropped, over every generation attempt (ADR 0021)."""
    degraded: bool
    """A fallback or an error on some node (ADR 0011). Excluded from aggregates."""

    @property
    def refused(self) -> bool:
        return self.status == "refused"


@dataclass(frozen=True)
class RefusalSummary:
    """Abstention, both ways round.

    The correct-refusal rate is what the project argues for; the false-refusal
    rate is what it costs. Neither is readable without the other.
    """

    unanswerable: int
    refused_correctly: int
    answerable: int
    refused_falsely: int
    reasons: Mapping[str, int]
    """Refusal reason counts over the unanswerable questions."""

    @property
    def correct_refusal_rate(self) -> float | None:
        return self.refused_correctly / self.unanswerable if self.unanswerable else None

    @property
    def false_refusal_rate(self) -> float | None:
        return self.refused_falsely / self.answerable if self.answerable else None


def summarize_refusals(results: Sequence[AnswerResult]) -> RefusalSummary:
    unanswerable = [r for r in results if r.unanswerable]
    answerable = [r for r in results if not r.unanswerable]
    return RefusalSummary(
        unanswerable=len(unanswerable),
        refused_correctly=sum(1 for r in unanswerable if r.refused),
        answerable=len(answerable),
        refused_falsely=sum(1 for r in answerable if r.refused),
        reasons=Counter(r.refusal_reason for r in unanswerable if r.refusal_reason),
    )


@dataclass(frozen=True)
class GroundednessSummary:
    """`verify_grounding`'s aggregate, over the attempt that decided each run.

    Split by outcome: tau separates them by construction, and a report that gave
    one mean over both would hide how far apart the two sides sit.
    """

    tau: float
    scored: int
    mean: float | None
    min: float | None
    max: float | None
    answered_mean: float | None
    refused_mean: float | None


def summarize_groundedness(results: Sequence[AnswerResult], tau: float) -> GroundednessSummary:
    scored = [r for r in results if r.groundedness is not None]
    scores = [r.groundedness for r in scored if r.groundedness is not None]
    answered = [r.groundedness for r in scored if r.status == "answered" and r.groundedness]
    refused = [r.groundedness for r in scored if r.refused and r.groundedness is not None]
    return GroundednessSummary(
        tau=tau,
        scored=len(scored),
        mean=fmean(scores) if scores else None,
        min=min(scores) if scores else None,
        max=max(scores) if scores else None,
        answered_mean=fmean(answered) if answered else None,
        refused_mean=fmean(refused) if refused else None,
    )


@dataclass(frozen=True)
class CitationSummary:
    """Whether a persisted citation still points at the evidence it was bound to."""

    answered: int
    citations: int
    unresolved: int
    fabricated: int

    @property
    def validity(self) -> float | None:
        if not self.citations:
            return None
        return (self.citations - self.unresolved) / self.citations


def summarize_citations(results: Sequence[AnswerResult]) -> CitationSummary:
    return CitationSummary(
        answered=sum(1 for r in results if r.status == "answered"),
        citations=sum(r.citations for r in results),
        unresolved=sum(r.unresolved_citations for r in results),
        fabricated=sum(r.fabricated_citations for r in results),
    )


@dataclass(frozen=True)
class AttemptsSummary:
    """How many passes each loop took, counted independently."""

    max_attempts: int
    retrieval: Mapping[int, int]
    grounding: Mapping[int, int]


def summarize_attempts(results: Sequence[AnswerResult], max_attempts: int) -> AttemptsSummary:
    return AttemptsSummary(
        max_attempts=max_attempts,
        retrieval=Counter(r.retrieval_attempts for r in results),
        grounding=Counter(r.grounding_attempts for r in results),
    )


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile. At 31 points, interpolation would invent precision."""
    if not values:
        return None
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


@dataclass(frozen=True)
class BudgetSummary:
    """Cost and latency per query, against the configured budgets.

    `unpriced` counts runs whose total is NULL — one call we could not price makes
    the total unknown, not smaller (ADR 0013), so those are left out of the mean
    rather than counted as zero.
    """

    runs: int
    unpriced: int
    mean_cost_usd: Decimal | None
    max_cost_usd: Decimal | None
    cost_budget_usd: Decimal
    over_cost_budget: int
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    mean_nodes: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    max_latency_ms: float | None
    latency_budget_ms: int
    over_latency_budget: int


def summarize_budget(
    results: Sequence[AnswerResult], *, cost_budget_usd: float, latency_budget_s: float
) -> BudgetSummary:
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    latencies = [float(r.latency_ms) for r in results if r.latency_ms is not None]
    budget = Decimal(str(cost_budget_usd))
    latency_budget_ms = int(latency_budget_s * 1000)
    return BudgetSummary(
        runs=len(results),
        unpriced=sum(1 for r in results if r.cost_usd is None),
        mean_cost_usd=(sum(costs, Decimal(0)) / len(costs)) if costs else None,
        max_cost_usd=max(costs) if costs else None,
        cost_budget_usd=budget,
        over_cost_budget=sum(1 for cost in costs if cost > budget),
        mean_input_tokens=fmean(r.input_tokens for r in results) if results else None,
        mean_output_tokens=fmean(r.output_tokens for r in results) if results else None,
        mean_nodes=fmean(r.nodes for r in results) if results else None,
        p50_latency_ms=percentile(latencies, 0.5),
        p95_latency_ms=percentile(latencies, 0.95),
        max_latency_ms=max(latencies) if latencies else None,
        latency_budget_ms=latency_budget_ms,
        over_latency_budget=sum(1 for ms in latencies if ms > latency_budget_ms),
    )
