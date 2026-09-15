"""Retrieval metrics. Pure functions over ranked page references.

`recall@k` and `hit@k` are reported separately and are not the same number.
Recall is the fraction of a question's relevant pages found in the top k; hit is
whether *any* of them was. A question with four relevant pages cannot score
above 0.25 recall@1 however perfect the ranking, so recall alone reads as a
failure where hit reads as a success — both are true and the pair is the honest
report.

Ranks are chunk ranks, not page ranks: two chunks from one page occupy two
slots, because that is what they cost at retrieval time.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import fmean

from prism.eval.golden import PageRef

__all__ = [
    "AnswerableSummary",
    "FloorSummary",
    "QuestionResult",
    "UnanswerableSummary",
    "above_floor",
    "hit_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "summarize_answerable",
    "summarize_floor",
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
