"""Metric math. Pure functions, so no database and no model."""

import pytest

from prism.eval.golden import PageRef
from prism.eval.metrics import (
    QuestionResult,
    above_floor,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_answerable,
    summarize_floor,
    summarize_unanswerable,
)


def page(doc: str, number: int) -> PageRef:
    return PageRef(doc=doc, page=number)


A1, A2, A3 = page("a.pdf", 1), page("a.pdf", 2), page("a.pdf", 3)
B1 = page("b.pdf", 1)


def result(
    *,
    relevant: frozenset[PageRef] = frozenset(),
    retrieved: tuple[PageRef, ...] = (),
    scores: tuple[float, ...] = (),
    unanswerable: bool = False,
    quote_found: bool | None = None,
) -> QuestionResult:
    return QuestionResult(
        question_id="gq-x",
        question="q",
        unanswerable=unanswerable,
        relevant=relevant,
        retrieved=retrieved,
        scores=scores or tuple(1.0 for _ in retrieved),
        quote_found=quote_found,
        unmappable_chunks=0,
    )


class TestRecallAtK:
    def test_counts_the_fraction_of_relevant_pages_found(self) -> None:
        assert recall_at_k(frozenset({A1, A2}), (A1, B1), k=2) == 0.5

    def test_is_one_when_every_relevant_page_is_in_the_window(self) -> None:
        assert recall_at_k(frozenset({A1, A2}), (A1, A2, B1), k=3) == 1.0

    def test_is_zero_when_nothing_relevant_is_retrieved(self) -> None:
        assert recall_at_k(frozenset({A1}), (B1, B1), k=2) == 0.0

    def test_only_the_first_k_count(self) -> None:
        assert recall_at_k(frozenset({A1}), (B1, A1), k=1) == 0.0
        assert recall_at_k(frozenset({A1}), (B1, A1), k=2) == 1.0

    def test_two_chunks_from_one_page_are_one_page(self) -> None:
        """Both slots are spent, but the page is found once."""
        assert recall_at_k(frozenset({A1, A2}), (A1, A1), k=2) == 0.5

    def test_cannot_exceed_one_when_k_exceeds_the_corpus(self) -> None:
        assert recall_at_k(frozenset({A1}), (A1,), k=100) == 1.0

    def test_rejects_a_question_with_no_relevant_pages(self) -> None:
        with pytest.raises(ValueError, match="no relevant pages"):
            recall_at_k(frozenset(), (A1,), k=1)

    def test_rejects_a_non_positive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(frozenset({A1}), (A1,), k=0)


class TestHitAtK:
    def test_is_true_when_any_relevant_page_is_found(self) -> None:
        assert hit_at_k(frozenset({A1, A2}), (B1, A2), k=2) is True

    def test_is_false_when_none_is(self) -> None:
        assert hit_at_k(frozenset({A1}), (B1,), k=1) is False

    def test_distinguishes_itself_from_recall(self) -> None:
        """The case the report exists to separate: full hit, partial recall."""
        relevant = frozenset({A1, A2, A3})
        assert hit_at_k(relevant, (A1,), k=1) is True
        assert recall_at_k(relevant, (A1,), k=1) == pytest.approx(1 / 3)


class TestReciprocalRank:
    def test_is_one_over_the_rank_of_the_first_relevant_chunk(self) -> None:
        assert reciprocal_rank(frozenset({A1}), (B1, B1, A1)) == pytest.approx(1 / 3)

    def test_is_one_when_the_first_chunk_is_relevant(self) -> None:
        assert reciprocal_rank(frozenset({A1}), (A1, B1)) == 1.0

    def test_is_zero_when_nothing_relevant_was_retrieved(self) -> None:
        assert reciprocal_rank(frozenset({A1}), (B1,)) == 0.0


class TestSummarizeAnswerable:
    def test_averages_over_questions_not_over_pages(self) -> None:
        results = [
            result(relevant=frozenset({A1}), retrieved=(A1,)),
            result(relevant=frozenset({A1, A2}), retrieved=(A1,)),
        ]
        summary = summarize_answerable(results, k=1)
        assert summary.recall == pytest.approx((1.0 + 0.5) / 2)
        assert summary.hit_rate == 1.0
        assert summary.questions == 2

    def test_ceiling_reports_the_best_recall_k_allows(self) -> None:
        """Two relevant pages cannot both be found in one slot."""
        results = [result(relevant=frozenset({A1, A2}), retrieved=(A1,))]
        assert summarize_answerable(results, k=1).ceiling == 0.5
        assert summarize_answerable(results, k=2).ceiling == 1.0

    def test_ignores_unanswerable_questions(self) -> None:
        results = [
            result(relevant=frozenset({A1}), retrieved=(A1,)),
            result(unanswerable=True, retrieved=(B1,)),
        ]
        assert summarize_answerable(results, k=1).questions == 1

    def test_rejects_a_set_with_nothing_to_measure(self) -> None:
        with pytest.raises(ValueError, match="no answerable questions"):
            summarize_answerable([result(unanswerable=True)], k=1)


class TestSummarizeUnanswerable:
    def test_counts_those_that_look_answerable_by_score(self) -> None:
        results = [
            result(unanswerable=True, retrieved=(B1,), scores=(0.71,)),
            result(unanswerable=True, retrieved=(B1,), scores=(0.42,)),
            result(relevant=frozenset({A1}), retrieved=(A1,), scores=(0.9,)),
        ]
        summary = summarize_unanswerable(results, threshold=0.58)

        assert summary.questions == 2
        assert summary.above_threshold == 1
        assert summary.max_score == 0.71
        assert summary.min_score == 0.42

    def test_survives_a_question_that_retrieved_nothing(self) -> None:
        summary = summarize_unanswerable([result(unanswerable=True)], threshold=0.58)
        assert summary.questions == 1
        assert summary.max_score is None
        assert summary.above_threshold == 0

    def test_is_empty_when_the_set_has_no_unanswerable_questions(self) -> None:
        summary = summarize_unanswerable([result(relevant=frozenset({A1}))], threshold=0.58)
        assert summary.questions == 0
        assert summary.above_threshold == 0


class TestAboveFloor:
    def test_keeps_the_prefix_at_or_above_the_floor(self) -> None:
        kept = above_floor(result(retrieved=(A1, A2, A3), scores=(0.9, 0.44, 0.2)), floor=0.44)
        assert kept.retrieved == (A1, A2)
        assert kept.scores == (0.9, 0.44)

    def test_can_leave_nothing(self) -> None:
        kept = above_floor(result(retrieved=(A1,), scores=(0.1,)), floor=0.44)
        assert kept.retrieved == () and kept.top_score is None

    def test_rejects_results_without_a_score_per_chunk(self) -> None:
        unscored = QuestionResult(
            question_id="gq-x",
            question="q",
            unanswerable=False,
            relevant=frozenset({A1}),
            retrieved=(A1,),
            scores=(),
            quote_found=None,
            unmappable_chunks=0,
        )
        with pytest.raises(ValueError, match="one score per retrieved chunk"):
            above_floor(unscored, floor=0.44)


class TestSummarizeFloor:
    def test_counts_emptied_questions_per_class(self) -> None:
        results = [
            result(relevant=frozenset({A1}), retrieved=(A1,), scores=(0.9,)),
            result(relevant=frozenset({A1}), retrieved=(A1,), scores=(0.3,)),
            result(unanswerable=True, retrieved=(B1,), scores=(0.1,)),
            result(unanswerable=True, retrieved=(B1,), scores=(0.5,)),
            result(unanswerable=True),
        ]
        summary = summarize_floor(results, floor=0.44)
        assert (summary.answerable, summary.answerable_emptied) == (2, 1)
        assert (summary.unanswerable, summary.unanswerable_emptied) == (3, 2)
