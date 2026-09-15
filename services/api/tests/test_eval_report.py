"""Unit: what a rerank run's report scores, and against which threshold."""

import json

from prism.config import Settings
from prism.eval.golden import GoldenQuestion, GoldenSet, PageRef
from prism.eval.metrics import QuestionResult
from prism.eval.report import Report, build_report, render_json, render_text
from prism.eval.runner import CorpusStats

A1, B1 = PageRef("a.pdf", 1), PageRef("b.pdf", 1)
SETTINGS = Settings(rerank_score_floor=0.44, abstention_threshold=0.58)


def _question(qid: str, *, unanswerable: bool) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        question=qid,
        unanswerable=unanswerable,
        expected_answer=None if unanswerable else "x",
        relevant=frozenset() if unanswerable else frozenset({A1}),
        supporting_quote=None,
        tags=(),
    )


def _result(
    q: GoldenQuestion, retrieved: tuple[PageRef, ...], scores: tuple[float, ...]
) -> QuestionResult:
    return QuestionResult(
        question_id=q.id,
        question=q.question,
        unanswerable=q.unanswerable,
        relevant=q.relevant,
        retrieved=retrieved,
        scores=scores,
        quote_found=None,
        unmappable_chunks=0,
    )


def _report() -> Report:
    kept = _question("gq-1", unanswerable=False)
    floored = _question("gq-2", unanswerable=False)
    refused = _question("gq-3", unanswerable=True)
    golden = GoldenSet(collection="c", questions=(kept, floored, refused))
    results = [
        _result(kept, (A1, B1), (0.9, 0.1)),
        # Found, but below the floor: ordering credits it, the node does not.
        _result(floored, (A1,), (0.4,)),
        # Above the floor and below tau: must count as clearing, since tau is not its unit.
        _result(refused, (B1,), (0.5,)),
    ]
    return build_report(
        golden=golden,
        results=results,
        stats=CorpusStats(documents=2, chunks=2, unembedded=0),
        settings=SETTINGS,
        ks=[1],
        retriever="rerank",
    )


def test_recall_is_scored_after_the_floor_and_ordering_before_it() -> None:
    report = _report()
    assert report.answerable[0].hit_rate == 0.5
    assert report.ordering[0].hit_rate == 1.0


def test_rerank_scores_are_compared_with_the_floor_never_tau() -> None:
    report = _report()
    assert report.unanswerable.threshold == SETTINGS.rerank_score_floor
    assert report.unanswerable.above_threshold == 1
    assert report.floor is not None
    assert (report.floor.answerable_emptied, report.floor.unanswerable_emptied) == (1, 0)


def test_json_records_the_reranker_and_the_unfloored_top_score() -> None:
    payload = json.loads(render_json(_report()))
    assert payload["schema"] == 3
    assert payload["run"]["rerank"]["score_floor"] == 0.44
    assert payload["run"]["rerank"]["quantization"] == "int8"
    emptied = next(q for q in payload["questions"] if q["id"] == "gq-2")
    assert emptied["retrieved"] == []
    assert emptied["top_score"] == 0.4


def test_text_labels_the_gate_as_the_floor() -> None:
    text = render_text(_report())
    assert "at floor=0.44" in text
    assert "tau" not in text
