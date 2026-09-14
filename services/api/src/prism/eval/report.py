"""Rendering a run: a table to read, and JSON to diff.

The JSON is the artifact that matters. A recall number is only evidence of an
improvement if the run that produced it is comparable to the next one, so every
report records the embedding model, the chunking settings and the corpus size
alongside the metrics — change any of those and the comparison is void, whatever
the numbers do.
"""

import json
from collections.abc import Sequence
from typing import Any

from prism.config import Settings
from prism.eval.golden import GoldenSet
from prism.eval.metrics import (
    AnswerableSummary,
    QuestionResult,
    UnanswerableSummary,
    hit_at_k,
    recall_at_k,
    summarize_answerable,
    summarize_unanswerable,
)
from prism.eval.runner import CorpusStats, Retriever

__all__ = ["Report", "build_report", "render_json", "render_text"]


class Report:
    """A completed run: the metrics, and everything needed to say what they mean."""

    def __init__(
        self,
        *,
        golden: GoldenSet,
        results: Sequence[QuestionResult],
        stats: CorpusStats,
        settings: Settings,
        ks: Sequence[int],
        retriever: Retriever,
    ) -> None:
        self.retriever = retriever
        self.golden = golden
        self.results = tuple(results)
        self.stats = stats
        self.settings = settings
        self.ks = tuple(ks)
        self.answerable: tuple[AnswerableSummary, ...] = tuple(
            summarize_answerable(self.results, k) for k in self.ks
        )
        self.unanswerable: UnanswerableSummary = summarize_unanswerable(
            self.results, settings.abstention_threshold
        )


def build_report(
    *,
    golden: GoldenSet,
    results: Sequence[QuestionResult],
    stats: CorpusStats,
    settings: Settings,
    ks: Sequence[int],
    retriever: Retriever,
) -> Report:
    return Report(
        golden=golden,
        results=results,
        stats=stats,
        settings=settings,
        ks=ks,
        retriever=retriever,
    )


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def render_text(report: Report) -> str:
    settings = report.settings
    golden = report.golden
    lines: list[str] = []

    lines.append(
        f"Golden set   {len(golden.questions)} questions "
        f"({len(golden.answerable)} answerable, {len(golden.unanswerable)} unanswerable)"
    )
    lines.append(
        f"Collection   {golden.collection} — "
        f"{report.stats.documents} documents, {report.stats.chunks} chunks"
    )
    lines.append(
        f"Embedding    {settings.embedding_model}/{settings.embedding_dim}-dim, no task prefix"
    )
    lines.append(
        f"Chunking     {settings.chunk_size_chars} chars, {settings.chunk_overlap_chars} overlap"
    )
    if report.retriever == "hybrid":
        lines.append(
            f"Retriever    hybrid — FTS + vector, RRF k={settings.rrf_k}, "
            f"{settings.retrieval_candidate_k} candidates per half"
        )
    else:
        lines.append("Retriever    vector only — the naive baseline")
    if report.stats.unembedded:
        lines.append(
            f"WARNING      {report.stats.unembedded} chunks have no vector "
            "and are invisible to retrieval"
        )
    lines.append("")

    largest_k = max(report.ks)
    lines.append("Retrieval — relevance judged per (document, page)")
    lines.append("   k   recall@k    hit@k   ceiling")
    for summary in report.answerable:
        lines.append(
            f"  {summary.k:>2}    {_pct(summary.recall)}   {_pct(summary.hit_rate)}   "
            f"{_pct(summary.ceiling)}"
        )
    lines.append("")
    # Reported once, not per row: rank of the first relevant chunk does not
    # depend on k, so a per-k column would repeat one number and read as a bug.
    lines.append(f"  MRR@{largest_k} {report.answerable[-1].mrr:.3f}")
    lines.append("  recall@k counts every relevant page found; hit@k counts finding any one.")
    lines.append("  ceiling is the best recall@k reachable — a question with more relevant")
    lines.append("  pages than k cannot reach 1.0 however good the ranking.")
    lines.append("")

    calibration = report.unanswerable
    if calibration.questions:
        lines.append(f"Refusal calibration — {calibration.questions} unanswerable questions")
        if calibration.max_score is None:
            lines.append("  no top-1 similarity: fusion yields an ordering, not a score.")
            lines.append("  groundedness is verify_grounding's call, never retrieval's.")
        if calibration.max_score is not None and calibration.mean_score is not None:
            lines.append(
                f"  top-1 similarity   max {calibration.max_score:.3f}   "
                f"mean {calibration.mean_score:.3f}   min {calibration.min_score:.3f}"
            )
        lines.append(
            f"  at tau={calibration.threshold:.2f}, {calibration.above_threshold} of "
            f"{calibration.questions} retrieve evidence that scores as answerable"
        )
        if calibration.max_score is not None:
            lines.append(
                f"  a similarity-only threshold would have to exceed "
                f"{calibration.max_score:.3f} to refuse all {calibration.questions}"
            )
        lines.append("")

    largest = max(report.ks)
    misses = [
        r
        for r in report.results
        if not r.unanswerable and not hit_at_k(r.relevant, r.retrieved, largest)
    ]
    if misses:
        lines.append(f"Retrieved nothing relevant at k={largest} — {len(misses)} question(s)")
        for result in misses:
            expected = ", ".join(str(page) for page in sorted(result.relevant))
            lines.append(f"  {result.question_id}  {result.question[:64]}")
            lines.append(f"      expected {expected}")
        lines.append("")

    quoted = [r for r in report.results if r.quote_found is not None]
    if quoted:
        missing = [r for r in quoted if not r.quote_found]
        lines.append(
            f"Supporting quotes   {len(quoted) - len(missing)}/{len(quoted)} found verbatim "
            f"in a retrieved chunk at k={largest}"
        )
        if missing:
            lines.append("  not found (may straddle a chunk boundary — diagnostic, not a failure):")
            for result in missing:
                lines.append(f"    {result.question_id}")

    return "\n".join(lines)


def _question_payload(result: QuestionResult, ks: Sequence[int]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": result.question_id,
        "unanswerable": result.unanswerable,
        "retrieved": [str(page) for page in result.retrieved],
        "top_score": result.top_score,
        "quote_found": result.quote_found,
    }
    if not result.unanswerable:
        payload["relevant"] = [str(page) for page in sorted(result.relevant)]
        payload["recall"] = {str(k): recall_at_k(result.relevant, result.retrieved, k) for k in ks}
        payload["hit"] = {str(k): hit_at_k(result.relevant, result.retrieved, k) for k in ks}
    return payload


def render_json(report: Report) -> str:
    settings = report.settings
    payload: dict[str, Any] = {
        # 2 added `retriever`. A schema-1 report predates hybrid and is vector.
        "schema": 2,
        "run": {
            "collection": report.golden.collection,
            "retriever": report.retriever,
            "documents": report.stats.documents,
            "chunks": report.stats.chunks,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "query_prefix": None,
            "chunk_size_chars": settings.chunk_size_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "questions": len(report.golden.questions),
            "answerable": len(report.golden.answerable),
            "unanswerable": len(report.golden.unanswerable),
        },
        "retrieval": [
            {
                "k": s.k,
                "recall_at_k": s.recall,
                "hit_at_k": s.hit_rate,
                "ceiling": s.ceiling,
            }
            for s in report.answerable
        ],
        # Independent of k: the rank of the first relevant chunk in the full list.
        "mrr": report.answerable[-1].mrr,
        "refusal_calibration": {
            "questions": report.unanswerable.questions,
            "threshold": report.unanswerable.threshold,
            "above_threshold": report.unanswerable.above_threshold,
            "max_score": report.unanswerable.max_score,
            "mean_score": report.unanswerable.mean_score,
            "min_score": report.unanswerable.min_score,
        },
        "questions": [_question_payload(r, report.ks) for r in report.results],
    }
    return json.dumps(payload, indent=2, sort_keys=False)
