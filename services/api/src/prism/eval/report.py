"""Rendering a run: a table to read, and JSON to diff.

The JSON is the artifact that matters. A recall number is only evidence of an
improvement if the run that produced it is comparable to the next one, so every
report records the embedding model, the chunking settings and the corpus size
alongside the metrics — change any of those and the comparison is void, whatever
the numbers do.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from prism.config import Settings
from prism.eval.golden import GoldenSet
from prism.eval.metrics import (
    AnswerableSummary,
    AnswerResult,
    AttemptsSummary,
    BudgetSummary,
    CitationSummary,
    FloorSummary,
    GroundednessSummary,
    QuestionResult,
    RefusalSummary,
    UnanswerableSummary,
    above_floor,
    hit_at_k,
    recall_at_k,
    summarize_answerable,
    summarize_attempts,
    summarize_budget,
    summarize_citations,
    summarize_floor,
    summarize_groundedness,
    summarize_refusals,
    summarize_unanswerable,
)
from prism.eval.runner import CorpusStats, Retriever

__all__ = ["AnswerBlock", "Report", "build_report", "render_json", "render_text"]


class AnswerBlock:
    """What the graph did with the same question set.

    A degraded run is excluded from every aggregate and counted, never dropped
    (ADR 0011): losing an observation is itself a number worth reading.
    """

    def __init__(self, results: Sequence[AnswerResult], settings: Settings) -> None:
        self.settings = settings
        self.results = tuple(results)
        self.scored = tuple(r for r in self.results if not r.degraded)
        self.excluded = tuple(r for r in self.results if r.degraded)
        self.refusal: RefusalSummary = summarize_refusals(self.scored)
        self.groundedness: GroundednessSummary = summarize_groundedness(
            self.scored, settings.abstention_threshold
        )
        self.citations: CitationSummary = summarize_citations(self.scored)
        self.attempts: AttemptsSummary = summarize_attempts(self.scored, settings.max_attempts)
        self.budget: BudgetSummary = summarize_budget(
            self.scored,
            cost_budget_usd=settings.cost_budget_usd,
            latency_budget_s=settings.latency_budget_s,
        )


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
        answers: Sequence[AnswerResult] | None = None,
    ) -> None:
        self.retriever = retriever
        self.answers = None if answers is None else AnswerBlock(answers, settings)
        self.golden = golden
        self.stats = stats
        self.settings = settings
        self.ks = tuple(ks)
        self.unfloored = tuple(results)
        self.ordering: tuple[AnswerableSummary, ...] = ()
        self.floor: FloorSummary | None = None

        if retriever == "rerank":
            # Recall is scored on what the floor keeps, since that is all
            # generation sees; the ordering alone is reported beside it.
            floor = settings.rerank_score_floor
            self.results = tuple(above_floor(r, floor) for r in results)
            self.ordering = tuple(summarize_answerable(self.unfloored, k) for k in self.ks)
            self.floor = summarize_floor(self.unfloored, floor)
            # A rerank score is compared with the floor, never with tau.
            threshold = floor
        else:
            self.results = self.unfloored
            threshold = settings.abstention_threshold

        self.answerable: tuple[AnswerableSummary, ...] = tuple(
            summarize_answerable(self.results, k) for k in self.ks
        )
        self.unanswerable: UnanswerableSummary = summarize_unanswerable(self.unfloored, threshold)


def build_report(
    *,
    golden: GoldenSet,
    results: Sequence[QuestionResult],
    stats: CorpusStats,
    settings: Settings,
    ks: Sequence[int],
    retriever: Retriever,
    answers: Sequence[AnswerResult] | None = None,
) -> Report:
    return Report(
        golden=golden,
        results=results,
        stats=stats,
        settings=settings,
        ks=ks,
        retriever=retriever,
        answers=answers,
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
    if report.retriever == "rerank":
        lines.append(
            f"Retriever    hybrid, then {settings.reranker_model} "
            f"({settings.reranker_quantization} @ {settings.reranker_revision[:8]}), "
            f"{settings.rerank_candidate_k} fused candidates, "
            f"floor {settings.rerank_score_floor:.2f}"
        )
    elif report.retriever == "hybrid":
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
    if report.ordering:
        ordering = report.ordering[-1]
        lines.append(
            f"  before the floor: recall@{largest_k} {_pct(ordering.recall)}   "
            f"hit@{largest_k} {_pct(ordering.hit_rate)}   MRR {ordering.mrr:.3f}"
        )
    lines.append("  recall@k counts every relevant page found; hit@k counts finding any one.")
    lines.append("  ceiling is the best recall@k reachable — a question with more relevant")
    lines.append("  pages than k cannot reach 1.0 however good the ranking.")
    lines.append("")

    if report.floor is not None:
        gate = report.floor
        lines.append(f"Rerank floor {gate.floor:.2f} — questions with no candidate above it")
        lines.append(
            f"  answerable     {gate.answerable_emptied} of {gate.answerable}   "
            "would re-enter the retrieval loop"
        )
        lines.append(
            f"  unanswerable   {gate.unanswerable_emptied} of {gate.unanswerable}   "
            "refused before generation"
        )
        lines.append("")

    calibration = report.unanswerable
    unit = "rerank score" if report.retriever == "rerank" else "similarity"
    gate_name = "floor" if report.retriever == "rerank" else "tau"
    if calibration.questions:
        lines.append(f"Refusal calibration — {calibration.questions} unanswerable questions")
        if calibration.max_score is None:
            lines.append("  no top-1 similarity: fusion yields an ordering, not a score.")
            lines.append("  groundedness is verify_grounding's call, never retrieval's.")
        if calibration.max_score is not None and calibration.mean_score is not None:
            lines.append(
                f"  top-1 {unit}   max {calibration.max_score:.3f}   "
                f"mean {calibration.mean_score:.3f}   min {calibration.min_score:.3f}"
            )
        lines.append(
            f"  at {gate_name}={calibration.threshold:.2f}, {calibration.above_threshold} of "
            f"{calibration.questions} retrieve evidence that scores as answerable"
        )
        if calibration.max_score is not None:
            lines.append(
                f"  a {unit}-only threshold would have to exceed "
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

    if report.answers is not None:
        lines.extend(_answer_lines(report.answers))

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


def _rate(value: float | None) -> str:
    return "  n/a " if value is None else _pct(value)


def _distribution(counts: Mapping[int, int], ceiling: int) -> str:
    return "  ".join(f"{n}:{counts.get(n, 0)}" for n in range(0, ceiling + 1))


def _answer_lines(block: AnswerBlock) -> list[str]:
    """The graph's own numbers. Correct refusal first — it is the headline."""
    refusal, grounding = block.refusal, block.groundedness
    citations, budget = block.citations, block.budget
    settings = block.settings
    lines = [
        f"Answer — {len(block.scored)} of {len(block.results)} runs scored",
        f"  Models    plan {settings.planner_model}   grade/verify {settings.grader_model}   "
        f"generate {settings.generator_model}",
        f"  Policy    tau {settings.abstention_threshold:.2f}   "
        f"attempts {settings.max_attempts} per loop   "
        f"doc relevance {settings.doc_relevance_threshold:.2f}",
    ]
    if block.excluded:
        lines.append(
            f"  {len(block.excluded)} excluded as degraded (ADR 0011): "
            + ", ".join(r.question_id for r in block.excluded)
        )
    lines.append("")

    lines.append(
        f"  correct refusal   {_rate(refusal.correct_refusal_rate)}   "
        f"{refusal.refused_correctly} of {refusal.unanswerable} unanswerable refused"
    )
    lines.append(
        f"  false refusal     {_rate(refusal.false_refusal_rate)}   "
        f"{refusal.refused_falsely} of {refusal.answerable} answerable refused"
    )
    for reason, count in sorted(refusal.reasons.items()):
        lines.append(f"      {reason:<24} {count}")
    lines.append("")

    lines.append(f"Groundedness — verify_grounding's aggregate, tau {grounding.tau:.2f}")
    if grounding.scored:
        lines.append(
            f"  {grounding.scored} scored   mean {grounding.mean:.3f}   "
            f"min {grounding.min:.3f}   max {grounding.max:.3f}"
        )
        answered = "n/a" if grounding.answered_mean is None else f"{grounding.answered_mean:.3f}"
        refused = "n/a" if grounding.refused_mean is None else f"{grounding.refused_mean:.3f}"
        lines.append(f"  answered {answered}   refused {refused}")
    else:
        lines.append("  no run reached the gate")
    lines.append("")

    lines.append("Citations — every citation must resolve to a chunk generate was shown")
    lines.append(
        f"  {citations.citations} over {citations.answered} answered runs   "
        f"validity {_rate(citations.validity)}   unresolved {citations.unresolved}"
    )
    lines.append(f"  {citations.fabricated} label(s) dropped by the binder before persisting")
    lines.append("")

    ceiling = block.attempts.max_attempts
    lines.append(f"Attempts — counted independently, max {ceiling} each")
    lines.append(f"  retrieval   {_distribution(block.attempts.retrieval, ceiling)}")
    lines.append(f"  grounding   {_distribution(block.attempts.grounding, ceiling)}")
    lines.append("")

    lines.append("Cost and latency per query, from the trace rows")
    if budget.mean_cost_usd is None:
        lines.append(f"  cost      no priced run ({budget.unpriced} unpriced)")
    else:
        lines.append(
            f"  cost      mean ${budget.mean_cost_usd:.8f}   max ${budget.max_cost_usd:.8f}   "
            f"{budget.over_cost_budget} over the ${budget.cost_budget_usd:.4f} budget"
        )
        if budget.unpriced:
            lines.append(f"            {budget.unpriced} run(s) unpriced, left out of the mean")
    if budget.mean_input_tokens is not None and budget.mean_output_tokens is not None:
        lines.append(
            f"  tokens    mean {budget.mean_input_tokens:.0f} in / "
            f"{budget.mean_output_tokens:.0f} out over {budget.mean_nodes:.1f} nodes"
        )
    if budget.p50_latency_ms is not None and budget.p95_latency_ms is not None:
        lines.append(
            f"  latency   p50 {budget.p50_latency_ms / 1000:.2f}s   "
            f"p95 {budget.p95_latency_ms / 1000:.2f}s   "
            f"max {(budget.max_latency_ms or 0) / 1000:.2f}s   "
            f"{budget.over_latency_budget} over the "
            f"{budget.latency_budget_ms / 1000:.0f}s budget"
        )
    lines.append("")
    return lines


def _question_payload(
    result: QuestionResult, unfloored: QuestionResult, ks: Sequence[int]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": result.question_id,
        "unanswerable": result.unanswerable,
        "retrieved": [str(page) for page in result.retrieved],
        # The best candidate's, whether or not a floor kept it.
        "top_score": unfloored.top_score,
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
        # 2 added `retriever`; 3 the rerank retriever and its blocks; 4 `answers`.
        "schema": 4,
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
        "questions": [
            _question_payload(r, u, report.ks)
            for r, u in zip(report.results, report.unfloored, strict=True)
        ],
    }
    if report.answers is not None:
        payload["answers"] = _answer_payload(report.answers)
    if report.floor is not None:
        payload["run"]["rerank"] = {
            "model": settings.reranker_model,
            "revision": settings.reranker_revision,
            "quantization": settings.reranker_quantization,
            "candidate_k": settings.rerank_candidate_k,
            "max_tokens": settings.rerank_max_tokens,
            "score_floor": report.floor.floor,
        }
        payload["ordering"] = [
            {"k": s.k, "recall_at_k": s.recall, "hit_at_k": s.hit_rate, "mrr": s.mrr}
            for s in report.ordering
        ]
        payload["rerank_floor"] = {
            "floor": report.floor.floor,
            "answerable_emptied": report.floor.answerable_emptied,
            "unanswerable_emptied": report.floor.unanswerable_emptied,
        }
    return json.dumps(payload, indent=2, sort_keys=False)


def _answer_payload(block: AnswerBlock) -> dict[str, Any]:
    refusal, grounding = block.refusal, block.groundedness
    citations, budget, attempts = block.citations, block.budget, block.attempts
    settings = block.settings
    return {
        "runs": len(block.results),
        # A refusal rate is comparable only against the models that graded and
        # generated it, and the constants they decided under.
        "models": {
            "planner": settings.planner_model,
            "grader": settings.grader_model,
            "generator": settings.generator_model,
            "reranker": settings.reranker_model,
        },
        "policy": {
            "abstention_threshold": settings.abstention_threshold,
            "max_attempts": settings.max_attempts,
            "doc_relevance_threshold": settings.doc_relevance_threshold,
            "rerank_score_floor": settings.rerank_score_floor,
        },
        "scored": len(block.scored),
        "excluded_degraded": [r.question_id for r in block.excluded],
        "refusal": {
            "unanswerable": refusal.unanswerable,
            "refused_correctly": refusal.refused_correctly,
            "correct_refusal_rate": refusal.correct_refusal_rate,
            "answerable": refusal.answerable,
            "refused_falsely": refusal.refused_falsely,
            "false_refusal_rate": refusal.false_refusal_rate,
            "reasons": dict(sorted(refusal.reasons.items())),
        },
        "groundedness": {
            "tau": grounding.tau,
            "scored": grounding.scored,
            "mean": grounding.mean,
            "min": grounding.min,
            "max": grounding.max,
            "answered_mean": grounding.answered_mean,
            "refused_mean": grounding.refused_mean,
        },
        "citations": {
            "answered": citations.answered,
            "citations": citations.citations,
            "unresolved": citations.unresolved,
            "validity": citations.validity,
            "fabricated": citations.fabricated,
        },
        "attempts": {
            "max_attempts": attempts.max_attempts,
            "retrieval": {str(k): v for k, v in sorted(attempts.retrieval.items())},
            "grounding": {str(k): v for k, v in sorted(attempts.grounding.items())},
        },
        "budget": {
            "unpriced": budget.unpriced,
            "mean_cost_usd": None if budget.mean_cost_usd is None else str(budget.mean_cost_usd),
            "max_cost_usd": None if budget.max_cost_usd is None else str(budget.max_cost_usd),
            "cost_budget_usd": str(budget.cost_budget_usd),
            "over_cost_budget": budget.over_cost_budget,
            "mean_input_tokens": budget.mean_input_tokens,
            "mean_output_tokens": budget.mean_output_tokens,
            "mean_nodes": budget.mean_nodes,
            "p50_latency_ms": budget.p50_latency_ms,
            "p95_latency_ms": budget.p95_latency_ms,
            "max_latency_ms": budget.max_latency_ms,
            "latency_budget_ms": budget.latency_budget_ms,
            "over_latency_budget": budget.over_latency_budget,
        },
        "questions": [
            {
                "id": r.question_id,
                "unanswerable": r.unanswerable,
                "status": r.status,
                "refusal_reason": r.refusal_reason,
                "retrieval_attempts": r.retrieval_attempts,
                "grounding_attempts": r.grounding_attempts,
                "groundedness": r.groundedness,
                "citations": r.citations,
                "unresolved_citations": r.unresolved_citations,
                "fabricated_citations": r.fabricated_citations,
                "latency_ms": r.latency_ms,
                "cost_usd": None if r.cost_usd is None else str(r.cost_usd),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "nodes": r.nodes,
                "degraded": r.degraded,
            }
            for r in block.results
        ],
    }
