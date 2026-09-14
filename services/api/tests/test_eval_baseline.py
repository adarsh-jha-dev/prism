"""The pinned baselines in `eval/runs/baseline-*.json`.

A baseline is only evidence if it was recorded against the question set that is
committed beside it. These assert that correspondence, so editing golden.yaml
without re-recording fails here instead of silently voiding the comparison.

Report JSON only — no corpus, no database, no network.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from prism.eval.golden import Corpus, GoldenSet, PageRef, load_corpus, load_golden_set

EVAL = Path(__file__).resolve().parent.parent / "eval"
BASELINES = sorted(EVAL.glob("runs/baseline-*.json"))


def _page_ref(value: str) -> PageRef:
    doc, page = value.rsplit(":p", 1)
    return PageRef(doc=doc, page=int(page))


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_corpus(EVAL / "corpus.yaml")


@pytest.fixture(scope="module")
def golden(corpus: Corpus) -> GoldenSet:
    return load_golden_set(EVAL / "golden.yaml", corpus)


@pytest.fixture(params=BASELINES, ids=lambda p: p.name)
def baseline(request: pytest.FixtureRequest) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(Path(request.param).read_text(encoding="utf-8"))
    return report


def test_at_least_one_baseline_is_pinned() -> None:
    assert BASELINES, "eval/runs/ holds no baseline-*.json"


def test_covers_exactly_the_committed_questions(
    baseline: dict[str, Any], golden: GoldenSet
) -> None:
    recorded = {q["id"] for q in baseline["questions"]}
    committed = {q.id for q in golden.questions}
    assert recorded == committed, (
        f"missing from the baseline: {sorted(committed - recorded)}; "
        f"no longer in golden.yaml: {sorted(recorded - committed)}"
    )


def test_labels_match_the_committed_set(baseline: dict[str, Any], golden: GoldenSet) -> None:
    by_id = {q.id: q for q in golden.questions}
    drifted = []
    for entry in baseline["questions"]:
        question = by_id[entry["id"]]
        if entry["unanswerable"] != question.unanswerable:
            drifted.append(f"{question.id}: unanswerable differs")
            continue
        recorded = frozenset(_page_ref(ref) for ref in entry.get("relevant", []))
        if recorded != question.relevant:
            drifted.append(f"{question.id}: scored against {sorted(map(str, recorded))}")
    assert not drifted, "baseline scored against different labels:\n  " + "\n  ".join(drifted)


def test_run_block_matches_the_committed_set(
    baseline: dict[str, Any], golden: GoldenSet, corpus: Corpus
) -> None:
    run = baseline["run"]
    assert run["collection"] == golden.collection
    assert run["questions"] == len(golden.questions)
    assert run["answerable"] == len(golden.answerable)
    assert run["unanswerable"] == len(golden.unanswerable)
    assert run["documents"] == len(corpus.documents)


def test_records_what_makes_a_run_comparable(baseline: dict[str, Any]) -> None:
    # Changing any of these voids the comparison, so a baseline must carry them.
    assert baseline["schema"] in (1, 2)
    for field in ("embedding_model", "embedding_dim", "chunk_size_chars", "chunk_overlap_chars"):
        assert baseline["run"][field] is not None
    assert "query_prefix" in baseline["run"]


def test_schema_2_names_its_retriever(baseline: dict[str, Any]) -> None:
    """Recall is not comparable across retrievers, so a run has to say which ran.

    Schema 1 predates the hybrid retriever and is vector-only by construction.
    """
    if baseline["schema"] < 2:
        assert "retriever" not in baseline["run"]
        return
    assert baseline["run"]["retriever"] in ("vector", "hybrid")


def test_a_hybrid_baseline_reports_no_similarity_calibration(baseline: dict[str, Any]) -> None:
    """ADR 0010: fusion yields an ordering. A score here would mean one leaked."""
    if baseline["run"].get("retriever") != "hybrid":
        return
    calibration = baseline["refusal_calibration"]
    assert calibration["max_score"] is None
    assert calibration["above_threshold"] == 0
    assert all(q["top_score"] is None for q in baseline["questions"])
