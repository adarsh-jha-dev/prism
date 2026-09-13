"""The committed golden set, checked against the manifest and the PDFs.

`test_eval_golden.py` covers the loader against synthetic YAML. This covers the
one failure the loader cannot see: a `supporting_quote` filed under a page it is
not on, which scores zero forever and reads as a retrieval miss.

The quote checks need the PDFs, which are gitignored, so they skip when
`eval/corpus/` is empty. Nothing here touches the network.
"""

import re
from pathlib import Path

import pytest

from prism.eval.golden import Corpus, GoldenSet, load_corpus, load_golden_set
from prism.ingestion.pdf import extract_pages

EVAL = Path(__file__).resolve().parent.parent / "eval"
WHITESPACE = re.compile(r"\s+")


def _normalize(value: str) -> str:
    # Extraction breaks lines mid-sentence, so quotes never match literally.
    return WHITESPACE.sub(" ", value).strip().casefold()


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_corpus(EVAL / "corpus.yaml")


@pytest.fixture(scope="module")
def golden(corpus: Corpus) -> GoldenSet:
    return load_golden_set(EVAL / "golden.yaml", corpus)


@pytest.fixture(scope="module")
def pages(corpus: Corpus) -> dict[str, dict[int, str]]:
    missing = [d.filename for d in corpus.documents if not (EVAL / "corpus" / d.filename).exists()]
    if missing:
        pytest.skip(f"corpus not populated ({len(missing)} of {len(corpus.documents)} missing)")
    return {d.filename: dict(extract_pages(EVAL / "corpus" / d.filename)) for d in corpus.documents}


class TestCommittedFiles:
    def test_the_committed_golden_set_loads(self, golden: GoldenSet) -> None:
        assert golden.questions

    def test_ids_are_never_reused(self, golden: GoldenSet) -> None:
        ids = [q.id for q in golden.questions]
        assert len(ids) == len(set(ids))

    def test_every_answerable_question_carries_a_quote(self, golden: GoldenSet) -> None:
        # gq-011 is the deliberate exception: its evidence is a table.
        without = {q.id for q in golden.answerable if q.supporting_quote is None}
        assert without == {"gq-011"}


class TestQuotesAgainstTheCorpus:
    def test_every_supporting_quote_is_on_a_gold_page(
        self, golden: GoldenSet, pages: dict[str, dict[int, str]]
    ) -> None:
        misfiled = []
        for question in golden.answerable:
            if question.supporting_quote is None:
                continue
            needle = _normalize(question.supporting_quote)
            if not any(needle in _normalize(pages[p.doc][p.page]) for p in question.relevant):
                found = [
                    f"{filename}:p{number}"
                    for filename, doc_pages in pages.items()
                    for number, text in doc_pages.items()
                    if needle in _normalize(text)
                ]
                gold = ", ".join(str(p) for p in sorted(question.relevant))
                misfiled.append(f"{question.id}: filed under {gold}, found on {found or 'nothing'}")

        # Reported together; fixing one label per run is how a set stays broken.
        assert not misfiled, "supporting quotes not on their gold page:\n  " + "\n  ".join(misfiled)

    def test_gold_pages_exist_in_their_document(
        self, golden: GoldenSet, pages: dict[str, dict[int, str]]
    ) -> None:
        overruns = [
            f"{q.id}: {p} but {p.doc} has {len(pages[p.doc])} pages"
            for q in golden.answerable
            for p in sorted(q.relevant)
            if p.page > len(pages[p.doc])
        ]
        assert not overruns, "gold pages past the end of the document:\n  " + "\n  ".join(overruns)

    def test_no_gold_page_is_empty(
        self, golden: GoldenSet, pages: dict[str, dict[int, str]]
    ) -> None:
        blank = [
            f"{q.id}: {p}"
            for q in golden.answerable
            for p in sorted(q.relevant)
            if not pages[p.doc][p.page].strip()
        ]
        assert not blank, "gold pages with no text layer:\n  " + "\n  ".join(blank)
