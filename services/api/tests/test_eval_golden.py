"""Loading and validating the golden set.

The loader's job is to refuse a file that would produce a plausible-looking
number for the wrong reason, so most of these assert on rejection.
"""

from pathlib import Path

import pytest

from prism.eval.golden import Corpus, GoldenSetError, PageRef, load_corpus, load_golden_set

DIGEST = "a" * 64

CORPUS = f"""
version: 1
documents:
  - filename: paper.pdf
    sha256: {DIGEST}
    title: A Paper
"""

GOLDEN = """
version: 1
collection: prism-eval
questions:
  - id: gq-001
    question: What does it say?
    unanswerable: false
    expected_answer: It says a thing.
    relevant:
      - doc: paper.pdf
        pages: [3, 4]
    supporting_quote: a thing
    tags: [architecture]
  - id: gq-002
    question: What does it not say?
    unanswerable: true
"""


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    return load_corpus(write(tmp_path, "corpus.yaml", CORPUS))


class TestLoadCorpus:
    def test_reads_documents(self, corpus: Corpus) -> None:
        assert corpus.filenames == {"paper.pdf"}
        document = corpus.by_filename("paper.pdf")
        assert document is not None
        assert document.sha256 == DIGEST
        assert document.title == "A Paper"

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenSetError, match="does not exist"):
            load_corpus(tmp_path / "absent.yaml")

    def test_rejects_a_wrong_schema_version(self, tmp_path: Path) -> None:
        path = write(tmp_path, "c.yaml", "version: 2\ndocuments: []\n")
        with pytest.raises(GoldenSetError, match="version must be 1"):
            load_corpus(path)

    def test_rejects_a_digest_that_is_not_sha256(self, tmp_path: Path) -> None:
        path = write(
            tmp_path, "c.yaml", "version: 1\ndocuments:\n  - {filename: a.pdf, sha256: nope}\n"
        )
        with pytest.raises(GoldenSetError, match="64 hex characters"):
            load_corpus(path)

    def test_rejects_a_duplicate_filename(self, tmp_path: Path) -> None:
        content = f"""
version: 1
documents:
  - {{filename: a.pdf, sha256: {DIGEST}}}
  - {{filename: a.pdf, sha256: {"b" * 64}}}
"""
        with pytest.raises(GoldenSetError, match="duplicate filename"):
            load_corpus(write(tmp_path, "c.yaml", content))


class TestLoadGoldenSet:
    def test_reads_both_kinds_of_question(self, tmp_path: Path, corpus: Corpus) -> None:
        golden = load_golden_set(write(tmp_path, "g.yaml", GOLDEN), corpus)

        assert golden.collection == "prism-eval"
        assert len(golden.questions) == 2
        assert len(golden.answerable) == 1
        assert len(golden.unanswerable) == 1

    def test_flattens_relevant_pages_into_page_refs(self, tmp_path: Path, corpus: Corpus) -> None:
        golden = load_golden_set(write(tmp_path, "g.yaml", GOLDEN), corpus)
        assert golden.answerable[0].relevant == {
            PageRef("paper.pdf", 3),
            PageRef("paper.pdf", 4),
        }

    def test_rejects_a_relevant_document_absent_from_the_corpus(
        self, tmp_path: Path, corpus: Corpus
    ) -> None:
        """The typo that would otherwise score a silent zero."""
        content = """
version: 1
collection: c
questions:
  - id: gq-001
    question: q
    unanswerable: false
    expected_answer: a
    relevant:
      - {doc: typo.pdf, pages: [1]}
"""
        with pytest.raises(GoldenSetError, match="not in the corpus manifest"):
            load_golden_set(write(tmp_path, "g.yaml", content), corpus)

    def test_rejects_an_unanswerable_question_carrying_evidence(
        self, tmp_path: Path, corpus: Corpus
    ) -> None:
        content = """
version: 1
collection: c
questions:
  - id: gq-001
    question: q
    unanswerable: true
    relevant:
      - {doc: paper.pdf, pages: [1]}
"""
        with pytest.raises(GoldenSetError, match="must not carry relevant"):
            load_golden_set(write(tmp_path, "g.yaml", content), corpus)

    def test_rejects_an_answerable_question_with_no_relevant_pages(
        self, tmp_path: Path, corpus: Corpus
    ) -> None:
        content = """
version: 1
collection: c
questions:
  - {id: gq-001, question: q, unanswerable: false, expected_answer: a}
"""
        with pytest.raises(GoldenSetError, match="relevant must be a non-empty list"):
            load_golden_set(write(tmp_path, "g.yaml", content), corpus)

    def test_requires_unanswerable_to_be_stated(self, tmp_path: Path, corpus: Corpus) -> None:
        content = """
version: 1
collection: c
questions:
  - {id: gq-001, question: q, expected_answer: a}
"""
        with pytest.raises(GoldenSetError, match="unanswerable must be true or false"):
            load_golden_set(write(tmp_path, "g.yaml", content), corpus)

    def test_rejects_a_page_that_is_not_a_positive_integer(
        self, tmp_path: Path, corpus: Corpus
    ) -> None:
        content = """
version: 1
collection: c
questions:
  - id: gq-001
    question: q
    unanswerable: false
    expected_answer: a
    relevant:
      - {doc: paper.pdf, pages: [0]}
"""
        with pytest.raises(GoldenSetError, match="must be a positive integer"):
            load_golden_set(write(tmp_path, "g.yaml", content), corpus)

    def test_rejects_a_duplicate_question_id(self, tmp_path: Path, corpus: Corpus) -> None:
        content = """
version: 1
collection: c
questions:
  - {id: gq-001, question: q, unanswerable: true}
  - {id: gq-001, question: r, unanswerable: true}
"""
        with pytest.raises(GoldenSetError, match="duplicate id"):
            load_golden_set(write(tmp_path, "g.yaml", content), corpus)

    def test_reports_every_problem_at_once(self, tmp_path: Path, corpus: Corpus) -> None:
        """One run of the loader should be enough to fix the file."""
        content = """
version: 1
collection: c
questions:
  - {id: gq-001, question: q, unanswerable: true, expected_answer: a}
  - {id: gq-002, question: "", unanswerable: true}
"""
        with pytest.raises(GoldenSetError) as caught:
            load_golden_set(write(tmp_path, "g.yaml", content), corpus)

        assert "2 problem(s)" in str(caught.value)
        assert "must not carry expected_answer" in str(caught.value)
        assert "question must be a non-empty string" in str(caught.value)
