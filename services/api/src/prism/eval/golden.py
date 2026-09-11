"""The golden set and the corpus manifest it is written against.

Relevance is recorded as `(filename, page)`, never as a chunk id. Chunk ids are
UUIDv7 minted at ingest, so they change on every re-ingest and on any change to
`chunk_size_chars` — a golden set keyed on them would silently decay into
measuring nothing. Pages survive both.

Loading validates the whole file and raises once with every problem found. A
half-checked golden set is worse than an unparsed one: the run completes, the
number looks plausible, and one typo'd filename has quietly scored zero.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Corpus",
    "CorpusDocument",
    "GoldenQuestion",
    "GoldenSet",
    "GoldenSetError",
    "PageRef",
    "load_corpus",
    "load_golden_set",
]

SCHEMA_VERSION = 1

_HEX = set("0123456789abcdef")


class GoldenSetError(ValueError):
    """A golden set or corpus manifest that cannot be trusted to measure anything."""


@dataclass(frozen=True, order=True)
class PageRef:
    """One page of one document — the unit relevance is judged in."""

    doc: str
    page: int

    def __str__(self) -> str:
        return f"{self.doc}:p{self.page}"


@dataclass(frozen=True)
class CorpusDocument:
    filename: str
    sha256: str
    title: str | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class Corpus:
    documents: tuple[CorpusDocument, ...]

    @property
    def filenames(self) -> frozenset[str]:
        return frozenset(d.filename for d in self.documents)

    def by_filename(self, filename: str) -> CorpusDocument | None:
        return next((d for d in self.documents if d.filename == filename), None)


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    unanswerable: bool
    expected_answer: str | None
    relevant: frozenset[PageRef]
    supporting_quote: str | None
    tags: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return not self.unanswerable


@dataclass(frozen=True)
class GoldenSet:
    collection: str
    questions: tuple[GoldenQuestion, ...]

    @property
    def answerable(self) -> tuple[GoldenQuestion, ...]:
        return tuple(q for q in self.questions if q.answerable)

    @property
    def unanswerable(self) -> tuple[GoldenQuestion, ...]:
        return tuple(q for q in self.questions if q.unanswerable)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenSetError(f"{path} does not exist") from exc
    except yaml.YAMLError as exc:
        raise GoldenSetError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise GoldenSetError(f"{path} must be a mapping at the top level, got {type(raw).__name__}")

    version = raw.get("version")
    if version != SCHEMA_VERSION:
        raise GoldenSetError(f"{path}: version must be {SCHEMA_VERSION}, got {version!r}")
    return raw


def _text(value: Any, field: str, problems: list[str], *, where: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{where}: {field} must be a non-empty string")
        return None
    return value


def load_corpus(path: Path) -> Corpus:
    """Read the corpus manifest. Raises GoldenSetError listing every problem."""
    raw = _load_mapping(path)
    entries = raw.get("documents")
    if not isinstance(entries, list) or not entries:
        raise GoldenSetError(f"{path}: `documents` must be a non-empty list")

    problems: list[str] = []
    documents: list[CorpusDocument] = []
    seen: set[str] = set()

    for index, entry in enumerate(entries):
        where = f"{path}: documents[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} must be a mapping")
            continue

        filename = _text(entry.get("filename"), "filename", problems, where=where)
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or not set(digest.lower()) <= _HEX:
            problems.append(f"{where}: sha256 must be 64 hex characters")
            digest = None

        if filename is not None:
            if filename in seen:
                problems.append(f"{where}: duplicate filename {filename!r}")
            seen.add(filename)

        if filename is None or digest is None:
            continue

        documents.append(
            CorpusDocument(
                filename=filename,
                sha256=digest.lower(),
                title=entry.get("title"),
                source_url=entry.get("source_url"),
            )
        )

    if problems:
        raise GoldenSetError(_joined(problems))
    return Corpus(documents=tuple(documents))


def _parse_relevant(
    value: Any, *, where: str, corpus: Corpus, problems: list[str]
) -> frozenset[PageRef]:
    if not isinstance(value, list) or not value:
        problems.append(f"{where}: relevant must be a non-empty list")
        return frozenset()

    refs: set[PageRef] = set()
    for index, entry in enumerate(value):
        entry_where = f"{where}: relevant[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{entry_where} must be a mapping of doc and pages")
            continue

        doc = _text(entry.get("doc"), "doc", problems, where=entry_where)
        if doc is not None and doc not in corpus.filenames:
            problems.append(f"{entry_where}: {doc!r} is not in the corpus manifest")
            doc = None

        pages = entry.get("pages")
        if not isinstance(pages, list) or not pages:
            problems.append(f"{entry_where}: pages must be a non-empty list")
            continue

        for page in pages:
            # bool is an int in Python; `pages: [true]` is a typo, not page 1.
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                problems.append(f"{entry_where}: page {page!r} must be a positive integer")
                continue
            if doc is not None:
                refs.add(PageRef(doc=doc, page=page))

    return frozenset(refs)


def _parse_question(
    entry: Any, *, where: str, corpus: Corpus, problems: list[str]
) -> GoldenQuestion | None:
    if not isinstance(entry, dict):
        problems.append(f"{where} must be a mapping")
        return None

    question_id = _text(entry.get("id"), "id", problems, where=where)
    question = _text(entry.get("question"), "question", problems, where=where)

    unanswerable = entry.get("unanswerable")
    if not isinstance(unanswerable, bool):
        problems.append(f"{where}: unanswerable must be true or false, and is required")
        return None

    expected_answer: str | None = None
    relevant: frozenset[PageRef] = frozenset()

    if unanswerable:
        # Both would be a contradiction rather than extra detail: a question the
        # corpus cannot answer has no answer to expect and no page to find.
        for field in ("expected_answer", "relevant"):
            if entry.get(field) is not None:
                problems.append(f"{where}: unanswerable questions must not carry {field}")
    else:
        expected_answer = _text(
            entry.get("expected_answer"), "expected_answer", problems, where=where
        )
        relevant = _parse_relevant(
            entry.get("relevant"), where=where, corpus=corpus, problems=problems
        )

    quote = entry.get("supporting_quote")
    if quote is not None and (not isinstance(quote, str) or not quote.strip()):
        problems.append(f"{where}: supporting_quote must be a non-empty string when present")
        quote = None

    tags = entry.get("tags") or []
    if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
        problems.append(f"{where}: tags must be a list of strings")
        tags = []

    if question_id is None or question is None:
        return None

    return GoldenQuestion(
        id=question_id,
        question=question,
        unanswerable=unanswerable,
        expected_answer=expected_answer,
        relevant=relevant,
        supporting_quote=quote,
        tags=tuple(tags),
    )


def load_golden_set(path: Path, corpus: Corpus) -> GoldenSet:
    """Read the golden set, checked against `corpus`.

    Raises GoldenSetError listing every problem found, so one run of the loader
    is enough to fix the file.
    """
    raw = _load_mapping(path)

    problems: list[str] = []
    collection = _text(raw.get("collection"), "collection", problems, where=str(path))

    entries = raw.get("questions")
    if not isinstance(entries, list) or not entries:
        raise GoldenSetError(f"{path}: `questions` must be a non-empty list")

    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"{path}: questions[{index}]"
        parsed = _parse_question(entry, where=where, corpus=corpus, problems=problems)
        if parsed is None:
            continue
        if parsed.id in seen:
            problems.append(f"{where}: duplicate id {parsed.id!r}")
        seen.add(parsed.id)
        questions.append(parsed)

    if problems or collection is None:
        raise GoldenSetError(_joined(problems))
    return GoldenSet(collection=collection, questions=tuple(questions))


def _joined(problems: list[str]) -> str:
    return f"{len(problems)} problem(s):\n" + "\n".join(f"  - {p}" for p in problems)
