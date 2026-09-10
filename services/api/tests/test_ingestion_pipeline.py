"""Unit: the composition itself — no database, no network, no model."""

import io
from collections.abc import Sequence

import pytest

from pdf_builder import build_pdf
from prism.config import Settings
from prism.core.ids import uuid7
from prism.ingestion.pipeline import (
    IngestionError,
    _embed_all,
    ingest_document,
    plan_chunks,
)

DIM = 8


def _settings(size: int = 10, overlap: int = 2, batch: int = 64) -> Settings:
    return Settings(chunk_size_chars=size, chunk_overlap_chars=overlap, embed_batch_size=batch)


class RecordingProvider:
    """Counts batches; the vectors themselves are meaningless here."""

    model = "nomic-embed-text"
    dim = DIM

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text))] * DIM for text in texts]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


def test_page_numbers_travel_with_their_chunks() -> None:
    pages = [(1, "alpha"), (2, "beta"), (3, "gamma")]
    assert plan_chunks(pages, _settings()) == pages


def test_a_long_page_yields_several_chunks_all_on_that_page() -> None:
    planned = plan_chunks([(7, "abcdefghijklmno")], _settings(size=6, overlap=2))
    assert planned == [(7, "abcdef"), (7, "efghij"), (7, "ijklmn"), (7, "mno")]


def test_chunks_never_span_a_page_boundary() -> None:
    """Page 1's tail must not be glued to page 2's head — a citation would lie."""
    planned = plan_chunks([(1, "abc"), (2, "def")], _settings(size=10, overlap=2))
    assert planned == [(1, "abc"), (2, "def")]


def test_blank_pages_contribute_nothing_but_do_not_shift_numbering() -> None:
    planned = plan_chunks([(1, "front"), (2, ""), (3, "back")], _settings())
    assert planned == [(1, "front"), (3, "back")]


def test_whitespace_only_windows_are_dropped() -> None:
    """The provider refuses a blank string; one such window would fail the document."""
    # Windows: "abc   ", "      ", "   def" — the middle one carries nothing.
    planned = plan_chunks([(1, "abc" + " " * 12 + "def")], _settings(size=6, overlap=0))
    assert planned == [(1, "abc   "), (1, "   def")]


def test_chunk_text_is_verbatim_not_stripped() -> None:
    ((_, chunk),) = plan_chunks([(1, "  padded  ")], _settings(size=100, overlap=0))
    assert chunk == "  padded  "


def test_reading_order_is_preserved_across_pages() -> None:
    pages = [(1, "abcdef"), (2, "ghijkl")]
    assert [chunk for _, chunk in plan_chunks(pages, _settings(size=3, overlap=0))] == [
        "abc",
        "def",
        "ghi",
        "jkl",
    ]


def test_nothing_to_chunk_is_an_error_not_an_empty_document() -> None:
    with pytest.raises(IngestionError, match="no chunks produced from 2 page"):
        plan_chunks([(1, ""), (2, "   ")], _settings())


def test_no_pages_at_all_is_an_error() -> None:
    with pytest.raises(IngestionError, match="no chunks produced from 0 page"):
        plan_chunks([], _settings())


def test_invalid_chunk_settings_surface_from_chunk_text() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller than size"):
        plan_chunks([(1, "abc")], _settings(size=4, overlap=4))


async def test_embedding_is_batched_at_the_configured_size() -> None:
    provider = RecordingProvider()
    vectors = await _embed_all(["a", "bb", "ccc", "dddd", "e"], provider, _settings(batch=2))
    assert [len(batch) for batch in provider.batches] == [2, 2, 1]
    assert len(vectors) == 5


async def test_batching_preserves_order() -> None:
    provider = RecordingProvider()
    vectors = await _embed_all(["a", "bb", "ccc"], provider, _settings(batch=2))
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]


async def test_a_single_batch_is_one_call() -> None:
    provider = RecordingProvider()
    await _embed_all(["a", "b"], provider, _settings(batch=64))
    assert len(provider.batches) == 1


async def test_non_positive_batch_size_is_rejected() -> None:
    with pytest.raises(IngestionError, match="embed_batch_size must be positive"):
        await _embed_all(["a"], RecordingProvider(), _settings(batch=0))


async def test_unsupported_mime_type_is_refused_before_anything_is_written() -> None:
    """Rejected ahead of the database: no row, no connection, no half-ingest."""
    with pytest.raises(IngestionError, match="unsupported mime type 'image/png'"):
        await ingest_document(
            io.BytesIO(build_pdf([["x"]])),
            collection_id=uuid7(),
            filename="x.png",
            mime_type="image/png",
        )
