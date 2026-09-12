"""Document ingestion: PDF -> pages -> chunks -> embeddings -> rows.

Composition only. Extraction, chunking and embedding are unchanged and still
tested in isolation; what is new here is the ordering, the transaction
boundaries, and the failure record.

A `documents` row is created by the uploader, not here: `create_document`
records a `pending` document, and `ingest_document` claims it and carries it to
`ready` or `failed`. That split is the queue boundary.

Pages are chunked independently of one another, so a chunk never spans a page
boundary and `chunks.page_number` is exact rather than approximate. Citations
are only as trustworthy as that number.

Reading order is recorded as `metadata.chunk_index`, not left to the chunk ids:
a document's chunks are generated inside one millisecond, and UUIDv7 orders by
random bits below that resolution.

Figure-bearing pages are optionally rendered and read by a vision provider in a
second pass; each parsed figure becomes its own chunk, typed and marked as
model-written. A vision failure fails the document — see ADR 0004.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.collections import (
    CollectionNotFoundError,
    EmbeddingModelMismatchError,
    assert_compatible,
)
from prism.config import Settings, get_settings
from prism.core.ids import uuid7
from prism.db import get_engine
from prism.embeddings import EmbeddingProvider, get_embedding_provider
from prism.ingestion.chunking import chunk_text
from prism.ingestion.pdf import extract_pages
from prism.ingestion.render import RenderedPage, RenderError, render_figure_pages
from prism.vision import ParsedFigure, VisionError, VisionProvider, get_vision_provider

__all__ = [
    "PDF_MIME_TYPE",
    "CollectionNotFoundError",
    "EmbeddingModelMismatchError",
    "IngestionError",
    "IngestionResult",
    "PlannedChunk",
    "RenderError",
    "VisionError",
    "create_document",
    "ingest_document",
    "parse_figures",
    "plan_chunks",
    "plan_document_chunks",
]

log = structlog.get_logger(__name__)

PDF_MIME_TYPE = "application/pdf"

_INSERT_DOCUMENT = text(
    "INSERT INTO documents "
    "(id, collection_id, filename, mime_type, status, storage_path, size_bytes, sha256) "
    "VALUES (:id, :collection_id, :filename, :mime_type, 'pending', "
    ":storage_path, :size_bytes, :sha256)"
)
_SELECT_DOCUMENT = text(
    "SELECT collection_id, filename, mime_type, status FROM documents WHERE id = :id"
)
# Conditional, so the claim is the check: two workers cannot both ingest one
# document and double its chunks.
_CLAIM_DOCUMENT = text(
    "UPDATE documents SET status = 'processing' WHERE id = :id AND status IN ('pending', 'failed')"
)
_INSERT_CHUNK = text(
    "INSERT INTO chunks "
    "(id, document_id, collection_id, content, chunk_type, page_number, embedding, metadata) "
    "VALUES (:id, :document_id, :collection_id, :content, :chunk_type, :page_number, "
    "CAST(:embedding AS vector), CAST(:metadata AS jsonb))"
)
_MARK_READY = text("UPDATE documents SET status = 'ready', ingested_at = now() WHERE id = :id")
_MARK_FAILED = text("UPDATE documents SET status = 'failed' WHERE id = :id")


class IngestionError(RuntimeError):
    """A document could not be ingested. Never softened into a partial ingest —
    retrieval cannot tell "not in the corpus" from "dropped on the way in"."""


@dataclass(frozen=True)
class IngestionResult:
    document_id: UUID
    collection_id: UUID
    pages: int
    chunks: int
    figures: int = 0  # vision-written chunks among `chunks`


@dataclass(frozen=True)
class PlannedChunk:
    """A row-to-be. `metadata` excludes `chunk_index`, assigned once the whole
    document is planned."""

    page_number: int
    content: str
    chunk_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


def plan_chunks(pages: list[tuple[int, str]], settings: Settings) -> list[tuple[int, str]]:
    """Map extracted pages onto `(page_number, chunk)` pairs, in reading order.

    Chunk text is verbatim; whitespace-only windows are dropped rather than
    stripped, since the provider refuses a blank string and one such window
    would fail the whole document.

    Raises IngestionError if nothing survives — `extract_pages` already rejects
    a document with no text layer, so an empty plan means the settings are
    wrong, not the PDF.
    """
    planned = [
        (number, chunk)
        for number, page_text in pages
        for chunk in chunk_text(page_text, settings.chunk_size_chars, settings.chunk_overlap_chars)
        if chunk.strip()
    ]
    if not planned:
        raise IngestionError(f"no chunks produced from {len(pages)} page(s)")
    return planned


def plan_document_chunks(
    pages: list[tuple[int, str]],
    figures: Mapping[int, Sequence[ParsedFigure]],
    settings: Settings,
    *,
    vision_model: str | None = None,
) -> list[PlannedChunk]:
    """Interleave text windows and parsed figures into one reading order.

    Within a page, text precedes its figures. Figure chunks carry
    `source: "vision"`; extracted text carries no marker, which is how
    verbatim content stays distinguishable from generated content.
    """
    by_page: dict[int, list[PlannedChunk]] = {}
    for number, content in plan_chunks(pages, settings):
        by_page.setdefault(number, []).append(PlannedChunk(page_number=number, content=content))

    for number, parsed in figures.items():
        for figure in parsed:
            metadata: dict[str, Any] = {"source": "vision"}
            if vision_model is not None:
                metadata["vision_model"] = vision_model
            if figure.caption is not None:
                # Beside the content, not inside it: the caption is the
                # document's words, the content is the model's.
                metadata["caption"] = figure.caption
            by_page.setdefault(number, []).append(
                PlannedChunk(
                    page_number=number,
                    content=figure.content,
                    chunk_type=figure.kind,
                    metadata=metadata,
                )
            )

    return [chunk for number in sorted(by_page) for chunk in by_page[number]]


async def parse_figures(
    source: str | Path | IO[bytes],
    *,
    provider: VisionProvider,
    settings: Settings,
) -> dict[int, list[ParsedFigure]]:
    """Render the pages that look like they carry figures, and read them.

    Concurrency is capped at the `gemini` lane's width; a failure cancels the
    pages still in flight rather than paying for them.
    """
    rendered = render_figure_pages(source, settings=settings)
    if not rendered:
        return {}

    limit = asyncio.Semaphore(max(1, settings.concurrency_gemini))

    async def parse(page: RenderedPage) -> tuple[int, list[ParsedFigure]]:
        async with limit:
            return page.page_number, await provider.parse_page(page.png)

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(parse(page)) for page in rendered]
    except* Exception as failures:
        # Unwrapped: the route matches on VisionError, not on ExceptionGroup.
        raise _first_leaf(failures) from None

    return {number: parsed for number, parsed in (task.result() for task in tasks) if parsed}


def _first_leaf(group: BaseExceptionGroup[Exception]) -> Exception:
    for exc in group.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            return _first_leaf(exc)
        return exc
    raise AssertionError("an exception group with no exceptions")


async def create_document(
    *,
    document_id: UUID,
    collection_id: UUID,
    filename: str,
    mime_type: str,
    storage_path: str | None = None,
    size_bytes: int | None = None,
    sha256: str | None = None,
    engine: AsyncEngine | None = None,
) -> None:
    """Record an uploaded document as `pending` — stored, not yet ingested.

    The id is the caller's because the blob is written first and named after it:
    a crash then leaves an unreferenced file rather than a row whose bytes never
    arrived.
    """
    try:
        async with (engine or get_engine()).begin() as conn:
            await conn.execute(
                _INSERT_DOCUMENT,
                {
                    "id": document_id,
                    "collection_id": collection_id,
                    "filename": filename,
                    "mime_type": mime_type,
                    "storage_path": storage_path,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                },
            )
    except IntegrityError as exc:
        raise CollectionNotFoundError(f"collection {collection_id} does not exist") from exc


async def ingest_document(
    source: str | Path | IO[bytes],
    *,
    document_id: UUID,
    engine: AsyncEngine | None = None,
    provider: EmbeddingProvider | None = None,
    vision: VisionProvider | None = None,
    settings: Settings | None = None,
) -> IngestionResult:
    """Ingest a `pending` document's bytes, returning what was written.

    The row moves `pending` -> `processing` -> `ready`, or -> `failed` with the
    error re-raised. Chunks and the `ready` transition share one transaction, so
    a document is never readable with a partial chunk set.

    With `vision_enabled`, figure-bearing pages are also parsed by `vision`; a
    failure there fails the document like any other.
    """
    settings = settings or get_settings()
    provider = provider or get_embedding_provider()
    engine = engine or get_engine()
    if settings.vision_enabled and vision is None:
        vision = get_vision_provider()

    async with engine.begin() as conn:
        document = (await conn.execute(_SELECT_DOCUMENT, {"id": document_id})).first()
        if document is None:
            raise IngestionError(f"document {document_id} does not exist")
        if document.mime_type != PDF_MIME_TYPE:
            raise IngestionError(
                f"unsupported mime type {document.mime_type!r}, expected {PDF_MIME_TYPE!r}"
            )
        await assert_compatible(conn, document.collection_id, provider)
        if (await conn.execute(_CLAIM_DOCUMENT, {"id": document_id})).rowcount != 1:
            raise IngestionError(
                f"document {document_id} is {document.status}, not pending — refusing to "
                "ingest it twice"
            )

    collection_id: UUID = document.collection_id
    try:
        pages = extract_pages(source)
        figures: dict[int, list[ParsedFigure]] = {}
        if settings.vision_enabled and vision is not None:
            figures = await parse_figures(source, provider=vision, settings=settings)

        planned = plan_document_chunks(
            pages,
            figures,
            settings,
            vision_model=vision.model if vision is not None else None,
        )
        vectors = await _embed_all([chunk.content for chunk in planned], provider, settings)

        async with engine.begin() as conn:
            await conn.execute(
                _INSERT_CHUNK,
                [
                    {
                        "id": uuid7(),
                        "document_id": document_id,
                        "collection_id": collection_id,
                        "content": chunk.content,
                        "chunk_type": chunk.chunk_type,
                        "page_number": chunk.page_number,
                        "embedding": str(vector),
                        "metadata": json.dumps({"chunk_index": index, **chunk.metadata}),
                    }
                    for index, (chunk, vector) in enumerate(zip(planned, vectors, strict=True))
                ],
            )
            await conn.execute(_MARK_READY, {"id": document_id})
    except Exception as exc:
        await _mark_failed(engine, document_id)
        log.warning(
            "ingestion_failed",
            document_id=str(document_id),
            collection_id=str(collection_id),
            filename=document.filename,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    return IngestionResult(
        document_id=document_id,
        collection_id=collection_id,
        pages=len(pages),
        chunks=len(planned),
        figures=sum(len(parsed) for parsed in figures.values()),
    )


async def _embed_all(
    contents: list[str], provider: EmbeddingProvider, settings: Settings
) -> list[list[float]]:
    batch_size = settings.embed_batch_size
    if batch_size <= 0:
        raise IngestionError(f"embed_batch_size must be positive, got {batch_size}")

    vectors: list[list[float]] = []
    for start in range(0, len(contents), batch_size):
        vectors.extend(await provider.embed(contents[start : start + batch_size]))
    return vectors


async def _mark_failed(engine: AsyncEngine, document_id: UUID) -> None:
    """Best effort: the original failure is what the caller needs to see."""
    try:
        async with engine.begin() as conn:
            await conn.execute(_MARK_FAILED, {"id": document_id})
    except SQLAlchemyError as exc:
        log.error("could_not_mark_failed", document_id=str(document_id), error=str(exc))
