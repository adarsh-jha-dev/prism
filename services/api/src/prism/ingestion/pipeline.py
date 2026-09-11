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
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO
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

__all__ = [
    "PDF_MIME_TYPE",
    "CollectionNotFoundError",
    "EmbeddingModelMismatchError",
    "IngestionError",
    "IngestionResult",
    "create_document",
    "ingest_document",
    "plan_chunks",
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
    "VALUES (:id, :document_id, :collection_id, :content, 'text', :page_number, "
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
    settings: Settings | None = None,
) -> IngestionResult:
    """Ingest a `pending` document's bytes, returning what was written.

    The row moves `pending` -> `processing` -> `ready`, or -> `failed` with the
    error re-raised. Chunks and the `ready` transition share one transaction, so
    a document is never readable with a partial chunk set.
    """
    settings = settings or get_settings()
    provider = provider or get_embedding_provider()
    engine = engine or get_engine()

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
        planned = plan_chunks(pages, settings)
        vectors = await _embed_all([chunk for _, chunk in planned], provider, settings)

        async with engine.begin() as conn:
            await conn.execute(
                _INSERT_CHUNK,
                [
                    {
                        "id": uuid7(),
                        "document_id": document_id,
                        "collection_id": collection_id,
                        "content": chunk,
                        "page_number": number,
                        "embedding": str(vector),
                        "metadata": json.dumps({"chunk_index": index}),
                    }
                    for index, ((number, chunk), vector) in enumerate(
                        zip(planned, vectors, strict=True)
                    )
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
