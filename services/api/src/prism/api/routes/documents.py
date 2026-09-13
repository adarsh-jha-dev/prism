"""Document upload, listing and status.

Ingestion runs inside the request for now, in the ordering a queue would use:
blob written, row recorded `pending`, then ingested.

The read endpoints are how ingestion becomes observable: a document that failed,
or that is `ready` with no chunks, is visible over the API instead of only in
the log of whoever ran the upload.

Failures are classified by whose fault they are. A document this service cannot
read is 422 and will never succeed on retry; a provider or database that is down
is 503 and will. Both leave an inspectable `failed` row.
"""

from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from prism import tenancy
from prism.api.deps import IngestDep, ProviderDep, ReadDep, SettingsDep, VisionDep
from prism.collections import (
    CollectionNotFoundError,
    EmbeddingModelMismatchError,
    assert_collection_compatible,
)
from prism.core.ids import uuid7
from prism.embeddings import EmbeddingError
from prism.ingestion import (
    PDF_MIME_TYPE,
    ExtractionError,
    IngestionError,
    RenderError,
    create_document,
    ingest_document,
)
from prism.storage import UploadTooLargeError, write_blob
from prism.tenancy import DocumentRow
from prism.vision import VisionError

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/collections/{collection_id}/documents", tags=["documents"])

_FALLBACK_FILENAME = "upload.pdf"


class DocumentUploaded(BaseModel):
    document_id: UUID
    collection_id: UUID
    filename: str
    status: Literal["ready"]
    pages: int
    chunks: int
    figures: int  # figure/table/equation chunks among `chunks`
    size_bytes: int
    sha256: str


class Document(BaseModel):
    """A document as the API reports it, including where ingestion got to."""

    id: UUID
    collection_id: UUID
    filename: str
    mime_type: str
    status: str
    chunks: int
    size_bytes: int | None
    sha256: str | None
    created_at: str
    ingested_at: str | None

    @classmethod
    def of(cls, row: DocumentRow) -> "Document":
        return cls(
            id=row.id,
            collection_id=row.collection_id,
            filename=row.filename,
            mime_type=row.mime_type,
            status=row.status,
            chunks=row.chunks,
            size_bytes=row.size_bytes,
            sha256=row.sha256,
            created_at=row.created_at.isoformat(),
            ingested_at=row.ingested_at.isoformat() if row.ingested_at else None,
        )


def _failed(document_id: UUID, exc: Exception) -> dict[str, str]:
    return {
        "document_id": str(document_id),
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    collection_id: UUID,
    key: IngestDep,
    provider: ProviderDep,
    vision: VisionDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="A PDF with a text layer.")],
) -> DocumentUploaded:
    """Upload a PDF, ingest it, and return the document it became."""
    if file.content_type != PDF_MIME_TYPE:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported content type {file.content_type!r}, expected {PDF_MIME_TYPE!r}",
        )

    # Starlette has already buffered the form by now, so this saves the copy into
    # storage, not the transfer. A chunked upload declares nothing at all, so
    # write_blob enforces the cap against what actually arrives.
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"upload exceeds {settings.max_upload_bytes} bytes",
        )

    # Before the bytes land, so a refusal here leaves neither a blob nor a row.
    try:
        await assert_collection_compatible(collection_id, provider, tenant_id=key.tenant_id)
    except CollectionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EmbeddingModelMismatchError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    # A client filename is a label, never a path component.
    filename = Path(file.filename or "").name or _FALLBACK_FILENAME
    document_id = uuid7()

    try:
        blob = await write_blob(
            file,
            root=settings.storage_dir,
            collection_id=collection_id,
            document_id=document_id,
            suffix=".pdf",
            max_bytes=settings.max_upload_bytes,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except OSError as exc:
        log.error("blob_write_failed", document_id=str(document_id), error=str(exc))
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail="could not store the uploaded file"
        ) from exc

    try:
        await create_document(
            document_id=document_id,
            collection_id=collection_id,
            tenant_id=key.tenant_id,
            filename=filename,
            mime_type=PDF_MIME_TYPE,
            storage_path=blob.storage_path,
            size_bytes=blob.size_bytes,
            sha256=blob.sha256,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    # The blob outlives a failed ingest: a retry needs the bytes, not a re-upload.
    try:
        result = await ingest_document(
            blob.path,
            document_id=document_id,
            provider=provider,
            vision=vision,
            settings=settings,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_failed(document_id, exc)) from exc
    except EmbeddingModelMismatchError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=_failed(document_id, exc)) from exc
    except (ExtractionError, IngestionError, RenderError) as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail=_failed(document_id, exc)
        ) from exc
    # A provider that is down is this service's problem, not the document's:
    # the same bytes ingest on retry.
    except (EmbeddingError, VisionError, SQLAlchemyError) as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=_failed(document_id, exc)
        ) from exc

    log.info(
        "document_ingested",
        document_id=str(document_id),
        collection_id=str(collection_id),
        filename=filename,
        pages=result.pages,
        chunks=result.chunks,
        figures=result.figures,
        size_bytes=blob.size_bytes,
    )
    return DocumentUploaded(
        document_id=document_id,
        collection_id=collection_id,
        filename=filename,
        status="ready",
        pages=result.pages,
        chunks=result.chunks,
        figures=result.figures,
        size_bytes=blob.size_bytes,
        sha256=blob.sha256,
    )


@router.get("")
async def list_documents(collection_id: UUID, key: ReadDep) -> list[Document]:
    """Every document in the collection, with its ingestion status.

    Empty for a collection the caller does not own — the tenant is part of the
    query, so there is no case where another tenant's documents are listed.
    """
    try:
        rows = await tenancy.list_documents(collection_id=collection_id, tenant_id=key.tenant_id)
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return [Document.of(row) for row in rows]


@router.get("/{document_id}")
async def read_document(collection_id: UUID, document_id: UUID, key: ReadDep) -> Document:
    """404 for another tenant's document, the same as for one that is absent."""
    try:
        row = await tenancy.read_document(
            document_id, collection_id=collection_id, tenant_id=key.tenant_id
        )
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"document {document_id} does not exist"
        )
    return Document.of(row)
