"""Document upload.

Ingestion runs inside the request for now, in the ordering a queue would use:
blob written, row recorded `pending`, then ingested.

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

from prism.api.deps import ProviderDep, SettingsDep
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
    create_document,
    ingest_document,
)
from prism.storage import UploadTooLargeError, write_blob

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
    size_bytes: int
    sha256: str


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
    provider: ProviderDep,
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
        await assert_collection_compatible(collection_id, provider)
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
            blob.path, document_id=document_id, provider=provider, settings=settings
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_failed(document_id, exc)) from exc
    except EmbeddingModelMismatchError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=_failed(document_id, exc)) from exc
    except (ExtractionError, IngestionError) as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail=_failed(document_id, exc)
        ) from exc
    except (EmbeddingError, SQLAlchemyError) as exc:
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
        size_bytes=blob.size_bytes,
    )
    return DocumentUploaded(
        document_id=document_id,
        collection_id=collection_id,
        filename=filename,
        status="ready",
        pages=result.pages,
        chunks=result.chunks,
        size_bytes=blob.size_bytes,
        sha256=blob.sha256,
    )
