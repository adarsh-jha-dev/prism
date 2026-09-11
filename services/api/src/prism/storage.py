"""Uploaded document blobs.

Keyed by ids, never by the client's filename: no part of a client-supplied name
reaches a path. Local filesystem now, S3 in the deployed shape.

Ordering and what survives a failure: docs/decisions/0003-uploaded-files-on-local-disk.md
"""

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

import structlog

__all__ = ["ByteStream", "StoredBlob", "UploadTooLargeError", "blob_path", "write_blob"]

log = structlog.get_logger(__name__)

_READ_CHUNK_BYTES = 1024 * 1024


class UploadTooLargeError(RuntimeError):
    """More bytes arrived than `max_upload_bytes` allows."""


class ByteStream(Protocol):
    """The part of Starlette's UploadFile this module needs."""

    async def read(self, size: int = -1, /) -> bytes: ...


@dataclass(frozen=True)
class StoredBlob:
    path: Path
    storage_path: str
    size_bytes: int
    sha256: str


def blob_path(root: Path, collection_id: UUID, document_id: UUID, suffix: str) -> Path:
    """Where a document's bytes live. Sharded by collection so one directory
    does not accumulate every tenant's uploads."""
    return root / str(collection_id) / f"{document_id}{suffix}"


async def write_blob(
    stream: ByteStream,
    *,
    root: Path,
    collection_id: UUID,
    document_id: UUID,
    suffix: str,
    max_bytes: int,
) -> StoredBlob:
    """Stream `stream` to storage, returning what was written.

    Written to a `.part` file and renamed on completion, so a path that exists
    is a complete blob. Raises UploadTooLargeError past `max_bytes`, leaving
    nothing behind; the partial file is removed on any failure.
    """
    target = blob_path(root, collection_id, document_id, suffix)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    size = 0

    await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
    try:
        handle = await asyncio.to_thread(partial.open, "wb")
        try:
            while chunk := await stream.read(_READ_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLargeError(f"upload exceeds {max_bytes} bytes")
                digest.update(chunk)
                await asyncio.to_thread(handle.write, chunk)
        finally:
            await asyncio.to_thread(handle.close)
        await asyncio.to_thread(partial.replace, target)
    except BaseException:
        await asyncio.to_thread(partial.unlink, True)
        raise

    return StoredBlob(
        path=target,
        storage_path=str(target.relative_to(root)),
        size_bytes=size,
        sha256=digest.hexdigest(),
    )
