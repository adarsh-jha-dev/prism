"""Blob storage: pure filesystem, no database, no network."""

import hashlib
from pathlib import Path

import pytest

from prism.core.ids import uuid7
from prism.storage import UploadTooLargeError, blob_path, write_blob


class Stream:
    """Hands out `content` in fixed slices, the way UploadFile does."""

    def __init__(self, content: bytes, slice_size: int = 8) -> None:
        self._content = content
        self._slice = slice_size
        self._offset = 0

    async def read(self, size: int = -1, /) -> bytes:
        take = self._slice if size < 0 else min(size, self._slice)
        chunk = self._content[self._offset : self._offset + take]
        self._offset += len(chunk)
        return chunk


async def test_a_blob_is_written_with_its_size_and_digest(tmp_path: Path) -> None:
    collection_id, document_id = uuid7(), uuid7()
    content = b"%PDF-1.7 some bytes"

    blob = await write_blob(
        Stream(content),
        root=tmp_path,
        collection_id=collection_id,
        document_id=document_id,
        suffix=".pdf",
        max_bytes=1024,
    )

    assert blob.path.read_bytes() == content
    assert blob.size_bytes == len(content)
    assert blob.sha256 == hashlib.sha256(content).hexdigest()


async def test_the_recorded_path_is_relative_to_the_root(tmp_path: Path) -> None:
    """Absolute paths in the database would pin every row to one machine."""
    collection_id, document_id = uuid7(), uuid7()
    blob = await write_blob(
        Stream(b"bytes"),
        root=tmp_path,
        collection_id=collection_id,
        document_id=document_id,
        suffix=".pdf",
        max_bytes=1024,
    )
    assert blob.storage_path == f"{collection_id}/{document_id}.pdf"
    assert tmp_path / blob.storage_path == blob.path


async def test_the_path_is_built_from_ids_only(tmp_path: Path) -> None:
    """No part of a client-supplied filename reaches the filesystem."""
    collection_id, document_id = uuid7(), uuid7()
    assert blob_path(tmp_path, collection_id, document_id, ".pdf") == (
        tmp_path / str(collection_id) / f"{document_id}.pdf"
    )


async def test_an_oversized_upload_is_refused_and_leaves_nothing_behind(
    tmp_path: Path,
) -> None:
    collection_id, document_id = uuid7(), uuid7()
    with pytest.raises(UploadTooLargeError, match="exceeds 16 bytes"):
        await write_blob(
            Stream(b"x" * 64),
            root=tmp_path,
            collection_id=collection_id,
            document_id=document_id,
            suffix=".pdf",
            max_bytes=16,
        )
    assert list((tmp_path / str(collection_id)).iterdir()) == []


async def test_a_failed_write_leaves_no_partial_file(tmp_path: Path) -> None:
    """A path that exists is a complete blob — the rename is the commit."""

    class Failing(Stream):
        async def read(self, size: int = -1, /) -> bytes:
            await super().read(size)
            raise ConnectionError("client went away")

    collection_id, document_id = uuid7(), uuid7()
    with pytest.raises(ConnectionError):
        await write_blob(
            Failing(b"partial content"),
            root=tmp_path,
            collection_id=collection_id,
            document_id=document_id,
            suffix=".pdf",
            max_bytes=1024,
        )
    assert list((tmp_path / str(collection_id)).iterdir()) == []


async def test_an_empty_upload_is_stored_rather_than_guessed_at(tmp_path: Path) -> None:
    """Zero bytes is a real upload; it fails later, as an unreadable PDF."""
    blob = await write_blob(
        Stream(b""),
        root=tmp_path,
        collection_id=uuid7(),
        document_id=uuid7(),
        suffix=".pdf",
        max_bytes=1024,
    )
    assert blob.size_bytes == 0
    assert blob.path.exists()
