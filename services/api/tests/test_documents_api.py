"""POST /collections/{id}/documents.

Two tiers in one file. The validation the endpoint does before touching
anything — content type, declared size — is a unit test; everything past that
writes a row and is marked `integration`.

The embedding provider is stubbed throughout, and the storage root is a
tmp_path, so no test leaves bytes behind.
"""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import text

from pdf_builder import build_pdf
from prism.api.deps import embedding_provider, vision_provider
from prism.config import Settings, get_settings
from prism.db import get_engine
from stub_provider import DIM, MODEL, StubProvider, vector_for
from stub_vision import TABLE, StubVision

URL = "/collections/{}/documents"


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def provider() -> StubProvider:
    return StubProvider()


@pytest.fixture
def configured(
    app: FastAPI, storage_root: Path, provider: StubProvider
) -> Iterator[dict[str, object]]:
    """Wire the app to a throwaway storage root and a stub provider."""
    # Pinned rather than inherited: a developer's .env must not decide whether
    # these tests reach a vision provider.
    overrides: dict[str, object] = {
        "max_upload_bytes": 25 * 1024 * 1024,
        "vision_enabled": False,
    }

    def settings() -> Settings:
        return Settings(
            storage_dir=storage_root,
            embedding_model=MODEL,
            embedding_dim=DIM,
            vision_render_dpi=72,
            **overrides,  # type: ignore[arg-type]
        )

    app.dependency_overrides[get_settings] = settings
    app.dependency_overrides[embedding_provider] = lambda: provider
    yield overrides
    app.dependency_overrides.clear()


def upload(
    content: bytes, filename: str = "doc.pdf", content_type: str = "application/pdf"
) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (filename, content, content_type)}


async def fetch_document(document_id: UUID | str) -> dict[str, object] | None:
    async with get_engine().connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT filename, mime_type, status, storage_path, size_bytes, sha256, "
                    "ingested_at FROM documents WHERE id = :id"
                ),
                {"id": str(document_id)},
            )
        ).first()
    return None if row is None else dict(row._mapping)


async def test_a_non_pdf_is_refused_before_any_storage_or_database_call(
    client: AsyncClient, configured: dict[str, object], storage_root: Path
) -> None:
    """415 is decided from the request alone — no row, no bytes, no connection."""
    response = await client.post(
        URL.format("00000000-0000-7000-8000-000000000000"),
        files=upload(b"GIF89a", filename="cat.gif", content_type="image/gif"),
    )
    assert response.status_code == 415
    assert "image/gif" in response.json()["detail"]
    assert not storage_root.exists()


async def test_an_upload_over_the_cap_is_refused_from_its_declared_length(
    client: AsyncClient, configured: dict[str, object], storage_root: Path
) -> None:
    configured["max_upload_bytes"] = 128
    response = await client.post(
        URL.format("00000000-0000-7000-8000-000000000000"),
        files=upload(b"%PDF-1.7" + b"x" * 4096),
    )
    assert response.status_code == 413
    assert not storage_root.exists()


@pytest.mark.integration
async def test_a_pdf_upload_becomes_a_ready_document_with_chunks(
    client: AsyncClient, configured: dict[str, object], collection_id: UUID, storage_root: Path
) -> None:
    content = build_pdf([["alpha"], ["beta"]])

    response = await client.post(URL.format(collection_id), files=upload(content, "paper.pdf"))

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["filename"] == "paper.pdf"
    assert UUID(body["collection_id"]) == collection_id
    assert (body["pages"], body["chunks"]) == (2, 2)
    assert body["size_bytes"] == len(content)

    document = await fetch_document(body["document_id"])
    assert document is not None
    assert document["status"] == "ready"
    assert document["mime_type"] == "application/pdf"
    assert document["ingested_at"] is not None

    stored = storage_root / str(document["storage_path"])
    assert stored.read_bytes() == content
    assert document["size_bytes"] == len(content)
    assert document["sha256"] == body["sha256"]


@pytest.mark.integration
async def test_the_uploaded_chunks_are_retrievable_by_nearest_neighbour(
    client: AsyncClient, configured: dict[str, object], collection_id: UUID
) -> None:
    """The point of uploading: what went in comes back out of the index."""
    response = await client.post(
        URL.format(collection_id), files=upload(build_pdf([["needle"], ["haystack"]]))
    )
    assert response.status_code == 201

    async with get_engine().connect() as conn:
        top = (
            await conn.execute(
                text(
                    "SELECT content FROM chunks WHERE collection_id = :c "
                    "ORDER BY embedding <=> CAST(:p AS vector) LIMIT 1"
                ),
                {"c": collection_id, "p": str(vector_for("needle"))},
            )
        ).scalar_one()
    assert top == "needle"


@pytest.mark.integration
async def test_the_blob_is_named_by_id_not_by_the_clients_filename(
    client: AsyncClient, configured: dict[str, object], collection_id: UUID, storage_root: Path
) -> None:
    """A filename is a label. It is recorded, stripped of any path, and never
    used to build one."""
    response = await client.post(
        URL.format(collection_id), files=upload(build_pdf([["x"]]), "../../../etc/passwd.pdf")
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "passwd.pdf"

    document = await fetch_document(body["document_id"])
    assert document is not None
    assert document["storage_path"] == f"{collection_id}/{body['document_id']}.pdf"
    assert [p.name for p in storage_root.rglob("*.pdf")] == [f"{body['document_id']}.pdf"]


@pytest.mark.integration
async def test_an_unknown_collection_is_404_and_stores_nothing(
    client: AsyncClient, configured: dict[str, object], storage_root: Path
) -> None:
    missing = "01890000-0000-7000-8000-00000000dead"
    response = await client.post(URL.format(missing), files=upload(build_pdf([["x"]])))

    assert response.status_code == 404
    assert missing in response.json()["detail"]
    assert not storage_root.exists()


@pytest.mark.integration
async def test_a_collection_built_for_another_model_is_409(
    client: AsyncClient,
    app: FastAPI,
    configured: dict[str, object],
    collection_id: UUID,
    storage_root: Path,
) -> None:
    """Vectors from the wrong model retrieve nothing meaningful, so the upload
    is refused rather than stored against the wrong index."""

    class WrongProvider(StubProvider):
        model = "bge-m3"
        dim = 1024

    app.dependency_overrides[embedding_provider] = WrongProvider
    response = await client.post(URL.format(collection_id), files=upload(build_pdf([["x"]])))

    assert response.status_code == 409
    assert "bge-m3" in response.json()["detail"]
    assert not storage_root.exists()


@pytest.mark.integration
async def test_a_scanned_pdf_is_422_and_leaves_a_failed_document(
    client: AsyncClient, configured: dict[str, object], collection_id: UUID, storage_root: Path
) -> None:
    """The client's fault and never retryable — but recorded, and the bytes are
    kept so the failure can be inspected."""
    response = await client.post(URL.format(collection_id), files=upload(build_pdf([[], []])))

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["status"] == "failed"
    assert "no text layer" in detail["error"]

    document = await fetch_document(detail["document_id"])
    assert document is not None
    assert document["status"] == "failed"
    assert document["ingested_at"] is None
    assert (storage_root / str(document["storage_path"])).exists()


@pytest.mark.integration
async def test_an_embedding_failure_is_503_not_a_partial_document(
    client: AsyncClient,
    app: FastAPI,
    configured: dict[str, object],
    collection_id: UUID,
    storage_root: Path,
) -> None:
    """Retryable, and distinguished from a bad document: the provider is down,
    the upload was fine."""
    app.dependency_overrides[embedding_provider] = lambda: StubProvider(fail_on="poison")
    response = await client.post(URL.format(collection_id), files=upload(build_pdf([["poison"]])))

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "failed"

    document = await fetch_document(detail["document_id"])
    assert document is not None
    assert document["status"] == "failed"
    assert (storage_root / str(document["storage_path"])).exists()

    async with get_engine().connect() as conn:
        chunks = (
            await conn.execute(
                text("SELECT count(*) FROM chunks WHERE document_id = :d"),
                {"d": detail["document_id"]},
            )
        ).scalar_one()
    assert chunks == 0


@pytest.mark.integration
async def test_a_chunked_upload_over_the_cap_is_refused_while_it_is_written(
    client: AsyncClient, configured: dict[str, object], collection_id: UUID, storage_root: Path
) -> None:
    """A chunked upload declares no length, so the cap can only be enforced
    against what arrives. Nothing is left in storage when it is not."""
    configured["max_upload_bytes"] = 512
    boundary = "prismtestboundary"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="big.pdf"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
        + build_pdf([["x" * 8192]])
        + f"\r\n--{boundary}--\r\n".encode()
    )

    async def chunks() -> AsyncIterator[bytes]:
        for start in range(0, len(body), 1024):
            yield body[start : start + 1024]

    response = await client.post(
        URL.format(collection_id),
        content=chunks(),
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 413
    assert list(storage_root.rglob("*.pdf")) == []


@pytest.mark.integration
async def test_reading_order_and_page_numbers_survive_the_endpoint(
    client: AsyncClient, configured: dict[str, object], collection_id: UUID
) -> None:
    response = await client.post(
        URL.format(collection_id), files=upload(build_pdf([["first"], ["second"], ["third"]]))
    )
    assert response.status_code == 201

    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT content, page_number, (metadata->>'chunk_index')::int AS chunk_index "
                "FROM chunks WHERE document_id = :d ORDER BY chunk_index"
            ),
            {"d": response.json()["document_id"]},
        )
    chunks = [dict(row._mapping) for row in rows]
    assert [c["content"] for c in chunks] == ["first", "second", "third"]
    assert [c["page_number"] for c in chunks] == [1, 2, 3]
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]


@pytest.mark.integration
async def test_the_response_is_the_documented_shape(
    client: AsyncClient, configured: dict[str, object], collection_id: UUID
) -> None:
    """The dashboard is a thin client: it reads these fields, not the database."""
    response = await client.post(URL.format(collection_id), files=upload(build_pdf([["x"]])))
    assert set(response.json()) == {
        "document_id",
        "collection_id",
        "filename",
        "status",
        "pages",
        "chunks",
        "figures",
        "size_bytes",
        "sha256",
    }


@pytest.mark.integration
async def test_a_figure_page_is_reported_in_the_upload_response(
    client: AsyncClient,
    app: FastAPI,
    configured: dict[str, object],
    collection_id: UUID,
) -> None:
    """The count the dashboard shows has to come from what was actually written."""
    configured["vision_enabled"] = True
    configured["vision_min_path_objects"] = 6
    app.dependency_overrides[vision_provider] = lambda: StubVision([TABLE])

    response = await client.post(
        URL.format(collection_id),
        files=upload(build_pdf([["Revenue"]], rules={0: 8}), "figures.pdf"),
    )

    assert response.status_code == 201
    body = response.json()
    assert (body["chunks"], body["figures"]) == (2, 1)


@pytest.mark.integration
async def test_vision_off_reports_no_figures(
    client: AsyncClient, configured: dict[str, object], collection_id: UUID
) -> None:
    response = await client.post(
        URL.format(collection_id), files=upload(build_pdf([["Revenue"]], rules={0: 8}))
    )
    assert response.json()["figures"] == 0


@pytest.mark.integration
async def test_a_vision_failure_is_503_not_a_document_without_its_figures(
    client: AsyncClient,
    app: FastAPI,
    configured: dict[str, object],
    collection_id: UUID,
    storage_root: Path,
) -> None:
    """Same classification as an embedding failure: the provider is down, the
    document is fine, and a retry is the fix."""
    configured["vision_enabled"] = True
    configured["vision_min_path_objects"] = 6
    app.dependency_overrides[vision_provider] = lambda: StubVision([], fail=True)

    response = await client.post(
        URL.format(collection_id), files=upload(build_pdf([["Revenue"]], rules={0: 8}))
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "failed"

    document = await fetch_document(detail["document_id"])
    assert document is not None
    assert document["status"] == "failed"
    # The bytes a retry needs outlive the failure.
    assert (storage_root / str(document["storage_path"])).exists()

    async with get_engine().connect() as conn:
        chunks = (
            await conn.execute(
                text("SELECT count(*) FROM chunks WHERE document_id = :d"),
                {"d": detail["document_id"]},
            )
        ).scalar_one()
    assert chunks == 0
