"""Integration: listing documents and reading one. Needs `make up`."""

from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import text

from conftest import Authenticate
from prism.api.deps import embedding_provider
from prism.auth import Scope
from prism.collections import CollectionRef
from prism.config import Settings, get_settings
from prism.core.ids import uuid7
from prism.db import get_engine
from seeding import seed
from stub_provider import DIM, MODEL, QueryProvider, graded

pytestmark = pytest.mark.integration

LIST = "/collections/{}/documents"
ONE = "/collections/{}/documents/{}"
UNKNOWN = "01890000-0000-7000-8000-00000000dead"


@pytest.fixture
def configured(app: FastAPI) -> Iterator[None]:
    def settings() -> Settings:
        return Settings(embedding_model=MODEL, embedding_dim=DIM)

    app.dependency_overrides[get_settings] = settings
    app.dependency_overrides[embedding_provider] = QueryProvider
    yield
    app.dependency_overrides.clear()


class TestList:
    async def test_an_empty_collection_lists_nothing(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        response = await client.get(LIST.format(collection.collection_id))
        assert response.status_code == 200
        assert response.json() == []

    async def test_it_reports_the_status_and_chunk_count(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        document_id = await seed(
            collection.collection_id, [("a", graded(0.5)), ("b", graded(0.4))], filename="x.pdf"
        )

        ((document,),) = ((await client.get(LIST.format(collection.collection_id))).json(),)

        assert UUID(document["id"]) == document_id
        assert document["filename"] == "x.pdf"
        assert document["status"] == "ready"
        assert document["chunks"] == 2

    async def test_a_pending_document_is_visible_before_its_chunks_exist(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        """The point of the endpoint: ingestion in progress is observable."""
        document_id = uuid7()
        async with get_engine().begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO documents (id, collection_id, tenant_id, filename, mime_type) "
                    "VALUES (:id, :c, :t, 'pending.pdf', 'application/pdf')"
                ),
                {
                    "id": document_id,
                    "c": collection.collection_id,
                    "t": collection.tenant_id,
                },
            )

        ((document,),) = ((await client.get(LIST.format(collection.collection_id))).json(),)

        assert document["status"] == "pending"
        assert document["chunks"] == 0
        assert document["ingested_at"] is None

    async def test_another_tenants_documents_are_not_listed(
        self,
        client: AsyncClient,
        configured: None,
        collection: CollectionRef,
        other_collection: CollectionRef,
    ) -> None:
        await seed(other_collection.collection_id, [("theirs", graded(0.9))])

        response = await client.get(LIST.format(other_collection.collection_id))

        assert response.status_code == 200
        assert response.json() == []

    async def test_listing_needs_the_read_scope(
        self, client: AsyncClient, configured: None, authenticate: Authenticate
    ) -> None:
        authenticate(uuid7(), Scope.INGEST)
        assert (await client.get(LIST.format(UNKNOWN))).status_code == 403


class TestRead:
    async def test_it_returns_the_document(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        document_id = await seed(collection.collection_id, [("a", graded(0.5))])

        response = await client.get(ONE.format(collection.collection_id, document_id))

        assert response.status_code == 200
        body = response.json()
        assert UUID(body["id"]) == document_id
        assert body["chunks"] == 1
        assert body["ingested_at"] is not None

    async def test_an_unknown_document_is_404(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        response = await client.get(ONE.format(collection.collection_id, UNKNOWN))
        assert response.status_code == 404

    async def test_another_tenants_document_is_404(
        self,
        client: AsyncClient,
        configured: None,
        collection: CollectionRef,
        other_collection: CollectionRef,
    ) -> None:
        """Addressed by its real id, from the wrong tenant."""
        document_id = await seed(other_collection.collection_id, [("theirs", graded(0.9))])

        response = await client.get(ONE.format(other_collection.collection_id, document_id))

        assert response.status_code == 404
        assert "theirs" not in response.text

    async def test_a_document_from_another_collection_is_404(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        """The same tenant, but the document is not in the collection addressed."""
        elsewhere = uuid7()
        async with get_engine().begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO collections (id, tenant_id, name, embedding_model, "
                    " embedding_dim) VALUES (:id, :t, :n, :m, :d)"
                ),
                {
                    "id": elsewhere,
                    "t": collection.tenant_id,
                    "n": f"elsewhere-{elsewhere}",
                    "m": MODEL,
                    "d": DIM,
                },
            )
        document_id = await seed(elsewhere, [("mine, but over there", graded(0.5))])

        response = await client.get(ONE.format(collection.collection_id, document_id))

        assert response.status_code == 404
