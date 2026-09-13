"""Integration: /tenants and /collections. Needs `make up`."""

from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import text

from conftest import Authenticate
from prism.api.deps import embedding_provider
from prism.auth import Scope, resolve_key
from prism.collections import CollectionRef
from prism.config import Settings, get_settings
from prism.core.ids import uuid7
from prism.db import get_engine
from stub_provider import DIM, MODEL, StubProvider

pytestmark = pytest.mark.integration

ADMIN = "test-admin-token"
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN}"}
UNKNOWN = "01890000-0000-7000-8000-00000000dead"


@pytest.fixture
def configured(app: FastAPI) -> Iterator[None]:
    def settings() -> Settings:
        return Settings(embedding_model=MODEL, embedding_dim=DIM, admin_token=ADMIN)

    app.dependency_overrides[get_settings] = settings
    app.dependency_overrides[embedding_provider] = StubProvider
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def created_tenants() -> Any:
    """Drop whatever a test created, by name prefix."""
    names: list[str] = []
    yield names
    if names:
        async with get_engine().begin() as conn:
            await conn.execute(
                text("DELETE FROM tenants WHERE name = ANY(:names)"), {"names": names}
            )


class TestCreateTenant:
    async def test_returns_the_tenant_and_a_usable_key(
        self, client: AsyncClient, configured: None, created_tenants: list[str]
    ) -> None:
        name = f"acme-{uuid7()}"
        created_tenants.append(name)

        response = await client.post("/tenants", json={"name": name}, headers=ADMIN_HEADERS)

        assert response.status_code == 201
        body = response.json()
        assert body["name"] == name
        assert body["api_key"].startswith("prism_ak_")

        resolved = await resolve_key(body["api_key"])
        assert str(resolved.tenant_id) == body["id"]
        assert resolved.scopes == frozenset(Scope)

    async def test_the_plaintext_is_not_stored(
        self, client: AsyncClient, configured: None, created_tenants: list[str]
    ) -> None:
        name = f"acme-{uuid7()}"
        created_tenants.append(name)
        plaintext = (
            await client.post("/tenants", json={"name": name}, headers=ADMIN_HEADERS)
        ).json()["api_key"]

        async with get_engine().connect() as conn:
            rows = (
                await conn.execute(
                    text("SELECT key_hash, key_prefix FROM api_keys WHERE key_prefix = :p"),
                    {"p": plaintext[:16]},
                )
            ).all()
        assert rows
        assert all(plaintext not in row.key_hash for row in rows)

    async def test_a_duplicate_name_is_409(
        self, client: AsyncClient, configured: None, created_tenants: list[str]
    ) -> None:
        name = f"acme-{uuid7()}"
        created_tenants.append(name)
        await client.post("/tenants", json={"name": name}, headers=ADMIN_HEADERS)

        again = await client.post("/tenants", json={"name": name}, headers=ADMIN_HEADERS)
        assert again.status_code == 409

    @pytest.mark.parametrize("body", [{}, {"name": ""}, {"name": "   "}, {"name": "x" * 201}])
    async def test_a_malformed_name_is_422(
        self, client: AsyncClient, configured: None, body: dict[str, Any]
    ) -> None:
        response = await client.post("/tenants", json=body, headers=ADMIN_HEADERS)
        assert response.status_code == 422


class TestAdminToken:
    async def test_a_tenant_key_cannot_create_a_tenant(
        self, client: AsyncClient, configured: None
    ) -> None:
        """The whole point of a separate credential."""
        response = await client.post("/tenants", json={"name": "sneaky"})
        assert response.status_code == 401

    async def test_a_wrong_token_cannot_create_a_tenant(
        self, client: AsyncClient, configured: None
    ) -> None:
        response = await client.post(
            "/tenants", json={"name": "sneaky"}, headers={"Authorization": "Bearer nope"}
        )
        assert response.status_code == 401

    async def test_a_tenant_key_cannot_list_tenants(
        self, client: AsyncClient, configured: None
    ) -> None:
        assert (await client.get("/tenants")).status_code == 401

    async def test_an_unset_token_disables_the_routes(
        self, client: AsyncClient, app: FastAPI
    ) -> None:
        """Unconfigured must not mean unprotected."""
        app.dependency_overrides[get_settings] = lambda: Settings(admin_token=None)

        response = await client.post("/tenants", json={"name": "x"}, headers=ADMIN_HEADERS)

        assert response.status_code == 503
        app.dependency_overrides.clear()


class TestReadOwnTenant:
    async def test_a_key_reads_its_own_tenant(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        response = await client.get("/tenants/me")
        assert response.status_code == 200
        assert UUID(response.json()["id"]) == collection.tenant_id

    async def test_it_needs_a_key(
        self, client: AsyncClient, configured: None, unauthenticated: None
    ) -> None:
        assert (await client.get("/tenants/me")).status_code == 401


class TestCollections:
    async def test_a_collection_is_created_under_the_callers_tenant(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        response = await client.post("/collections", json={"name": f"docs-{uuid7()}"})

        assert response.status_code == 201
        body = response.json()
        assert UUID(body["tenant_id"]) == collection.tenant_id
        assert body["embedding_model"] == MODEL
        assert body["embedding_dim"] == DIM

    async def test_the_tenant_cannot_be_chosen_by_the_caller(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        """A tenant_id in the body is ignored, not honoured."""
        response = await client.post(
            "/collections", json={"name": f"docs-{uuid7()}", "tenant_id": UNKNOWN}
        )

        assert response.status_code == 201
        assert UUID(response.json()["tenant_id"]) == collection.tenant_id

    async def test_a_duplicate_name_under_one_tenant_is_409(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        name = f"docs-{uuid7()}"
        await client.post("/collections", json={"name": name})
        assert (await client.post("/collections", json={"name": name})).status_code == 409

    async def test_creating_one_needs_the_admin_scope(
        self, client: AsyncClient, configured: None, authenticate: Authenticate
    ) -> None:
        authenticate(uuid7(), Scope.READ, Scope.INGEST)
        response = await client.post("/collections", json={"name": "nope"})
        assert response.status_code == 403

    async def test_listing_shows_only_the_callers_own(
        self,
        client: AsyncClient,
        configured: None,
        collection: CollectionRef,
        other_collection: CollectionRef,
    ) -> None:
        listed = (await client.get("/collections")).json()

        ids = {UUID(row["id"]) for row in listed}
        assert collection.collection_id in ids
        assert other_collection.collection_id not in ids

    async def test_reading_another_tenants_collection_is_404(
        self,
        client: AsyncClient,
        configured: None,
        collection: CollectionRef,
        other_collection: CollectionRef,
    ) -> None:
        """404 rather than 403 — a 403 would confirm it exists."""
        response = await client.get(f"/collections/{other_collection.collection_id}")
        assert response.status_code == 404

    async def test_reading_its_own_collection_works(
        self, client: AsyncClient, configured: None, collection: CollectionRef
    ) -> None:
        response = await client.get(f"/collections/{collection.collection_id}")
        assert response.status_code == 200
        assert UUID(response.json()["id"]) == collection.collection_id
