"""Unit tests are the default. Integration tests are marked and need `make up`.

Paid providers are never called from either tier — they replay from
tests/fixtures/.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("PRISM_ENV", "test")

if TYPE_CHECKING:
    from prism.auth import ResolvedKey, Scope
    from prism.collections import CollectionRef


class Authenticate(Protocol):
    def __call__(self, tenant_id: UUID, *scopes: "Scope") -> "ResolvedKey": ...


@pytest.fixture
def app() -> FastAPI:
    """A fresh app per test, so one test's dependency overrides cannot leak."""
    from prism.main import create_app

    return create_app()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """In-process client — never binds a port."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@asynccontextmanager
async def _throwaway_collection(name: str) -> AsyncIterator["CollectionRef"]:
    from sqlalchemy import text

    from prism.collections import CollectionRef
    from prism.core.ids import uuid7
    from prism.db import get_engine

    tenant_id, collection_id = uuid7(), uuid7()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": name},
        )
        await conn.execute(
            text("INSERT INTO collections (id, tenant_id, name) VALUES (:id, :tenant_id, :name)"),
            {"id": collection_id, "tenant_id": tenant_id, "name": name},
        )
    try:
        yield CollectionRef(tenant_id=tenant_id, collection_id=collection_id)
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


@pytest.fixture
async def collection() -> AsyncIterator["CollectionRef"]:
    """A throwaway tenant and collection. Integration only — needs `make up`."""
    async with _throwaway_collection("test") as value:
        yield value


@pytest.fixture
async def other_collection() -> AsyncIterator["CollectionRef"]:
    """A second tenant's collection, in the same table and the same index."""
    async with _throwaway_collection("other") as value:
        yield value


@pytest.fixture
def collection_id(collection: "CollectionRef") -> UUID:
    return collection.collection_id


@pytest.fixture
def other_collection_id(other_collection: "CollectionRef") -> UUID:
    return other_collection.collection_id


@pytest.fixture
def authenticate(app: FastAPI) -> "Authenticate":
    """Make subsequent requests carry a resolved key for `tenant_id`.

    Overrides the dependency rather than issuing a real key: these tests are
    about what the routes do with a tenant, not about key resolution, which
    test_auth_integration.py covers against the table.
    """
    from prism.api.deps import require_key
    from prism.auth import ResolvedKey, Scope
    from prism.core.ids import uuid7

    def authenticate(tenant_id: UUID, *scopes: Scope) -> ResolvedKey:
        key = ResolvedKey(
            id=uuid7(),
            tenant_id=tenant_id,
            scopes=frozenset(scopes or (Scope.READ, Scope.INGEST)),
            rate_limit_rpm=60,
        )
        app.dependency_overrides[require_key] = lambda: key
        return key

    return authenticate


@pytest.fixture(autouse=True)
def _authenticate_route_tests(request: pytest.FixtureRequest) -> None:
    """Every route test is authenticated unless it says otherwise.

    As the collection's own tenant when one is in play, so an integration test
    reads what it seeded. A test about rejection pops the override:

        app.dependency_overrides.pop(require_key)
    """
    if "client" not in request.fixturenames:
        return

    from prism.core.ids import uuid7

    authenticate = request.getfixturevalue("authenticate")
    tenant_id = (
        request.getfixturevalue("collection").tenant_id
        if "collection" in request.fixturenames
        else uuid7()
    )
    authenticate(tenant_id)
