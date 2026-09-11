"""Unit tests are the default. Integration tests are marked and need `make up`.

Paid providers are never called from either tier — they replay from
tests/fixtures/.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("PRISM_ENV", "test")


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
async def _throwaway_collection(name: str) -> AsyncIterator[UUID]:
    from sqlalchemy import text

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
        yield collection_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


@pytest.fixture
async def collection_id() -> AsyncIterator[UUID]:
    """A throwaway tenant and collection. Integration only — needs `make up`."""
    async with _throwaway_collection("test") as value:
        yield value


@pytest.fixture
async def other_collection_id() -> AsyncIterator[UUID]:
    """A second tenant's collection, in the same table and the same index."""
    async with _throwaway_collection("other") as value:
        yield value
