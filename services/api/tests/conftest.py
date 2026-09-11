"""Unit tests are the default. Integration tests are marked and need `make up`.

Paid providers are never called from either tier — they replay from
tests/fixtures/.
"""

import os
from collections.abc import AsyncIterator
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


@pytest.fixture
async def collection_id() -> AsyncIterator[UUID]:
    """A throwaway tenant and collection. Integration only — needs `make up`."""
    from sqlalchemy import text

    from prism.core.ids import uuid7
    from prism.db import get_engine

    tenant_id, collection_id = uuid7(), uuid7()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'test')"), {"id": tenant_id}
        )
        await conn.execute(
            text("INSERT INTO collections (id, tenant_id, name) VALUES (:id, :tenant_id, 'test')"),
            {"id": collection_id, "tenant_id": tenant_id},
        )
    try:
        yield collection_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})
