"""Integration: the api_keys columns migration 0003 adds. Needs `make up`."""

from collections.abc import AsyncIterator
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from prism.core.ids import uuid7
from prism.db import get_engine

pytestmark = pytest.mark.integration

INSERT = text(
    "INSERT INTO api_keys (id, tenant_id, key_hash, name, key_prefix, scopes) "
    "VALUES (:id, :tenant_id, :key_hash, :name, :key_prefix, CAST(:scopes AS text[]))"
)


@pytest.fixture
async def tenant_id() -> AsyncIterator[UUID]:
    tenant = uuid7()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"schema-test-{tenant}"},
        )
    try:
        yield tenant
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant})


async def _insert(tenant: UUID, *, prefix: str = "prism_ak", scopes: str = "{read}") -> UUID:
    key_id = uuid7()
    async with get_engine().begin() as conn:
        await conn.execute(
            INSERT,
            {
                "id": key_id,
                "tenant_id": tenant,
                "key_hash": f"hash-{key_id}",
                "name": "test key",
                "key_prefix": prefix,
                "scopes": scopes,
            },
        )
    return key_id


async def test_the_new_columns_exist(tenant_id: UUID) -> None:
    key_id = await _insert(tenant_id)
    async with get_engine().connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT key_prefix, scopes, last_used_at, expires_at "
                    "FROM api_keys WHERE id = :id"
                ),
                {"id": key_id},
            )
        ).one()
    assert row.key_prefix == "prism_ak"
    assert row.scopes == ["read"]
    assert row.last_used_at is None
    assert row.expires_at is None


async def test_scopes_defaults_to_read(tenant_id: UUID) -> None:
    key_id = uuid7()
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO api_keys (id, tenant_id, key_hash, name, key_prefix) "
                "VALUES (:id, :tenant_id, :key_hash, :name, :key_prefix)"
            ),
            {
                "id": key_id,
                "tenant_id": tenant_id,
                "key_hash": f"hash-{key_id}",
                "name": "test key",
                "key_prefix": "prism_ak",
            },
        )
    async with get_engine().connect() as conn:
        scopes = (
            await conn.execute(text("SELECT scopes FROM api_keys WHERE id = :id"), {"id": key_id})
        ).scalar_one()
    assert scopes == ["read"]


@pytest.mark.parametrize("scopes", ["{read,ingest}", "{admin}", "{read,ingest,admin}"])
async def test_accepts_the_defined_scopes(tenant_id: UUID, scopes: str) -> None:
    assert await _insert(tenant_id, scopes=scopes)


@pytest.mark.parametrize("scopes", ["{write}", "{read,write}", "{}", "{READ}"])
async def test_rejects_undefined_or_empty_scopes(tenant_id: UUID, scopes: str) -> None:
    with pytest.raises(IntegrityError):
        await _insert(tenant_id, scopes=scopes)


async def test_key_prefix_is_indexed_but_not_unique(tenant_id: UUID) -> None:
    """Prefixes collide by design; the hash is what identifies a key."""
    await _insert(tenant_id, prefix="prism_shared")
    await _insert(tenant_id, prefix="prism_shared")

    async with get_engine().connect() as conn:
        indexdef = (
            await conn.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE tablename = 'api_keys' AND indexname = 'api_keys_key_prefix_idx'"
                )
            )
        ).scalar_one_or_none()
    assert indexdef is not None, "api_keys_key_prefix_idx missing"
    assert "UNIQUE" not in indexdef.upper()


async def test_key_prefix_is_required(tenant_id: UUID) -> None:
    """The backfill default is dropped, so a new key cannot omit its prefix."""
    with pytest.raises(IntegrityError):
        async with get_engine().begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO api_keys (id, tenant_id, key_hash, name) "
                    "VALUES (:id, :tenant_id, :key_hash, :name)"
                ),
                {
                    "id": uuid7(),
                    "tenant_id": tenant_id,
                    "key_hash": "hash-no-prefix",
                    "name": "test key",
                },
            )
