"""Integration: needs `make up`."""

import pytest
from sqlalchemy import text

from prism.core.ids import uuid7
from prism.db import get_engine, get_redis

pytestmark = pytest.mark.integration

CORE_TABLES = {"tenants", "collections", "documents", "chunks", "api_keys"}
EMBEDDING_DIM = 768


async def test_pgvector_extension_is_installed() -> None:
    async with get_engine().connect() as conn:
        version = (
            await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
        ).scalar_one_or_none()
    assert version is not None, "pgvector missing — wrong Postgres image?"


async def test_core_tables_exist() -> None:
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
    present = {r[0] for r in rows}
    assert present >= CORE_TABLES, f"migration 0001 not applied: missing {CORE_TABLES - present}"


async def test_hnsw_index_exists_on_chunks() -> None:
    async with get_engine().connect() as conn:
        indexdef = (
            await conn.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE tablename = 'chunks' AND indexname = 'chunks_embedding_hnsw'"
                )
            )
        ).scalar_one_or_none()
    assert indexdef is not None, "chunks_embedding_hnsw missing"
    assert "hnsw" in indexdef.lower()


async def test_vector_roundtrip_and_nearest_neighbour() -> None:
    """Insert three chunks, confirm the nearest one comes back first."""
    tenant_id, collection_id, document_id = uuid7(), uuid7(), uuid7()
    near = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    mid = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    far = [0.0] * (EMBEDDING_DIM - 1) + [1.0]
    chunks = {"near": near, "mid": mid, "far": far}

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'test-tenant')"),
            {"id": tenant_id},
        )
        await conn.execute(
            text(
                "INSERT INTO collections (id, tenant_id, name) "
                "VALUES (:id, :tenant_id, 'test-collection')"
            ),
            {"id": collection_id, "tenant_id": tenant_id},
        )
        await conn.execute(
            text(
                "INSERT INTO documents (id, collection_id, tenant_id, filename, mime_type) "
                "SELECT :id, c.id, c.tenant_id, 'test.pdf', 'application/pdf' "
                "FROM collections c WHERE c.id = :collection_id"
            ),
            {"id": document_id, "collection_id": collection_id},
        )
        for label, vector in chunks.items():
            await conn.execute(
                text(
                    "INSERT INTO chunks "
                    "(id, document_id, collection_id, tenant_id, content, embedding) "
                    "SELECT :id, d.id, d.collection_id, d.tenant_id, :content, "
                    "       CAST(:embedding AS vector) "
                    "FROM documents d WHERE d.id = :document_id"
                ),
                {
                    "id": uuid7(),
                    "document_id": document_id,
                    "collection_id": collection_id,
                    "content": label,
                    "embedding": str(vector),
                },
            )

    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT content FROM chunks "
                    "WHERE collection_id = :collection_id "
                    "ORDER BY embedding <=> CAST(:probe AS vector) LIMIT 3"
                ),
                {"collection_id": collection_id, "probe": str(near)},
            )
            assert [r[0] for r in result] == ["near", "mid", "far"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


async def test_redis_responds_to_ping() -> None:
    assert await get_redis().ping() is True
