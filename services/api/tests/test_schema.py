"""Integration: needs `make up`."""

from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

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


async def _seed_document(conn: AsyncConnection) -> tuple[UUID, UUID, UUID]:
    """A tenant, collection and document to hang chunks off. Caller deletes the tenant."""
    tenant_id, collection_id, document_id = uuid7(), uuid7(), uuid7()
    await conn.execute(
        text("INSERT INTO tenants (id, name) VALUES (:id, 'fts-tenant')"),
        {"id": tenant_id},
    )
    await conn.execute(
        text(
            "INSERT INTO collections (id, tenant_id, name) "
            "VALUES (:id, :tenant_id, 'fts-collection')"
        ),
        {"id": collection_id, "tenant_id": tenant_id},
    )
    await conn.execute(
        text(
            "INSERT INTO documents (id, collection_id, tenant_id, filename, mime_type) "
            "VALUES (:id, :collection_id, :tenant_id, 'fts.pdf', 'application/pdf')"
        ),
        {"id": document_id, "collection_id": collection_id, "tenant_id": tenant_id},
    )
    return tenant_id, collection_id, document_id


async def test_btree_gin_extension_is_installed() -> None:
    async with get_engine().connect() as conn:
        version = (
            await conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'btree_gin'")
            )
        ).scalar_one_or_none()
    assert version is not None, "btree_gin missing — migration 0007 not applied"


async def test_fts_index_exists_on_chunks() -> None:
    async with get_engine().connect() as conn:
        indexdef = (
            await conn.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE tablename = 'chunks' AND indexname = 'chunks_content_tsv_gin'"
                )
            )
        ).scalar_one_or_none()
    assert indexdef is not None, "chunks_content_tsv_gin missing"
    assert "gin" in indexdef.lower()
    # Scope rides inside the index, not as a post-filter.
    for column in ("tenant_id", "collection_id", "content_tsv"):
        assert column in indexdef, f"{column} not in {indexdef}"


async def test_content_tsv_is_generated_and_stemmed() -> None:
    """The column populates itself, and 'revenue' matches 'revenues' via the english config."""
    engine = get_engine()
    async with engine.begin() as conn:
        tenant_id, collection_id, document_id = await _seed_document(conn)
        for content in ("Quarterly revenues rose sharply in Lisbon", "Unrelated filler text"):
            await conn.execute(
                text(
                    "INSERT INTO chunks "
                    "(id, document_id, collection_id, tenant_id, content) "
                    "VALUES (:id, :document_id, :collection_id, :tenant_id, :content)"
                ),
                {
                    "id": uuid7(),
                    "document_id": document_id,
                    "collection_id": collection_id,
                    "tenant_id": tenant_id,
                    "content": content,
                },
            )

    try:
        async with engine.connect() as conn:
            # Stemming proves the config was applied, not just that a column exists.
            matched = await conn.execute(
                text(
                    "SELECT content FROM chunks "
                    "WHERE tenant_id = :tenant_id AND collection_id = :collection_id "
                    "  AND content_tsv @@ websearch_to_tsquery('english', :q)"
                ),
                {"tenant_id": tenant_id, "collection_id": collection_id, "q": "revenue"},
            )
            assert [r[0] for r in matched] == ["Quarterly revenues rose sharply in Lisbon"]

            # Stopwords are dropped, so the lexeme count is below the word count.
            lexemes = (
                await conn.execute(
                    text(
                        "SELECT length(content_tsv) FROM chunks "
                        "WHERE tenant_id = :tenant_id AND content LIKE 'Quarterly%'"
                    ),
                    {"tenant_id": tenant_id},
                )
            ).scalar_one()
            assert lexemes == 5

            unmatched = await conn.execute(
                text(
                    "SELECT content FROM chunks "
                    "WHERE tenant_id = :tenant_id "
                    "  AND content_tsv @@ websearch_to_tsquery('english', :q)"
                ),
                {"tenant_id": tenant_id, "q": "Madrid"},
            )
            assert unmatched.all() == []
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


async def test_content_tsv_cannot_be_written_directly() -> None:
    """GENERATED ALWAYS is what stops the index drifting from the text it indexes."""
    engine = get_engine()
    async with engine.begin() as conn:
        tenant_id, collection_id, document_id = await _seed_document(conn)

    try:
        with pytest.raises(DBAPIError):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO chunks "
                        "(id, document_id, collection_id, tenant_id, content, content_tsv) "
                        "VALUES (:id, :document_id, :collection_id, :tenant_id, 'real text', "
                        "        to_tsvector('english', 'lies'))"
                    ),
                    {
                        "id": uuid7(),
                        "document_id": document_id,
                        "collection_id": collection_id,
                        "tenant_id": tenant_id,
                    },
                )
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


async def test_redis_responds_to_ping() -> None:
    assert await get_redis().ping() is True
