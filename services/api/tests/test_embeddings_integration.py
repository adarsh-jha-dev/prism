"""Integration: needs `make up`.

Tests marked `ollama` additionally need a local Ollama carrying
nomic-embed-text. That is free and self-hosted — the no-paid-call rule is not
what excludes it from CI; CI simply has no Ollama, so those are deselected
there and the semantic assertions are a local-only signal.
"""

import json
import math
import random
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import text

from prism.core.ids import uuid7
from prism.db import get_engine
from prism.embeddings import OllamaEmbeddingProvider

pytestmark = pytest.mark.integration

DIM = 768

CAT = "The cat sat on the mat."
CAT_ALIKE = "A cat is sitting on the rug."
FINANCE = "Quarterly revenue grew twelve percent in the third quarter."
FINANCE_ALIKE = "Q3 earnings rose 12% year over year."

# Measured locally at ~0.74 within topic against ~0.28 across. The assertion
# leaves room for model drift while still failing on a scrambled index.
MIN_SEPARATION = 0.15

# Enough rows that a sequential scan is genuinely the more expensive plan.
# Below this the planner is right to ignore the index, and the test proves
# nothing about production.
INDEXED_ROWS = 2000


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.fixture(scope="module")
def provider() -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider()


@pytest.fixture
async def collection() -> AsyncIterator[tuple[UUID, UUID]]:
    """A tenant/collection/document triple, dropped by cascade afterwards."""
    tenant_id, collection_id, document_id = uuid7(), uuid7(), uuid7()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'embed-test')"),
            {"id": tenant_id},
        )
        await conn.execute(
            text(
                "INSERT INTO collections (id, tenant_id, name) "
                "VALUES (:id, :tenant_id, 'embed-test')"
            ),
            {"id": collection_id, "tenant_id": tenant_id},
        )
        await conn.execute(
            text(
                "INSERT INTO documents (id, collection_id, filename, mime_type) "
                "VALUES (:id, :collection_id, 'embed-test.txt', 'text/plain')"
            ),
            {"id": document_id, "collection_id": collection_id},
        )
    try:
        yield collection_id, document_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


@pytest.mark.ollama
async def test_similar_sentences_outrank_dissimilar_ones(
    provider: OllamaEmbeddingProvider,
) -> None:
    cat, cat_alike, finance, finance_alike = await provider.embed(
        [CAT, CAT_ALIKE, FINANCE, FINANCE_ALIKE]
    )

    near_cat = cosine(cat, cat_alike)
    near_finance = cosine(finance, finance_alike)
    across = max(
        cosine(cat, finance),
        cosine(cat, finance_alike),
        cosine(cat_alike, finance),
        cosine(cat_alike, finance_alike),
    )

    assert near_cat - across > MIN_SEPARATION, f"{near_cat=} {across=}"
    assert near_finance - across > MIN_SEPARATION, f"{near_finance=} {across=}"


@pytest.mark.ollama
async def test_a_sentence_is_closest_to_itself(provider: OllamaEmbeddingProvider) -> None:
    once, twice = await provider.embed([CAT, CAT])
    assert cosine(once, twice) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.ollama
async def test_vectors_are_unit_norm(provider: OllamaEmbeddingProvider) -> None:
    """The HNSW index is built with vector_cosine_ops on this assumption."""
    for vector in await provider.embed([CAT, FINANCE]):
        assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.ollama
async def test_embedding_round_trips_through_pgvector(
    provider: OllamaEmbeddingProvider,
    collection: tuple[UUID, UUID],
) -> None:
    collection_id, document_id = collection
    chunk_id = uuid7()
    written = await provider.embed_one(CAT)
    assert len(written) == DIM

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO chunks (id, document_id, collection_id, content, embedding) "
                "VALUES (:id, :document_id, :collection_id, :content, CAST(:embedding AS vector))"
            ),
            {
                "id": chunk_id,
                "document_id": document_id,
                "collection_id": collection_id,
                "content": CAT,
                "embedding": str(written),
            },
        )

    async with engine.connect() as conn:
        stored = (
            await conn.execute(
                text("SELECT embedding FROM chunks WHERE id = :id"), {"id": chunk_id}
            )
        ).scalar_one()

    read_back = json.loads(stored)
    assert len(read_back) == DIM
    # pgvector stores float4, so the round trip is lossy by design.
    assert read_back == pytest.approx(written, abs=1e-6)
    assert cosine(read_back, written) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.ollama
async def test_nearest_neighbour_finds_the_semantic_match(
    provider: OllamaEmbeddingProvider,
    collection: tuple[UUID, UUID],
) -> None:
    """The point of the whole thing: query text retrieves related chunk text."""
    collection_id, document_id = collection
    corpus = [CAT, FINANCE]
    vectors = await provider.embed(corpus)

    engine = get_engine()
    async with engine.begin() as conn:
        for content, vector in zip(corpus, vectors, strict=True):
            await conn.execute(
                text(
                    "INSERT INTO chunks (id, document_id, collection_id, content, embedding) "
                    "VALUES (:id, :d, :c, :content, CAST(:e AS vector))"
                ),
                {
                    "id": uuid7(),
                    "d": document_id,
                    "c": collection_id,
                    "content": content,
                    "e": str(vector),
                },
            )

    probe = await provider.embed_one(CAT_ALIKE)
    async with engine.connect() as conn:
        top = (
            await conn.execute(
                text(
                    "SELECT content FROM chunks WHERE collection_id = :c "
                    "ORDER BY embedding <=> CAST(:p AS vector) LIMIT 1"
                ),
                {"c": collection_id, "p": str(probe)},
            )
        ).scalar_one()
    assert top == CAT


async def test_scoped_ann_query_uses_the_hnsw_index(collection: tuple[UUID, UUID]) -> None:
    """The retrieval query shape must reach the index, not fall back to a scan.

    Asserted without disabling seqscan: forcing the plan would prove only that
    the index is usable, not that the planner picks it.
    """
    collection_id, document_id = collection
    rng = random.Random(0)

    rows = []
    for _ in range(INDEXED_ROWS):
        raw = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
        norm = math.sqrt(sum(x * x for x in raw))
        rows.append(
            {
                "id": uuid7(),
                "d": document_id,
                "c": collection_id,
                "content": "filler",
                "e": str([x / norm for x in raw]),
            }
        )

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO chunks (id, document_id, collection_id, content, embedding) "
                "VALUES (:id, :d, :c, :content, CAST(:e AS vector))"
            ),
            rows,
        )
        await conn.execute(text("ANALYZE chunks"))

    probe = str([1.0 / math.sqrt(DIM)] * DIM)
    async with engine.connect() as conn:
        plan_json = (
            await conn.execute(
                text(
                    "EXPLAIN (FORMAT JSON) SELECT id FROM chunks "
                    "WHERE collection_id = :c "
                    "ORDER BY embedding <=> CAST(:p AS vector) LIMIT 10"
                ),
                {"c": collection_id, "p": probe},
            )
        ).scalar_one()

    plan = plan_json[0]["Plan"] if isinstance(plan_json, list) else json.loads(plan_json)[0]["Plan"]
    names = _index_names(plan)
    assert "chunks_embedding_hnsw" in names, f"planner avoided HNSW; used {names or 'a scan'}"


def _index_names(node: dict[str, Any]) -> set[str]:
    found = {node["Index Name"]} if "Index Name" in node else set()
    for child in node.get("Plans", []):
        found |= _index_names(child)
    return found
