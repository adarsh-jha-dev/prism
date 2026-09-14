"""Integration: hybrid retrieval against a real index. Needs `make up`."""

import pytest
from sqlalchemy import text

from prism.collections import CollectionNotFoundError
from prism.config import Settings
from prism.core.ids import uuid7
from prism.db import get_engine
from prism.embeddings import EmbeddingProvider
from prism.retrieval.hybrid import _LEXICAL_HALF, _VECTOR_HALF, hybrid_search

pytestmark = pytest.mark.integration

DIM = 768

# Lexically exact for "chinchilla", but deliberately far from every probe vector.
LEXICAL_ONLY = "The chinchilla provisioning ratio is twenty tokens per parameter."
# Near the probe vector, and shares no rare term with the query.
VECTOR_ONLY = "Scaling budgets are allocated across the training corpus."
# Nearer the probe than LEXICAL_ONLY, so it crowds the vector half's shortlist.
FILLER = "Unrelated material about harbour logistics."

# Both halves shortlist two candidates, so a chunk the vector half ranks third
# is genuinely absent from it — without this the halves both return everything
# and the test proves nothing.
NARROW = Settings(retrieval_candidate_k=2, retrieval_top_k=2)


class _FixedEmbeddings(EmbeddingProvider):
    """Deterministic vectors, so the vector half is a fact and not a model's mood."""

    model = "nomic-embed-text"
    dim = DIM

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed_one(self, text_: str) -> list[float]:
        return self._table[text_]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._table[t] for t in texts]


def _unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def _near(index: int) -> list[float]:
    """Cosine 0.707 to _unit(0) — ahead of anything orthogonal, behind the probe."""
    vector = [0.0] * DIM
    vector[0] = 0.707
    vector[index] = 0.707
    return vector


async def _seed(contents: dict[str, list[float]]) -> tuple:
    tenant_id, collection_id, document_id = uuid7(), uuid7(), uuid7()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": f"hybrid-{tenant_id}"},
        )
        await conn.execute(
            text(
                "INSERT INTO collections (id, tenant_id, name, embedding_model, embedding_dim) "
                "VALUES (:id, :tenant_id, 'hybrid', 'nomic-embed-text', :dim)"
            ),
            {"id": collection_id, "tenant_id": tenant_id, "dim": DIM},
        )
        await conn.execute(
            text(
                "INSERT INTO documents (id, collection_id, tenant_id, filename, mime_type) "
                "VALUES (:id, :collection_id, :tenant_id, 'hybrid.pdf', 'application/pdf')"
            ),
            {"id": document_id, "collection_id": collection_id, "tenant_id": tenant_id},
        )
        for content, vector in contents.items():
            await conn.execute(
                text(
                    "INSERT INTO chunks "
                    "(id, document_id, collection_id, tenant_id, content, embedding) "
                    "VALUES (:id, :document_id, :collection_id, :tenant_id, :content, "
                    "        CAST(:embedding AS vector))"
                ),
                {
                    "id": uuid7(),
                    "document_id": document_id,
                    "collection_id": collection_id,
                    "tenant_id": tenant_id,
                    "content": content,
                    "embedding": str(vector),
                },
            )
    return tenant_id, collection_id


async def _drop(tenant_id) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


async def test_each_half_contributes_what_the_other_misses() -> None:
    """The whole claim of hybrid retrieval, as one assertion."""
    probe = _unit(0)
    tenant_id, collection_id = await _seed(
        {LEXICAL_ONLY: _unit(700), VECTOR_ONLY: probe, FILLER: _near(701)}
    )
    provider = _FixedEmbeddings({"chinchilla ratio": probe})
    try:
        hits = await hybrid_search(
            "chinchilla ratio",
            tenant_id=tenant_id,
            collection_id=collection_id,
            provider=provider,
            settings=NARROW,
        )
        found = {hit.content: hit for hit in hits}
        assert LEXICAL_ONLY in found, "lexical half contributed nothing"
        assert VECTOR_ONLY in found, "vector half contributed nothing"

        # Each was found by exactly one half — that is what makes this hybrid.
        assert found[LEXICAL_ONLY].lexical_rank == 1
        assert found[LEXICAL_ONLY].vector_rank is None, "vector half was not actually narrow"
        assert found[VECTOR_ONLY].vector_rank == 1
        assert found[VECTOR_ONLY].lexical_rank is None
    finally:
        await _drop(tenant_id)


async def test_ranks_are_dense_and_start_at_one() -> None:
    probe = _unit(0)
    tenant_id, collection_id = await _seed({LEXICAL_ONLY: _unit(700), VECTOR_ONLY: probe})
    provider = _FixedEmbeddings({"chinchilla ratio": probe})
    try:
        hits = await hybrid_search(
            "chinchilla ratio",
            tenant_id=tenant_id,
            collection_id=collection_id,
            provider=provider,
        )
        assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    finally:
        await _drop(tenant_id)


async def test_no_lexical_match_is_not_an_error() -> None:
    """websearch_to_tsquery ANDs terms, so prose often matches nothing."""
    probe = _unit(0)
    tenant_id, collection_id = await _seed({VECTOR_ONLY: probe, FILLER: _unit(701)})
    question = "what is the provisioning ratio for a chinchilla optimal model"
    provider = _FixedEmbeddings({question: probe})
    try:
        hits = await hybrid_search(
            question, tenant_id=tenant_id, collection_id=collection_id, provider=provider
        )
        assert hits, "the vector half must still carry the query"
        assert all(hit.lexical_rank is None for hit in hits)
    finally:
        await _drop(tenant_id)


async def test_a_borrowed_collection_id_reads_nothing() -> None:
    """The actual attack: my key, their collection. collection_id alone would pass."""
    probe = _unit(0)
    mine_tenant, _ = await _seed({FILLER: _near(701)})
    theirs_tenant, theirs_collection = await _seed({LEXICAL_ONLY: probe})
    provider = _FixedEmbeddings({"chinchilla ratio": probe})
    try:
        with pytest.raises(CollectionNotFoundError):
            await hybrid_search(
                "chinchilla ratio",
                tenant_id=mine_tenant,
                collection_id=theirs_collection,
                provider=provider,
            )
    finally:
        await _drop(mine_tenant)
        await _drop(theirs_tenant)


async def test_both_halves_scope_by_tenant_inside_their_own_scan() -> None:
    """Runs the module's own statements with a mismatched tenant.

    The composite foreign keys from migration 0004 make a chunk whose tenant_id
    disagrees with its collection unconstructible, so the leak cannot be staged
    as data. What can be checked is that each half's predicate does the work:
    with the wrong tenant, the scan itself must come back empty.
    """
    probe = _unit(0)
    tenant_id, collection_id = await _seed({LEXICAL_ONLY: probe})
    stranger = uuid7()
    try:
        async with get_engine().begin() as conn:
            mine = {
                "collection_id": collection_id,
                "tenant_id": tenant_id,
                "candidates": 10,
            }
            theirs = {**mine, "tenant_id": stranger}

            lexical = {"query_text": "chinchilla"}
            vector = {"query_vector": str(probe)}

            assert (await conn.execute(_LEXICAL_HALF, {**mine, **lexical})).all(), "fixture"
            assert (await conn.execute(_VECTOR_HALF, {**mine, **vector})).all(), "fixture"

            assert (await conn.execute(_LEXICAL_HALF, {**theirs, **lexical})).all() == []
            assert (await conn.execute(_VECTOR_HALF, {**theirs, **vector})).all() == []
    finally:
        await _drop(tenant_id)
