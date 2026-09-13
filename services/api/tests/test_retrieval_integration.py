"""Integration: needs `make up`.

Chunks are seeded straight into the table with known vectors, so the expected
ranking is arithmetic rather than a property of any embedding model.
"""

from uuid import UUID

import pytest

from prism.collections import (
    CollectionNotFoundError,
    CollectionRef,
    EmbeddingModelMismatchError,
)
from prism.config import Settings
from prism.core.ids import uuid7
from prism.retrieval import search_chunks
from seeding import seed
from stub_provider import DIM, MODEL, QueryProvider, graded

pytestmark = pytest.mark.integration


def settings(top_k: int = 10) -> Settings:
    return Settings(retrieval_top_k=top_k, embedding_model=MODEL, embedding_dim=DIM)


async def test_the_nearest_chunks_come_back_first(
    collection_id: UUID, collection: CollectionRef
) -> None:
    await seed(
        collection_id,
        [("far", graded(0.1)), ("near", graded(0.9)), ("middling", graded(0.5))],
    )

    hits = await search_chunks(
        "anything",
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        provider=QueryProvider(),
        settings=settings(),
    )

    assert [hit.content for hit in hits] == ["near", "middling", "far"]
    assert [hit.score for hit in hits] == pytest.approx([0.9, 0.5, 0.1], abs=1e-5)


async def test_the_score_is_similarity_not_distance(
    collection_id: UUID, collection: CollectionRef
) -> None:
    """0.9 similar is 0.1 distant. Reported the way the abstention threshold and
    the rerank floor are expressed, or every later comparison is inverted."""
    await seed(collection_id, [("near", graded(0.9))])

    ((hit,)) = await search_chunks(
        "anything",
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        provider=QueryProvider(),
        settings=settings(),
    )
    assert hit.score == pytest.approx(0.9, abs=1e-5)


async def test_k_bounds_the_number_of_hits(collection_id: UUID, collection: CollectionRef) -> None:
    await seed(collection_id, [(f"chunk {n}", graded(0.9 - n / 100)) for n in range(20)])

    hits = await search_chunks(
        "anything",
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        k=3,
        provider=QueryProvider(),
        settings=settings(),
    )
    assert len(hits) == 3
    assert [hit.content for hit in hits] == ["chunk 0", "chunk 1", "chunk 2"]


async def test_k_defaults_to_the_configured_top_k(
    collection_id: UUID, collection: CollectionRef
) -> None:
    await seed(collection_id, [(f"chunk {n}", graded(0.9)) for n in range(20)])

    hits = await search_chunks(
        "anything",
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        provider=QueryProvider(),
        settings=settings(top_k=4),
    )
    assert len(hits) == 4


async def test_a_better_match_in_another_collection_is_never_returned(
    collection_id: UUID, other_collection_id: UUID, collection: CollectionRef
) -> None:
    """The scoping test. The other collection holds an exact match; this one
    holds only poor ones. A post-filter would have surfaced it and then dropped
    it, costing a slot. The predicate means it is never a candidate."""
    await seed(other_collection_id, [("exact match, wrong tenant", graded(1.0))])
    await seed(collection_id, [("weak but mine", graded(0.2))])

    hits = await search_chunks(
        "anything",
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        provider=QueryProvider(),
        settings=settings(),
    )

    assert [hit.content for hit in hits] == ["weak but mine"]


async def test_k_is_still_filled_when_another_collection_dominates_the_table(
    collection_id: UUID, other_collection_id: UUID, collection: CollectionRef
) -> None:
    """Every nearer neighbour belongs to someone else, and k must still be filled
    from this collection. Which plan delivers that is the planner's business —
    at this size it is the btree and an exact sort, not the ANN index."""
    await seed(other_collection_id, [(f"theirs {n}", graded(0.99)) for n in range(500)])
    await seed(collection_id, [(f"mine {n}", graded(0.3)) for n in range(15)])

    hits = await search_chunks(
        "anything",
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        k=10,
        provider=QueryProvider(),
        settings=settings(),
    )

    assert len(hits) == 10
    assert all(hit.content.startswith("mine") for hit in hits)


async def test_an_empty_collection_returns_nothing_rather_than_failing(
    collection_id: UUID, collection: CollectionRef
) -> None:
    hits = await search_chunks(
        "anything",
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        provider=QueryProvider(),
        settings=settings(),
    )
    assert hits == []


async def test_an_unknown_collection_is_refused() -> None:
    missing = uuid7()
    with pytest.raises(CollectionNotFoundError, match=f"collection {missing} does not exist"):
        await search_chunks(
            "anything",
            collection_id=missing,
            tenant_id=uuid7(),
            provider=QueryProvider(),
            settings=settings(),
        )


async def test_a_mismatched_provider_is_refused_before_the_query_is_embedded(
    collection_id: UUID, collection: CollectionRef
) -> None:
    """Embedding costs a model call; a query that cannot be answered should not
    pay for one."""

    class WrongProvider(QueryProvider):
        model = "bge-m3"
        dim = 1024

    provider = WrongProvider()
    with pytest.raises(EmbeddingModelMismatchError):
        await search_chunks(
            "anything",
            collection_id=collection_id,
            tenant_id=collection.tenant_id,
            provider=provider,
            settings=settings(),
        )
    assert provider.queries == []


async def test_a_hit_carries_what_a_citation_needs(
    collection_id: UUID, collection: CollectionRef
) -> None:
    document_id = await seed(collection_id, [("cited text", graded(0.9))], filename="source.pdf")

    ((hit,)) = await search_chunks(
        "anything",
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        provider=QueryProvider(),
        settings=settings(),
    )
    assert hit.document_id == document_id
    assert hit.filename == "source.pdf"
    assert hit.page_number == 1
    assert hit.chunk_index == 0
    assert UUID(str(hit.chunk_id)).version == 7


async def test_an_empty_query_is_refused_without_a_database_round_trip() -> None:
    with pytest.raises(ValueError, match="query is empty"):
        await search_chunks(
            "   ",
            tenant_id=uuid7(),
            collection_id=uuid7(),
            provider=QueryProvider(),
            settings=settings(),
        )


@pytest.mark.integration
async def test_another_tenants_chunks_are_unreachable_through_their_collection_id(
    collection: CollectionRef, other_collection: CollectionRef
) -> None:
    """The isolation bug: holding a collection UUID used to be enough to read it.

    Addressed exactly as an attacker would — the victim's real collection id,
    with the attacker's tenant.
    """
    await seed(other_collection.collection_id, [("their secret", graded(1.0))])

    with pytest.raises(CollectionNotFoundError):
        await search_chunks(
            "anything",
            tenant_id=collection.tenant_id,
            collection_id=other_collection.collection_id,
            provider=QueryProvider(),
            settings=settings(),
        )


@pytest.mark.integration
async def test_a_tenant_reads_its_own_collection_of_the_same_name(
    collection: CollectionRef, other_collection: CollectionRef
) -> None:
    """The predicate scopes, it does not simply refuse everything."""
    await seed(other_collection.collection_id, [("theirs", graded(1.0))])
    await seed(collection.collection_id, [("mine", graded(0.2))])

    hits = await search_chunks(
        "anything",
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        provider=QueryProvider(),
        settings=settings(),
    )
    assert [hit.content for hit in hits] == ["mine"]
