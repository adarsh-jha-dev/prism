"""Naive retrieval: top-k cosine over one collection's chunks.

Vector only, and it grades nothing. Hybrid BM25 belongs to the graph's
`retrieve` node; this is the baseline the benchmark measures against.

The collection scope is a predicate inside the ANN query, never a filter over
its results — a neighbour from another collection must not occupy a top-k slot,
and at this layer that predicate is the whole of tenant isolation.

The HNSW knobs are insurance, not something currently doing work: at the sizes
measured so far the planner prefers the `collection_id` btree and an exact
top-k. If it ever picks the ANN index, that index visits `ef_search` candidates
and only then discards what the predicate excludes, so without iterative scan a
small collection can come back short — looking exactly like one holding little.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.collections import assert_compatible
from prism.config import Settings, get_settings
from prism.db import get_engine
from prism.embeddings import EmbeddingProvider, get_embedding_provider

__all__ = ["SearchHit", "search_chunks"]

# Reported as similarity, the unit tau and the rerank floor are expressed in.
_SEARCH = text(
    """
    SELECT c.id AS chunk_id,
           c.document_id,
           d.filename,
           c.content,
           c.page_number,
           (c.metadata->>'chunk_index')::int AS chunk_index,
           1 - (c.embedding <=> CAST(:query_vector AS vector)) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.collection_id = :collection_id
      AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(:query_vector AS vector)
    LIMIT :k
    """
)


# set_config takes bind parameters; SET LOCAL would interpolate them.
_APPLY_SCAN_SETTINGS = text(
    "SELECT set_config('hnsw.ef_search', :ef_search, true), "
    "set_config('hnsw.iterative_scan', :scan, true)"
)


@dataclass(frozen=True)
class SearchHit:
    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    page_number: int | None
    chunk_index: int | None
    score: float


async def search_chunks(
    query: str,
    *,
    collection_id: UUID,
    k: int | None = None,
    provider: EmbeddingProvider | None = None,
    settings: Settings | None = None,
    engine: AsyncEngine | None = None,
) -> list[SearchHit]:
    """The k nearest chunks in `collection_id`, most similar first.

    Raises CollectionNotFoundError or EmbeddingModelMismatchError before
    embedding anything, and EmbeddingError if the query cannot be embedded. An
    empty list means the collection holds nothing near the query — never that
    the scope was applied too late.
    """
    settings = settings or get_settings()
    provider = provider or get_embedding_provider()
    engine = engine or get_engine()
    k = settings.retrieval_top_k if k is None else k

    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if not query.strip():
        raise ValueError("query is empty")

    async with engine.connect() as conn:
        await assert_compatible(conn, collection_id, provider)

    query_vector = await provider.embed_one(query)

    # HNSW never yields more rows than the candidates it visits.
    ef_search = max(settings.hnsw_ef_search, k)

    async with engine.begin() as conn:
        await conn.execute(
            _APPLY_SCAN_SETTINGS,
            {"ef_search": str(ef_search), "scan": settings.hnsw_iterative_scan},
        )
        rows = (
            await conn.execute(
                _SEARCH,
                {
                    "query_vector": str(query_vector),
                    "collection_id": collection_id,
                    "k": k,
                },
            )
        ).all()

    return [
        SearchHit(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            filename=row.filename,
            content=row.content,
            page_number=row.page_number,
            chunk_index=row.chunk_index,
            score=float(row.score),
        )
        for row in rows
    ]
