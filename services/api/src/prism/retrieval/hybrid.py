"""Hybrid retrieval: a lexical half and a vector half, fused by reciprocal rank.

This is the graph's `retrieve` node. `search.py` stays vector-only — it is the
baseline the benchmark measures against, not a component of this.

Fusion is RRF (ADR 0010) and it consumes ranks only. `ts_rank_cd` is unbounded
and shares no unit with cosine similarity, so adding or weighting the two
together would be arithmetic on incommensurable quantities. `HybridHit` therefore
carries positions and no fused magnitude: there is no number on it that could be
compared against `abstention_threshold` or `rerank_score_floor`, which is the
rule ADR 0010 states and this shape enforces.

Both halves scope by tenant and collection inside their own scan — the ANN query
through the `collection_id` btree or HNSW, the lexical one through the
multicolumn GIN index from migration 0007. A lexical hit from another tenant must
not occupy a top-k slot any more than a vector one may.

Zero lexical hits is a normal outcome, not an error. `websearch_to_tsquery` ANDs
unquoted terms, so a long prose question often matches nothing; the fix is fewer
and better terms from `plan_query`, never a looser parser. Until that node
exists, `terms` is unset and the raw question is parsed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.collections import assert_compatible
from prism.config import Settings, get_settings
from prism.db import get_engine
from prism.embeddings import EmbeddingProvider, get_embedding_provider

__all__ = ["HybridHit", "hybrid_search"]

_VECTOR_HALF = text(
    """
    SELECT c.id AS chunk_id,
           c.document_id,
           d.filename,
           c.content,
           c.page_number,
           (c.metadata->>'chunk_index')::int AS chunk_index
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.collection_id = :collection_id
      AND c.tenant_id = :tenant_id
      AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(:query_vector AS vector)
    LIMIT :candidates
    """
)

# ts_rank_cd's magnitude never leaves this query: it orders, nothing more.
# id breaks ties so the ordering is reproducible across eval runs.
_LEXICAL_HALF = text(
    """
    SELECT c.id AS chunk_id,
           c.document_id,
           d.filename,
           c.content,
           c.page_number,
           (c.metadata->>'chunk_index')::int AS chunk_index
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN websearch_to_tsquery('english', :query_text) AS q
    WHERE c.collection_id = :collection_id
      AND c.tenant_id = :tenant_id
      AND c.content_tsv @@ q
    ORDER BY ts_rank_cd(c.content_tsv, q) DESC, c.id
    LIMIT :candidates
    """
)

_APPLY_SCAN_SETTINGS = text(
    "SELECT set_config('hnsw.ef_search', :ef_search, true), "
    "set_config('hnsw.iterative_scan', :scan, true)"
)


@dataclass(frozen=True)
class HybridHit:
    """A fused candidate. Positions only — deliberately no score to threshold."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    page_number: int | None
    chunk_index: int | None
    rank: int
    vector_rank: int | None
    lexical_rank: int | None


def _fuse(ranked: Sequence[Sequence[UUID]], *, k: int) -> list[UUID]:
    """RRF: each list contributes 1/(k + position), positions 1-based."""
    scores: dict[UUID, float] = {}
    best: dict[UUID, int] = {}
    for half in ranked:
        for position, chunk_id in enumerate(half, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            best[chunk_id] = min(best.get(chunk_id, position), position)
    # Ties break on best position then id: eval reruns must not reshuffle.
    return sorted(scores, key=lambda c: (-scores[c], best[c], str(c)))


async def hybrid_search(
    query: str,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    terms: Sequence[str] | None = None,
    k: int | None = None,
    provider: EmbeddingProvider | None = None,
    settings: Settings | None = None,
    engine: AsyncEngine | None = None,
) -> list[HybridHit]:
    """The k best chunks in `collection_id` by fused lexical and vector rank.

    `terms` is what `plan_query` extracted, if it has run; unset means the raw
    question is parsed instead. Either way the text goes through
    `websearch_to_tsquery`, never `to_tsquery`, which raises on user punctuation.

    `tenant_id` is the caller's, resolved from their API key. An empty list means
    neither half found anything in scope — never that the scope was applied late.
    """
    settings = settings or get_settings()
    provider = provider or get_embedding_provider()
    engine = engine or get_engine()
    k = settings.retrieval_top_k if k is None else k

    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if not query.strip():
        raise ValueError("query is empty")

    query_text = " ".join(terms) if terms else query

    async with engine.connect() as conn:
        await assert_compatible(conn, collection_id, provider, tenant_id=tenant_id)

    query_vector = await provider.embed_one(query)

    candidates = max(settings.retrieval_candidate_k, k)
    ef_search = max(settings.hnsw_ef_search, candidates)
    scope = {"collection_id": collection_id, "tenant_id": tenant_id, "candidates": candidates}

    # One transaction, two statements: fusion needs both lists before it can
    # order anything, so there is nothing to overlap them with.
    async with engine.begin() as conn:
        await conn.execute(
            _APPLY_SCAN_SETTINGS,
            {"ef_search": str(ef_search), "scan": settings.hnsw_iterative_scan},
        )
        vector_rows = (
            await conn.execute(_VECTOR_HALF, {**scope, "query_vector": str(query_vector)})
        ).all()
        lexical_rows = (
            await conn.execute(_LEXICAL_HALF, {**scope, "query_text": query_text})
        ).all()

    vector_ids = [row.chunk_id for row in vector_rows]
    lexical_ids = [row.chunk_id for row in lexical_rows]
    by_id = {row.chunk_id: row for row in [*lexical_rows, *vector_rows]}

    vector_at = {chunk_id: i for i, chunk_id in enumerate(vector_ids, start=1)}
    lexical_at = {chunk_id: i for i, chunk_id in enumerate(lexical_ids, start=1)}

    fused = _fuse([vector_ids, lexical_ids], k=settings.rrf_k)[:k]
    return [
        HybridHit(
            chunk_id=chunk_id,
            document_id=by_id[chunk_id].document_id,
            filename=by_id[chunk_id].filename,
            content=by_id[chunk_id].content,
            page_number=by_id[chunk_id].page_number,
            chunk_index=by_id[chunk_id].chunk_index,
            rank=position,
            vector_rank=vector_at.get(chunk_id),
            lexical_rank=lexical_at.get(chunk_id),
        )
        for position, chunk_id in enumerate(fused, start=1)
    ]
