"""The graph's `rerank` node: a cross-encoder over the fused candidates, then the floor.

Fusion (ADR 0010) yields an ordering; this is the first place retrieval produces
a magnitude, and `rerank_score_floor` applies to it and to nothing upstream. An
empty `hits` means every candidate fell below the floor, which re-enters the
retrieval loop exactly as a `grade_docs` failure does (ADR 0011).

A RerankError propagates. Falling back to fusion order is the node's decision,
recorded on its trace row, and a query that took it is excluded from eval.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from prism.config import Settings, get_settings
from prism.rerank import Reranker, RerankError, get_reranker
from prism.retrieval.hybrid import HybridHit

__all__ = ["Reranked", "RerankedHit", "rerank"]


@dataclass(frozen=True)
class RerankedHit:
    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    page_number: int | None
    chunk_index: int | None
    rank: int
    fused_rank: int
    score: float


@dataclass(frozen=True)
class Reranked:
    ranked: tuple[RerankedHit, ...]
    """The top k by score, before the floor. Diagnostic: generation reads `hits`."""
    floor: float

    @property
    def hits(self) -> tuple[RerankedHit, ...]:
        return tuple(hit for hit in self.ranked if hit.score >= self.floor)


async def rerank(
    query: str,
    candidates: Sequence[HybridHit],
    *,
    k: int | None = None,
    reranker: Reranker | None = None,
    settings: Settings | None = None,
) -> Reranked:
    """Score the first `rerank_candidate_k` fused candidates and keep the best `k`."""
    settings = settings or get_settings()
    reranker = reranker or get_reranker()
    k = settings.retrieval_top_k if k is None else k
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")

    pool = list(candidates[: settings.rerank_candidate_k])
    scores = await reranker.score(query, [hit.content for hit in pool])
    if len(scores) != len(pool):
        raise RerankError(f"scored {len(scores)} of {len(pool)} candidates")

    # Fused rank breaks ties, so equal scores keep the order retrieval gave them.
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i].rank))[:k]
    ranked = tuple(
        RerankedHit(
            chunk_id=pool[i].chunk_id,
            document_id=pool[i].document_id,
            filename=pool[i].filename,
            content=pool[i].content,
            page_number=pool[i].page_number,
            chunk_index=pool[i].chunk_index,
            rank=position,
            fused_rank=pool[i].rank,
            score=scores[i],
        )
        for position, i in enumerate(order, start=1)
    )
    return Reranked(ranked=ranked, floor=settings.rerank_score_floor)
