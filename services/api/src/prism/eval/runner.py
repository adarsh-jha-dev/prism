"""Run the golden set against retrieval.

Either retriever, chosen per run: `vector` is the naive baseline the benchmark
measures against, `hybrid` is the graph's `retrieve` node. Which one ran is
recorded in the report, because a recall number compared across the two is not a
comparison of anything.

The hybrid retriever reports no scores. Fusion yields an ordering and no
magnitude (ADR 0010), so a hybrid run has no top-1 similarity and its refusal
calibration is empty rather than zero — see `eval/README.md`.

Calls `prism.retrieval.search_chunks` directly rather than the HTTP endpoint:
the number being measured is the retriever's, and a local ASGI hop would add
latency that belongs to neither the baseline nor the optimized path.

Questions are independent, so they run concurrently under the `ollama` lane's
own cap. Results are returned in golden-set order regardless, so two runs over
one index produce byte-identical reports.
"""

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.collections import CollectionRef
from prism.config import Settings, get_settings
from prism.db import get_engine
from prism.embeddings import EmbeddingProvider, get_embedding_provider
from prism.eval.golden import GoldenQuestion, GoldenSet, PageRef
from prism.eval.metrics import QuestionResult
from prism.retrieval import search_chunks
from prism.retrieval.hybrid import hybrid_search

__all__ = [
    "RETRIEVERS",
    "CollectionNotResolvedError",
    "CorpusStats",
    "Retriever",
    "collection_stats",
    "resolve_collection",
    "run_golden_set",
]

Retriever = Literal["vector", "hybrid"]
RETRIEVERS: tuple[Retriever, ...] = ("vector", "hybrid")


class _Hit(Protocol):
    """What both retrievers agree on. Scores are not in it, because one has none.

    Read-only properties, so the frozen result dataclasses satisfy it.
    """

    @property
    def filename(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def page_number(self) -> int | None: ...


_WHITESPACE = re.compile(r"\s+")

_SELECT_COLLECTION_BY_NAME = text(
    """
    SELECT c.id, c.tenant_id
    FROM collections c
    JOIN tenants t ON t.id = c.tenant_id
    WHERE c.name = :name AND t.name = :tenant
    """
)

_COLLECTION_STATS = text(
    """
    SELECT count(DISTINCT c.document_id) AS documents,
           count(*)                      AS chunks,
           count(*) FILTER (WHERE c.embedding IS NULL) AS unembedded
    FROM chunks c
    WHERE c.collection_id = :collection_id
    """
)


class CollectionNotResolvedError(RuntimeError):
    """The collection the golden set names does not exist yet."""


@dataclass(frozen=True)
class CorpusStats:
    documents: int
    chunks: int
    unembedded: int


async def resolve_collection(
    name: str, *, tenant: str, engine: AsyncEngine | None = None
) -> CollectionRef:
    """The collection the golden set is written against, with its owning tenant."""
    async with (engine or get_engine()).connect() as conn:
        row = (
            await conn.execute(_SELECT_COLLECTION_BY_NAME, {"name": name, "tenant": tenant})
        ).first()
    if row is None:
        raise CollectionNotResolvedError(
            f"no collection {name!r} under tenant {tenant!r} — run `make eval-ingest` first"
        )
    return CollectionRef(tenant_id=row.tenant_id, collection_id=row.id)


async def collection_stats(
    collection_id: UUID, *, engine: AsyncEngine | None = None
) -> CorpusStats:
    async with (engine or get_engine()).connect() as conn:
        row = (await conn.execute(_COLLECTION_STATS, {"collection_id": collection_id})).one()
    return CorpusStats(documents=row.documents, chunks=row.chunks, unembedded=row.unembedded)


def _normalize(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip().casefold()


def _quote_found(quote: str | None, hits: Sequence[_Hit]) -> bool | None:
    """Whether the supporting quote survives into a retrieved chunk.

    Reported as a diagnostic, never folded into recall: a quote can straddle a
    chunk boundary and be absent from every chunk while the page is retrieved
    perfectly well.
    """
    if quote is None:
        return None
    needle = _normalize(quote)
    return any(needle in _normalize(hit.content) for hit in hits)


def _to_result(
    question: GoldenQuestion, hits: Sequence[_Hit], *, scores: tuple[float, ...]
) -> QuestionResult:
    pages = tuple(
        PageRef(doc=hit.filename, page=hit.page_number)
        for hit in hits
        if hit.page_number is not None
    )
    return QuestionResult(
        question_id=question.id,
        question=question.question,
        unanswerable=question.unanswerable,
        relevant=question.relevant,
        retrieved=pages,
        scores=scores,
        quote_found=_quote_found(question.supporting_quote, hits),
        unmappable_chunks=sum(1 for hit in hits if hit.page_number is None),
    )


async def run_golden_set(
    golden: GoldenSet,
    *,
    collection: CollectionRef,
    k: int,
    retriever: Retriever = "vector",
    provider: EmbeddingProvider | None = None,
    settings: Settings | None = None,
    engine: AsyncEngine | None = None,
) -> tuple[QuestionResult, ...]:
    """Retrieve `k` chunks for every question, in golden-set order."""
    settings = settings or get_settings()
    provider = provider or get_embedding_provider()
    limit = asyncio.Semaphore(settings.concurrency_ollama_local)

    async def one(question: GoldenQuestion) -> QuestionResult:
        async with limit:
            if retriever == "hybrid":
                fused = await hybrid_search(
                    question.question,
                    tenant_id=collection.tenant_id,
                    collection_id=collection.collection_id,
                    k=k,
                    provider=provider,
                    settings=settings,
                    engine=engine,
                )
                # No scores: a fused rank is an ordering, never a magnitude.
                return _to_result(question, fused, scores=())

            hits = await search_chunks(
                question.question,
                tenant_id=collection.tenant_id,
                collection_id=collection.collection_id,
                k=k,
                provider=provider,
                settings=settings,
                engine=engine,
            )
        return _to_result(question, hits, scores=tuple(hit.score for hit in hits))

    return tuple(await asyncio.gather(*(one(q) for q in golden.questions)))
