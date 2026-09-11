"""Run the golden set against retrieval.

Calls `prism.retrieval.search_chunks` directly rather than the HTTP endpoint:
the number being measured is the retriever's, and a local ASGI hop would add
latency that belongs to neither the baseline nor the optimized path.

Questions are independent, so they run concurrently under the `ollama` lane's
own cap. Results are returned in golden-set order regardless, so two runs over
one index produce byte-identical reports.
"""

import asyncio
import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.config import Settings, get_settings
from prism.db import get_engine
from prism.embeddings import EmbeddingProvider, get_embedding_provider
from prism.eval.golden import GoldenQuestion, GoldenSet, PageRef
from prism.eval.metrics import QuestionResult
from prism.retrieval import SearchHit, search_chunks

__all__ = [
    "CollectionNotResolvedError",
    "CorpusStats",
    "collection_stats",
    "resolve_collection",
    "run_golden_set",
]

_WHITESPACE = re.compile(r"\s+")

_SELECT_COLLECTION_BY_NAME = text(
    """
    SELECT c.id
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


async def resolve_collection(name: str, *, tenant: str, engine: AsyncEngine | None = None) -> UUID:
    """The id of the collection the golden set is written against."""
    async with (engine or get_engine()).connect() as conn:
        row = (
            await conn.execute(_SELECT_COLLECTION_BY_NAME, {"name": name, "tenant": tenant})
        ).first()
    if row is None:
        raise CollectionNotResolvedError(
            f"no collection {name!r} under tenant {tenant!r} — run `make eval-ingest` first"
        )
    collection_id: UUID = row.id
    return collection_id


async def collection_stats(
    collection_id: UUID, *, engine: AsyncEngine | None = None
) -> CorpusStats:
    async with (engine or get_engine()).connect() as conn:
        row = (await conn.execute(_COLLECTION_STATS, {"collection_id": collection_id})).one()
    return CorpusStats(documents=row.documents, chunks=row.chunks, unembedded=row.unembedded)


def _normalize(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip().casefold()


def _quote_found(quote: str | None, hits: list[SearchHit]) -> bool | None:
    """Whether the supporting quote survives into a retrieved chunk.

    Reported as a diagnostic, never folded into recall: a quote can straddle a
    chunk boundary and be absent from every chunk while the page is retrieved
    perfectly well.
    """
    if quote is None:
        return None
    needle = _normalize(quote)
    return any(needle in _normalize(hit.content) for hit in hits)


def _to_result(question: GoldenQuestion, hits: list[SearchHit]) -> QuestionResult:
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
        scores=tuple(hit.score for hit in hits),
        quote_found=_quote_found(question.supporting_quote, hits),
        unmappable_chunks=sum(1 for hit in hits if hit.page_number is None),
    )


async def run_golden_set(
    golden: GoldenSet,
    *,
    collection_id: UUID,
    k: int,
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
            hits = await search_chunks(
                question.question,
                collection_id=collection_id,
                k=k,
                provider=provider,
                settings=settings,
                engine=engine,
            )
        return _to_result(question, hits)

    return tuple(await asyncio.gather(*(one(q) for q in golden.questions)))
