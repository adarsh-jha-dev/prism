"""Naive retrieval over one collection.

No grading and no generation: this is the retrieval baseline the benchmark
measures the correction loop against, and the scores are raw cosine similarity.

Scope is the caller's tenant and the collection in the path, both enforced as
predicates inside the ANN query rather than over its results. The tenant comes
from the bearer key; a collection the caller does not own is a 404.
"""

from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError

from prism.api.deps import ProviderDep, ReadDep, SettingsDep
from prism.collections import CollectionNotFoundError, EmbeddingModelMismatchError
from prism.embeddings import EmbeddingError
from prism.retrieval import search_chunks

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/collections/{collection_id}/search", tags=["retrieval"])

# Request bounds, not retrieval policy — the default k is a Settings value.
_MAX_TOP_K = 50
_MAX_QUERY_CHARS = 2000


class SearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=_MAX_QUERY_CHARS)]
    k: Annotated[int | None, Field(default=None, ge=1, le=_MAX_TOP_K)] = None

    @field_validator("query")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query is empty")
        return value


class Hit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    page_number: int | None
    chunk_index: int | None
    score: float


class SearchResults(BaseModel):
    collection_id: UUID
    k: int
    hits: list[Hit]


@router.post("")
async def search(
    collection_id: UUID,
    request: SearchRequest,
    key: ReadDep,
    provider: ProviderDep,
    settings: SettingsDep,
) -> SearchResults:
    """Embed the query and return the k nearest chunks in this collection."""
    k = request.k if request.k is not None else settings.retrieval_top_k

    try:
        hits = await search_chunks(
            request.query,
            tenant_id=key.tenant_id,
            collection_id=collection_id,
            k=k,
            provider=provider,
            settings=settings,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EmbeddingModelMismatchError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (EmbeddingError, SQLAlchemyError) as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"{type(exc).__name__}: {exc}"
        ) from exc

    log.info(
        "collection_searched",
        tenant_id=str(key.tenant_id),
        collection_id=str(collection_id),
        k=k,
        hits=len(hits),
        top_score=hits[0].score if hits else None,
    )
    return SearchResults(
        collection_id=collection_id,
        k=k,
        hits=[Hit(**vars(hit)) for hit in hits],
    )
