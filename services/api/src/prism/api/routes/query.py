"""The graph over one collection, as SSE or as one JSON body.

Scope and error mapping follow routes/search.py: the tenant comes from the bearer
key, the collection from the path, and a collection this tenant does not own is a
404 — a 403 confirms the id exists.

`Accept: text/event-stream` streams a `node` event per executed node and closes
with a `result` event holding the JSON variant's body. Anything else returns that
body alone.

Before the first byte, on both variants: 422 for a malformed, empty or oversized
question, 404 for an unknown or unowned collection, 409 for an embedding model
the collection was not built for, 503 for a database that cannot be read. All are
decided before the graph starts, so all still set a status code.

After it: every provider, lane, reranker and database failure inside the graph.
The status is already 200 and cannot be revised, so the stream emits an `error`
event and closes, leaving the query row `refused` with partial traces. The JSON
variant has sent nothing, so the same failures map to 503 there.

A refusal is a 200. A disconnect stops the stream, not the run (ADR 0023).
`thread_id` is in no response.

`rate_limit_rpm` is uniform per key, so a query spends the same allowance as a
search at orders of magnitude more cost. Weighting limits by cost is Phase 3's.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Annotated, Any
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError

from prism.api.deps import ProviderDep, ReadDep
from prism.chat import ChatError
from prism.collections import (
    CollectionNotFoundError,
    EmbeddingModelMismatchError,
    assert_collection_compatible,
)
from prism.embeddings import EmbeddingError
from prism.graph.events import EventChannel, NodeEvent
from prism.graph.run import QueryRun, run_query
from prism.graph.state import RefusalReason, TerminalStatus
from prism.providers import LaneError
from prism.rerank import RerankError

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/collections/{collection_id}/query", tags=["query"])

SSE_MEDIA_TYPE = "text/event-stream"

# A request bound, not retrieval policy.
_MAX_QUESTION_CHARS = 2000

# A failed run, as opposed to a refusal.
_GRAPH_FAILURES = (ChatError, EmbeddingError, LaneError, RerankError, SQLAlchemyError)

# Strong references, so a detached run is not collected mid-flight.
_DETACHED: set[asyncio.Task[QueryRun]] = set()


class QueryRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=_MAX_QUESTION_CHARS)]

    @field_validator("question")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question is empty")
        return value


class Citation(BaseModel):
    rank: int
    document_id: UUID
    filename: str | None
    page_number: int | None
    content: str


class QueryResult(BaseModel):
    query_id: UUID
    collection_id: UUID
    status: TerminalStatus
    refusal_reason: RefusalReason | None
    answer: str | None
    citations: list[Citation]


@router.post(
    "",
    response_model=QueryResult,
    responses={200: {"content": {"application/json": {}, SSE_MEDIA_TYPE: {}}}},
)
async def query(
    collection_id: UUID,
    request: QueryRequest,
    http_request: Request,
    key: ReadDep,
    provider: ProviderDep,
) -> Response:
    """Answer the question from this collection, or refuse with a reason."""
    try:
        await assert_collection_compatible(collection_id, provider, tenant_id=key.tenant_id)
    except CollectionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EmbeddingModelMismatchError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"{type(exc).__name__}: {exc}"
        ) from exc

    if SSE_MEDIA_TYPE in http_request.headers.get("accept", ""):
        return StreamingResponse(
            _stream(
                tenant_id=key.tenant_id,
                collection_id=collection_id,
                question=request.question,
            ),
            media_type=SSE_MEDIA_TYPE,
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    try:
        run = await run_query(
            tenant_id=key.tenant_id,
            collection_id=collection_id,
            question=request.question,
        )
    except _GRAPH_FAILURES as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"{type(exc).__name__}: {exc}"
        ) from exc
    return JSONResponse(_result(collection_id, run).model_dump(mode="json"))


async def _stream(*, tenant_id: UUID, collection_id: UUID, question: str) -> AsyncIterator[bytes]:
    channel = EventChannel()
    task = asyncio.create_task(
        run_query(
            tenant_id=tenant_id,
            collection_id=collection_id,
            question=question,
            events=channel,
        )
    )
    _detach(task, channel)

    query_id: UUID | None = None
    try:
        async for event in channel.events():
            query_id = event.query_id
            yield _frame("node", _node_payload(event))
        try:
            run = await task
        except Exception as exc:
            yield _frame(
                "error",
                {
                    "query_id": None if query_id is None else str(query_id),
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
            return
        yield _frame("result", _result(collection_id, run).model_dump(mode="json"))
    except (asyncio.CancelledError, GeneratorExit):
        log.info("query.abandoned", query_id=None if query_id is None else str(query_id))
        raise


def _detach(task: asyncio.Task[QueryRun], channel: EventChannel) -> None:
    """Keep the run alive independently of the response that started it."""
    _DETACHED.add(task)

    def finished(done: asyncio.Task[QueryRun]) -> None:
        _DETACHED.discard(done)
        channel.close()
        if not done.cancelled() and done.exception() is not None:
            log.warning("query.run_failed", error=str(done.exception()))

    task.add_done_callback(finished)


def _result(collection_id: UUID, run: QueryRun) -> QueryResult:
    return QueryResult(
        query_id=run.query_id,
        collection_id=collection_id,
        status=run.status,
        refusal_reason=run.refusal_reason,
        answer=run.answer,
        citations=[Citation(**asdict(citation)) for citation in run.citations],
    )


def _node_payload(event: NodeEvent) -> dict[str, Any]:
    return {
        "query_id": str(event.query_id),
        "node": event.node,
        "sequence": event.sequence,
        "attempt": event.attempt,
        "status": event.status,
        "verdict": event.verdict,
        "started_at": event.started_at.isoformat(),
        "duration_ms": event.duration_ms,
        "error": event.error,
        "provider": event.provider,
        "model": event.model,
        "cost_usd": None if event.cost_usd is None else float(event.cost_usd),
    }


def _frame(name: str, data: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()
