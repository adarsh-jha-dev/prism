"""POST /collections/{id}/search.

Request validation is a unit test — it is decided before any collection is
looked up. Everything that reaches the index is marked `integration`.
"""

import math
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from conftest import Authenticate
from prism.api.deps import embedding_provider
from prism.auth import Scope
from prism.collections import CollectionRef
from prism.config import Settings, get_settings
from prism.core.ids import uuid7
from prism.embeddings import EmbeddingError
from seeding import seed
from stub_provider import DIM, MODEL, QueryProvider, StubProvider, graded

URL = "/collections/{}/search"
UNKNOWN = "01890000-0000-7000-8000-00000000dead"


@pytest.fixture
def provider() -> QueryProvider:
    return QueryProvider()


@pytest.fixture
def configured(app: FastAPI, provider: QueryProvider) -> Iterator[dict[str, Any]]:
    overrides: dict[str, Any] = {"retrieval_top_k": 10}

    def settings() -> Settings:
        return Settings(embedding_model=MODEL, embedding_dim=DIM, **overrides)

    app.dependency_overrides[get_settings] = settings
    app.dependency_overrides[embedding_provider] = lambda: provider
    yield overrides
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({}, "no query at all"),
        ({"query": ""}, "empty query"),
        ({"query": "   "}, "whitespace-only query"),
        ({"query": "x" * 2001}, "query past the length bound"),
        ({"query": "ok", "k": 0}, "k below one"),
        ({"query": "ok", "k": 51}, "k past the bound"),
        ({"query": "ok", "k": -1}, "negative k"),
    ],
)
async def test_a_malformed_search_is_refused_before_anything_is_looked_up(
    client: AsyncClient,
    configured: dict[str, Any],
    provider: QueryProvider,
    body: dict[str, Any],
    why: str,
) -> None:
    response = await client.post(URL.format(UNKNOWN), json=body)
    assert response.status_code == 422, why
    assert provider.queries == []


@pytest.mark.integration
async def test_a_search_returns_scored_chunks_nearest_first(
    client: AsyncClient, configured: dict[str, Any], collection_id: UUID
) -> None:
    await seed(
        collection_id,
        [("far", graded(0.1)), ("near", graded(0.9)), ("middling", graded(0.5))],
        filename="source.pdf",
    )

    response = await client.post(URL.format(collection_id), json={"query": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert UUID(body["collection_id"]) == collection_id
    assert body["k"] == 10
    assert [hit["content"] for hit in body["hits"]] == ["near", "middling", "far"]
    assert [hit["score"] for hit in body["hits"]] == pytest.approx([0.9, 0.5, 0.1], abs=1e-5)

    top = body["hits"][0]
    assert set(top) == {
        "chunk_id",
        "document_id",
        "filename",
        "content",
        "page_number",
        "chunk_index",
        "score",
    }
    assert top["filename"] == "source.pdf"


@pytest.mark.integration
async def test_k_in_the_request_overrides_the_configured_default(
    client: AsyncClient, configured: dict[str, Any], collection_id: UUID
) -> None:
    await seed(collection_id, [(f"chunk {n}", graded(0.9 - n / 100)) for n in range(20)])

    response = await client.post(URL.format(collection_id), json={"query": "anything", "k": 2})

    assert response.status_code == 200
    assert response.json()["k"] == 2
    assert len(response.json()["hits"]) == 2


@pytest.mark.integration
async def test_another_collections_chunks_are_never_in_the_results(
    client: AsyncClient,
    configured: dict[str, Any],
    collection_id: UUID,
    other_collection_id: UUID,
) -> None:
    """Scoping is enforced in the query, so the exact match next door is not a
    candidate — not a candidate that gets filtered out afterwards."""
    await seed(other_collection_id, [("exact match, wrong tenant", graded(1.0))])
    await seed(collection_id, [("weak but mine", graded(0.2))])

    response = await client.post(URL.format(collection_id), json={"query": "anything"})

    assert response.status_code == 200
    assert [hit["content"] for hit in response.json()["hits"]] == ["weak but mine"]


@pytest.mark.integration
async def test_an_empty_collection_is_an_empty_result_not_an_error(
    client: AsyncClient, configured: dict[str, Any], collection_id: UUID
) -> None:
    """Nothing to retrieve is a fact about the corpus, not a failure."""
    response = await client.post(URL.format(collection_id), json={"query": "anything"})

    assert response.status_code == 200
    assert response.json()["hits"] == []


@pytest.mark.integration
async def test_an_unknown_collection_is_404(
    client: AsyncClient, configured: dict[str, Any]
) -> None:
    response = await client.post(URL.format(UNKNOWN), json={"query": "anything"})

    assert response.status_code == 404
    assert UNKNOWN in response.json()["detail"]


@pytest.mark.integration
async def test_a_collection_built_for_another_model_is_409(
    client: AsyncClient, app: FastAPI, configured: dict[str, Any], collection_id: UUID
) -> None:
    class WrongProvider(QueryProvider):
        model = "bge-m3"
        dim = 1024

    app.dependency_overrides[embedding_provider] = WrongProvider
    response = await client.post(URL.format(collection_id), json={"query": "anything"})

    assert response.status_code == 409
    assert "bge-m3" in response.json()["detail"]


@pytest.mark.integration
async def test_an_embedding_failure_is_503(
    client: AsyncClient, app: FastAPI, configured: dict[str, Any], collection_id: UUID
) -> None:
    """The provider is down, the query was fine — retryable, and said so."""

    class DownProvider(StubProvider):
        async def embed_one(self, text: str) -> list[float]:
            raise EmbeddingError("ollama is not answering")

    app.dependency_overrides[embedding_provider] = DownProvider
    response = await client.post(URL.format(collection_id), json={"query": "anything"})

    assert response.status_code == 503
    assert "EmbeddingError" in response.json()["detail"]


@pytest.mark.integration
async def test_scores_stay_within_the_cosine_range(
    client: AsyncClient, configured: dict[str, Any], collection_id: UUID
) -> None:
    """A score outside [-1, 1] would mean distance leaked through in place of
    similarity, and every later threshold would read it wrong."""
    await seed(collection_id, [(f"chunk {n}", graded(n / 10)) for n in range(10)])

    response = await client.post(URL.format(collection_id), json={"query": "anything"})

    scores = [hit["score"] for hit in response.json()["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 - 1e-9 <= score <= 1.0 + 1e-9 for score in scores)
    assert not any(math.isnan(score) for score in scores)


@pytest.mark.integration
async def test_a_tenant_cannot_search_another_tenants_collection(
    client: AsyncClient,
    configured: dict[str, Any],
    collection: CollectionRef,
    other_collection: CollectionRef,
) -> None:
    """The security fix, end to end.

    The caller is authenticated as their own tenant and addresses the victim's
    real collection id. It must read as absent, not as forbidden: a 403 would
    confirm the collection exists.
    """
    await seed(other_collection.collection_id, [("their secret", graded(1.0))])

    response = await client.post(
        URL.format(other_collection.collection_id), json={"query": "anything"}
    )

    assert response.status_code == 404
    assert "their secret" not in response.text


async def test_an_unauthenticated_search_is_401(
    client: AsyncClient, configured: dict[str, Any], unauthenticated: None, provider: QueryProvider
) -> None:

    response = await client.post(URL.format(UNKNOWN), json={"query": "anything"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert provider.queries == []


async def test_an_unparseable_key_is_401_without_a_database_round_trip(
    client: AsyncClient, configured: dict[str, Any], unauthenticated: None
) -> None:

    response = await client.post(
        URL.format(UNKNOWN),
        json={"query": "anything"},
        headers={"Authorization": "Bearer sk-somebody-elses-key"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid bearer key"


async def test_a_key_without_the_read_scope_is_403(
    client: AsyncClient, authenticate: Authenticate, provider: QueryProvider
) -> None:
    authenticate(uuid7(), Scope.INGEST)

    response = await client.post(URL.format(UNKNOWN), json={"query": "anything"})

    assert response.status_code == 403
    assert provider.queries == []
