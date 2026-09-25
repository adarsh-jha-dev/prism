"""POST /collections/{id}/query.

Request validation is a unit test — it is decided before any collection is looked
up. Everything that starts the graph is marked `integration`. The models are
scripted: the subject is the route, not what a model writes.
"""

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from prism.api.deps import embedding_provider
from prism.auth import Scope
from prism.config import get_settings
from seeding import seed
from stub_provider import StubProvider

if TYPE_CHECKING:
    from conftest import Authenticate, StubbedModels
    from prism.collections import CollectionRef

URL = "/collections/{}/query"
UNKNOWN = "01890000-0000-7000-8000-00000000dead"
SSE = {"Accept": "text/event-stream"}

QUESTION = "What is the chinchilla provisioning ratio?"
CHINCHILLA = "The chinchilla provisioning ratio is twenty tokens per parameter."
HARBOUR = "Unrelated material about harbour logistics and berth scheduling."

NOTHING_PASSES = '{"verdicts": [{"label": 1, "score": 0.02}]}'
DIM = 768


def _unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


@pytest.fixture
def stub_embedder(app: FastAPI) -> Iterator[StubProvider]:
    """The compatibility check reaches a provider; this one is not Ollama."""
    provider = StubProvider()
    app.dependency_overrides[embedding_provider] = lambda: provider
    yield provider
    app.dependency_overrides.pop(embedding_provider, None)


@pytest.fixture
async def answerable(collection: "CollectionRef") -> "CollectionRef":
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0))], filename="chinchilla.pdf")
    return collection


@pytest.fixture
async def unanswerable(collection: "CollectionRef") -> "CollectionRef":
    await seed(collection.collection_id, [(HARBOUR, _unit(1))])
    return collection


def _events(body: str) -> list[tuple[str, dict[str, Any]]]:
    """The stream's frames, as (event name, payload)."""
    parsed = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        parsed.append((lines["event"], json.loads(lines["data"])))
    return parsed


# ------------------------------------------------------------ the unit tier


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({}, "no question at all"),
        ({"question": ""}, "empty question"),
        ({"question": "   "}, "whitespace-only question"),
        ({"question": "\t\n "}, "whitespace-only question of other whitespace"),
        ({"question": "x" * 2001}, "question past the length bound"),
    ],
)
async def test_a_malformed_question_is_refused_before_the_graph_starts(
    client: AsyncClient, stub_embedder: StubProvider, body: dict[str, Any], why: str
) -> None:
    response = await client.post(URL.format(UNKNOWN), json=body)

    assert response.status_code == 422, why
    # Nothing was embedded, so no node ran.
    assert stub_embedder.batches == []


async def test_a_key_without_the_read_scope_is_refused(
    client: AsyncClient,
    stub_embedder: StubProvider,
    authenticate: "Authenticate",
) -> None:
    authenticate(UUID(UNKNOWN), Scope.INGEST)

    response = await client.post(URL.format(UNKNOWN), json={"question": QUESTION})

    assert response.status_code == 403
    assert stub_embedder.batches == []


# ----------------------------------------------------------- the answer path


@pytest.mark.integration
async def test_a_question_with_relevant_evidence_is_answered_with_ranked_citations(
    client: AsyncClient,
    stub_embedder: StubProvider,
    answerable: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    stubbed_models()

    response = await client.post(URL.format(answerable.collection_id), json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["refusal_reason"] is None
    assert body["answer"]
    assert UUID(body["collection_id"]) == answerable.collection_id

    assert [c["rank"] for c in body["citations"]] == [1]
    citation = body["citations"][0]
    assert set(citation) == {"rank", "document_id", "filename", "page_number", "content"}
    assert citation["filename"] == "chinchilla.pdf"
    assert citation["page_number"] == 1
    assert citation["content"] == CHINCHILLA


@pytest.mark.integration
async def test_thread_id_is_nowhere_in_the_response(
    client: AsyncClient,
    stub_embedder: StubProvider,
    answerable: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    """`run.py` mints it and never accepts it from a client."""
    stubbed_models()

    response = await client.post(URL.format(answerable.collection_id), json={"question": QUESTION})
    streamed = await client.post(
        URL.format(answerable.collection_id), json={"question": QUESTION}, headers=SSE
    )

    assert response.status_code == 200
    assert "thread_id" not in response.text
    assert "thread_id" not in streamed.text


# ---------------------------------------------------------- the refusal path


@pytest.mark.integration
async def test_no_relevant_evidence_is_a_200_with_a_reason(
    client: AsyncClient,
    stub_embedder: StubProvider,
    unanswerable: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    stubbed_models(grade=NOTHING_PASSES)

    response = await client.post(
        URL.format(unanswerable.collection_id), json={"question": QUESTION}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["refusal_reason"] == "no_relevant_evidence"
    assert body["answer"] is None
    assert body["citations"] == []


# ------------------------------------------------------------------- scoping


@pytest.mark.integration
async def test_another_tenants_collection_is_a_404(
    client: AsyncClient,
    stub_embedder: StubProvider,
    collection: "CollectionRef",
    other_collection: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    """404, not 403: a 403 confirms the id exists."""
    stubbed_models()

    response = await client.post(
        URL.format(other_collection.collection_id), json={"question": QUESTION}
    )

    assert response.status_code == 404


@pytest.mark.integration
async def test_an_unknown_collection_is_a_404_on_the_streaming_variant_too(
    client: AsyncClient,
    stub_embedder: StubProvider,
    collection: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    """Decided before the first byte, so it still sets a status code."""
    stubbed_models()

    response = await client.post(URL.format(UNKNOWN), json={"question": QUESTION}, headers=SSE)

    assert response.status_code == 404


# -------------------------------------------------------------- the stream


@pytest.mark.integration
async def test_the_stream_carries_an_event_per_executed_node_then_the_outcome(
    client: AsyncClient,
    stub_embedder: StubProvider,
    answerable: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    """The events are the trace rows, in commit order (ADR 0023)."""
    stubbed_models()

    response = await client.post(
        URL.format(answerable.collection_id), json={"question": QUESTION}, headers=SSE
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = _events(response.text)
    names = [name for name, _ in frames]
    assert names[-1] == "result"
    assert set(names[:-1]) == {"node"}

    nodes = [payload for _, payload in frames[:-1]]
    assert [n["node"] for n in nodes] == [
        "plan_query",
        "embed_query",
        "retrieve",
        "rerank",
        "grade_docs",
        "generate",
        "verify_grounding",
    ]
    assert [n["sequence"] for n in nodes] == list(range(1, len(nodes) + 1))
    assert all(n["status"] == "ok" for n in nodes)
    assert {n["query_id"] for n in nodes} == {frames[-1][1]["query_id"]}

    # The judging nodes carry a verdict and the rest carry none (migration 0009).
    verdicts = {n["node"]: n["verdict"] for n in nodes}
    assert verdicts["grade_docs"] == "pass"
    assert verdicts["verify_grounding"] == "pass"
    assert verdicts["retrieve"] is None

    outcome = frames[-1][1]
    assert outcome["status"] == "answered"
    assert [c["rank"] for c in outcome["citations"]] == [1]


@pytest.mark.integration
async def test_the_streamed_result_is_the_body_the_json_variant_returns(
    client: AsyncClient,
    stub_embedder: StubProvider,
    answerable: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    stubbed_models()

    plain = await client.post(URL.format(answerable.collection_id), json={"question": QUESTION})
    streamed = await client.post(
        URL.format(answerable.collection_id), json={"question": QUESTION}, headers=SSE
    )

    result = _events(streamed.text)[-1][1]
    assert set(result) == set(plain.json())
    # Two runs, so only the ids differ.
    assert {k: v for k, v in result.items() if k != "query_id"} == {
        k: v for k, v in plain.json().items() if k != "query_id"
    }


# ------------------------------------------------------- failure, either side


@pytest.mark.integration
async def test_a_provider_failure_before_the_first_byte_is_a_503(
    client: AsyncClient,
    stub_embedder: StubProvider,
    answerable: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    """Nothing has been sent, so the failure can still set a status code."""
    stubbed_models(plan="not a query plan")

    response = await client.post(URL.format(answerable.collection_id), json={"question": QUESTION})

    assert response.status_code == 503
    assert "ChatError" in response.json()["detail"]


@pytest.mark.integration
async def test_a_provider_failure_after_the_stream_opens_is_an_error_event(
    client: AsyncClient,
    stub_embedder: StubProvider,
    answerable: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    """The status is already 200 and cannot be revised."""
    stubbed_models(plan="not a query plan")

    response = await client.post(
        URL.format(answerable.collection_id), json={"question": QUESTION}, headers=SSE
    )

    assert response.status_code == 200
    frames = _events(response.text)
    assert [name for name, _ in frames] == ["node", "error"]

    # The failing node wrote its error row before raising, so it is in the stream.
    failed = frames[0][1]
    assert (failed["node"], failed["status"]) == ("plan_query", "error")
    assert "ChatError" in failed["error"]

    error = frames[1][1]
    assert "ChatError" in error["detail"]
    assert error["query_id"] == failed["query_id"]


@pytest.mark.integration
async def test_an_embedding_model_mismatch_is_a_409(
    client: AsyncClient,
    app: FastAPI,
    collection: "CollectionRef",
    stubbed_models: "StubbedModels",
) -> None:
    stubbed_models()

    class Other(StubProvider):
        model = "some-other-embedder"

    app.dependency_overrides[embedding_provider] = lambda: Other()

    response = await client.post(URL.format(collection.collection_id), json={"question": QUESTION})

    assert response.status_code == 409
    assert get_settings().embedding_model in response.json()["detail"]
