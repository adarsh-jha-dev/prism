"""Unit tests are the default. Integration tests are marked and need `make up`.

Paid providers are never called from either tier — they replay from
tests/fixtures/.
"""

import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("PRISM_ENV", "test")

if TYPE_CHECKING:
    from prism.auth import ResolvedKey, Scope
    from prism.collections import CollectionRef
    from stub_chat import ScriptedChat
    from stub_reranker import StubReranker


class Authenticate(Protocol):
    def __call__(
        self,
        tenant_id: UUID,
        *scopes: "Scope",
        rate_limit_rpm: int = 60,
        key_id: UUID | None = None,
    ) -> "ResolvedKey": ...


@pytest.fixture(autouse=True)
async def _sweep_orphan_checkpoints(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    """Checkpoints hold no foreign key, so dropping a tenant cannot cascade here.

    Retention (ADR 0012) is not built yet; until it is, the tests sweep their own
    threads.
    """
    yield
    if request.node.get_closest_marker("integration") is None:
        return  # the unit tier touches no database
    from sqlalchemy import text

    from prism.db import get_engine

    async with get_engine().begin() as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await conn.execute(
                text(f"DELETE FROM {table} WHERE thread_id NOT IN (SELECT thread_id FROM queries)")
            )


class StubbedModels(Protocol):
    """What `stubbed_models` hands back, so a test can script the run."""

    def __call__(
        self,
        *,
        plan: "str | Sequence[str]" = ...,
        grade: "str | Sequence[str]" = ...,
        rewrite: "str | Sequence[str]" = ...,
        reranker: "StubReranker | None" = ...,
    ) -> "ScriptedChat": ...


@pytest.fixture
def stubbed_models(monkeypatch: pytest.MonkeyPatch) -> "StubbedModels":
    """Run the graph without Ollama, for tests whose subject is the machinery.

    Patched where each default is resolved, so nothing reaches the network.
    tests/test_graph_nodes.py covers the live path.

    Calling the fixture re-scripts the replies; depending on it without calling
    it takes the defaults, which pass the first candidate and rewrite once.
    """
    from prism.config import Settings
    from prism.providers import Lane, ProviderRegistry
    from stub_chat import ScriptedChat
    from stub_provider import StubProvider
    from stub_reranker import StubReranker

    embedder = StubProvider()
    monkeypatch.setattr("prism.providers.registry.get_embedding_provider", lambda: embedder)
    monkeypatch.setattr("prism.retrieval.hybrid.get_embedding_provider", lambda: embedder)

    def script(
        *,
        plan: "str | Sequence[str]" = '{"terms": ["chinchilla", "ratio"]}',
        grade: "str | Sequence[str]" = '{"verdicts": [{"label": 1, "score": 0.9}]}',
        rewrite: "str | Sequence[str]" = '{"query": "rewritten query"}',
        reranker: "StubReranker | None" = None,
    ) -> "ScriptedChat":
        # Well clear of rerank_score_floor: a test about the loop should not have
        # to think about the floor, and a test about the floor scripts its own.
        scorer = reranker or StubReranker(default=0.9)
        monkeypatch.setattr("prism.graph.nodes.get_reranker", lambda: scorer)
        chat = ScriptedChat(
            {"QueryPlan": plan, "RelevanceVerdicts": grade, "QueryRewrite": rewrite}
        )
        lane = Lane(
            name="ollama",
            billing_unit="tokens",
            concurrency=8,
            queue_timeout_s=1.0,
            timeout_s=1.0,
            model="stub-model",
            factory=lambda: chat,
        )
        registry = ProviderRegistry({"ollama": lane}, Settings())
        monkeypatch.setattr("prism.graph.nodes.get_registry", lambda: registry)
        return chat

    script()
    return script


@pytest.fixture
def app() -> FastAPI:
    """A fresh app per test, so one test's dependency overrides cannot leak."""
    from prism.main import create_app

    return create_app()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """In-process client — never binds a port."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@asynccontextmanager
async def _throwaway_collection(name: str) -> AsyncIterator["CollectionRef"]:
    from sqlalchemy import text

    from prism.collections import CollectionRef
    from prism.core.ids import uuid7
    from prism.db import get_engine

    tenant_id, collection_id = uuid7(), uuid7()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": f"{name}-{tenant_id}"},
        )
        await conn.execute(
            text("INSERT INTO collections (id, tenant_id, name) VALUES (:id, :tenant_id, :name)"),
            {"id": collection_id, "tenant_id": tenant_id, "name": name},
        )
    try:
        yield CollectionRef(tenant_id=tenant_id, collection_id=collection_id)
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


@pytest.fixture
async def collection() -> AsyncIterator["CollectionRef"]:
    """A throwaway tenant and collection. Integration only — needs `make up`."""
    async with _throwaway_collection("test") as value:
        yield value


@pytest.fixture
async def other_collection() -> AsyncIterator["CollectionRef"]:
    """A second tenant's collection, in the same table and the same index."""
    async with _throwaway_collection("other") as value:
        yield value


@pytest.fixture
def collection_id(collection: "CollectionRef") -> UUID:
    return collection.collection_id


@pytest.fixture
def other_collection_id(other_collection: "CollectionRef") -> UUID:
    return other_collection.collection_id


@pytest.fixture
def authenticate(app: FastAPI) -> "Authenticate":
    """Make subsequent requests carry a resolved key for `tenant_id`.

    Overrides the dependency rather than issuing a real key: these tests are
    about what the routes do with a tenant, not about key resolution, which
    test_auth_integration.py covers against the table.

    Metering is stubbed out with it. A test about the limiter pops that override
    to get the real one back:

        app.dependency_overrides.pop(metered_key)
    """
    from prism.api.deps import metered_key, require_key
    from prism.auth import ResolvedKey, Scope
    from prism.core.ids import uuid7

    def authenticate(
        tenant_id: UUID,
        *scopes: Scope,
        rate_limit_rpm: int = 60,
        key_id: UUID | None = None,
    ) -> ResolvedKey:
        key = ResolvedKey(
            id=key_id or uuid7(),
            tenant_id=tenant_id,
            scopes=frozenset(scopes or Scope),
            rate_limit_rpm=rate_limit_rpm,
        )
        app.dependency_overrides[require_key] = lambda: key
        app.dependency_overrides[metered_key] = lambda: key
        return key

    return authenticate


@pytest.fixture
def unauthenticated(app: FastAPI) -> None:
    """Undo the autouse authentication, both halves of it."""
    from prism.api.deps import metered_key, require_key

    app.dependency_overrides.pop(require_key, None)
    app.dependency_overrides.pop(metered_key, None)
    return None


@pytest.fixture(autouse=True)
def _authenticate_route_tests(request: pytest.FixtureRequest) -> None:
    """Every route test is authenticated unless it says otherwise.

    As the collection's own tenant when one is in play, so an integration test
    reads what it seeded. A test about rejection pops the override:

        app.dependency_overrides.pop(require_key)
    """
    if "client" not in request.fixturenames:
        return

    from prism.core.ids import uuid7

    authenticate = request.getfixturevalue("authenticate")
    tenant_id = (
        request.getfixturevalue("collection").tenant_id
        if "collection" in request.fixturenames
        else uuid7()
    )
    authenticate(tenant_id)


@pytest.fixture(autouse=True)
async def _close_loop_bound_clients() -> AsyncIterator[None]:
    """A redis-py client and a psycopg pool each belong to one event loop.

    Each test gets its own loop, so both are closed between them.
    """
    from prism.db import close_checkpoint_pool, close_redis

    yield
    await close_redis()
    await close_checkpoint_pool()
