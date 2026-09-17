"""Integration: the query-time tables migrations 0008 and 0009 add. Needs `make up`.

These assert the constraints, not the columns. Every invariant ADR 0012 and ADR
0013 chose to enforce in the database is one a writer would otherwise have to be
trusted with, so the test is that the database refuses the bad write.
"""

from collections.abc import AsyncIterator
from decimal import Decimal
from uuid import UUID

import pytest
from sqlalchemy import TextClause, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from prism.core.ids import uuid7
from prism.db import get_engine

pytestmark = pytest.mark.integration

QUERY_TABLES = {"queries", "query_traces", "query_citations", "model_pricing"}

INSERT_QUERY = text(
    """
    INSERT INTO queries
        (id, tenant_id, collection_id, thread_id, question, status, refusal_reason,
         cache_entry_id, final_answer, citation_count, retrieval_attempts,
         grounding_attempts, total_cost_usd)
    VALUES
        (:id, :tenant_id, :collection_id, :thread_id, :question, :status, :refusal_reason,
         :cache_entry_id, :final_answer, :citation_count, :retrieval_attempts,
         :grounding_attempts, :total_cost_usd)
    """
)

INSERT_TRACE = text(
    """
    INSERT INTO query_traces
        (id, query_id, tenant_id, node_name, sequence, attempt, status, verdict,
         started_at, duration_ms, billing_unit, input_tokens, output_tokens, gpu_ms,
         price_id, cost_usd, cost_basis, error)
    VALUES
        (:id, :query_id, :tenant_id, :node_name, :sequence, :attempt, :status, :verdict,
         now(), :duration_ms, :billing_unit, :input_tokens, :output_tokens, :gpu_ms,
         :price_id, :cost_usd, :cost_basis, :error)
    """
)

INSERT_CITATION = text(
    """
    INSERT INTO query_citations
        (id, query_id, tenant_id, chunk_ref, chunk_id, document_id, page_number,
         chunk_index, rank, rerank_score, cited_content)
    VALUES
        (:id, :query_id, :tenant_id, :chunk_ref, :chunk_id, :document_id, :page_number,
         :chunk_index, :rank, :rerank_score, :cited_content)
    """
)


@pytest.fixture
async def scope() -> AsyncIterator[tuple[UUID, UUID]]:
    """A tenant and a collection to hang queries off. Deleting the tenant cascades."""
    tenant_id, collection_id = uuid7(), uuid7()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": f"query-schema-{tenant_id}"},
        )
        await conn.execute(
            text(
                "INSERT INTO collections (id, tenant_id, name) "
                "VALUES (:id, :tenant_id, 'query-schema')"
            ),
            {"id": collection_id, "tenant_id": tenant_id},
        )
    try:
        yield tenant_id, collection_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


def _query_row(scope: tuple[UUID, UUID], **overrides: object) -> dict[str, object]:
    tenant_id, collection_id = scope
    query_id = uuid7()
    row: dict[str, object] = {
        "id": query_id,
        "tenant_id": tenant_id,
        "collection_id": collection_id,
        "thread_id": f"thread-{query_id}",
        "question": "what did revenue do last quarter?",
        "status": "answered",
        "refusal_reason": None,
        "cache_entry_id": None,
        "final_answer": "It rose.",
        "citation_count": 1,
        "retrieval_attempts": 1,
        "grounding_attempts": 1,
        "total_cost_usd": None,
    }
    return row | overrides


def _trace_row(query_id: UUID, tenant_id: UUID, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": uuid7(),
        "query_id": query_id,
        "tenant_id": tenant_id,
        "node_name": "retrieve",
        "sequence": 1,
        "attempt": 1,
        "status": "ok",
        "verdict": None,
        "duration_ms": 12,
        "billing_unit": "none",
        "input_tokens": None,
        "output_tokens": None,
        "gpu_ms": None,
        "price_id": None,
        "cost_usd": None,
        "cost_basis": None,
        "error": None,
    }
    return row | overrides


async def _insert(statement: TextClause, row: dict[str, object]) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(statement, row)


async def _rejects(statement: TextClause, row: dict[str, object]) -> None:
    with pytest.raises(IntegrityError):
        await _insert(statement, row)


async def test_query_tables_exist() -> None:
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
    present = {r[0] for r in rows}
    missing = QUERY_TABLES - present
    assert not missing, f"migrations 0008/0009 not applied: missing {missing}"


async def test_btree_gist_extension_is_installed() -> None:
    async with get_engine().connect() as conn:
        version = (
            await conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'btree_gist'")
            )
        ).scalar_one_or_none()
    assert version is not None, "btree_gist missing — migration 0008 not applied"


# --- queries -----------------------------------------------------------------


async def test_a_refusal_carries_a_reason_and_nothing_else_does(
    scope: tuple[UUID, UUID],
) -> None:
    """The equivalence is the point: a nullable enum alone permits the boolean it replaces."""
    await _rejects(INSERT_QUERY, _query_row(scope, status="refused", citation_count=0))
    await _rejects(INSERT_QUERY, _query_row(scope, refusal_reason="insufficient_evidence"))
    await _insert(
        INSERT_QUERY,
        _query_row(
            scope,
            status="refused",
            refusal_reason="insufficient_evidence",
            citation_count=0,
            final_answer=None,
        ),
    )


async def test_a_cached_query_names_the_entry_that_served_it(scope: tuple[UUID, UUID]) -> None:
    await _rejects(INSERT_QUERY, _query_row(scope, status="cached"))
    await _rejects(INSERT_QUERY, _query_row(scope, cache_entry_id=uuid7()))
    await _insert(INSERT_QUERY, _query_row(scope, status="cached", cache_entry_id=uuid7()))


async def test_an_answer_without_citations_cannot_be_stored(scope: tuple[UUID, UUID]) -> None:
    """Including a cached one — a cache entry is an answer with its citations."""
    await _rejects(INSERT_QUERY, _query_row(scope, citation_count=0))
    await _rejects(
        INSERT_QUERY,
        _query_row(scope, status="cached", cache_entry_id=uuid7(), citation_count=0),
    )
    await _rejects(
        INSERT_QUERY,
        _query_row(
            scope,
            status="refused",
            refusal_reason="no_relevant_evidence",
            citation_count=1,
            final_answer=None,
        ),
    )


async def test_refusal_reasons_are_the_two_the_graph_can_reach(scope: tuple[UUID, UUID]) -> None:
    await _rejects(
        INSERT_QUERY,
        _query_row(scope, status="refused", refusal_reason="low_confidence", citation_count=0),
    )


async def test_status_has_exactly_three_terminal_values(scope: tuple[UUID, UUID]) -> None:
    await _rejects(INSERT_QUERY, _query_row(scope, status="error"))


async def test_the_two_loops_count_independently_and_the_schema_holds_no_limit(
    scope: tuple[UUID, UUID],
) -> None:
    """max_attempts lives in Settings. A <= 3 check here would be it, hardcoded."""
    await _insert(INSERT_QUERY, _query_row(scope, retrieval_attempts=3, grounding_attempts=1))
    await _insert(INSERT_QUERY, _query_row(scope, retrieval_attempts=9, grounding_attempts=7))
    await _rejects(INSERT_QUERY, _query_row(scope, retrieval_attempts=-1))


async def test_thread_id_is_unique(scope: tuple[UUID, UUID]) -> None:
    first = _query_row(scope)
    await _insert(INSERT_QUERY, first)
    await _rejects(INSERT_QUERY, _query_row(scope, thread_id=first["thread_id"]))


async def test_cost_keeps_eight_decimal_places(scope: tuple[UUID, UUID]) -> None:
    """At a $0.00019 mean, six places round a node row to zero and the sum stops matching."""
    row = _query_row(scope, total_cost_usd=Decimal("0.00000019"))
    await _insert(INSERT_QUERY, row)
    async with get_engine().connect() as conn:
        stored = (
            await conn.execute(
                text("SELECT total_cost_usd FROM queries WHERE id = :id"), {"id": row["id"]}
            )
        ).scalar_one()
    assert stored == Decimal("0.00000019")


# --- query_traces ------------------------------------------------------------


async def test_an_errored_node_says_why_and_a_healthy_one_does_not(
    scope: tuple[UUID, UUID],
) -> None:
    tenant_id, _ = scope
    row = _query_row(scope)
    await _insert(INSERT_QUERY, row)
    query_id = row["id"]
    assert isinstance(query_id, UUID)

    await _rejects(INSERT_TRACE, _trace_row(query_id, tenant_id, status="error"))
    await _rejects(INSERT_TRACE, _trace_row(query_id, tenant_id, error="boom"))
    await _insert(
        INSERT_TRACE, _trace_row(query_id, tenant_id, status="error", error="model failed to load")
    )


async def test_sequence_orders_the_waterfall_and_cannot_tie(scope: tuple[UUID, UUID]) -> None:
    tenant_id, _ = scope
    row = _query_row(scope)
    await _insert(INSERT_QUERY, row)
    query_id = row["id"]
    assert isinstance(query_id, UUID)

    await _insert(INSERT_TRACE, _trace_row(query_id, tenant_id, node_name="plan_query", sequence=1))
    # Same node, second attempt of the retrieval loop: a distinct row.
    await _insert(
        INSERT_TRACE,
        _trace_row(query_id, tenant_id, node_name="retrieve", sequence=2, attempt=2),
    )
    await _rejects(INSERT_TRACE, _trace_row(query_id, tenant_id, node_name="rerank", sequence=2))


async def test_a_verdict_is_pass_or_fail_only(scope: tuple[UUID, UUID]) -> None:
    """pruned is a status, not a judgement, and the designed skip verdict was that mistake."""
    tenant_id, _ = scope
    row = _query_row(scope)
    await _insert(INSERT_QUERY, row)
    query_id = row["id"]
    assert isinstance(query_id, UUID)

    await _rejects(
        INSERT_TRACE, _trace_row(query_id, tenant_id, node_name="grade_docs", verdict="skip")
    )
    await _insert(
        INSERT_TRACE, _trace_row(query_id, tenant_id, node_name="grade_docs", verdict="fail")
    )
    await _insert(
        INSERT_TRACE,
        _trace_row(query_id, tenant_id, node_name="rerank", sequence=2, status="pruned"),
    )


async def test_the_meter_matches_the_billing_unit(scope: tuple[UUID, UUID]) -> None:
    """Tokens cannot express GPU-time, and the in-process lane has neither."""
    tenant_id, _ = scope
    row = _query_row(scope)
    await _insert(INSERT_QUERY, row)
    query_id = row["id"]
    assert isinstance(query_id, UUID)

    await _rejects(INSERT_TRACE, _trace_row(query_id, tenant_id, billing_unit="tokens", gpu_ms=400))
    await _rejects(
        INSERT_TRACE, _trace_row(query_id, tenant_id, billing_unit="gpu_ms", input_tokens=100)
    )
    await _rejects(INSERT_TRACE, _trace_row(query_id, tenant_id, input_tokens=100))
    await _insert(
        INSERT_TRACE,
        _trace_row(query_id, tenant_id, billing_unit="gpu_ms", gpu_ms=400),
    )


async def test_a_cost_without_a_price_basis_cannot_be_stored(scope: tuple[UUID, UUID]) -> None:
    """price_id IS NULL means we could not price the row. Unpriced is not free."""
    tenant_id, _ = scope
    row = _query_row(scope)
    await _insert(INSERT_QUERY, row)
    query_id = row["id"]
    assert isinstance(query_id, UUID)

    await _rejects(
        INSERT_TRACE,
        _trace_row(query_id, tenant_id, cost_usd=Decimal("0"), cost_basis="metered"),
    )

    price_id = await _price_id("in-process", "bge-reranker-v2-m3")
    await _rejects(INSERT_TRACE, _trace_row(query_id, tenant_id, price_id=price_id))
    await _insert(
        INSERT_TRACE,
        _trace_row(
            query_id,
            tenant_id,
            node_name="rerank",
            price_id=price_id,
            cost_usd=Decimal("0"),
            cost_basis="metered",
        ),
    )


async def test_nothing_holds_a_foreign_key_to_query_traces() -> None:
    """The constraint that keeps partitioning a one-table change (ADR 0012)."""
    async with get_engine().connect() as conn:
        referencing = (
            await conn.execute(
                text(
                    "SELECT conrelid::regclass::text FROM pg_constraint "
                    "WHERE contype = 'f' AND confrelid = 'query_traces'::regclass"
                )
            )
        ).scalars()
    assert list(referencing) == []


# --- query_citations ---------------------------------------------------------


async def test_a_citation_outlives_the_chunk_it_cited(scope: tuple[UUID, UUID]) -> None:
    """Deleting a document must not rewrite the history of every answer that cited it."""
    tenant_id, collection_id = scope
    document_id, chunk_id = uuid7(), uuid7()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO documents (id, collection_id, tenant_id, filename, mime_type) "
                "VALUES (:id, :collection_id, :tenant_id, 'cited.pdf', 'application/pdf')"
            ),
            {"id": document_id, "collection_id": collection_id, "tenant_id": tenant_id},
        )
        await conn.execute(
            text(
                "INSERT INTO chunks (id, document_id, collection_id, tenant_id, content) "
                "VALUES (:id, :document_id, :collection_id, :tenant_id, 'Revenue rose 4%.')"
            ),
            {
                "id": chunk_id,
                "document_id": document_id,
                "collection_id": collection_id,
                "tenant_id": tenant_id,
            },
        )

    row = _query_row(scope)
    await _insert(INSERT_QUERY, row)
    query_id = row["id"]
    await _insert(
        INSERT_CITATION,
        {
            "id": uuid7(),
            "query_id": query_id,
            "tenant_id": tenant_id,
            "chunk_ref": chunk_id,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "page_number": 3,
            "chunk_index": 7,
            "rank": 1,
            "rerank_score": 0.81,
            "cited_content": "Revenue rose 4%.",
        },
    )

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})

    async with engine.connect() as conn:
        stored = (
            await conn.execute(
                text(
                    "SELECT chunk_ref, chunk_id, cited_content FROM query_citations "
                    "WHERE query_id = :query_id"
                ),
                {"query_id": query_id},
            )
        ).one()
    assert stored.chunk_ref == chunk_id, "what was cited is not recoverable"
    assert stored.chunk_id is None, "the chunk is gone and the row should say so"
    assert stored.cited_content == "Revenue rose 4%."


async def test_citation_rank_is_unique_per_query(scope: tuple[UUID, UUID]) -> None:
    tenant_id, _ = scope
    row = _query_row(scope)
    await _insert(INSERT_QUERY, row)
    citation: dict[str, object] = {
        "id": uuid7(),
        "query_id": row["id"],
        "tenant_id": tenant_id,
        "chunk_ref": uuid7(),
        "chunk_id": None,
        "document_id": None,
        "page_number": None,
        "chunk_index": None,
        "rank": 1,
        "rerank_score": 0.62,
        "cited_content": "A span that was cited.",
    }
    await _insert(INSERT_CITATION, citation)
    await _rejects(INSERT_CITATION, citation | {"id": uuid7(), "chunk_ref": uuid7()})


async def test_traces_and_citations_go_with_their_query(scope: tuple[UUID, UUID]) -> None:
    tenant_id, _ = scope
    row = _query_row(scope)
    await _insert(INSERT_QUERY, row)
    query_id = row["id"]
    assert isinstance(query_id, UUID)
    await _insert(INSERT_TRACE, _trace_row(query_id, tenant_id))
    await _insert(
        INSERT_CITATION,
        {
            "id": uuid7(),
            "query_id": query_id,
            "tenant_id": tenant_id,
            "chunk_ref": uuid7(),
            "chunk_id": None,
            "document_id": None,
            "page_number": None,
            "chunk_index": None,
            "rank": 1,
            "rerank_score": None,
            "cited_content": "A span that was cited.",
        },
    )

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM queries WHERE id = :id"), {"id": query_id})
    async with engine.connect() as conn:
        for table in ("query_traces", "query_citations"):
            remaining = (
                await conn.execute(
                    text(f"SELECT count(*) FROM {table} WHERE query_id = :id"), {"id": query_id}
                )
            ).scalar_one()
            assert remaining == 0, f"{table} survived its query"


# --- model_pricing -----------------------------------------------------------


async def _price_id(provider: str, model: str) -> UUID:
    async with get_engine().connect() as conn:
        price_id = (
            await conn.execute(
                text(
                    "SELECT id FROM model_pricing "
                    "WHERE provider = :provider AND model = :model AND effective_to IS NULL"
                ),
                {"provider": provider, "model": model},
            )
        ).scalar_one()
    assert isinstance(price_id, UUID)
    return price_id


async def test_the_local_lanes_are_priced_at_zero_not_left_unpriced() -> None:
    """Free is priced; unpriced is not free. Asserting cost > 0 here would be the bug."""
    async with get_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT model, billing_unit, input_per_mtok, output_per_mtok, source "
                    "FROM model_pricing WHERE provider IN ('ollama', 'in-process')"
                )
            )
        ).all()
    priced = {r.model for r in rows}
    assert priced >= {
        "qwen2.5:14b",
        "llama3.1:8b",
        "qwen2.5:32b",
        "nomic-embed-text",
        "qwen2.5vl:7b",
        "bge-reranker-v2-m3",
    }
    for row in rows:
        assert row.source, "a price without provenance is a guess wearing a decimal point"
        if row.billing_unit == "tokens":
            assert row.input_per_mtok == Decimal("0")
            assert row.output_per_mtok == Decimal("0")


async def test_two_prices_cannot_cover_one_moment(scope: tuple[UUID, UUID]) -> None:
    """Overlap would make "the price at time T" ambiguous, and wrong rather than loud."""
    insert = text(
        "INSERT INTO model_pricing "
        "(id, provider, model, billing_unit, input_per_mtok, output_per_mtok, "
        " effective_from, effective_to, source) "
        "VALUES (:id, 'gemini', :model, 'tokens', 0.3, 2.5, :start, :end, 'test')"
    )
    model = f"gemini-test-{uuid7()}"
    await _insert(
        insert,
        {
            "id": uuid7(),
            "model": model,
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-06-01T00:00:00+00:00",
        },
    )
    try:
        await _rejects(
            insert,
            {
                "id": uuid7(),
                "model": model,
                "start": "2026-05-01T00:00:00+00:00",
                "end": None,
            },
        )
        # Abutting, not overlapping: the successor starts where the first ends.
        await _insert(
            insert,
            {
                "id": uuid7(),
                "model": model,
                "start": "2026-06-01T00:00:00+00:00",
                "end": None,
            },
        )
    finally:
        async with get_engine().begin() as conn:
            await conn.execute(
                text("DELETE FROM model_pricing WHERE model = :model"), {"model": model}
            )


async def test_a_price_can_only_be_closed_never_edited() -> None:
    """Editing one retroactively rewrites every cost that points at it."""
    price_id = await _price_id("ollama", "qwen2.5:32b")
    engine = get_engine()

    with pytest.raises(DBAPIError) as raised:
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE model_pricing SET input_per_mtok = 0.5 WHERE id = :id"),
                {"id": price_id},
            )
    assert "append-only" in str(raised.value)

    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE model_pricing SET effective_to = :end WHERE id = :id"),
            {"id": price_id, "end": "2099-01-01T00:00:00+00:00"},
        )
        await conn.execute(
            text("UPDATE model_pricing SET effective_to = NULL WHERE id = :id"),
            {"id": price_id},
        )


async def test_a_price_a_trace_used_cannot_be_deleted(scope: tuple[UUID, UUID]) -> None:
    tenant_id, _ = scope
    row = _query_row(scope)
    await _insert(INSERT_QUERY, row)
    query_id = row["id"]
    assert isinstance(query_id, UUID)
    price_id = await _price_id("ollama", "llama3.1:8b")
    await _insert(
        INSERT_TRACE,
        _trace_row(
            query_id,
            tenant_id,
            node_name="grade_docs",
            verdict="pass",
            billing_unit="tokens",
            input_tokens=900,
            output_tokens=12,
            price_id=price_id,
            cost_usd=Decimal("0"),
            cost_basis="metered",
        ),
    )
    with pytest.raises(IntegrityError):
        async with get_engine().begin() as conn:
            await conn.execute(text("DELETE FROM model_pricing WHERE id = :id"), {"id": price_id})
