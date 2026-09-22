"""The graph machinery: trace rows, checkpoints, and a refusal with a reason.

These run against an empty collection, so retrieval finds nothing, grading
fails with no model call, and the loop runs its full course. The loop itself
is tests/test_retrieval_loop.py.
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from sqlalchemy import text

from prism.config import get_settings
from prism.db import get_engine
from prism.graph.checkpointer import CHECKPOINTER_SCHEMA_VERSION, get_checkpointer
from prism.graph.graph import NODES, compile_graph
from prism.graph.run import run_query
from prism.graph.state import GraphState
from prism.graph.trace import TraceContext, traced

if TYPE_CHECKING:
    from conftest import StubbedModels
    from prism.collections import CollectionRef


def _exhausted() -> list[str]:
    """Every node an empty collection visits, in order: the loop, then abstain."""
    attempts = get_settings().max_attempts
    nodes: list[str] = []
    for attempt in range(1, attempts + 1):
        nodes += ["plan_query", "embed_query", "retrieve", "rerank", "grade_docs"]
        if attempt < attempts:
            nodes.append("rewrite_query")
    return [*nodes, "abstain"]


@pytest.fixture(autouse=True)
def _no_live_models(stubbed_models: "StubbedModels") -> None:
    """This module's subject is the machinery around a node."""


def test_checkpointer_schema_version_matches_the_pin() -> None:
    """A library bump must fail here, not at query time in a node."""
    from langgraph.checkpoint.postgres.base import MIGRATIONS

    assert len(MIGRATIONS) == CHECKPOINTER_SCHEMA_VERSION


async def _traces(query_id: UUID) -> list[dict[str, Any]]:
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT node_name, sequence, attempt, status, verdict, error,
                       duration_ms, started_at, tenant_id, provider, model,
                       billing_unit, input_tokens, output_tokens, gpu_ms,
                       price_id, cost_usd, cost_basis, checkpoint_ref
                  FROM query_traces
                 WHERE query_id = :query_id
                 ORDER BY sequence
                """
            ),
            {"query_id": query_id},
        )
        return [dict(row) for row in rows.mappings()]


@pytest.mark.integration
async def test_every_node_writes_one_contiguous_trace_row(collection: "CollectionRef") -> None:
    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="what does the skeleton do?",
    )

    rows = await _traces(run.query_id)
    assert [row["node_name"] for row in rows] == _exhausted()
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["error"] is None for row in rows)
    assert all(row["duration_ms"] >= 0 for row in rows)


@pytest.mark.integration
async def test_a_stub_node_records_no_meter_and_no_price(collection: "CollectionRef") -> None:
    """Stubs call no model, so the meter and price columns stay NULL.

    Migration 0009's meters_check and priced_check enforce the pairing. The nodes
    that do call one are asserted in tests/test_graph_nodes.py.
    """
    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="what does a stub cost?",
    )

    for row in await _traces(run.query_id):
        if row["node_name"] in ("plan_query", "embed_query", "rewrite_query"):
            continue
        # grade_docs is here on purpose: an empty candidate set is graded
        # without a model call, so its row must look like a node that made none.
        assert row["billing_unit"] == "none"
        assert row["provider"] is None and row["model"] is None
        assert row["input_tokens"] is None and row["output_tokens"] is None
        assert row["gpu_ms"] is None
        assert row["price_id"] is None and row["cost_usd"] is None
        assert row["cost_basis"] is None
        assert row["verdict"] is None


@pytest.mark.integration
async def test_query_finalizes_as_refused_with_no_citations(collection: "CollectionRef") -> None:
    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="is there any evidence here?",
    )
    assert run.status == "refused"
    assert run.refusal_reason == "no_relevant_evidence"

    async with get_engine().connect() as conn:
        row = (
            (
                await conn.execute(
                    text(
                        """
                    SELECT status, refusal_reason, citation_count, latency_ms,
                           retrieval_attempts, grounding_attempts, thread_id
                      FROM queries WHERE id = :id
                    """
                    ),
                    {"id": run.query_id},
                )
            )
            .mappings()
            .one()
        )
        citations = (
            await conn.execute(
                text("SELECT count(*) FROM query_citations WHERE query_id = :id"),
                {"id": run.query_id},
            )
        ).scalar_one()

    assert row["status"] == "refused"
    assert row["refusal_reason"] == "no_relevant_evidence"
    assert row["citation_count"] == 0
    assert citations == 0
    assert row["latency_ms"] is not None and row["latency_ms"] >= 0
    # Separate columns: retrieval exhausted its attempts, generation never ran.
    assert row["retrieval_attempts"] == get_settings().max_attempts
    assert row["grounding_attempts"] == 0
    assert row["thread_id"] == run.thread_id


@pytest.mark.integration
async def test_checkpoint_exists_and_the_run_is_resumable(collection: "CollectionRef") -> None:
    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="can this be re-entered?",
    )
    config = {"configurable": {"thread_id": run.thread_id}}

    # A graph built from nothing but the thread id, as a replay would be.
    graph = compile_graph(await get_checkpointer())

    tuple_ = await graph.checkpointer.aget_tuple(config)  # type: ignore[union-attr]
    assert tuple_ is not None

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()  # the run finished
    assert snapshot.values["question"] == "can this be re-entered?"
    assert snapshot.values["status"] == "refused"
    assert snapshot.values["sequence"] == len(_exhausted())

    # Forkable from any node: every node boundary is a checkpoint whose `next`
    # names the node that would run. The pass path is not on this run's route.
    pending = [s.next[0] for s in [s async for s in graph.aget_state_history(config)] if s.next]
    assert set(_exhausted()).issubset(pending)
    # LangGraph's own tasks are not nodes of ours; everything else is one.
    assert {node for node in pending if not node.startswith("__")}.issubset(set(NODES))


@pytest.mark.integration
async def test_every_trace_row_points_at_the_checkpoint_it_would_fork_from(
    collection: "CollectionRef",
) -> None:
    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="where would a fork start?",
    )

    rows = await _traces(run.query_id)
    assert all(row["checkpoint_ref"] for row in rows)

    graph = compile_graph(await get_checkpointer())
    for row in rows:
        snapshot = await graph.aget_state(
            {
                "configurable": {
                    "thread_id": run.thread_id,
                    "checkpoint_id": row["checkpoint_ref"],
                }
            }
        )
        # The ref is the state as it was before that node ran.
        assert snapshot.next == (row["node_name"],)


@pytest.mark.integration
async def test_a_node_that_raises_writes_an_error_row_and_stops(
    collection: "CollectionRef", monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run fails loudly, and what it wrote stays readable."""

    @traced("retrieve")
    async def boom(state: GraphState, trace: TraceContext) -> dict[str, Any]:
        raise RuntimeError("retrieval exploded")

    monkeypatch.setattr("prism.graph.nodes.retrieve", boom)

    with pytest.raises(RuntimeError, match="retrieval exploded"):
        await run_query(
            tenant_id=collection.tenant_id,
            collection_id=collection.collection_id,
            question="what happens when a node fails?",
        )

    async with get_engine().connect() as conn:
        row = (
            (
                await conn.execute(
                    text(
                        """
                    SELECT q.id, q.status, q.refusal_reason, q.latency_ms
                      FROM queries q WHERE q.question = :question AND q.tenant_id = :tenant_id
                    """
                    ),
                    {
                        "question": "what happens when a node fails?",
                        "tenant_id": collection.tenant_id,
                    },
                )
            )
            .mappings()
            .one()
        )

    rows = await _traces(row["id"])
    assert [r["node_name"] for r in rows] == ["plan_query", "embed_query", "retrieve"]
    assert [r["status"] for r in rows] == ["ok", "ok", "error"]
    # The row an error writes carries the attempt it failed on, like any other.
    assert [r["attempt"] for r in rows] == [1, 1, 1]
    assert rows[-1]["error"] == "RuntimeError: retrieval exploded"
    # The node that raised is the one a fork most wants to start from.
    assert rows[-1]["checkpoint_ref"]
    # A failed run does not finalize, so the row stays as it was inserted.
    assert row["status"] == "refused"
    assert row["refusal_reason"] == "no_relevant_evidence"
    assert row["latency_ms"] is None


@pytest.mark.integration
async def test_traces_are_scoped_to_the_querys_tenant(
    collection: "CollectionRef", other_collection: "CollectionRef"
) -> None:
    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="whose run is this?",
    )
    other = await run_query(
        tenant_id=other_collection.tenant_id,
        collection_id=other_collection.collection_id,
        question="and whose is this?",
    )

    rows = await _traces(run.query_id)
    assert all(row["tenant_id"] == collection.tenant_id for row in rows)

    async with get_engine().connect() as conn:
        # The same query id under the wrong tenant is no rows.
        visible = (
            await conn.execute(
                text(
                    """
                    SELECT count(*) FROM query_traces
                     WHERE query_id = :query_id AND tenant_id = :tenant_id
                    """
                ),
                {"query_id": run.query_id, "tenant_id": other_collection.tenant_id},
            )
        ).scalar_one()
        threads = (
            await conn.execute(
                text("SELECT count(*) FROM queries WHERE id = :id AND tenant_id = :tenant_id"),
                {"id": run.query_id, "tenant_id": other_collection.tenant_id},
            )
        ).scalar_one()

    assert visible == 0
    assert threads == 0
    assert run.thread_id != other.thread_id
