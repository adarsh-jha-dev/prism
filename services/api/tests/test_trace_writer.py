"""Integration: what the trace writer records for a node that reports. Needs `make up`.

Every row here goes through `traced`, not a raw INSERT: test_schema_queries.py
shows the constraints refuse bad rows, and these show the writer never makes one.
"""

from collections.abc import AsyncIterator
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from prism.chat import Usage
from prism.config import get_settings
from prism.core.ids import uuid7
from prism.db import get_engine
from prism.graph.checkpointer import get_checkpointer
from prism.graph.graph import compile_graph
from prism.graph.run import mint_query, run_query
from prism.graph.state import GraphState, search_params_from
from prism.graph.trace import TraceContext, traced
from stub_chat import usage

if TYPE_CHECKING:
    from prism.collections import CollectionRef

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
async def _sweep_orphan_checkpoints() -> AsyncIterator[None]:
    """Checkpoints hold no foreign key; see test_graph_skeleton.py."""
    yield
    async with get_engine().begin() as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await conn.execute(
                text(f"DELETE FROM {table} WHERE thread_id NOT IN (SELECT thread_id FROM queries)")
            )


@pytest.fixture
async def paid_model() -> AsyncIterator[str]:
    """A priced, non-zero model for a total worth summing. No call is ever made to it."""
    model, price_id = f"test-{uuid7()}", uuid7()
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO model_pricing
                    (id, provider, model, billing_unit, input_per_mtok, output_per_mtok,
                     effective_from, source)
                VALUES
                    (:id, 'gemini', :model, 'tokens', 0.30, 2.50,
                     now() - interval '1 day', 'test fixture')
                """
            ),
            {"id": price_id, "model": model},
        )
    try:
        yield model
    finally:
        async with get_engine().begin() as conn:
            # price_id is ON DELETE RESTRICT; the traces go first.
            await conn.execute(
                text("DELETE FROM query_traces WHERE price_id = :id"), {"id": price_id}
            )
            await conn.execute(text("DELETE FROM model_pricing WHERE id = :id"), {"id": price_id})


async def _state(collection: "CollectionRef") -> GraphState:
    query_id, thread_id = await mint_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="what did the node report?",
    )
    return {
        "query_id": query_id,
        "tenant_id": collection.tenant_id,
        "collection_id": collection.collection_id,
        "thread_id": thread_id,
        "question": "what did the node report?",
        "retrieval_attempts": 0,
        "grounding_attempts": 0,
        "sequence": 0,
        "search_terms": [],
        "search_params": search_params_from(get_settings()),
        "query_embedding": None,
        "candidates": [],
        "status": "refused",
        "refusal_reason": "no_relevant_evidence",
    }


def _reporting(
    *,
    meter: Usage | None = None,
    payload_in: object = None,
    payload_out: object = None,
    raises: Exception | None = None,
) -> Any:
    async def node(state: GraphState, trace: TraceContext) -> dict[str, Any]:
        if meter is not None:
            trace.record_usage(meter)
        trace.record_input(payload_in)
        trace.record_output(payload_out)
        if raises is not None:
            raise raises
        return {}

    return node


async def _rows(query_id: UUID) -> list[dict[str, Any]]:
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT node_name, status, provider, model, billing_unit,
                       input_tokens, output_tokens, gpu_ms,
                       price_id, cost_usd, cost_basis,
                       input_json, output_json, input_truncated, output_truncated
                  FROM query_traces WHERE query_id = :query_id ORDER BY sequence
                """
            ),
            {"query_id": query_id},
        )
        return [dict(row) for row in rows.mappings()]


async def _total(query_id: UUID) -> Decimal | None:
    async with get_engine().connect() as conn:
        total: Decimal | None = (
            await conn.execute(
                text("SELECT total_cost_usd FROM queries WHERE id = :id"), {"id": query_id}
            )
        ).scalar_one()
        return total


async def test_a_node_reporting_usage_writes_provider_model_unit_and_meters(
    collection: "CollectionRef",
) -> None:
    state = await _state(collection)
    await traced("generate")(_reporting(meter=usage(input_tokens=120, output_tokens=40)))(state)

    [row] = await _rows(state["query_id"])
    assert (row["provider"], row["model"], row["billing_unit"]) == (
        "ollama",
        "qwen2.5:32b",
        "tokens",
    )
    assert (row["input_tokens"], row["output_tokens"], row["gpu_ms"]) == (120, 40, None)


async def test_a_local_call_is_priced_at_exactly_zero_not_left_unpriced(
    collection: "CollectionRef",
) -> None:
    state = await _state(collection)
    await traced("generate")(_reporting(meter=usage()))(state)

    [row] = await _rows(state["query_id"])
    assert row["price_id"] is not None
    assert row["cost_usd"] == Decimal("0")
    assert row["cost_basis"] == "metered"


async def test_a_node_reporting_nothing_writes_none_with_every_meter_null(
    collection: "CollectionRef",
) -> None:
    """Never zeros: a zero meter under 'none' is one meters_check refuses."""
    state = await _state(collection)
    await traced("plan_query")(_reporting())(state)

    [row] = await _rows(state["query_id"])
    assert row["billing_unit"] == "none"
    assert row["provider"] is None and row["model"] is None
    assert (row["input_tokens"], row["output_tokens"], row["gpu_ms"]) == (None, None, None)
    assert (row["price_id"], row["cost_usd"], row["cost_basis"]) == (None, None, None)
    assert row["input_json"] is None and row["output_json"] is None
    assert not row["input_truncated"] and not row["output_truncated"]


async def test_a_meter_the_unit_does_not_own_is_refused_by_meters_check(
    collection: "CollectionRef",
) -> None:
    """The writer passes the meter through as reported; the schema is the check."""
    state = await _state(collection)
    mismatched = Usage(
        model="qwen2.5:32b",
        provider="ollama",
        billing_unit="tokens",
        input_tokens=10,
        output_tokens=5,
        gpu_ms=400,
        duration_ms=5,
        cost_basis="metered",
    )

    with pytest.raises(IntegrityError, match="query_traces_meters_check"):
        await traced("generate")(_reporting(meter=mismatched))(state)
    assert await _rows(state["query_id"]) == []


async def test_price_id_cost_and_basis_are_written_together_or_not_at_all(
    collection: "CollectionRef",
) -> None:
    state = await _state(collection)
    unpriced = usage("not-a-priced-model")
    for sequence, meter in enumerate((usage(), unpriced, None)):
        await traced("generate")(_reporting(meter=meter))({**state, "sequence": sequence})

    priced_row, unpriced_row, silent_row = await _rows(state["query_id"])
    assert None not in (priced_row["price_id"], priced_row["cost_usd"], priced_row["cost_basis"])
    # A call we could not price: recorded, with its meter, and no cost at all.
    assert unpriced_row["provider"] == "ollama" and unpriced_row["input_tokens"] == 11
    for row in (unpriced_row, silent_row):
        assert (row["price_id"], row["cost_usd"], row["cost_basis"]) == (None, None, None)

    # And the schema refuses the half-written row the writer never produces.
    with pytest.raises(IntegrityError, match="query_traces_priced_check"):
        async with get_engine().begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO query_traces
                        (id, query_id, tenant_id, node_name, sequence, status,
                         started_at, duration_ms, provider, model, billing_unit,
                         price_id, cost_usd, cost_basis)
                    VALUES
                        (:id, :query_id, :tenant_id, 'generate', 99, 'ok',
                         now(), 1, 'ollama', 'qwen2.5:32b', 'tokens',
                         :price_id, NULL, 'metered')
                    """
                ),
                {
                    "id": uuid7(),
                    "query_id": state["query_id"],
                    "tenant_id": state["tenant_id"],
                    "price_id": priced_row["price_id"],
                },
            )


async def test_a_node_that_raises_after_a_call_still_writes_its_meter(
    collection: "CollectionRef",
) -> None:
    """ADR 0016's reason for a context object: spent money survives the exception."""
    state = await _state(collection)
    node = _reporting(meter=usage(), raises=ValueError("verdict did not validate"))

    with pytest.raises(ValueError, match="verdict did not validate"):
        await traced("verify_grounding")(node)(state)

    [row] = await _rows(state["query_id"])
    assert row["status"] == "error"
    assert (row["provider"], row["input_tokens"], row["output_tokens"]) == ("ollama", 11, 7)
    assert row["price_id"] is not None and row["cost_usd"] == Decimal("0")


async def test_payloads_are_stored_as_references(collection: "CollectionRef") -> None:
    state = await _state(collection)
    chunk_id = uuid7()
    node = _reporting(
        payload_in={"query_vector_of": "embed_query"},
        payload_out={"chunks": [{"id": chunk_id, "score": 0.83}]},
    )
    await traced("retrieve")(node)(state)

    [row] = await _rows(state["query_id"])
    assert row["input_json"] == {"query_vector_of": "embed_query"}
    assert row["output_json"] == {"chunks": [{"id": str(chunk_id), "score": 0.83}]}
    assert not row["input_truncated"] and not row["output_truncated"]


async def test_an_oversized_payload_is_capped_valid_json_and_flagged(
    collection: "CollectionRef",
) -> None:
    from prism.config import get_settings

    cap = get_settings().trace_payload_max_bytes
    state = await _state(collection)
    chunks = [{"id": str(uuid7()), "score": 0.5} for _ in range(cap // 20)]
    await traced("retrieve")(_reporting(payload_in={"k": 10}, payload_out={"chunks": chunks}))(
        state
    )

    [row] = await _rows(state["query_id"])
    assert row["output_truncated"] is True
    assert row["input_truncated"] is False
    # jsonb would have refused anything that did not parse; the driver hands
    # back the parsed value.
    kept = row["output_json"]["chunks"]
    assert 0 < len(kept) < len(chunks)
    assert kept == chunks[: len(kept)]
    assert row["input_json"] == {"k": 10}


def _patch_node(monkeypatch: pytest.MonkeyPatch, name: str, meter: Usage) -> None:
    monkeypatch.setattr(f"prism.graph.nodes.{name}", traced(name)(_reporting(meter=meter)))


async def test_the_query_total_is_the_sum_of_its_priced_trace_rows(
    collection: "CollectionRef",
    paid_model: str,
    monkeypatch: pytest.MonkeyPatch,
    stubbed_models: None,
) -> None:
    _patch_node(monkeypatch, "embed_query", usage("nomic-embed-text"))
    _patch_node(
        monkeypatch,
        "generate",
        Usage(
            model=paid_model,
            provider="gemini",
            billing_unit="tokens",
            input_tokens=2_000,
            output_tokens=300,
            gpu_ms=None,
            duration_ms=900,
            cost_basis="metered",
        ),
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="what does a query cost?",
    )

    rows = await _rows(run.query_id)
    priced = [row["cost_usd"] for row in rows if row["price_id"] is not None]
    # plan_query meters too, and prices to zero on the local lane.
    assert len(priced) == 3
    total = await _total(run.query_id)
    assert total == sum(priced, Decimal(0))
    # 2000 * 0.30 / 1e6 + 300 * 2.50 / 1e6, and the local embed adds exactly 0.
    assert total == Decimal("0.00135000")


async def test_one_unpriced_call_makes_the_total_null_not_smaller(
    collection: "CollectionRef", monkeypatch: pytest.MonkeyPatch, stubbed_models: None
) -> None:
    """ADR 0013: a partial sum is wrong in the flattering direction."""
    _patch_node(monkeypatch, "embed_query", usage("nomic-embed-text"))
    _patch_node(monkeypatch, "generate", usage("not-a-priced-model"))

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="what if a call cannot be priced?",
    )
    assert await _total(run.query_id) is None


async def test_a_query_of_only_free_calls_totals_exactly_zero(
    collection: "CollectionRef", stubbed_models: None
) -> None:
    """ADR 0016: a row with no provider made no call and is not an unpriced one.

    The total is 0 — a number — rather than NULL.
    """
    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question="what do the stubs cost?",
    )
    assert await _total(run.query_id) == Decimal("0")


async def test_usage_and_payloads_never_reach_a_checkpoint(
    collection: "CollectionRef", monkeypatch: pytest.MonkeyPatch, stubbed_models: None
) -> None:
    marker = f"trace-only-{uuid7()}"
    question = f"state-{uuid7()}"
    monkeypatch.setattr(
        "prism.graph.nodes.retrieve",
        traced("retrieve")(
            _reporting(
                meter=usage(f"{marker}-model"),
                payload_in={"marker": marker},
                payload_out={"chunks": [{"id": marker, "score": 0.5}]},
            )
        ),
    )

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=question,
    )

    async def occurrences(needle: str) -> int:
        async with get_engine().connect() as conn:
            count: int = (
                await conn.execute(
                    text(
                        """
                        SELECT
                          (SELECT count(*) FROM checkpoints
                            WHERE thread_id = :thread_id
                              AND (checkpoint::text LIKE :like OR metadata::text LIKE :like))
                        + (SELECT count(*) FROM checkpoint_blobs
                            WHERE thread_id = :thread_id
                              AND position(convert_to(:needle, 'UTF8') IN blob) > 0)
                        + (SELECT count(*) FROM checkpoint_writes
                            WHERE thread_id = :thread_id
                              AND position(convert_to(:needle, 'UTF8') IN blob) > 0)
                        """
                    ),
                    {"thread_id": run.thread_id, "needle": needle, "like": f"%{needle}%"},
                )
            ).scalar_one()
            return count

    # The search can see state, so not finding the marker means something.
    assert await occurrences(question) > 0
    assert await occurrences(marker) == 0

    rows = await _rows(run.query_id)
    assert any(row["output_json"] == {"chunks": [{"id": marker, "score": 0.5}]} for row in rows)

    snapshot = await compile_graph(await get_checkpointer()).aget_state(
        {"configurable": {"thread_id": run.thread_id}}
    )
    assert set(snapshot.values) == set(GraphState.__annotations__)
