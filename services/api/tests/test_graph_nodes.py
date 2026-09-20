"""plan_query, embed_query and retrieve, against a real index and a real model.

Marked `ollama`: these run the configured `planner_model` and `embedding_model`,
because a stub cannot tell us whether the pulled model honours Ollama's `format`.
"""

import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from prism.config import get_settings
from prism.db import get_engine
from prism.graph.checkpointer import get_checkpointer
from prism.graph.graph import NODE_SEQUENCE, compile_graph
from prism.graph.nodes import _clean_terms
from prism.graph.run import QueryRun, run_query
from seeding import seed

if TYPE_CHECKING:
    from prism.collections import CollectionRef

# Lexically distinctive, so the lexical half can find it.
CHINCHILLA = "The chinchilla provisioning ratio is twenty tokens per parameter."
HARBOUR = "Unrelated material about harbour logistics and berth scheduling."
QUESTION = "What is the chinchilla provisioning ratio?"

DIM = 768


def _unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


async def _traces(query_id: UUID) -> dict[str, dict[str, Any]]:
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT node_name, sequence, status, error, provider, model,
                       billing_unit, input_tokens, output_tokens, gpu_ms,
                       price_id, cost_usd, cost_basis, input_json, output_json,
                       input_truncated, output_truncated
                  FROM query_traces
                 WHERE query_id = :query_id
                 ORDER BY sequence
                """
            ),
            {"query_id": query_id},
        )
        return {row["node_name"]: dict(row) for row in rows.mappings()}


# ---------------------------------------------------------------- unit tier


def test_terms_are_stripped_deduplicated_and_capped() -> None:
    assert _clean_terms(["Chinchilla", " ratio ", "chinchilla", ""], max_terms=4) == [
        "Chinchilla",
        "ratio",
    ]
    assert _clean_terms(["a", "b", "c", "d", "e"], max_terms=4) == ["a", "b", "c", "d"]
    # A sentence echoed back as a "term" ANDs to nothing, so it is not a term.
    assert _clean_terms(["x" * 65, "ratio"], max_terms=4) == ["ratio"]
    # Internal whitespace collapses; a phrase stays one term.
    assert _clean_terms(["provisioning   ratio"], max_terms=4) == ["provisioning ratio"]


def test_no_terms_is_a_normal_plan() -> None:
    """The lexical half then parses the raw question."""
    assert _clean_terms([], max_terms=4) == []


# --------------------------------------------------------- integration tier


@pytest.fixture
async def run(collection: "CollectionRef") -> QueryRun:
    await seed(collection.collection_id, [(CHINCHILLA, _unit(0)), (HARBOUR, _unit(1))])
    return await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )


@pytest.mark.integration
@pytest.mark.ollama
async def test_the_nodes_that_call_a_model_leave_a_meter(run: QueryRun) -> None:
    rows = await _traces(run.query_id)

    for name in ("plan_query", "embed_query"):
        row = rows[name]
        assert row["status"] == "ok", row["error"]
        assert row["provider"] == "ollama"
        assert row["billing_unit"] == "tokens"
        assert row["model"]
        # Metered, not estimated: Ollama reports its own counts.
        assert row["input_tokens"] is not None
        assert row["cost_basis"] == "metered"
        assert row["gpu_ms"] is None

    # An embedding generates nothing: a structural zero, not a missing reading.
    assert rows["embed_query"]["output_tokens"] == 0
    assert rows["embed_query"]["model"] == get_settings().embedding_model
    assert rows["plan_query"]["output_tokens"] is not None
    assert rows["plan_query"]["model"] == get_settings().planner_model

    # retrieve calls no model, and must stay distinguishable from a node that
    # forgot to report one.
    retrieve = rows["retrieve"]
    assert retrieve["status"] == "ok"
    assert retrieve["provider"] is None and retrieve["model"] is None
    assert retrieve["billing_unit"] == "none"
    assert retrieve["price_id"] is None and retrieve["cost_usd"] is None


@pytest.mark.integration
@pytest.mark.ollama
async def test_embed_query_prices_to_exactly_zero_against_a_real_price_row(
    run: QueryRun,
) -> None:
    """Free is priced, not unpriced (ADR 0013): a NULL price would NULL the total."""
    row = (await _traces(run.query_id))["embed_query"]
    assert row["price_id"] is not None
    assert row["cost_usd"] == Decimal(0)

    async with get_engine().connect() as conn:
        price = (
            (
                await conn.execute(
                    text(
                        "SELECT provider, model, billing_unit, input_per_mtok, "
                        "output_per_mtok FROM model_pricing WHERE id = :id"
                    ),
                    {"id": row["price_id"]},
                )
            )
            .mappings()
            .one()
        )
    assert price["provider"] == "ollama"
    assert price["model"] == get_settings().embedding_model
    assert price["billing_unit"] == "tokens"
    assert price["input_per_mtok"] == 0 and price["output_per_mtok"] == 0

    # The query total is a number, not NULL: every row that named a provider priced.
    async with get_engine().connect() as conn:
        total = (
            await conn.execute(
                text("SELECT total_cost_usd FROM queries WHERE id = :id"),
                {"id": run.query_id},
            )
        ).scalar_one()
    assert total == Decimal(0)


@pytest.mark.integration
@pytest.mark.ollama
async def test_embed_query_records_the_vectors_shape_and_never_the_vector(
    run: QueryRun,
) -> None:
    row = (await _traces(run.query_id))["embed_query"]
    assert row["input_json"] == {"question": QUESTION}
    assert row["output_json"]["dim"] == get_settings().embedding_dim
    # A zero-norm vector retrieves arbitrary neighbours.
    assert row["output_json"]["norm"] > 0
    assert set(row["output_json"]) == {"dim", "norm"}
    assert not row["output_truncated"]


@pytest.mark.integration
@pytest.mark.ollama
async def test_plan_query_records_the_terms_and_the_parameters_it_pinned(
    run: QueryRun,
) -> None:
    row = (await _traces(run.query_id))["plan_query"]
    settings = get_settings()

    assert row["input_json"] == {"question": QUESTION}
    assert row["output_json"]["params"] == {
        "k": settings.retrieval_top_k,
        "candidate_k": settings.retrieval_candidate_k,
        "rrf_k": settings.rrf_k,
        "rerank_score_floor": settings.rerank_score_floor,
    }
    terms = row["output_json"]["terms"]
    assert isinstance(terms, list)
    assert len(terms) <= settings.planner_max_terms
    assert all(isinstance(term, str) and term.strip() for term in terms)


@pytest.mark.integration
@pytest.mark.ollama
async def test_retrieve_records_ids_and_positions_and_no_chunk_text(
    run: QueryRun,
) -> None:
    row = (await _traces(run.query_id))["retrieve"]

    assert row["input_json"]["k"] == get_settings().retrieval_top_k
    assert "terms" in row["input_json"]

    candidates = row["output_json"]
    assert candidates, "the seeded collection has chunks; fusion returned none"
    for candidate in candidates:
        assert set(candidate) == {
            "chunk_id",
            "document_id",
            "rank",
            "vector_rank",
            "lexical_rank",
        }
        UUID(candidate["chunk_id"])
    assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))

    # References, not copies (ADR 0012). The chunks are rows in `chunks` already.
    serialized = json.dumps(candidates)
    assert "chinchilla" not in serialized.casefold()
    assert "harbour" not in serialized.casefold()


@pytest.mark.integration
@pytest.mark.ollama
async def test_empty_retrieval_is_ok_and_the_run_still_refuses(
    collection: "CollectionRef",
) -> None:
    """A collection with nothing in it is a refusal, not an error."""
    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    row = (await _traces(run.query_id))["retrieve"]
    assert row["status"] == "ok"
    assert row["error"] is None
    assert row["output_json"] == []

    assert run.status == "refused"
    assert run.refusal_reason == "no_relevant_evidence"

    async with get_engine().connect() as conn:
        final = (
            (
                await conn.execute(
                    text(
                        "SELECT status, refusal_reason, citation_count FROM queries WHERE id = :id"
                    ),
                    {"id": run.query_id},
                )
            )
            .mappings()
            .one()
        )
    assert final["status"] == "refused"
    assert final["refusal_reason"] == "no_relevant_evidence"
    assert final["citation_count"] == 0


@pytest.mark.integration
@pytest.mark.ollama
async def test_a_run_cannot_retrieve_another_tenants_chunks(
    collection: "CollectionRef",
    other_collection: "CollectionRef",
) -> None:
    """The scope is a predicate inside the scan, not a filter over its results.

    The other tenant holds the better match by both halves, so a post-filter
    would have given those rows the top-k slots and then dropped them.
    """
    await seed(collection.collection_id, [(HARBOUR, _unit(1))])
    await seed(
        other_collection.collection_id,
        [(CHINCHILLA, _unit(0)), (CHINCHILLA, _unit(2)), (CHINCHILLA, _unit(3))],
    )

    async with get_engine().connect() as conn:
        theirs = {
            row[0]
            for row in await conn.execute(
                text("SELECT id FROM chunks WHERE tenant_id = :tenant_id"),
                {"tenant_id": other_collection.tenant_id},
            )
        }
    assert theirs, "the other tenant's chunks were not seeded"

    run = await run_query(
        tenant_id=collection.tenant_id,
        collection_id=collection.collection_id,
        question=QUESTION,
    )

    candidates = (await _traces(run.query_id))["retrieve"]["output_json"]
    retrieved = {UUID(candidate["chunk_id"]) for candidate in candidates}
    assert retrieved and retrieved.isdisjoint(theirs)


@pytest.mark.integration
async def test_naming_another_tenants_collection_never_starts_a_run(
    collection: "CollectionRef",
    other_collection: "CollectionRef",
) -> None:
    """The composite foreign key refuses it at mint time, before any node runs.

    No model is called, so this one needs no `ollama` marker.
    """
    await seed(other_collection.collection_id, [(CHINCHILLA, _unit(0))])

    with pytest.raises(IntegrityError, match="queries_collection_tenant_fkey"):
        await run_query(
            tenant_id=collection.tenant_id,
            collection_id=other_collection.collection_id,
            question=QUESTION,
        )

    async with get_engine().connect() as conn:
        started = (
            await conn.execute(
                text("SELECT count(*) FROM queries WHERE question = :question"),
                {"question": QUESTION},
            )
        ).scalar_one()
        traced_rows = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM query_traces WHERE tenant_id = :tenant_id",
                ),
                {"tenant_id": collection.tenant_id},
            )
        ).scalar_one()
    assert started == 0
    assert traced_rows == 0


@pytest.mark.integration
@pytest.mark.ollama
async def test_sequences_stay_contiguous_and_the_checkpoint_round_trips_the_state(
    run: QueryRun,
) -> None:
    rows = await _traces(run.query_id)
    assert [row["sequence"] for row in rows.values()] == list(range(1, len(NODE_SEQUENCE) + 1))
    assert list(rows) == list(NODE_SEQUENCE)

    graph = compile_graph(await get_checkpointer())
    values = (await graph.aget_state({"configurable": {"thread_id": run.thread_id}})).values

    embedding = values["query_embedding"]
    assert isinstance(embedding, list)
    assert len(embedding) == get_settings().embedding_dim
    assert all(isinstance(x, float) for x in embedding)

    assert values["search_params"]["k"] == get_settings().retrieval_top_k
    assert isinstance(values["search_terms"], list)

    candidates = values["candidates"]
    assert candidates
    # UUIDs survive the checkpoint as UUIDs.
    assert all(isinstance(candidate["chunk_id"], UUID) for candidate in candidates)
    assert [candidate["rank"] for candidate in candidates] == list(range(1, len(candidates) + 1))
