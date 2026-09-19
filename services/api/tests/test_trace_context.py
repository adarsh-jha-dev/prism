"""Unit: what a node reports beside its state update, and how payloads are capped."""

import json
from uuid import UUID

import pytest

from prism.graph.trace import TraceContext, cap_payload
from stub_chat import usage


def test_a_node_reports_one_meter_at_most() -> None:
    """One row, one provider column: a second meter would be misattributed."""
    trace = TraceContext()
    trace.record_usage(usage())
    with pytest.raises(RuntimeError, match="already recorded"):
        trace.record_usage(usage("llama3.1:8b"))
    assert trace.usage is not None and trace.usage.model == "qwen2.5:32b"


def test_a_context_nobody_reported_into_is_empty() -> None:
    trace = TraceContext()
    assert (trace.usage, trace.input, trace.output) == (None, None, None)


def test_no_payload_is_sql_null_not_json_null() -> None:
    assert cap_payload(None, 1024) == (None, False)


def test_a_payload_under_the_cap_is_stored_whole() -> None:
    chunk_id = UUID("01920000-0000-7000-8000-000000000001")
    encoded, truncated = cap_payload({"chunks": [{"id": chunk_id, "score": 0.5}]}, 1024)
    assert not truncated
    assert encoded is not None
    assert json.loads(encoded) == {"chunks": [{"id": str(chunk_id), "score": 0.5}]}


def test_an_oversized_payload_is_cut_to_valid_json_and_flagged() -> None:
    chunks = [{"id": f"{i:036d}", "score": i / 1000} for i in range(500)]
    encoded, truncated = cap_payload({"query": "q", "chunks": chunks}, 1024)

    assert truncated
    assert encoded is not None and len(encoded.encode()) <= 1024
    parsed = json.loads(encoded)
    # Cut from the list of references, which keeps its leading entries whole.
    assert parsed["query"] == "q"
    assert 0 < len(parsed["chunks"]) < len(chunks)
    assert parsed["chunks"] == chunks[: len(parsed["chunks"])]


def test_a_single_oversized_string_is_shortened_not_split_mid_json() -> None:
    encoded, truncated = cap_payload({"plan": "x" * 5000}, 512)
    assert truncated
    assert encoded is not None and len(encoded.encode()) <= 512
    assert json.loads(encoded)["plan"].startswith("x")


def test_a_payload_with_nothing_left_to_cut_records_its_size() -> None:
    wide = {f"key-{i}": i for i in range(200)}
    encoded, truncated = cap_payload(wide, 256)
    assert truncated
    assert encoded is not None
    assert json.loads(encoded) == {"omitted_bytes": len(json.dumps(wide, separators=(",", ":")))}


def test_nan_is_refused_rather_than_stored() -> None:
    """jsonb cannot hold NaN, and a NaN score is a bug upstream."""
    with pytest.raises(ValueError):
        cap_payload({"score": float("nan")}, 1024)
