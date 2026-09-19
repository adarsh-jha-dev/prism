"""Pricing: the arithmetic is unit; resolving a row is integration."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from prism.chat import Usage
from prism.db import get_engine
from prism.pricing import cost_of, price


def _usage(**overrides: object) -> Usage:
    fields: dict[str, object] = {
        "model": "gemini-3.6-flash",
        "provider": "gemini",
        "billing_unit": "tokens",
        "input_tokens": 1_000,
        "output_tokens": 500,
        "gpu_ms": None,
        "duration_ms": 10,
        "cost_basis": "metered",
    }
    return Usage(**(fields | overrides))  # type: ignore[arg-type]


def test_tokens_are_priced_per_million_as_published() -> None:
    cost = cost_of(
        _usage(),
        input_per_mtok=Decimal("0.30"),
        output_per_mtok=Decimal("2.50"),
        usd_per_gpu_hour=None,
    )
    # 1000 * 0.30 / 1e6 + 500 * 2.50 / 1e6
    assert cost == Decimal("0.00155000")


def test_gpu_time_is_priced_per_hour_as_published() -> None:
    usage = _usage(
        provider="ollama-cloud",
        billing_unit="gpu_ms",
        input_tokens=None,
        output_tokens=None,
        gpu_ms=1_800,
    )
    cost = cost_of(
        usage, input_per_mtok=None, output_per_mtok=None, usd_per_gpu_hour=Decimal("2.00")
    )
    assert cost == Decimal("0.00100000")


def test_cost_is_rounded_to_the_columns_eight_places() -> None:
    cost = cost_of(
        _usage(input_tokens=1, output_tokens=0),
        input_per_mtok=Decimal("0.075"),
        output_per_mtok=Decimal("0.30"),
        usd_per_gpu_hour=None,
    )
    assert cost == Decimal("0.00000008")


def test_a_missing_meter_against_a_real_rate_is_unpriced_not_free() -> None:
    cost = cost_of(
        _usage(output_tokens=None),
        input_per_mtok=Decimal("0.30"),
        output_per_mtok=Decimal("2.50"),
        usd_per_gpu_hour=None,
    )
    assert cost is None


def test_a_missing_meter_against_a_zero_rate_is_zero() -> None:
    cost = cost_of(
        _usage(provider="ollama", model="qwen2.5:32b", input_tokens=None),
        input_per_mtok=Decimal("0"),
        output_per_mtok=Decimal("0"),
        usd_per_gpu_hour=None,
    )
    assert cost == Decimal("0")


@pytest.mark.integration
async def test_a_local_call_prices_to_exactly_zero_with_a_price_row() -> None:
    """ADR 0013: free is priced. A NULL price_id here would mean unpriced."""
    local = _usage(provider="ollama", model="qwen2.5:32b")
    async with get_engine().connect() as conn:
        priced = await price(conn, local, at=datetime.now(UTC))
    assert priced is not None
    assert priced.price_id is not None
    assert priced.cost_usd == Decimal("0")
    assert priced.cost_basis == "metered"


@pytest.mark.integration
async def test_the_in_process_lane_is_priced_with_no_meter() -> None:
    rerank = _usage(
        provider="in-process",
        model="bge-reranker-v2-m3",
        billing_unit="none",
        input_tokens=None,
        output_tokens=None,
    )
    async with get_engine().connect() as conn:
        priced = await price(conn, rerank, at=datetime.now(UTC))
    assert priced is not None and priced.cost_usd == Decimal("0")


@pytest.mark.integration
async def test_the_basis_is_carried_from_the_meter() -> None:
    estimated = _usage(provider="ollama", model="qwen2.5:32b", cost_basis="estimated")
    async with get_engine().connect() as conn:
        priced = await price(conn, estimated, at=datetime.now(UTC))
    assert priced is not None and priced.cost_basis == "estimated"


@pytest.mark.integration
async def test_no_row_in_force_is_unpriced() -> None:
    async with get_engine().connect() as conn:
        # Before the seed migration's effective_from.
        before = await price(
            conn,
            _usage(provider="ollama", model="qwen2.5:32b"),
            at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        unknown = await price(
            conn, _usage(provider="ollama", model="not-a-model"), at=datetime.now(UTC)
        )
        # The unit is part of the key: tokens cannot be priced as GPU-time.
        wrong_unit = await price(
            conn,
            _usage(
                provider="ollama",
                model="qwen2.5:32b",
                billing_unit="gpu_ms",
                input_tokens=None,
                output_tokens=None,
                gpu_ms=10,
            ),
            at=datetime.now(UTC),
        )
    assert (before, unknown, wrong_unit) == (None, None, None)
