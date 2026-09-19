"""A meter reading priced against model_pricing.

Rationale: docs/decisions/0013-cost-basis-and-model-pricing.md, amended by 0016.

None means the reading could not be priced: no row in force for that provider,
model and unit at that moment, or a meter missing against a non-zero rate. It is
never zero — unpriced is not free.
"""

from datetime import datetime
from decimal import Decimal
from typing import NamedTuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from prism.chat.base import Usage

__all__ = ["Priced", "cost_of", "price"]

# query_traces.cost_usd is numeric(14,8).
_COST_PLACES = Decimal("0.00000001")
_TOKENS_PER_MTOK = 1_000_000
_MS_PER_HOUR = 3_600_000

# Ranges are [effective_from, effective_to), matching the exclusion constraint's
# tstzrange default, so a closing moment belongs to the next row.
_LOOKUP = text(
    """
    SELECT id, input_per_mtok, output_per_mtok, usd_per_gpu_hour
      FROM model_pricing
     WHERE provider = :provider AND model = :model AND billing_unit = :billing_unit
       AND effective_from <= :at
       AND (effective_to IS NULL OR effective_to > :at)
    """
)


class Priced(NamedTuple):
    """Written as a unit: migration 0009's priced_check ties the three together."""

    price_id: UUID
    cost_usd: Decimal
    cost_basis: str


async def price(conn: AsyncConnection, usage: Usage, *, at: datetime) -> Priced | None:
    row = (
        await conn.execute(
            _LOOKUP,
            {
                "provider": usage.provider,
                "model": usage.model,
                "billing_unit": usage.billing_unit,
                "at": at,
            },
        )
    ).one_or_none()
    if row is None:
        return None

    cost = cost_of(
        usage,
        input_per_mtok=row.input_per_mtok,
        output_per_mtok=row.output_per_mtok,
        usd_per_gpu_hour=row.usd_per_gpu_hour,
    )
    if cost is None:
        return None
    return Priced(price_id=row.id, cost_usd=cost, cost_basis=usage.cost_basis)


def cost_of(
    usage: Usage,
    *,
    input_per_mtok: Decimal | None,
    output_per_mtok: Decimal | None,
    usd_per_gpu_hour: Decimal | None,
) -> Decimal | None:
    """Rates as published; model_pricing_rates_check guarantees the unit's are set."""
    match usage.billing_unit:
        case "tokens":
            parts = [
                _metered(usage.input_tokens, input_per_mtok, _TOKENS_PER_MTOK),
                _metered(usage.output_tokens, output_per_mtok, _TOKENS_PER_MTOK),
            ]
        case "gpu_ms":
            parts = [_metered(usage.gpu_ms, usd_per_gpu_hour, _MS_PER_HOUR)]
        case "none":
            parts = []

    total = Decimal(0)
    for part in parts:
        if part is None:
            return None
        total += part
    return total.quantize(_COST_PLACES)


def _metered(meter: int | None, rate: Decimal | None, per: int) -> Decimal | None:
    if rate is None:
        return None
    # Zero times a count the provider left out is still zero.
    if rate == 0:
        return Decimal(0)
    if meter is None:
        return None
    return rate * meter / per
