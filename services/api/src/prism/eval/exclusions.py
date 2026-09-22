"""Which graph runs may not be counted.

A degraded run is a failed observation, not a cheap one (ADR 0011): a query that
fell back to fusion order is not measuring the optimized path, so eval and
benchmark aggregates exclude it rather than averaging it in.

Degradation is a property of a node's row, not of the query row — the query
succeeded, which is the whole problem. Two shapes count, and they are the only
two: a node that recorded `fallback` on its output, and a node that errored.
"""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.db import get_engine

__all__ = ["degraded_query_ids"]

_DEGRADED = text(
    """
    SELECT DISTINCT query_id
      FROM query_traces
     WHERE query_id = ANY(CAST(:query_ids AS uuid[]))
       AND (status = 'error' OR output_json->>'fallback' = 'true')
    """
)


async def degraded_query_ids(
    query_ids: Sequence[UUID], *, engine: AsyncEngine | None = None
) -> set[UUID]:
    """The subset of `query_ids` that any node degraded on.

    Callers filter with this before aggregating; nothing here deletes or hides a
    run. The rows stay readable in the dashboard, which is where a degraded run
    is worth seeing.
    """
    if not query_ids:
        return set()

    async with (engine or get_engine()).connect() as conn:
        rows = await conn.execute(
            _DEGRADED, {"query_ids": [str(query_id) for query_id in query_ids]}
        )
        return {row.query_id for row in rows}
