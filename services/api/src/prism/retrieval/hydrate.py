"""Chunk text by id, under the same predicate that retrieved the id.

State and trace payloads carry references, not chunk text (ADR 0012, ADR 0017),
so `grade_docs`, `rerank` and `generate` each fetch the text they need. This is
a second read of tenant data and gets the same treatment as the first: tenant
and collection are predicates inside the query, never a filter over its results.
A bare lookup by chunk id would hand back everything the ANN predicate bought.

Hydration returns what it finds, in the order asked for. A chunk deleted or
re-ingested since retrieval hydrates short rather than raising: state is not an
archive, and what was cited is snapshotted in `query_citations` (ADR 0017).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.db import get_engine

__all__ = ["HydratedChunk", "hydrate_chunks"]

# The id list is cast rather than passed as uuid[]: the driver sends a text
# array and the cast is where it becomes uuids, or fails loudly.
_HYDRATE = text(
    """
    SELECT c.id AS chunk_id,
           c.document_id,
           d.filename,
           c.content,
           c.page_number,
           (c.metadata->>'chunk_index')::int AS chunk_index
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.tenant_id = :tenant_id
      AND c.collection_id = :collection_id
      AND c.id = ANY(CAST(:chunk_ids AS uuid[]))
    """
)


@dataclass(frozen=True)
class HydratedChunk:
    """One chunk's text and the fields a citation is built from."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    page_number: int | None
    chunk_index: int | None


async def hydrate_chunks(
    chunk_ids: Sequence[UUID],
    *,
    tenant_id: UUID,
    collection_id: UUID,
    engine: AsyncEngine | None = None,
) -> list[HydratedChunk]:
    """The chunks among `chunk_ids` this tenant owns in this collection."""
    if not chunk_ids:
        return []

    async with (engine or get_engine()).connect() as conn:
        rows = await conn.execute(
            _HYDRATE,
            {
                "tenant_id": tenant_id,
                "collection_id": collection_id,
                "chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
            },
        )
        found = {
            row["chunk_id"]: HydratedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                filename=row["filename"],
                content=row["content"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
            )
            for row in rows.mappings()
        }

    # Caller order is candidate order; a missing id is dropped, never padded.
    return [found[chunk_id] for chunk_id in chunk_ids if chunk_id in found]
