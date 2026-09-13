"""Chunks with known vectors, written straight to the table.

Retrieval tests want a ranking that is arithmetic rather than a property of any
embedding model, so they seed vectors instead of ingesting documents.
"""

import json
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import text

from prism.core.ids import uuid7
from prism.db import get_engine

# tenant_id is copied from the parent rather than passed, so a seeded row cannot
# disagree with it — the same shape the ingestion pipeline uses.
_INSERT_DOCUMENT = text(
    "INSERT INTO documents (id, collection_id, tenant_id, filename, mime_type, status, "
    " ingested_at) "
    "SELECT :id, c.id, c.tenant_id, :filename, 'application/pdf', 'ready', now() "
    "FROM collections c WHERE c.id = :collection_id"
)
_INSERT_CHUNK = text(
    "INSERT INTO chunks (id, document_id, collection_id, tenant_id, content, chunk_type, "
    " page_number, embedding, metadata) "
    "SELECT :id, d.id, d.collection_id, d.tenant_id, :content, 'text', :page, "
    "       CAST(:embedding AS vector), CAST(:metadata AS jsonb) "
    "FROM documents d WHERE d.id = :document_id"
)


async def seed(
    collection_id: UUID,
    chunks: Sequence[tuple[str, list[float]]],
    *,
    filename: str = "seed.pdf",
) -> UUID:
    """One ready document and its chunks, with the vectors given."""
    document_id = uuid7()
    async with get_engine().begin() as conn:
        await conn.execute(
            _INSERT_DOCUMENT,
            {"id": document_id, "collection_id": collection_id, "filename": filename},
        )
        await conn.execute(
            _INSERT_CHUNK,
            [
                {
                    "id": uuid7(),
                    "document_id": document_id,
                    "content": content,
                    "page": index + 1,
                    "embedding": str(vector),
                    "metadata": json.dumps({"chunk_index": index}),
                }
                for index, (content, vector) in enumerate(chunks)
            ],
        )
    return document_id
