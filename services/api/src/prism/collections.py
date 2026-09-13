"""What a collection will accept, and from which provider.

Vectors from one model are meaningless against an index built by another: the
neighbours come back, they just mean nothing. Both the write path and the read
path check this before doing any work, so the check lives here rather than in
either of them.

`assert_compatible` takes a connection, for callers already inside one.
`assert_collection_compatible` opens its own.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from prism.db import get_engine
from prism.embeddings import EmbeddingProvider

__all__ = [
    "CollectionError",
    "CollectionNotFoundError",
    "CollectionRef",
    "EmbeddingModelMismatchError",
    "assert_collection_compatible",
    "assert_compatible",
]

# tenant_id is part of the lookup, not a check after it: a collection owned by
# another tenant must be indistinguishable from one that does not exist.
_SELECT_COLLECTION = text(
    "SELECT embedding_model, embedding_dim FROM collections "
    "WHERE id = :id AND tenant_id = :tenant_id"
)


@dataclass(frozen=True)
class CollectionRef:
    """A collection and the tenant that owns it.

    Kept together because every scoped query needs both: a collection id on its
    own is not enough to address anything safely.
    """

    tenant_id: UUID
    collection_id: UUID


class CollectionError(RuntimeError):
    """A collection cannot serve the request as addressed."""


class CollectionNotFoundError(CollectionError):
    """No such collection. The caller addressed something that does not exist."""


class EmbeddingModelMismatchError(CollectionError):
    """The provider does not produce the vectors this collection was built for."""


async def assert_compatible(
    conn: AsyncConnection,
    collection_id: UUID,
    provider: EmbeddingProvider,
    *,
    tenant_id: UUID,
) -> None:
    """Raises CollectionNotFoundError or EmbeddingModelMismatchError."""
    row = (
        await conn.execute(_SELECT_COLLECTION, {"id": collection_id, "tenant_id": tenant_id})
    ).first()
    if row is None:
        raise CollectionNotFoundError(f"collection {collection_id} does not exist")

    model, dim = row
    if (model, dim) != (provider.model, provider.dim):
        raise EmbeddingModelMismatchError(
            f"collection {collection_id} is {model}/{dim}-dim, "
            f"provider is {provider.model}/{provider.dim}-dim"
        )


async def assert_collection_compatible(
    collection_id: UUID,
    provider: EmbeddingProvider,
    *,
    tenant_id: UUID,
    engine: AsyncEngine | None = None,
) -> None:
    async with (engine or get_engine()).connect() as conn:
        await assert_compatible(conn, collection_id, provider, tenant_id=tenant_id)
