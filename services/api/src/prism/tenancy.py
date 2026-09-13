"""Creating tenants and collections.

The routes in `prism.api.routes.tenants` and `.collections` are the only network
path to these, but the eval harness needs the same rows without standing up a
server. Both call this, so there is exactly one place that writes them and one
definition of what a valid tenant or collection is.

Nothing here authorizes anything. Callers decide who may create what — the
routes do it with `admin_token` and key scopes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.auth import GeneratedKey, Scope, generate_key
from prism.collections import CollectionRef
from prism.core.ids import uuid7
from prism.db import get_engine

__all__ = [
    "AlreadyExistsError",
    "CollectionRow",
    "TenantRow",
    "create_collection",
    "create_tenant",
    "ensure_collection",
    "ensure_tenant",
    "list_collections",
    "list_tenants",
    "read_collection",
    "read_tenant",
]


class AlreadyExistsError(ValueError):
    """A tenant or collection with that name is already present."""


@dataclass(frozen=True)
class TenantRow:
    id: UUID
    name: str
    created_at: datetime


@dataclass(frozen=True)
class CollectionRow:
    id: UUID
    tenant_id: UUID
    name: str
    embedding_model: str
    embedding_dim: int
    abstention_threshold: float
    created_at: datetime

    @property
    def ref(self) -> CollectionRef:
        return CollectionRef(tenant_id=self.tenant_id, collection_id=self.id)


_INSERT_TENANT = text("INSERT INTO tenants (id, name) VALUES (:id, :name)")
_INSERT_KEY = text(
    "INSERT INTO api_keys (id, tenant_id, key_hash, name, key_prefix, scopes) "
    "VALUES (:id, :tenant_id, :key_hash, :name, :key_prefix, CAST(:scopes AS text[]))"
)
_SELECT_TENANT = text("SELECT id, name, created_at FROM tenants WHERE id = :id")
_SELECT_TENANT_BY_NAME = text("SELECT id, name, created_at FROM tenants WHERE name = :name")
_LIST_TENANTS = text("SELECT id, name, created_at FROM tenants ORDER BY id")

_COLLECTION_COLUMNS = (
    "id, tenant_id, name, embedding_model, embedding_dim, abstention_threshold, created_at"
)
_INSERT_COLLECTION = text(
    "INSERT INTO collections (id, tenant_id, name, embedding_model, embedding_dim) "
    "VALUES (:id, :tenant_id, :name, :embedding_model, :embedding_dim)"
)
_SELECT_COLLECTION = text(
    f"SELECT {_COLLECTION_COLUMNS} FROM collections WHERE id = :id AND tenant_id = :tenant_id"
)
_SELECT_COLLECTION_BY_NAME = text(
    f"SELECT {_COLLECTION_COLUMNS} FROM collections WHERE tenant_id = :tenant_id AND name = :name"
)
_LIST_COLLECTIONS = text(
    f"SELECT {_COLLECTION_COLUMNS} FROM collections WHERE tenant_id = :tenant_id ORDER BY id"
)


def _tenant(row: Any) -> TenantRow:
    return TenantRow(id=row.id, name=row.name, created_at=row.created_at)


def _collection(row: Any) -> CollectionRow:
    return CollectionRow(
        id=row.id,
        tenant_id=row.tenant_id,
        name=row.name,
        embedding_model=row.embedding_model,
        embedding_dim=row.embedding_dim,
        abstention_threshold=row.abstention_threshold,
        created_at=row.created_at,
    )


async def create_tenant(
    name: str, *, engine: AsyncEngine | None = None
) -> tuple[TenantRow, GeneratedKey]:
    """A tenant and its first key, which carries every scope.

    The key's plaintext is on the returned object and nowhere else.
    """
    tenant_id = uuid7()
    key = generate_key()
    try:
        async with (engine or get_engine()).begin() as conn:
            await conn.execute(_INSERT_TENANT, {"id": tenant_id, "name": name})
            await conn.execute(
                _INSERT_KEY,
                {
                    "id": uuid7(),
                    "tenant_id": tenant_id,
                    "key_hash": key.key_hash,
                    "name": "initial key",
                    "key_prefix": key.prefix,
                    "scopes": "{" + ",".join(s.value for s in Scope) + "}",
                },
            )
            row = (await conn.execute(_SELECT_TENANT, {"id": tenant_id})).one()
    except IntegrityError as exc:
        raise AlreadyExistsError(f"tenant {name!r} already exists") from exc
    return _tenant(row), key


async def ensure_tenant(name: str, *, engine: AsyncEngine | None = None) -> TenantRow:
    """The tenant of that name, created if absent. For dev helpers, not routes."""
    engine = engine or get_engine()
    async with engine.connect() as conn:
        row = (await conn.execute(_SELECT_TENANT_BY_NAME, {"name": name})).first()
    if row is not None:
        return _tenant(row)
    try:
        tenant, _ = await create_tenant(name, engine=engine)
    except AlreadyExistsError:
        # Raced with another creator; theirs is as good as ours.
        async with engine.connect() as conn:
            return _tenant((await conn.execute(_SELECT_TENANT_BY_NAME, {"name": name})).one())
    return tenant


async def read_tenant(tenant_id: UUID, *, engine: AsyncEngine | None = None) -> TenantRow | None:
    async with (engine or get_engine()).connect() as conn:
        row = (await conn.execute(_SELECT_TENANT, {"id": tenant_id})).first()
    return None if row is None else _tenant(row)


async def list_tenants(*, engine: AsyncEngine | None = None) -> list[TenantRow]:
    async with (engine or get_engine()).connect() as conn:
        return [_tenant(row) for row in (await conn.execute(_LIST_TENANTS)).all()]


async def create_collection(
    *,
    tenant_id: UUID,
    name: str,
    embedding_model: str,
    embedding_dim: int,
    engine: AsyncEngine | None = None,
) -> CollectionRow:
    collection_id = uuid7()
    try:
        async with (engine or get_engine()).begin() as conn:
            await conn.execute(
                _INSERT_COLLECTION,
                {
                    "id": collection_id,
                    "tenant_id": tenant_id,
                    "name": name,
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                },
            )
            row = (
                await conn.execute(
                    _SELECT_COLLECTION, {"id": collection_id, "tenant_id": tenant_id}
                )
            ).one()
    except IntegrityError as exc:
        raise AlreadyExistsError(f"collection {name!r} already exists") from exc
    return _collection(row)


async def ensure_collection(
    *,
    tenant_id: UUID,
    name: str,
    embedding_model: str,
    embedding_dim: int,
    engine: AsyncEngine | None = None,
) -> CollectionRow:
    """The collection of that name under `tenant_id`, created if absent."""
    engine = engine or get_engine()
    async with engine.connect() as conn:
        row = (
            await conn.execute(_SELECT_COLLECTION_BY_NAME, {"tenant_id": tenant_id, "name": name})
        ).first()
    if row is not None:
        return _collection(row)
    try:
        return await create_collection(
            tenant_id=tenant_id,
            name=name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            engine=engine,
        )
    except AlreadyExistsError:
        async with engine.connect() as conn:
            return _collection(
                (
                    await conn.execute(
                        _SELECT_COLLECTION_BY_NAME, {"tenant_id": tenant_id, "name": name}
                    )
                ).one()
            )


async def read_collection(
    collection_id: UUID, *, tenant_id: UUID, engine: AsyncEngine | None = None
) -> CollectionRow | None:
    """None for another tenant's collection, the same as for one that is absent."""
    async with (engine or get_engine()).connect() as conn:
        row = (
            await conn.execute(_SELECT_COLLECTION, {"id": collection_id, "tenant_id": tenant_id})
        ).first()
    return None if row is None else _collection(row)


async def list_collections(
    *, tenant_id: UUID, engine: AsyncEngine | None = None
) -> list[CollectionRow]:
    async with (engine or get_engine()).connect() as conn:
        rows = (await conn.execute(_LIST_COLLECTIONS, {"tenant_id": tenant_id})).all()
    return [_collection(row) for row in rows]
