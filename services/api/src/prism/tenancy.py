"""Creating tenants and collections.

The routes in `prism.api.routes.tenants` and `.collections` are the only network
path to these, but the eval harness needs the same rows without standing up a
server. Both call this, so there is exactly one place that writes them and one
definition of what a valid tenant or collection is.

Nothing here authorizes anything. Callers decide who may create what — the
routes do it with `admin_token` and key scopes.
"""

from collections.abc import Iterable
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
    "DocumentRow",
    "KeyRow",
    "LastAdminKeyError",
    "TenantRow",
    "create_collection",
    "create_key",
    "create_tenant",
    "ensure_collection",
    "ensure_tenant",
    "list_collections",
    "list_documents",
    "list_keys",
    "list_tenants",
    "read_collection",
    "read_document",
    "read_tenant",
    "revoke_key",
]


class AlreadyExistsError(ValueError):
    """A tenant or collection with that name is already present."""


class LastAdminKeyError(ValueError):
    """The revoke would leave the tenant no unrevoked, non-expiring admin key."""


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
    "INSERT INTO api_keys (id, tenant_id, key_hash, name, key_prefix, scopes, expires_at) "
    "VALUES (:id, :tenant_id, :key_hash, :name, :key_prefix, CAST(:scopes AS text[]), "
    " :expires_at)"
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
                _key_params(uuid7(), tenant_id, key, name="initial key", scopes=Scope),
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


@dataclass(frozen=True)
class KeyRow:
    id: UUID
    tenant_id: UUID
    name: str
    key_prefix: str
    scopes: tuple[Scope, ...]
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None
    expires_at: datetime | None


# key_hash is deliberately absent: nothing that reads keys back can return it.
_KEY_COLUMNS = (
    "id, tenant_id, name, key_prefix, scopes, created_at, last_used_at, revoked_at, expires_at"
)
_SELECT_KEY = text(f"SELECT {_KEY_COLUMNS} FROM api_keys WHERE id = :id AND tenant_id = :tenant_id")
_LIST_KEYS = text(f"SELECT {_KEY_COLUMNS} FROM api_keys WHERE tenant_id = :tenant_id ORDER BY id")
_LOCK_TENANT = text("SELECT id FROM tenants WHERE id = :id FOR NO KEY UPDATE")
_COUNT_OTHER_PERMANENT_ADMIN_KEYS = text(
    "SELECT count(*) FROM api_keys "
    "WHERE tenant_id = :tenant_id AND id <> :id AND revoked_at IS NULL "
    "AND expires_at IS NULL AND 'admin' = ANY(scopes)"
)
_REVOKE_KEY = text(
    "UPDATE api_keys SET revoked_at = now() WHERE id = :id AND tenant_id = :tenant_id "
    f"RETURNING {_KEY_COLUMNS}"
)


def _key(row: Any) -> KeyRow:
    return KeyRow(
        id=row.id,
        tenant_id=row.tenant_id,
        name=row.name,
        key_prefix=row.key_prefix,
        scopes=tuple(Scope(s) for s in row.scopes),
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
        expires_at=row.expires_at,
    )


def _key_params(
    key_id: UUID,
    tenant_id: UUID,
    key: GeneratedKey,
    *,
    name: str,
    scopes: Iterable[Scope],
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    chosen = set(scopes)
    return {
        "id": key_id,
        "tenant_id": tenant_id,
        "key_hash": key.key_hash,
        "name": name,
        "key_prefix": key.prefix,
        "scopes": "{" + ",".join(s.value for s in Scope if s in chosen) + "}",
        "expires_at": expires_at,
    }


async def create_key(
    *,
    tenant_id: UUID,
    name: str,
    scopes: Iterable[Scope],
    expires_at: datetime | None = None,
    engine: AsyncEngine | None = None,
) -> tuple[KeyRow, GeneratedKey]:
    """A new key under `tenant_id`. The plaintext is on the GeneratedKey only."""
    key_id = uuid7()
    key = generate_key()
    params = _key_params(key_id, tenant_id, key, name=name, scopes=scopes, expires_at=expires_at)
    async with (engine or get_engine()).begin() as conn:
        await conn.execute(_INSERT_KEY, params)
        row = (await conn.execute(_SELECT_KEY, {"id": key_id, "tenant_id": tenant_id})).one()
    return _key(row), key


async def list_keys(*, tenant_id: UUID, engine: AsyncEngine | None = None) -> list[KeyRow]:
    async with (engine or get_engine()).connect() as conn:
        rows = (await conn.execute(_LIST_KEYS, {"tenant_id": tenant_id})).all()
    return [_key(row) for row in rows]


async def revoke_key(
    key_id: UUID, *, tenant_id: UUID, engine: AsyncEngine | None = None
) -> KeyRow | None:
    """Revoke a key; an already-revoked one comes back unchanged.

    None for another tenant's key, the same as for one that is absent. Raises
    LastAdminKeyError rather than leave the tenant no permanent admin key (ADR
    0009). The tenant row lock serializes concurrent revokes, so two cannot each
    count the other as the survivor.
    """
    params = {"id": key_id, "tenant_id": tenant_id}
    async with (engine or get_engine()).begin() as conn:
        await conn.execute(_LOCK_TENANT, {"id": tenant_id})
        row = (await conn.execute(_SELECT_KEY, params)).first()
        if row is None:
            return None
        if row.revoked_at is not None:
            return _key(row)
        if row.expires_at is None and Scope.ADMIN.value in row.scopes:
            others = (await conn.execute(_COUNT_OTHER_PERMANENT_ADMIN_KEYS, params)).scalar_one()
            if others == 0:
                raise LastAdminKeyError("the tenant's last non-expiring admin key")
        return _key((await conn.execute(_REVOKE_KEY, params)).one())


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


@dataclass(frozen=True)
class DocumentRow:
    id: UUID
    collection_id: UUID
    tenant_id: UUID
    filename: str
    mime_type: str
    status: str
    size_bytes: int | None
    sha256: str | None
    chunks: int
    created_at: datetime
    ingested_at: datetime | None


# The chunk count is what makes a status observable rather than asserted: a
# `ready` document with no chunks is a failure that the status word hides.
_DOCUMENT_COLUMNS = """
    d.id, d.collection_id, d.tenant_id, d.filename, d.mime_type, d.status,
    d.size_bytes, d.sha256, d.created_at, d.ingested_at,
    (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunks
"""
_LIST_DOCUMENTS = text(
    f"SELECT {_DOCUMENT_COLUMNS} FROM documents d "
    "WHERE d.collection_id = :collection_id AND d.tenant_id = :tenant_id "
    "ORDER BY d.id"
)
_SELECT_DOCUMENT = text(
    f"SELECT {_DOCUMENT_COLUMNS} FROM documents d "
    "WHERE d.id = :id AND d.collection_id = :collection_id AND d.tenant_id = :tenant_id"
)


def _document(row: Any) -> DocumentRow:
    return DocumentRow(
        id=row.id,
        collection_id=row.collection_id,
        tenant_id=row.tenant_id,
        filename=row.filename,
        mime_type=row.mime_type,
        status=row.status,
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        chunks=row.chunks,
        created_at=row.created_at,
        ingested_at=row.ingested_at,
    )


async def list_documents(
    *, collection_id: UUID, tenant_id: UUID, engine: AsyncEngine | None = None
) -> list[DocumentRow]:
    async with (engine or get_engine()).connect() as conn:
        rows = (
            await conn.execute(
                _LIST_DOCUMENTS, {"collection_id": collection_id, "tenant_id": tenant_id}
            )
        ).all()
    return [_document(row) for row in rows]


async def read_document(
    document_id: UUID,
    *,
    collection_id: UUID,
    tenant_id: UUID,
    engine: AsyncEngine | None = None,
) -> DocumentRow | None:
    """None for another tenant's document, the same as for one that is absent."""
    async with (engine or get_engine()).connect() as conn:
        row = (
            await conn.execute(
                _SELECT_DOCUMENT,
                {"id": document_id, "collection_id": collection_id, "tenant_id": tenant_id},
            )
        ).first()
    return None if row is None else _document(row)
