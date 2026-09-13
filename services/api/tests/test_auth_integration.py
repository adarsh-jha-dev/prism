"""Integration: resolving a key against the real table. Needs `make up`."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import text

from prism.auth import (
    GeneratedKey,
    KeyExpired,
    KeyNotFound,
    KeyRevoked,
    Scope,
    generate_key,
    resolve_key,
)
from prism.core.ids import uuid7
from prism.db import get_engine

pytestmark = pytest.mark.integration

INSERT = text(
    """
    INSERT INTO api_keys
        (id, tenant_id, key_hash, name, key_prefix, scopes, revoked_at, expires_at)
    VALUES
        (:id, :tenant_id, :key_hash, :name, :key_prefix, CAST(:scopes AS text[]),
         :revoked_at, :expires_at)
    """
)


@pytest.fixture
async def tenant_id() -> AsyncIterator[UUID]:
    tenant = uuid7()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"auth-test-{tenant}"},
        )
    try:
        yield tenant
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant})


async def _issue(
    tenant: UUID,
    *,
    scopes: str = "{read}",
    revoked_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> GeneratedKey:
    key = generate_key()
    await _insert(tenant, key.prefix, key.key_hash, scopes, revoked_at, expires_at)
    return key


async def _insert(
    tenant: UUID,
    prefix: str,
    key_hash: str,
    scopes: str = "{read}",
    revoked_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(
            INSERT,
            {
                "id": uuid7(),
                "tenant_id": tenant,
                "key_hash": key_hash,
                "name": "test key",
                "key_prefix": prefix,
                "scopes": scopes,
                "revoked_at": revoked_at,
                "expires_at": expires_at,
            },
        )


async def test_resolves_to_its_tenant_and_scopes(tenant_id: UUID) -> None:
    key = await _issue(tenant_id, scopes="{read,ingest}")
    resolved = await resolve_key(key.plaintext)
    assert resolved.tenant_id == tenant_id
    assert resolved.scopes == frozenset({Scope.READ, Scope.INGEST})
    assert resolved.permits(Scope.INGEST)


async def test_an_unissued_key_does_not_resolve(tenant_id: UUID) -> None:
    await _issue(tenant_id)
    with pytest.raises(KeyNotFound):
        await resolve_key(generate_key().plaintext)


async def test_a_revoked_key_is_distinguishable(tenant_id: UUID) -> None:
    key = await _issue(tenant_id, revoked_at=datetime.now(UTC) - timedelta(days=1))
    with pytest.raises(KeyRevoked):
        await resolve_key(key.plaintext)


async def test_an_expired_key_is_distinguishable(tenant_id: UUID) -> None:
    key = await _issue(tenant_id, expires_at=datetime.now(UTC) - timedelta(seconds=1))
    with pytest.raises(KeyExpired):
        await resolve_key(key.plaintext)


async def test_a_key_expiring_later_still_resolves(tenant_id: UUID) -> None:
    key = await _issue(tenant_id, expires_at=datetime.now(UTC) + timedelta(days=30))
    assert (await resolve_key(key.plaintext)).tenant_id == tenant_id


async def test_a_prefix_collision_resolves_to_the_right_key(tenant_id: UUID) -> None:
    """A decoy row sharing the prefix must not authenticate: only the digest does."""
    mine = generate_key()
    decoy = generate_key()
    await _insert(tenant_id, mine.prefix, decoy.key_hash, scopes="{admin}")
    await _insert(tenant_id, mine.prefix, mine.key_hash, scopes="{read}")

    resolved = await resolve_key(mine.plaintext)
    assert resolved.scopes == frozenset({Scope.READ})
    assert not resolved.permits(Scope.ADMIN)


async def test_a_key_matching_no_digest_under_a_live_prefix_is_rejected(
    tenant_id: UUID,
) -> None:
    """The prefix index returning a row is not itself an authentication."""
    stored = generate_key()
    impostor = generate_key()
    await _insert(tenant_id, stored.prefix, stored.key_hash)
    await _insert(tenant_id, stored.prefix, impostor.key_hash)

    with pytest.raises(KeyNotFound):
        await resolve_key(generate_key().plaintext)


async def test_the_prefix_alone_does_not_authenticate(tenant_id: UUID) -> None:
    key = await _issue(tenant_id)
    with pytest.raises(KeyNotFound):
        await resolve_key(key.prefix)
