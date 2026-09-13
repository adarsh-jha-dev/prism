"""API key generation, hashing and resolution.

Format and digest are fixed by docs/decisions/0007-api-key-format-and-hashing.md:
changing any constant here invalidates every key already issued.

The plaintext exists only in the return value of `generate_key`. Nothing in this
module writes it anywhere, and there is no path back to it from the database.
"""

import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.db import get_engine

__all__ = [
    "KEY_PREFIX_LENGTH",
    "AuthError",
    "GeneratedKey",
    "KeyExpired",
    "KeyNotFound",
    "KeyRevoked",
    "ResolvedKey",
    "Scope",
    "generate_key",
    "hash_key",
    "prefix_of",
    "resolve_key",
    "verify_key",
]

KEY_LITERAL_PREFIX = "prism_ak_"
KEY_ENTROPY_BYTES = 32
KEY_PREFIX_LENGTH = 16


class Scope(StrEnum):
    READ = "read"
    INGEST = "ingest"
    ADMIN = "admin"


class AuthError(Exception):
    """A bearer key that does not resolve to a usable tenant."""


class KeyNotFound(AuthError):
    """No stored key matches. Also raised for a malformed key."""


class KeyRevoked(AuthError):
    pass


class KeyExpired(AuthError):
    pass


@dataclass(frozen=True)
class GeneratedKey:
    """A freshly minted key. `plaintext` is shown once and never persisted."""

    plaintext: str
    prefix: str
    key_hash: str


@dataclass(frozen=True)
class ResolvedKey:
    id: UUID
    tenant_id: UUID
    scopes: frozenset[Scope]
    rate_limit_rpm: int

    def permits(self, scope: Scope) -> bool:
        return Scope.ADMIN in self.scopes or scope in self.scopes


def prefix_of(plaintext: str) -> str:
    return plaintext[:KEY_PREFIX_LENGTH]


def hash_key(plaintext: str) -> str:
    return sha256(plaintext.encode("utf-8")).hexdigest()


def verify_key(plaintext: str, key_hash: str) -> bool:
    """Constant-time comparison, so a wrong key leaks nothing by timing."""
    return hmac.compare_digest(hash_key(plaintext), key_hash)


def generate_key() -> GeneratedKey:
    plaintext = KEY_LITERAL_PREFIX + secrets.token_urlsafe(KEY_ENTROPY_BYTES)
    return GeneratedKey(
        plaintext=plaintext,
        prefix=prefix_of(plaintext),
        key_hash=hash_key(plaintext),
    )


_BY_PREFIX = text(
    """
    SELECT id, tenant_id, key_hash, scopes, rate_limit_rpm, revoked_at, expires_at
    FROM api_keys
    WHERE key_prefix = :key_prefix
    """
)


async def resolve_key(plaintext: str, *, engine: AsyncEngine | None = None) -> ResolvedKey:
    """Resolve a bearer key to its tenant and scopes.

    Looks up by indexed prefix, then compares digests. The comparison runs even
    when the prefix returns one row: the prefix is not a credential.

    Raises KeyNotFound, KeyRevoked or KeyExpired. Revocation and expiry are
    distinguished from absence so an operator can tell why a key stopped
    working; callers facing the network must not pass that distinction on.
    """
    if not plaintext.startswith(KEY_LITERAL_PREFIX):
        raise KeyNotFound("not a prism key")

    engine = engine or get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(_BY_PREFIX, {"key_prefix": prefix_of(plaintext)})).all()

    match = next((row for row in rows if verify_key(plaintext, row.key_hash)), None)
    if match is None:
        raise KeyNotFound("no key matches")
    if match.revoked_at is not None:
        raise KeyRevoked(f"revoked at {match.revoked_at.isoformat()}")
    if match.expires_at is not None and match.expires_at <= datetime.now(UTC):
        raise KeyExpired(f"expired at {match.expires_at.isoformat()}")

    return ResolvedKey(
        id=match.id,
        tenant_id=match.tenant_id,
        scopes=frozenset(Scope(s) for s in match.scopes),
        rate_limit_rpm=match.rate_limit_rpm,
    )
