"""Shared route dependencies.

Thin wrappers over the module-level getters: a route that declares what it needs
can be handed a stub in a test without reaching into another module's globals.
"""

from collections.abc import Awaitable, Callable
from hmac import compare_digest
from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from prism.auth import AuthError, ResolvedKey, Scope, resolve_key
from prism.config import Settings, get_settings
from prism.embeddings import EmbeddingProvider, get_embedding_provider
from prism.vision import VisionProvider, get_vision_provider

__all__ = [
    "AdminDep",
    "AdminTokenDep",
    "IngestDep",
    "KeyDep",
    "ProviderDep",
    "ReadDep",
    "SettingsDep",
    "VisionDep",
    "embedding_provider",
    "require_key",
    "require_scope",
    "vision_provider",
]

log = structlog.get_logger(__name__)

_bearer = HTTPBearer(auto_error=False, description="API key issued by prism.auth")
_UNAUTHENTICATED = {"WWW-Authenticate": "Bearer"}


def embedding_provider() -> EmbeddingProvider:
    return get_embedding_provider()


SettingsDep = Annotated[Settings, Depends(get_settings)]


def vision_provider(settings: SettingsDep) -> VisionProvider | None:
    """None when vision is off."""
    return get_vision_provider() if settings.vision_enabled else None


ProviderDep = Annotated[EmbeddingProvider, Depends(embedding_provider)]
VisionDep = Annotated[VisionProvider | None, Depends(vision_provider)]


async def require_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> ResolvedKey:
    """Resolve the bearer key, or refuse the request.

    Every failure is one 401 with one message. auth.resolve_key distinguishes
    absent, revoked and expired keys for the operator; telling the caller which
    one it was would confirm that a key exists.
    """
    if credentials is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing bearer key", headers=_UNAUTHENTICATED
        )
    try:
        return await resolve_key(credentials.credentials)
    except AuthError as exc:
        log.info("key_rejected", reason=type(exc).__name__)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "invalid bearer key", headers=_UNAUTHENTICATED
        ) from exc


KeyDep = Annotated[ResolvedKey, Depends(require_key)]


def require_scope(scope: Scope) -> Callable[[ResolvedKey], Awaitable[ResolvedKey]]:
    async def dependency(key: KeyDep) -> ResolvedKey:
        if not key.permits(scope):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"key lacks the {scope} scope")
        return key

    return dependency


ReadDep = Annotated[ResolvedKey, Depends(require_scope(Scope.READ))]
IngestDep = Annotated[ResolvedKey, Depends(require_scope(Scope.INGEST))]
AdminDep = Annotated[ResolvedKey, Depends(require_scope(Scope.ADMIN))]


async def require_admin_token(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Gate the platform routes on `admin_token`.

    Deliberately not a scope on an API key: every key belongs to a tenant, and
    a tenant-scoped credential that can create other tenants is not a tenant
    boundary. An unset token disables these routes rather than opening them.
    """
    if settings.admin_token is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "admin_token is not configured")
    if credentials is None or not compare_digest(credentials.credentials, settings.admin_token):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "invalid admin token", headers=_UNAUTHENTICATED
        )


AdminTokenDep = Annotated[None, Depends(require_admin_token)]
