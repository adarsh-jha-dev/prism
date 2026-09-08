"""Health endpoints."""

import asyncio
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from prism import __version__
from prism.config import get_settings
from prism.db import get_engine, get_redis

log = structlog.get_logger(__name__)
router = APIRouter(tags=["health"])

# Driver errors run to hundreds of characters. The dashboard renders this in a
# table cell; the full exception goes to the log.
_MAX_ERROR_CHARS = 160


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def _failed(exc: Exception) -> dict[str, Any]:
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    message = f"{type(exc).__name__}: {first_line}"
    if len(message) > _MAX_ERROR_CHARS:
        message = message[: _MAX_ERROR_CHARS - 1].rstrip() + "…"
    return {"ok": False, "error": message}


async def _check_postgres() -> dict[str, Any]:
    try:
        async with get_engine().connect() as conn:
            version = (
                await conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
            ).scalar_one_or_none()
    except Exception as exc:
        return _failed(exc)

    if version is None:
        return {"ok": False, "error": "pgvector extension not installed"}
    return {"ok": True, "detail": f"pgvector {version}"}


async def _check_redis() -> dict[str, Any]:
    try:
        pong = await get_redis().ping()
    except Exception as exc:
        return _failed(exc)
    return {"ok": bool(pong)} if pong else {"ok": False, "error": "PING returned falsey"}


async def _check_ollama() -> dict[str, Any]:
    """Reachable, and carrying the embedding model."""
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=settings.dep_check_timeout_s) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return _failed(exc)

    # Ollama reports "name" as "model:tag"; an untagged pull lands as ":latest".
    installed = {str(m.get("name", "")).split(":")[0] for m in payload.get("models", [])}
    wanted = settings.embedding_model.split(":")[0]
    if wanted not in installed:
        return {
            "ok": False,
            "error": f"model not pulled — run: ollama pull {settings.embedding_model}",
            "detail": f"{len(installed)} model(s) present",
        }
    return {"ok": True, "detail": f"{len(installed)} model(s) present, {wanted} ready"}


@router.get("/health/deps")
async def health_deps(response: Response) -> dict[str, Any]:
    # Concurrently: run serially, three 2s ceilings become a 6s worst case.
    names = ("postgres", "redis", "ollama")
    results = await asyncio.gather(_check_postgres(), _check_redis(), _check_ollama())
    checks = dict(zip(names, results, strict=True))

    ok = all(check["ok"] for check in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("dependency_check_failed", checks=checks)
    return {"ok": ok, "checks": checks}
