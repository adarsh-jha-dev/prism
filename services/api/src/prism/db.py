"""Shared Postgres and Redis clients."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from prism.config import get_settings

_engine: AsyncEngine | None = None
_redis: aioredis.Redis | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            get_settings().database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=5,
        )
    return _engine


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis


async def close_redis() -> None:
    """Close the shared client. The next get_redis() opens a fresh one.

    A redis-py client binds to the event loop it was created on, so anything
    that runs each test on its own loop has to close it between them.
    """
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


@asynccontextmanager
async def lifespan_resources() -> AsyncIterator[None]:
    """Open shared clients on startup, close them on shutdown."""
    get_engine()
    get_redis()
    try:
        yield
    finally:
        global _engine
        await close_redis()
        if _engine is not None:
            await _engine.dispose()
            _engine = None
