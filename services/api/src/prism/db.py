"""Shared Postgres and Redis clients."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from prism.config import get_settings

_engine: AsyncEngine | None = None
_redis: aioredis.Redis | None = None
_checkpoint_pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None


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


async def get_checkpoint_pool() -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    """The checkpointer's own pool (ADR 0015).

    Separate from the engine's: the saver wants autocommit and no prepared
    statements, which SQLAlchemy's pooled connections must not carry back.
    """
    global _checkpoint_pool
    if _checkpoint_pool is None:
        pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
            # psycopg wants libpq conninfo, not a SQLAlchemy URL.
            make_url(get_settings().database_url)
            .set(drivername="postgresql")
            .render_as_string(hide_password=False),
            min_size=1,
            max_size=get_settings().checkpointer_pool_max_size,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        await pool.open()
        _checkpoint_pool = pool
    return _checkpoint_pool


async def close_checkpoint_pool() -> None:
    """Close the shared pool. The next get_checkpoint_pool() opens a fresh one."""
    global _checkpoint_pool
    if _checkpoint_pool is not None:
        await _checkpoint_pool.close()
        _checkpoint_pool = None


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
    await get_checkpoint_pool()
    try:
        yield
    finally:
        global _engine
        await close_checkpoint_pool()
        await close_redis()
        if _engine is not None:
            await _engine.dispose()
            _engine = None
