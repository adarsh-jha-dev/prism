"""The checkpointer.

Rationale: docs/decisions/0015-checkpointer-tables-and-connection.md

Migration 0010 owns setup(); nothing here calls it. setup() takes no lock, so
concurrent callers collide on checkpoint_migrations.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from prism.db import get_checkpoint_pool

__all__ = ["CHECKPOINTER_SCHEMA_VERSION", "get_checkpointer"]

# len(langgraph.checkpoint.postgres.base.MIGRATIONS) that migration 0010 was
# written against. A test guards it: raising the pin means a new migration
# calling setup() again, never an edit to 0010.
CHECKPOINTER_SCHEMA_VERSION = 10


async def get_checkpointer() -> AsyncPostgresSaver:
    """A saver over the shared pool, built per run.

    The saver holds one asyncio.Lock around every cursor and binds to the loop it
    was constructed on, so concurrent runs each get their own handle.
    """
    return AsyncPostgresSaver(await get_checkpoint_pool())
