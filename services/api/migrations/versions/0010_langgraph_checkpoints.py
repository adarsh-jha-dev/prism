"""LangGraph checkpoint tables, created by the library's own setup()

Rationale: docs/decisions/0015-checkpointer-tables-and-connection.md

The DDL for the four tables is the library's; this migration decides only when
it runs, so `alembic upgrade head` stays the whole schema.

setup() is idempotent and tracks its own version: raising the pin means a new
migration that calls it again, not an edit here.

Revision ID: 0010
Revises: 0009
"""

from collections.abc import Sequence

from alembic import op
from langgraph.checkpoint.postgres import PostgresSaver

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "checkpoint_migrations")


def upgrade() -> None:
    # Three of the library's ten statements are CREATE INDEX CONCURRENTLY, which
    # Postgres refuses inside a transaction block. setup() also assumes
    # autocommit for the DDL it commits.
    with op.get_context().autocommit_block():
        PostgresSaver(op.get_bind().connection.driver_connection).setup()


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")
