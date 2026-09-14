"""Full-text column and GIN index on chunks

Rationale: docs/decisions/0010-full-text-is-postgres-fts.md

The regconfig is pinned in the column definition rather than left to
default_text_search_config: the one-argument to_tsvector is only STABLE, so it is
neither indexable nor legal in a generated column, and pinning it also stops a
server-side GUC change reinterpreting the whole corpus.

tenant_id and collection_id ride in the same GIN index so the scope is a
predicate inside the index scan, as it already is for the ANN query. A lexical
hit from another tenant must not occupy a top-k slot either.

No retrieval change here — search.py stays vector-only until the hybrid
`retrieve` node lands.

Revision ID: 0007
Revises: 0006
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # GIN opclasses for uuid.
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")

    # content is NOT NULL, so no coalesce. Rewrites the table.
    op.execute(
        """
        ALTER TABLE chunks
            ADD COLUMN content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )

    op.execute(
        """
        CREATE INDEX chunks_content_tsv_gin
            ON chunks
            USING gin (tenant_id, collection_id, content_tsv)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_content_tsv_gin")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
    # btree_gin stays: dropping a shared extension is not this migration's call.
