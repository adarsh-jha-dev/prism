"""Record where a document's uploaded bytes live

Rationale: docs/decisions/0003-uploaded-files-on-local-disk.md

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable: a document ingested from a path rather than an upload has no blob
    # of its own. Root-relative, so moving the storage root is not a backfill.
    op.execute(
        """
        ALTER TABLE documents
            ADD COLUMN storage_path text,
            ADD COLUMN size_bytes   bigint,
            ADD COLUMN sha256       text,
            ADD CONSTRAINT documents_size_bytes_check CHECK (size_bytes IS NULL OR size_bytes >= 0)
        """
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE documents "
        "DROP CONSTRAINT IF EXISTS documents_size_bytes_check, "
        "DROP COLUMN IF EXISTS storage_path, "
        "DROP COLUMN IF EXISTS size_bytes, "
        "DROP COLUMN IF EXISTS sha256"
    )
