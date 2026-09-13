"""Carry tenant_id on documents and chunks

Rationale: docs/decisions/0006-postgres-rls-for-tenant-isolation.md

Tenant scoping has to be a predicate inside the ANN query, and a predicate on a
joined table is not inside the chunks scan. chunks already carries collection_id
denormalized for exactly this reason; tenant_id follows it.

The composite foreign keys are what make the denormalization safe: a row whose
tenant_id disagrees with its parent cannot be written at all, so the column the
isolation predicate reads cannot drift away from the truth.

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Composite FK targets have to be unique.
    op.execute(
        "ALTER TABLE collections ADD CONSTRAINT collections_id_tenant_key UNIQUE (id, tenant_id)"
    )

    op.execute("ALTER TABLE documents ADD COLUMN tenant_id uuid")
    op.execute(
        "UPDATE documents d SET tenant_id = c.tenant_id "
        "FROM collections c WHERE c.id = d.collection_id"
    )
    op.execute("ALTER TABLE documents ALTER COLUMN tenant_id SET NOT NULL")
    op.execute(
        "ALTER TABLE documents "
        "DROP CONSTRAINT documents_collection_id_fkey, "
        "ADD CONSTRAINT documents_collection_tenant_fkey "
        "    FOREIGN KEY (collection_id, tenant_id) "
        "    REFERENCES collections (id, tenant_id) ON DELETE CASCADE, "
        "ADD CONSTRAINT documents_id_tenant_key UNIQUE (id, tenant_id)"
    )

    op.execute("ALTER TABLE chunks ADD COLUMN tenant_id uuid")
    op.execute(
        "UPDATE chunks ch SET tenant_id = c.tenant_id "
        "FROM collections c WHERE c.id = ch.collection_id"
    )
    op.execute("ALTER TABLE chunks ALTER COLUMN tenant_id SET NOT NULL")
    op.execute(
        "ALTER TABLE chunks "
        "DROP CONSTRAINT chunks_collection_id_fkey, "
        "DROP CONSTRAINT chunks_document_id_fkey, "
        "ADD CONSTRAINT chunks_collection_tenant_fkey "
        "    FOREIGN KEY (collection_id, tenant_id) "
        "    REFERENCES collections (id, tenant_id) ON DELETE CASCADE, "
        "ADD CONSTRAINT chunks_document_tenant_fkey "
        "    FOREIGN KEY (document_id, tenant_id) "
        "    REFERENCES documents (id, tenant_id) ON DELETE CASCADE"
    )

    # collection_id leads, so this also serves the collection-only lookups that
    # chunks_collection_id_idx served.
    op.execute("CREATE INDEX chunks_collection_tenant_idx ON chunks (collection_id, tenant_id)")
    op.execute("DROP INDEX chunks_collection_id_idx")
    op.execute(
        "CREATE INDEX documents_collection_tenant_idx ON documents (collection_id, tenant_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS documents_collection_tenant_idx")
    op.execute("CREATE INDEX IF NOT EXISTS chunks_collection_id_idx ON chunks (collection_id)")
    op.execute("DROP INDEX IF EXISTS chunks_collection_tenant_idx")

    op.execute(
        "ALTER TABLE chunks "
        "DROP CONSTRAINT IF EXISTS chunks_collection_tenant_fkey, "
        "DROP CONSTRAINT IF EXISTS chunks_document_tenant_fkey, "
        "ADD CONSTRAINT chunks_collection_id_fkey "
        "    FOREIGN KEY (collection_id) REFERENCES collections (id) ON DELETE CASCADE, "
        "ADD CONSTRAINT chunks_document_id_fkey "
        "    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE, "
        "DROP COLUMN IF EXISTS tenant_id"
    )

    op.execute(
        "ALTER TABLE documents "
        "DROP CONSTRAINT IF EXISTS documents_collection_tenant_fkey, "
        "DROP CONSTRAINT IF EXISTS documents_id_tenant_key, "
        "ADD CONSTRAINT documents_collection_id_fkey "
        "    FOREIGN KEY (collection_id) REFERENCES collections (id) ON DELETE CASCADE, "
        "DROP COLUMN IF EXISTS tenant_id"
    )

    op.execute("ALTER TABLE collections DROP CONSTRAINT IF EXISTS collections_id_tenant_key")
