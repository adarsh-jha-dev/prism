"""Enable pgvector and create the core schema

Scope and omissions: docs/decisions/0001-phase-0-schema-scope.md

Revision ID: 0001
Revises:
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match Settings. pgvector caps HNSW at 2000 dims.
EMBEDDING_DIM = 768
EMBEDDING_MODEL = "nomic-embed-text"
ABSTENTION_THRESHOLD = 0.58


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Raw DDL: `vector` is not a native SQLAlchemy type. Ids are UUIDv7 from
    # prism.core.ids — no DEFAULT, since gen_random_uuid() would silently emit
    # v4 for any insert that forgot one.
    op.execute(
        """
        CREATE TABLE tenants (
            id          uuid PRIMARY KEY,
            name        text NOT NULL,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        f"""
        CREATE TABLE collections (
            id                    uuid PRIMARY KEY,
            tenant_id             uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            name                  text NOT NULL,
            embedding_model       text NOT NULL DEFAULT '{EMBEDDING_MODEL}',
            embedding_dim         integer NOT NULL DEFAULT {EMBEDDING_DIM},
            abstention_threshold  real NOT NULL DEFAULT {ABSTENTION_THRESHOLD},
            created_at            timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT collections_tenant_name_key UNIQUE (tenant_id, name)
        )
        """
    )
    op.execute("CREATE INDEX collections_tenant_id_idx ON collections (tenant_id)")

    op.execute(
        """
        CREATE TABLE api_keys (
            id              uuid PRIMARY KEY,
            tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            key_hash        text NOT NULL,
            name            text NOT NULL,
            rate_limit_rpm  integer NOT NULL DEFAULT 60,
            created_at      timestamptz NOT NULL DEFAULT now(),
            revoked_at      timestamptz,
            CONSTRAINT api_keys_key_hash_key UNIQUE (key_hash)
        )
        """
    )
    op.execute("CREATE INDEX api_keys_tenant_id_idx ON api_keys (tenant_id)")

    op.execute(
        """
        CREATE TABLE documents (
            id             uuid PRIMARY KEY,
            collection_id  uuid NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            filename       text NOT NULL,
            mime_type      text NOT NULL,
            status         text NOT NULL DEFAULT 'pending',
            created_at     timestamptz NOT NULL DEFAULT now(),
            ingested_at    timestamptz,
            CONSTRAINT documents_status_check
                CHECK (status IN ('pending', 'processing', 'ready', 'failed'))
        )
        """
    )
    op.execute("CREATE INDEX documents_collection_id_idx ON documents (collection_id)")

    # collection_id is denormalized from documents: scoping has to be a WHERE
    # predicate inside the ANN query, and joining out would defeat the index.
    op.execute(
        f"""
        CREATE TABLE chunks (
            id             uuid PRIMARY KEY,
            document_id    uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            collection_id  uuid NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            content        text NOT NULL,
            chunk_type     text NOT NULL DEFAULT 'text',
            page_number    integer,
            embedding      vector({EMBEDDING_DIM}),
            metadata       jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            created_at     timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT chunks_chunk_type_check
                CHECK (chunk_type IN ('text', 'figure', 'table', 'equation'))
        )
        """
    )
    op.execute("CREATE INDEX chunks_document_id_idx ON chunks (document_id)")
    op.execute("CREATE INDEX chunks_collection_id_idx ON chunks (collection_id)")

    # Cosine: the embedding models in play produce normalized vectors.
    op.execute(
        """
        CREATE INDEX chunks_embedding_hnsw
            ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    for table in ("chunks", "documents", "api_keys", "collections", "tenants"):
        op.execute(f"DROP TABLE IF EXISTS {table}")
