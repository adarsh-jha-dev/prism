"""Per-key hourly usage counters

Rationale: docs/decisions/0008-rate-limiting-and-usage-metering.md

Revision ID: 0006
Revises: 0005
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # tenant_id is denormalized from api_keys, with a composite FK, so a tenant's
    # usage can be summed without a join and cannot be attributed to the wrong one.
    op.execute("ALTER TABLE api_keys ADD CONSTRAINT api_keys_id_tenant_key UNIQUE (id, tenant_id)")
    op.execute(
        """
        CREATE TABLE usage_records (
            id            uuid PRIMARY KEY,
            api_key_id    uuid NOT NULL,
            tenant_id     uuid NOT NULL,
            window_start  timestamptz NOT NULL,
            requests      bigint NOT NULL DEFAULT 0,
            throttled     bigint NOT NULL DEFAULT 0,
            created_at    timestamptz NOT NULL DEFAULT now(),
            updated_at    timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT usage_records_api_key_tenant_fkey
                FOREIGN KEY (api_key_id, tenant_id)
                REFERENCES api_keys (id, tenant_id) ON DELETE CASCADE,
            CONSTRAINT usage_records_counts_check
                CHECK (requests >= 0 AND throttled >= 0),
            CONSTRAINT usage_records_window_key UNIQUE (api_key_id, window_start)
        )
        """
    )
    op.execute(
        "CREATE INDEX usage_records_tenant_window_idx "
        "ON usage_records (tenant_id, window_start DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS usage_records")
    op.execute("ALTER TABLE api_keys DROP CONSTRAINT IF EXISTS api_keys_id_tenant_key")
