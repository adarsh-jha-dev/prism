"""Make a tenant's name unique

The eval harness and `prism.tenancy.ensure_tenant` both resolve a tenant by
name. Two tenants of one name make that lookup ambiguous rather than wrong-ish,
so it is a constraint, not a convention.

Revision ID: 0005
Revises: 0004
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE tenants ADD CONSTRAINT tenants_name_key UNIQUE (name)")


def downgrade() -> None:
    op.execute("ALTER TABLE tenants DROP CONSTRAINT IF EXISTS tenants_name_key")
