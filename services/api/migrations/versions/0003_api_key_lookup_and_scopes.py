"""Give api_keys a lookup prefix, scopes, and a lifetime

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCOPES = ("read", "ingest", "admin")


def upgrade() -> None:
    # key_prefix is the plaintext head of the key, stored so lookup is an index
    # hit rather than a hash of every row. Deliberately not unique: prefixes can
    # collide, and the hash comparison is what identifies the key.
    #
    # DEFAULT '' then DROP DEFAULT backfills any pre-existing row. A prefix
    # cannot be recovered from a hash, so such a key can never be looked up and
    # has to be re-issued.
    op.execute(
        f"""
        ALTER TABLE api_keys
            ADD COLUMN key_prefix   text NOT NULL DEFAULT '',
            ADD COLUMN scopes       text[] NOT NULL DEFAULT ARRAY['read']::text[],
            ADD COLUMN last_used_at timestamptz,
            ADD COLUMN expires_at   timestamptz,
            ADD CONSTRAINT api_keys_scopes_check
                CHECK (cardinality(scopes) > 0
                       AND scopes <@ ARRAY[{", ".join(f"'{s}'" for s in SCOPES)}]::text[])
        """
    )
    op.execute("ALTER TABLE api_keys ALTER COLUMN key_prefix DROP DEFAULT")
    op.execute("CREATE INDEX api_keys_key_prefix_idx ON api_keys (key_prefix)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS api_keys_key_prefix_idx")
    op.execute(
        "ALTER TABLE api_keys "
        "DROP CONSTRAINT IF EXISTS api_keys_scopes_check, "
        "DROP COLUMN IF EXISTS key_prefix, "
        "DROP COLUMN IF EXISTS scopes, "
        "DROP COLUMN IF EXISTS last_used_at, "
        "DROP COLUMN IF EXISTS expires_at"
    )
