"""Effective-dated price basis for cost_usd

Rationale: docs/decisions/0013-cost-basis-and-model-pricing.md

Here rather than in a later migration because query_traces.price_id lands in
0009: adding it afterwards means backfilling a table with no recoverable basis
to backfill from.

$0 is a price, not an absence. The local Ollama models and the in-process lane
get real rows at zero, so a null price_id downstream means exactly one thing —
we could not price this row — and it is an error rather than a freebie.

Revision ID: 0008
Revises: 0007
"""

from collections.abc import Sequence

from alembic import op

from prism.core.ids import uuid7

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# This migration's own date: the zero rows are in force from it, and nothing
# priced against them predates it.
PRICED_FROM = "2026-09-17T00:00:00+00:00"

SELF_HOSTED = "self-hosted; no marginal price per token"
IN_PROCESS = "in-process; no provider and no meter (ADR 0011)"

# Must match Settings: planner, grader, generator, embedding, vision, reranker.
ZERO_PRICED = (
    ("ollama", "qwen2.5:14b", "tokens", SELF_HOSTED),
    ("ollama", "llama3.1:8b", "tokens", SELF_HOSTED),
    ("ollama", "qwen2.5:32b", "tokens", SELF_HOSTED),
    ("ollama", "nomic-embed-text", "tokens", SELF_HOSTED),
    ("ollama", "qwen2.5vl:7b", "tokens", SELF_HOSTED),
    ("in-process", "bge-reranker-v2-m3", "none", IN_PROCESS),
)


def upgrade() -> None:
    # gist opclasses for the scalar halves of the exclusion constraint.
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    # Prices are stored as published — per million tokens, per GPU-hour.
    # Converting at write time bakes in a rounding decision that cannot be
    # recovered from the stored value.
    op.execute(
        """
        CREATE TABLE model_pricing (
            id                uuid PRIMARY KEY,
            provider          text NOT NULL,
            model             text NOT NULL,
            billing_unit      text NOT NULL,
            input_per_mtok    numeric(12,6),
            output_per_mtok   numeric(12,6),
            usd_per_gpu_hour  numeric(12,6),
            effective_from    timestamptz NOT NULL,
            effective_to      timestamptz,
            source            text NOT NULL,
            created_at        timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT model_pricing_provider_check
                CHECK (provider IN ('ollama', 'ollama-cloud', 'gemini', 'openai', 'in-process')),
            CONSTRAINT model_pricing_billing_unit_check
                CHECK (billing_unit IN ('tokens', 'gpu_ms', 'none')),
            CONSTRAINT model_pricing_rates_check CHECK (
                CASE billing_unit
                    WHEN 'tokens' THEN input_per_mtok IS NOT NULL
                                   AND output_per_mtok IS NOT NULL
                                   AND usd_per_gpu_hour IS NULL
                    WHEN 'gpu_ms' THEN usd_per_gpu_hour IS NOT NULL
                                   AND input_per_mtok IS NULL
                                   AND output_per_mtok IS NULL
                    ELSE input_per_mtok IS NULL
                     AND output_per_mtok IS NULL
                     AND usd_per_gpu_hour IS NULL
                END
            ),
            CONSTRAINT model_pricing_rates_nonnegative_check CHECK (
                coalesce(input_per_mtok, 0) >= 0
                AND coalesce(output_per_mtok, 0) >= 0
                AND coalesce(usd_per_gpu_hour, 0) >= 0
            ),
            CONSTRAINT model_pricing_range_check
                CHECK (effective_to IS NULL OR effective_to > effective_from)
        )
        """
    )

    # Two overlapping rows for one model make "the price at time T" ambiguous,
    # and the ambiguity would surface as a silently wrong benchmark. Gaps are
    # allowed: a trace falling in one is unpriced, which is the loud failure.
    op.execute(
        """
        ALTER TABLE model_pricing
            ADD CONSTRAINT model_pricing_no_overlap
            EXCLUDE USING gist (
                provider WITH =,
                model WITH =,
                tstzrange(effective_from, effective_to) WITH &&
            )
        """
    )

    op.execute(
        "CREATE INDEX model_pricing_lookup_idx "
        "ON model_pricing (provider, model, effective_from DESC)"
    )

    # Editing a price row retroactively rewrites every historical cost that
    # points at it — the precise bug this table exists to prevent. Closing a
    # range is the only permitted UPDATE, and the tempting wrong move is a
    # one-line UPDATE ... SET input_per_mtok, so a comment will not do.
    op.execute(
        """
        CREATE FUNCTION model_pricing_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF ROW(NEW.id, NEW.provider, NEW.model, NEW.billing_unit,
                   NEW.input_per_mtok, NEW.output_per_mtok, NEW.usd_per_gpu_hour,
                   NEW.effective_from, NEW.source)
               IS DISTINCT FROM
               ROW(OLD.id, OLD.provider, OLD.model, OLD.billing_unit,
                   OLD.input_per_mtok, OLD.output_per_mtok, OLD.usd_per_gpu_hour,
                   OLD.effective_from, OLD.source)
            THEN
                RAISE EXCEPTION
                    'model_pricing is append-only: only effective_to may be updated';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER model_pricing_append_only
            BEFORE UPDATE ON model_pricing
            FOR EACH ROW EXECUTE FUNCTION model_pricing_append_only()
        """
    )

    for provider, model, billing_unit, source in ZERO_PRICED:
        rates = "0, 0, NULL" if billing_unit == "tokens" else "NULL, NULL, NULL"
        op.execute(
            f"""
            INSERT INTO model_pricing
                (id, provider, model, billing_unit,
                 input_per_mtok, output_per_mtok, usd_per_gpu_hour,
                 effective_from, source)
            VALUES
                ('{uuid7()}', '{provider}', '{model}', '{billing_unit}',
                 {rates},
                 '{PRICED_FROM}'::timestamptz, '{source}')
            """
        )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS model_pricing_append_only ON model_pricing")
    op.execute("DROP FUNCTION IF EXISTS model_pricing_append_only()")
    op.execute("DROP TABLE IF EXISTS model_pricing")
    # btree_gist stays: dropping a shared extension is not this migration's call.
