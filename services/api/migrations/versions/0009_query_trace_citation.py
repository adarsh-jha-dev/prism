"""Query-time schema: queries, query_traces, query_citations

Rationale: docs/decisions/0012-query-trace-and-citation-model.md, and the
price columns from docs/decisions/0013-cost-basis-and-model-pricing.md.

Where an invariant can be a database constraint, it is one. These tables are
written by graph nodes, eval runs, migrations and psql prompts, and an invariant
enforced only by the writer is enforced by the least careful writer.

No policy constant is in the schema: both attempt counters are checked >= 0 and
nothing more. A <= 3 check would be max_attempts hardcoded outside Settings, in
the one place that needs a migration to change.

Not partitioned. started_at is NOT NULL from this first migration so it can
become the range key with no backfill, and nothing anywhere gets a foreign key
to query_traces — that is what would otherwise make partitioning a schema-wide
change.

Revision ID: 0009
Revises: 0008
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # tenant_id is denormalized with a composite FK, as migration 0004 did for
    # documents and chunks: RLS (ADR 0006) needs a column comparison rather than
    # a per-row subquery, and these become the highest-volume tables here.
    #
    # cache_entry_id carries no FK — cache_entries is an open design gap
    # (docs/design/REVIEW.md B1) and this column is shaped to take one later.
    #
    # The two CHECK equivalences are the point. A nullable enum on its own
    # permits a refusal with no reason, which is the boolean this replaces.
    op.execute(
        """
        CREATE TABLE queries (
            id                  uuid PRIMARY KEY,
            tenant_id           uuid NOT NULL,
            collection_id       uuid NOT NULL,
            thread_id           text NOT NULL,
            question            text NOT NULL,
            status              text NOT NULL,
            refusal_reason      text,
            cache_entry_id      uuid,
            final_answer        text,
            citation_count      smallint NOT NULL DEFAULT 0,
            retrieval_attempts  smallint NOT NULL DEFAULT 0,
            grounding_attempts  smallint NOT NULL DEFAULT 0,
            total_cost_usd      numeric(14,8),
            latency_ms          integer,
            created_at          timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT queries_collection_tenant_fkey
                FOREIGN KEY (collection_id, tenant_id)
                REFERENCES collections (id, tenant_id) ON DELETE CASCADE,
            CONSTRAINT queries_id_tenant_key UNIQUE (id, tenant_id),
            CONSTRAINT queries_thread_id_key UNIQUE (thread_id),
            CONSTRAINT queries_status_check
                CHECK (status IN ('cached', 'answered', 'refused')),
            CONSTRAINT queries_refusal_reason_check
                CHECK (refusal_reason IN ('no_relevant_evidence', 'insufficient_evidence')),
            CONSTRAINT queries_refusal_reason_present_check
                CHECK ((status = 'refused') = (refusal_reason IS NOT NULL)),
            CONSTRAINT queries_cache_entry_present_check
                CHECK ((status = 'cached') = (cache_entry_id IS NOT NULL)),
            CONSTRAINT queries_attempts_check
                CHECK (retrieval_attempts >= 0 AND grounding_attempts >= 0),
            CONSTRAINT queries_latency_check
                CHECK (latency_ms IS NULL OR latency_ms >= 0),
            CONSTRAINT queries_cost_check
                CHECK (total_cost_usd IS NULL OR total_cost_usd >= 0),
            -- A cross-table "has at least one child" rule is not expressible as
            -- a CHECK. The counter is written in the same transaction as the
            -- citation rows, and is also the number the dashboard lists. Note
            -- what the non-refused branch commits to: a cached answer carries
            -- citations too.
            CONSTRAINT queries_citation_count_check
                CHECK ((status = 'refused' AND citation_count = 0)
                    OR (status <> 'refused' AND citation_count > 0))
        )
        """
    )
    op.execute("CREATE INDEX queries_tenant_created_idx ON queries (tenant_id, created_at DESC)")
    op.execute("CREATE INDEX queries_collection_tenant_idx ON queries (collection_id, tenant_id)")

    # status is universal; verdict is nullable and meaningful only on grading
    # nodes. The designed pass|fail|skip was one column doing two jobs —
    # plan_query, retrieve, rerank and generate have an outcome but no verdict,
    # and pruned is a status, not a judgement.
    #
    # input_json/output_json store references, not copies: a retrieve output is
    # chunk ids with scores, not ten chunks of text. The truncation flags keep a
    # capped payload distinguishable from a node that returned little.
    #
    # cost_usd is numeric(14,8): six decimal places round a cheap local or flash
    # node row to zero, and then the node rows stop summing to the query total.
    op.execute(
        """
        CREATE TABLE query_traces (
            id                uuid PRIMARY KEY,
            query_id          uuid NOT NULL,
            tenant_id         uuid NOT NULL,
            node_name         text NOT NULL,
            sequence          integer NOT NULL,
            attempt           smallint NOT NULL DEFAULT 1,
            status            text NOT NULL,
            verdict           text,
            started_at        timestamptz NOT NULL,
            duration_ms       integer NOT NULL,
            provider          text,
            model             text,
            billing_unit      text NOT NULL DEFAULT 'none',
            input_tokens      integer,
            output_tokens     integer,
            gpu_ms            integer,
            price_id          uuid REFERENCES model_pricing (id) ON DELETE RESTRICT,
            cost_usd          numeric(14,8),
            cost_basis        text,
            error             text,
            input_json        jsonb,
            output_json       jsonb,
            input_truncated   boolean NOT NULL DEFAULT false,
            output_truncated  boolean NOT NULL DEFAULT false,
            checkpoint_ref    text,
            created_at        timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT query_traces_query_tenant_fkey
                FOREIGN KEY (query_id, tenant_id)
                REFERENCES queries (id, tenant_id) ON DELETE CASCADE,
            -- started_at ties under clock resolution, so sequence is what
            -- actually orders the waterfall and it has to be unique.
            CONSTRAINT query_traces_sequence_key UNIQUE (query_id, sequence),
            CONSTRAINT query_traces_status_check
                CHECK (status IN ('ok', 'error', 'pruned', 'skipped')),
            CONSTRAINT query_traces_verdict_check
                CHECK (verdict IN ('pass', 'fail')),
            CONSTRAINT query_traces_error_present_check
                CHECK ((status = 'error') = (error IS NOT NULL)),
            CONSTRAINT query_traces_attempt_check CHECK (attempt >= 1),
            CONSTRAINT query_traces_duration_check CHECK (duration_ms >= 0),
            CONSTRAINT query_traces_billing_unit_check
                CHECK (billing_unit IN ('tokens', 'gpu_ms', 'none')),
            -- Forbids the meters that do not belong to the unit, without
            -- requiring the ones that do: an errored node reports neither.
            CONSTRAINT query_traces_meters_check CHECK (
                CASE billing_unit
                    WHEN 'tokens' THEN gpu_ms IS NULL
                    WHEN 'gpu_ms' THEN input_tokens IS NULL AND output_tokens IS NULL
                    ELSE input_tokens IS NULL
                     AND output_tokens IS NULL
                     AND gpu_ms IS NULL
                END
            ),
            CONSTRAINT query_traces_meters_nonnegative_check CHECK (
                coalesce(input_tokens, 0) >= 0
                AND coalesce(output_tokens, 0) >= 0
                AND coalesce(gpu_ms, 0) >= 0
            ),
            -- price_id IS NULL means exactly one thing: we could not price this
            -- row. Unpriced is not free (ADR 0013).
            CONSTRAINT query_traces_priced_check
                CHECK ((price_id IS NULL) = (cost_usd IS NULL)),
            CONSTRAINT query_traces_cost_basis_check
                CHECK (cost_basis IN ('metered', 'estimated')),
            CONSTRAINT query_traces_cost_basis_present_check
                CHECK ((price_id IS NULL) = (cost_basis IS NULL)),
            CONSTRAINT query_traces_cost_nonnegative_check
                CHECK (cost_usd IS NULL OR cost_usd >= 0)
        )
        """
    )
    # Eval results reference (query_id, node_name, attempt) rather than a trace
    # id, since nothing may hold a foreign key here.
    op.execute(
        "CREATE INDEX query_traces_node_attempt_idx ON query_traces (query_id, node_name, attempt)"
    )
    op.execute(
        "CREATE INDEX query_traces_tenant_started_idx ON query_traces (tenant_id, started_at DESC)"
    )

    # chunk_ref is what was cited at answer time and is never null; chunk_id
    # says whether that chunk still exists. Chunks cascade-delete with their
    # document and re-ingestion replaces them, so a hard FK alone would rewrite
    # the history of every answer that cited one. "The source has since been
    # deleted" is true and useful; "this answer had no sources" is a lie.
    #
    # chunk_index is the document-wide reading-order index from chunks.metadata
    # (ADR 0002), snapshotted like the rest of the row. Character offsets are a
    # known gap: they arrive as two nullable columns against cited_content,
    # additive and without a backfill.
    op.execute(
        """
        CREATE TABLE query_citations (
            id             uuid PRIMARY KEY,
            query_id       uuid NOT NULL,
            tenant_id      uuid NOT NULL,
            chunk_ref      uuid NOT NULL,
            chunk_id       uuid REFERENCES chunks (id) ON DELETE SET NULL,
            document_id    uuid,
            page_number    integer,
            chunk_index    integer,
            rank           smallint NOT NULL,
            rerank_score   real,
            cited_content  text NOT NULL,
            created_at     timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT query_citations_query_tenant_fkey
                FOREIGN KEY (query_id, tenant_id)
                REFERENCES queries (id, tenant_id) ON DELETE CASCADE,
            CONSTRAINT query_citations_rank_key UNIQUE (query_id, rank),
            CONSTRAINT query_citations_rank_check CHECK (rank >= 1)
        )
        """
    )
    op.execute("CREATE INDEX query_citations_chunk_ref_idx ON query_citations (chunk_ref)")


def downgrade() -> None:
    for table in ("query_citations", "query_traces", "queries"):
        op.execute(f"DROP TABLE IF EXISTS {table}")
