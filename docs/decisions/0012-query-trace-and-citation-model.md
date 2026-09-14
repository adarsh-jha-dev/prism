# 0012 — Query, trace and citation model: constraints over conventions, and refusals are not cached

- **Status:** accepted
- **Date:** 2026-09-14
- **Relates to:** [0001](0001-phase-0-schema-scope.md),
  [0006](0006-postgres-rls-for-tenant-isolation.md),
  [0011](0011-rerank-runs-in-process.md)

## Context

Five core tables exist. The query-time tables are next, and they are the
substrate for three deliverables at once: the trace viewer, the eval harness, and
the benchmark that is the whole point. `REVIEW.md` finds defects in all three
designed shapes — `was_refused boolean` cannot express three terminal states, one
`retry_count` cannot express two independent loops, `QUERY_TRACES` cannot render
its own mockup, and the highest-volume table in the system has no retention plan.

It also leaves one question genuinely open rather than merely underspecified:
**are refusals cached?** "It saves money and locks in a wrong abstention" is the
entire tension.

## Decision

The governing choice, applied throughout: **where an invariant can be a database
constraint, it is one.** These tables are written by graph nodes, eval runs,
migrations and psql prompts — the same argument ADR 0006 makes for RLS. An
invariant enforced only by the writer is enforced by the least careful writer.

### `queries` — three terminal states, and a reason that cannot go missing

`status` is a checked enum of `cached | answered | refused`; `refusal_reason` is
nullable and checked against `no_relevant_evidence | insufficient_evidence`. The
two are tied together:

```
CHECK ((status = 'refused') = (refusal_reason IS NOT NULL))
CHECK ((status = 'cached')  = (cache_entry_id IS NOT NULL))
```

The equivalence is the point. A nullable enum on its own permits a refusal with
no reason, which is precisely the boolean the design is replacing — `CLAUDE.md`
says a refusal carries a reason, not a flag, and an unenforced enum makes that a
hope. `cache_entry_id` gets the same treatment so a cached query can always name
the entry that served it.

### Two attempt counters, and no policy constants in the schema

`retrieval_attempts` and `grounding_attempts`, separately, because the loops count
independently. Both are checked `>= 0` and nothing more. A `<= 3` check would
look like rigour and would actually be `max_attempts` hardcoded outside
`Settings` — in the one place that needs a migration to change, and that silently
diverges the moment the constant moves.

### `thread_id` is the replay handle

`thread_id text NOT NULL UNIQUE`. LangGraph's Postgres checkpointer owns its own
tables keyed by `thread_id`, and we do not put a foreign key into them: that is a
library's schema and it may migrate it. `checkpoint_ref` stays per trace row, so
a fork can target a specific node rather than only the head of the thread.

### Record the meter, not just the money

`cost_usd numeric(14,8)`. Six decimal places round a cheap local or flash node row
to zero, and then the node rows do not sum to the query total — against a
$0.0050 budget and a $0.00019 mean, the per-node numbers are exactly where the
precision has to survive.

Each trace row also carries `billing_unit` (`tokens | gpu_ms | none`) with
`input_tokens`, `output_tokens` and `gpu_ms`. Ollama Cloud is metered in GPU-time
windows and tokens cannot express it; `none` covers the in-process nodes, which
cost nothing and still take time (ADR 0011).

`cost_usd` stays **derived and advisory** until the provider/model/pricing tables
exist — that gap is still open in `CLAUDE.md` and this ADR does not close it. The
metered columns are the ground truth, and the reason they matter is that a priced
number cannot be recomputed after a price change while a meter reading can. A
benchmark whose cost figure cannot be re-derived is not reproducible.

### `query_traces` — a shape that can render its own waterfall

- `status` (`ok | error | pruned | skipped`) on every row, and `verdict`
  (`pass | fail`) nullable and meaningful only on grading nodes. The designed
  `verdict` of `pass|fail|skip` was one column doing two jobs: `plan_query`,
  `retrieve`, `rerank` and `generate` have an outcome but no verdict, and
  `pruned` — the greyed nodes in the mockup — is a status, not a judgement.
- `started_at timestamptz NOT NULL` alongside `duration_ms`. A waterfall needs
  offsets; durations alone cannot place a bar, and nodes that overlap are
  invisible to a sequence number.
- `sequence int NOT NULL` as well, because `started_at` ties under clock
  resolution. ADR 0002 already paid for that lesson: below millisecond
  resolution, ordering by a timestamp-derived value is ordering by chance.
- `attempt smallint NOT NULL DEFAULT 1`. One column suffices — the node name
  identifies which loop the attempt belongs to.
- `error text`, non-null exactly when `status = 'error'`.
- `input_json` / `output_json jsonb` for the node inspector, under one rule:
  **trace payloads store references, not copies.** A `retrieve` output is chunk
  ids with scores, not ten chunks of text; otherwise every query writes ~12KB
  into the highest-volume table to duplicate rows that already exist. Payloads
  are size-capped with an explicit truncation marker, so a truncated inspector
  panel is distinguishable from a node that returned little.

### Citations snapshot what was cited

`query_citations` carries `chunk_ref uuid NOT NULL` — a plain copy of the chunk id
with no foreign key — beside `chunk_id uuid NULL REFERENCES chunks(id) ON DELETE
SET NULL`, plus `document_id`, `page_number`, `chunk_index`, the `rerank_score`,
the rank, and `cited_content text NOT NULL`.

Chunks cascade-delete with their document, and re-ingestion replaces them. With a
hard FK and no snapshot, deleting one document rewrites the history of every
answer that ever cited it, and the audit trail for the single thing this system
claims — that an answer is grounded in a citable span — disappears retroactively.
The two id columns separate two facts that a single nullable column would
conflate: `chunk_ref` is what was cited at answer time and is never null;
`chunk_id` says whether that chunk still exists. "The source has since been
deleted" is a true and useful thing for the UI to say. "This answer had no
sources" is a lie, and it is the one a bare `ON DELETE SET NULL` would tell.

The nullable `chunk_id` is not a door for citing something that was never a chunk.
There is no web-search node, that is settled, and nothing here reopens it.

### "Answered implies citations" is a constraint

`queries.citation_count smallint NOT NULL DEFAULT 0`, written in the same
transaction as the citation rows, with:

```
CHECK ((status = 'refused' AND citation_count = 0)
    OR (status <> 'refused' AND citation_count > 0))
```

A cross-table "has at least one child" rule is not expressible as a CHECK, and
the alternatives are a trigger or a convention. A counter the schema checks is
simpler than a trigger and is also the number the dashboard lists. Note what the
`status <> 'refused'` branch commits to: a **cached** answer must carry citations
too. A cached answer returned without them would break the same promise as an
uncited fresh one, so the cache entry stores its citations and replays them.

### Refusals are not cached

A cached refusal is a wrong abstention with a TTL. Both reasons are statements
about the corpus at a moment — ingest one document and `no_relevant_evidence` and
`insufficient_evidence` can both become false. A stale cached answer is a quality
regression. A stale cached refusal makes the system permanently deny knowledge it
now holds, and does it silently, on the path that is supposed to be the
trustworthy one.

It also corrupts the measurement. Correct refusal is the primary correctness
criterion, so an eval run that reads refusals out of a cache is measuring the
cache and reporting it as abstention quality.

And it is the smallest saving on offer. A refusal is already the cheap path: it
abstains before or instead of the escalated paid generation. Caching refusals
memoizes exactly the queries that cost the least, in exchange for the risk on the
metric that matters most.

Finally it keeps the cache's invariant single: **a cache entry is an answer with
its citations.** Admitting refusals means a nullable answer, a second copy of the
refusal-reason enum, and a branch in every consumer, to store rows whose value is
near zero.

Rejected alternative: cache refusals with a short TTL plus invalidation on
ingest. It is the right design when refusal is expensive — it is not here, and
the invalidation surface it depends on (which collection, which document version)
is itself an open design gap. Building it for the cheapest case first is the wrong
order.

The consequence is that a repeated unanswerable question re-runs the graph every
time, the 72 injection payloads included. That is rate limiting's job (ADR 0008),
not the cache's, and it is the regime the robustness numbers should be measured
under anyway: a cache hit is not a successful defence.

### Partitioning: not yet, and here is the arithmetic

~612 queries/24h at ten to fifteen node rows each is roughly 6–9k trace rows a
day, about 3M a year. An unpartitioned table indexed on `(query_id, sequence)` is
not troubled by that, and partitioning now is not free: the partition key must
appear in every unique constraint, so the primary key becomes
`(id, started_at)` and "UUIDv7 everywhere" stops being a sole primary key,
composite for everything that wants to reference a trace row.

So the decision is not to partition, plus the two things that keep it cheap
later:

1. `started_at NOT NULL` from the first migration, so it can become the range key
   with no backfill.
2. **Nothing gets a foreign key to `query_traces`.** Eval results reference
   `(query_id, node_name, attempt)`; `queries` is the low-volume table and stays
   the FK-able one. This is the constraint that would otherwise make partitioning
   a schema-wide change.

Revisit when `query_traces` passes tens of millions of rows, or when a retention
pass takes longer than the window it reclaims.

### Retention deletes traces and checkpoints together

`CLAUDE.md` requires any query to be replayable and forkable from any node. A
trace row without its checkpoint is a picture of a run you cannot re-enter; a
checkpoint without its trace is a run you cannot read. Retention is therefore per
query, keyed by `thread_id`, and drops both — half-swept queries are worse than
absent ones, because they look replayable until you try.

Queries belonging to an eval or benchmark run are exempt. They are the
reproducibility record, and a benchmark whose traces were swept is a number with
no evidence behind it. The horizon for everything else lives in `Settings`, not
in the schema.

### `tenant_id` on all three tables

Denormalized with composite foreign keys, as migration 0004 did for `documents`
and `chunks`. RLS step 3 is still outstanding (ADR 0006) and needs a column
comparison rather than a per-row subquery, and these are about to become the
highest-volume tables in the system.

## Consequences

- `queries` is wide and denormalized — `citation_count`, `tenant_id`, both
  attempt counters. Each one is there to make a constraint or a policy expressible
  without a join, and each has to be written in the same transaction as the rows
  it summarizes.
- The writer must set `citation_count` and insert citations atomically. A partial
  write is rejected by the CHECK rather than stored, which is the intended
  failure: an answered query with no citations should be impossible to persist.
- Trace payloads holding references means the inspector resolves chunk ids at read
  time, and a chunk deleted since the run shows as missing. Citations do not have
  that problem, because they snapshot on purpose.
- Storing `question` and `final_answer` puts user text and injection payloads in a
  table the dashboard reads. Retention is the only control over that, which is one
  more reason the horizon is a setting rather than a hardcode.
- Not closed here, and not built over: the provider/model/pricing tables, the
  cache's scoping columns, the benchmark's data model,
  `GOLDEN_QUESTIONS.is_unanswerable`, and citation character offsets. The
  citations table is shaped so offsets arrive as two nullable columns against
  `cited_content`, additive and without a backfill.
- Refusals re-running the graph makes refusal latency a real p95 contributor
  rather than a cached constant. That is the honest number to publish, and it is
  the one the abstention path should be optimized against.
