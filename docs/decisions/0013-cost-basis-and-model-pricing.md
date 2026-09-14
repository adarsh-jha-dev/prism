# 0013 — `cost_usd` needs a price basis: `model_pricing`, effective-dated and append-only

- **Status:** accepted
- **Date:** 2026-09-14
- **Relates to:** [0011](0011-rerank-runs-in-process.md),
  [0012](0012-query-trace-and-citation-model.md)
- **Closes:** B4 in `docs/design/REVIEW.md`, and the pricing half of the open gap
  ADR 0012 deliberately left standing

## Context

ADR 0012 settled the meter: every trace row carries `billing_unit`
(`tokens | gpu_ms | none`) with `input_tokens`, `output_tokens` and `gpu_ms`, and
`cost_usd numeric(14,8)`. It explicitly did not settle the price, and said why —
a priced number cannot be recomputed after a price change, while a meter reading
can.

That leaves `cost_usd` as a number with no recorded basis. Cost-per-query is half
the headline benchmark, and "−95% cost versus a single-shot paid-API baseline" is
a claim about dollars derived from prices that move. Today those prices would
live in whatever constant the writer happened to reference at the time, which
means the benchmark is a number nobody can re-derive and nobody can audit.

Phase 2 only ever writes local-lane rows at $0. That is exactly why the shape has
to be right now: the columns are cheap to add before traces accumulate and
expensive afterwards, and a table full of $0 rows with no price basis looks
correct right up to the first paid call.

## Decision

### `$0` is a price, not an absence

Local Ollama and the in-process lane get real `model_pricing` rows at zero, with
effective dates, like every other provider. This is the load-bearing choice in
the ADR, because it makes one distinction enforceable:

```
CHECK ((price_id IS NULL) = (cost_usd IS NULL))
```

`price_id IS NULL` then means exactly one thing — **we could not price this row** —
and it is an error, not a freebie. If unknown prices defaulted to zero instead,
a missing price row would quietly deflate cost-per-query, and the direction of
that error flatters the headline number. A cost figure that fails toward "cheaper"
is the one failure mode this project cannot afford to have silently.

Free is therefore priced, and unpriced is not free.

### `model_pricing`, effective-dated

```
id             uuid PK
provider       text NOT NULL     -- ollama | ollama-cloud | gemini | openai | in-process
model          text NOT NULL
billing_unit   text NOT NULL     -- tokens | gpu_ms | none
input_per_mtok    numeric(12,6)  -- tokens only
output_per_mtok   numeric(12,6)  -- tokens only
usd_per_gpu_hour  numeric(12,6)  -- gpu_ms only
effective_from timestamptz NOT NULL
effective_to   timestamptz       -- null = current
source         text NOT NULL
```

A CHECK keyed on `billing_unit` requires the matching columns and forbids the
others — `tokens` rows carry the two per-Mtok figures and no GPU rate, `gpu_ms`
rows the reverse, `none` rows neither. Same reasoning as ADR 0012: an invariant
that can be a constraint is one, because these rows will be inserted by
migrations and by hand.

**Prices are stored as published — per million tokens, per GPU-hour.** Converting
to per-token or per-millisecond at write time bakes in a rounding decision that
nobody records and that cannot be recovered from the stored value. The conversion
happens once, in the pricing function, against `numeric`.

`source` is `NOT NULL` because a price without provenance is a guess wearing a
decimal point. It records where the figure came from and when it was read.

### Overlap is an exclusion constraint, not a convention

```
EXCLUDE USING gist (provider WITH =, model WITH =,
                    tstzrange(effective_from, effective_to) WITH &&)
```

via `btree_gist` (contrib; present in the pgvector image and allowlisted on RDS).
Two overlapping rows for one model make "the price at time T" ambiguous, and the
ambiguity would surface as a silently wrong benchmark rather than an error.

Gaps between ranges are *allowed*. A trace whose timestamp falls in no range is
unpriced, `price_id` and `cost_usd` are both null, and that is the loud failure
the equivalence CHECK above exists to produce.

### Rows are append-only; closing a range is the only update

A price row is never edited in place. Updating one retroactively rewrites every
historical cost that points at it, which is the precise bug this table exists to
prevent. A price change closes the current row by setting `effective_to` and
inserts a successor — that one column is the only permitted UPDATE, and it is
worth enforcing with a trigger rather than a comment, since the tempting wrong
move is a one-line `UPDATE model_pricing SET input_per_mtok = …`.

### The trace row records which price it used

`query_traces.price_id` is a nullable FK to `model_pricing`. Cost is computed at
write time and frozen, *and* the basis is recorded. Both halves are needed and
they answer different questions:

- The frozen `cost_usd` is what the query cost when it ran. It must not move.
- The `price_id` makes the number re-derivable, so a reader can verify it was not
  fabricated, and so a historical benchmark can be deliberately re-priced under a
  different date's prices.

Compute-at-read alone was rejected: deriving cost from current prices at query
time means a benchmark's published figure changes retroactively when a provider
cuts its rates, which is reproducibility failing in the other direction.

### GPU-time is not wall-clock, and it is not queue wait

Ollama Cloud is metered on GPU-time and pinned to concurrency 1, which
`CLAUDE.md` notes will dominate p95 under concurrent load. Those two facts
interact: the node's `duration_ms` includes time spent waiting on the local
semaphore, and billing that time would charge money for our own serialization
point — counting the architectural bottleneck twice, once as latency and once as
cost.

So `gpu_ms` is the time the request was actually being served, excluding local
queue wait, and the provider's reported billed window is preferred over our
stopwatch whenever it reports one. A row priced from our own measurement is
marked `cost_basis = 'estimated'` rather than `'metered'`, and estimated rows are
excluded from the headline cost figure — the same rule ADR 0011 applies to
degraded nodes. An estimate is publishable as an estimate or not at all.

### The baseline's dollars come from the same table

The benchmark compares against a single-shot paid-API baseline, and CI must never
make a paid call. Those are compatible because the record-and-replay fixtures
carry token counts: the baseline's meter is real, recorded once, and its dollar
figure is computed from `model_pricing` with no key present and nothing spent.

This makes the pricing table load-bearing for the headline number rather than
decoration. It also means paid-provider price rows exist in the schema long
before any paid lane is wired up, which is fine — a price row is not a
credential.

### A partial sum is worse than no sum

`queries.total_cost_usd` equals the sum of its trace rows' `cost_usd`, exactly —
8 decimal places in `numeric` sum without loss. If **any** trace row is unpriced,
the total is null, not a partial sum. A partial sum is a plausible-looking
number that is wrong in the flattering direction, and it would pass every
eyeball test.

### `model_pricing` is not tenant-scoped

It is the first table in the schema with no `tenant_id`: prices are global
reference data, not tenant data. RLS does not apply to it, and the RLS work still
outstanding from ADR 0006 should not grow a policy for it. Read access is
effectively public within the deployment; write access belongs to migrations.

### What phase 2 actually writes

A seed migration inserts `$0` rows for the local Ollama models and the in-process
lane, `effective_from` set to the migration's own date. Paid-provider rows land
with their lanes, each with a `source`. Until then every trace row is priced,
every cost is zero, and every zero has a basis — which is the state the schema
has to be able to express before it can express anything harder.

A test asserting `cost_usd > 0` for a local-lane row is a bug. A test asserting
`price_id IS NOT NULL` for every trace row is the one worth writing.

## Consequences

- `query_traces` gains `price_id` and `cost_basis`, both settled now, while the
  only rows being written are free ones. That is the point: adding them later
  means backfilling a table with no recoverable basis to backfill from.
- Every new model or provider needs a price row before its first call, or its
  traces are unpriced and its queries have a null total. This will be felt as
  friction the first time a model is swapped, and the friction is the feature.
- `btree_gist` joins `btree_gin` (ADR 0010) as a required contrib extension. Both
  ship with the pgvector image and both are allowlisted on RDS, so neither
  constrains the deployment target.
- Re-pricing is a deliberate, explicit operation against a chosen date — not a
  side effect of reading. "What would this benchmark have cost at today's prices"
  becomes answerable, which is a more interesting claim than the frozen figure
  and was previously not computable at all.
- The `−95% cost` headline now has an auditable derivation: recorded meters,
  recorded prices, recorded provenance. Anyone can recompute it, and anyone can
  find the row that made it wrong.
- Still open, and still not built over: the cache's scoping columns, the
  benchmark's own data model, `GOLDEN_QUESTIONS.is_unanswerable`, and citation
  character offsets. This ADR closes the price basis only.
