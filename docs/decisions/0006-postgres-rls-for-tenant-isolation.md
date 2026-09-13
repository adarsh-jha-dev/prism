# 0006 — Postgres RLS for tenant isolation

- **Status:** accepted
- **Date:** 2026-09-13
- **Relates to:** [0001](0001-phase-0-schema-scope.md)

## Context

Tenant isolation today is foreign-key convention and nothing else. `collections`
and `api_keys` carry `tenant_id`; `documents` and `chunks` do not, reaching the
tenant only through `collections`. Every query is expected to carry the right
predicate, and `REVIEW.md` flags this as the one design gap that "affects every
query and session setup — cheaper to decide early".

The gap is not hypothetical. `search_chunks` scopes by `collection_id` with no
check that the caller's tenant owns that collection, and its own docstring
asserts that "at this layer that predicate is the whole of tenant isolation".
Any caller holding a collection UUID reads that collection. Fixing that
specific query is a separate piece of work; this ADR decides whether the
database should also refuse it independently.

Two mechanisms are available:

- **Explicit predicates.** Every query names the tenant. Enforcement lives in
  application code and is only as good as its least careful query.
- **Row-level security.** Policies on each tenant-scoped table read a session
  GUC (`SET LOCAL prism.tenant_id`), and Postgres filters rows regardless of
  what the query asked for.

## The case for RLS

**It fails closed.** A query that forgets its predicate returns nothing rather
than everything. That inverts the default: today the safe outcome depends on
remembering, and the unsafe one is what you get by omission.

**It covers paths that are not routes.** `eval/ingest.py`, Alembic data
migrations, the eventual LangGraph nodes and anything run from a psql prompt all
reach the same tables. An application-layer rule binds only the code that
imports it.

**It is a real answer to "prove the isolation".** A test asserting that tenant A
cannot read tenant B's chunks proves something much stronger when the query is
issued without any tenant predicate at all and still comes back empty.

**Cross-tenant leakage here is the worst failure the system has.** CLAUDE.md
already singles out the semantic cache, where a cross-tenant hit is a data leak.
Defence in depth is proportionate to that.

**It is the phase's demonstrable artifact.** REVIEW.md's framing is right:
multi-tenant RAG platforms are judged on isolation, and "RLS with session GUCs"
is a more credible claim than "we were careful".

## The case against

**RLS cannot be the ANN query's scoping mechanism, and that is the query that
matters.** CLAUDE.md requires tenant scoping to be a predicate *inside* the ANN
query, never a post-filter. An RLS policy is a qual applied to rows the scan
produces. With `ORDER BY embedding <=> :q LIMIT k` served from the HNSW index,
the index yields its candidates in distance order and the policy then discards
the ones belonging to other tenants — so a query can come back with fewer than
`k` rows, looking exactly like a collection holding little. `search.py` already
documents this failure mode for the `collection_id` predicate. RLS reintroduces
it at a layer the query cannot fix.

**`chunks` has no `tenant_id` to write a policy against.** The only policy
expressible today is a subquery — `collection_id IN (SELECT id FROM collections
WHERE tenant_id = ...)` — evaluated per candidate row inside the hot retrieval
path. Making it a plain column comparison means denormalizing `tenant_id` onto
`chunks` (and `documents`), which is the right schema change but is a migration
over the largest table, not a policy toggle.

**Session setup becomes a correctness requirement of every connection.** The GUC
must be set inside the same transaction on the same pooled connection as the
query. `search_chunks` currently acquires two connections — one for
`assert_compatible`, one for the search — from a `pool_size=10` engine. Under
RLS that is no longer an implementation detail: a missed `SET LOCAL` on either
turns a working query into an empty result, and empty results are the one
symptom this system is designed to produce legitimately. A refusal caused by a
plumbing bug is indistinguishable from a correct refusal.

**It lands on top of the test suite.** `conftest.py` inserts tenants and
collections directly, `eval/ingest.py` writes its own rows, and Alembic runs as
the table owner, which bypasses RLS unless `FORCE ROW LEVEL SECURITY` is set —
at which point the migrations themselves need a bypass role. None of that is
hard; all of it is work that buys no isolation the explicit predicate does not
already buy for the routes.

**It is not free to reverse either way.** Adopting RLS is a migration plus
session plumbing on every query path. Removing it later is another migration
plus unwinding that plumbing. The expensive half is the plumbing, not the
policies.

## Recommendation

**Adopt RLS, but as a backstop behind explicit predicates — and not yet.**

1. **Now:** enforce tenancy as an explicit predicate inside the ANN query, and
   resolve the tenant from the API key rather than trusting a caller-supplied
   `collection_id`. This closes the actual vulnerability, keeps the scope inside
   the index where CLAUDE.md requires it, and does not depend on session state.
2. **Next, as its own change:** denormalize `tenant_id` onto `documents` and
   `chunks`, so a policy can be a column comparison rather than a subquery and
   so the composite index can carry it.
3. **Then:** enable RLS with `FORCE ROW LEVEL SECURITY`, a `prism.tenant_id`
   GUC set per request, and a separate migration role that bypasses it.

The ordering is the substance of the recommendation. RLS first would delay the
security fix behind a schema migration and a session-plumbing change, and would
risk shipping the short-result failure mode into the one query whose result
count is load-bearing. Predicates first makes the isolation correct; RLS
afterwards makes it hard to get wrong again.

The case against RLS as a *mechanism* is strong. The case against it as a
*second layer* is only that it costs work — and the failure it catches is the
exact bug now sitting in `search_chunks`, which is the best evidence available
that the first layer is not sufficient on its own.

## Consequences

- Step 7's tenant scoping is written to be correct on its own, not as a
  placeholder for RLS.
- A follow-up migration adds `tenant_id` to `documents` and `chunks`. Both are
  `NOT NULL` backfills over the largest tables in the schema.
- Request handling gains a per-transaction `SET LOCAL`, and the two-connection
  shape of `search_chunks` has to become one transaction.
- Tests gain a case asserting that a query issued with *no* tenant predicate
  still returns nothing across tenants — the assertion only RLS can satisfy.

Step 2 landed with the step-7 security fix rather than after it: the predicate
needed a column to name, so migration 0004 denormalizes `tenant_id` onto
`documents` and `chunks` with composite foreign keys against
`collections (id, tenant_id)` and `documents (id, tenant_id)`. A row whose
`tenant_id` disagrees with its parent is now rejected by the database, so the
column the isolation predicate reads cannot drift.

Step 3 — the policies, the GUC and the bypass role — remains outstanding.
