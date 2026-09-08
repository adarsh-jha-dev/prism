# 0001 — Phase 0 creates the five core tables, not a smoke table

- **Status:** accepted
- **Date:** 2026-09-08
- **Supersedes:** the "Phase 0 database scope: smoke test only" line in `CLAUDE.md`

## Context

Phase 0's job is to prove the environment works. The original plan was a single
migration that enabled pgvector and created a throwaway `_smoke_vectors` table
purely to prove an HNSW round-trip, deferring all real tables until the design
contradictions in `docs/design/REVIEW.md` were resolved.

Those contradictions are real, but they are not evenly distributed. Every one of
them sits in a table Phase 0 does not need:

| Blocked table | Blocked on |
|---|---|
| `cache_entries` | no `collection_id`, no `embedding_model`, no `expires_at` |
| `query_traces` | nothing models providers/models/pricing, so `cost_usd` has no basis |
| `query_citations` | no character offsets, but the UI highlights an exact span |
| `golden_questions` | no `is_unanswerable`, so refusal rate is uncomputable |
| benchmark tables | no data model at all |

The five tables ingestion and tenancy need — `tenants`, `collections`,
`documents`, `chunks`, `api_keys` — are not blocked on any of them. The two ER
diagrams agree on their shape, and the one place they disagreed (`string` vs
`bigint` ids) is already settled as UUIDv7.

## Decision

Create those five tables in migration `0001`, and drop the smoke table.

The HNSW index on `chunks.embedding` *is* the round-trip proof, so nothing is
lost by deleting `_smoke_vectors` — the integration test now seeds a real
tenant → collection → document → chunk chain and asserts nearest-neighbour
ordering against the index Phase 2 will actually query.

Everything in the table above stays unbuilt until its contradiction is resolved.

## Consequences

- Phase 1 starts against a schema that exists, instead of writing it first.
- The integration test exercises the real index, not a parallel one that could
  drift from it.
- `chunks` carries `collection_id` denormalized from `documents`. This is the
  one shape decision not taken from the ER diagrams. Tenant scoping must be a
  `WHERE` predicate *inside* the ANN query — joining out to `documents` to
  discover the collection would defeat the HNSW index and turn every retrieval
  into a scan. Adding the column later would mean a backfill.
- Ids have no database-side default. Postgres 16 has no native `uuidv7()`, and a
  `gen_random_uuid()` default would silently emit v4 ids for any insert that
  forgot one, quietly losing the time-ordering the index locality depends on.
  When the target moves to Postgres 18 this can become a column default with no
  data migration — the wire format is identical.
