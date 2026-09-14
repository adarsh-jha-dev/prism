# 0010 — Full-text is Postgres FTS, and nothing is allowed to say BM25

- **Status:** accepted
- **Date:** 2026-09-14
- **Relates to:** [0001](0001-phase-0-schema-scope.md),
  [0006](0006-postgres-rls-for-tenant-isolation.md)
- **Closes:** "BM25 vs Postgres FTS" in `docs/design/REVIEW.md`

## Context

`retrieve` is specified as "hybrid BM25 + HNSW over pgvector". The vector half
exists (`chunks.embedding`, HNSW, cosine). The lexical half does not: no
`tsvector` column, no GIN index, no fusion step. Before building it, two things
have to be settled together, because the second is a consequence of the first.

**Postgres FTS is not BM25.** `ts_rank` and `ts_rank_cd` score a document from
the term frequency and position of its own matching lexemes. They use no
corpus-level statistics at all — no document frequency, so no IDF. BM25 is
defined by three things Postgres FTS has none of: saturating term frequency
(`k1`), document-length normalization (`b`), and IDF. A rare discriminating term
and a near-stopword contribute the same weight under `ts_rank_cd`. Calling it
BM25 is not a naming imprecision, it is a claim about the ranking function that
does not hold.

**The alternative is a different database.** Real BM25 in Postgres means
ParadeDB's `pg_search` — Tantivy-backed, `@@@`, `paradedb.score()`. The cost is
not the extension, it is the distribution. `pg_search` is not in the extension
allowlist on RDS or Aurora, and it is not in `pgvector/pgvector:pg16`. Adopting
it replaces the database image for dev, CI and production, and gives up managed
Postgres on the cloud this is designed for.

Corpus size does not break the tie. At honest portfolio scale — a 31-query
golden set over a small corpus — IDF has little to separate, and any measured
delta would be noise dressed as a finding. The tie is broken on deployability
and on what the benchmark is allowed to claim.

## Decision

**Postgres FTS.** `chunks` gains a stored generated `tsvector` column over
`content` and a GIN index; `retrieve` ranks with `ts_rank_cd`.

Generated and stored, not an expression index, and always the two-argument
`to_tsvector('english', content)` — the one-argument form reads
`default_text_search_config` and is only STABLE, so it is not indexable and not
usable in a generated column. Pinning the config in the column definition also
means a server-side GUC change cannot silently reinterpret the corpus.

**Tenant scope goes inside the index scan, same rule as the ANN query.** The
GIN index is multicolumn over `(tenant_id, collection_id, content_tsv)` via
`btree_gin` (contrib; present in the pgvector image and allowlisted on RDS). A
lexical hit from another tenant must not occupy a top-k slot any more than a
vector one may, and a post-filter over a `LIMIT`ed lexical scan has the same
defect as a post-filter over ANN results.

**User text is parsed with `websearch_to_tsquery`, never `to_tsquery`.**
`to_tsquery` raises on unbalanced input, which turns a question mark in a
question into a 500.

**Fusion is reciprocal rank fusion, and it consumes ranks only.** `ts_rank_cd`
is unbounded and has no shared unit with cosine similarity; adding or weighting
them together would be arithmetic on incommensurable quantities. RRF
(`1/(k + rank)`, `k = 60`) never looks at either score's magnitude.

**No score produced by the lexical half or by fusion is ever compared against a
threshold.** Not `abstention_threshold`, not `rerank_score_floor`. Relevance is
decided by `grade_docs` and by the reranker's own score; groundedness is decided
by tau at `verify_grounding`. A fused rank is an ordering, not a measurement —
this is the same failure mode `CLAUDE.md` already forbids for cosine similarity
against tau, and the lexical ranker adds a second quantity that invites it.

### What the UI, the docs and the benchmark are allowed to claim

Forbidden everywhere user-visible: **BM25**, Okapi, and any phrasing implying
IDF or corpus-weighted term scoring. This covers the dashboard, `CLAUDE.md`, the
API docs, the eval report and the benchmark writeup.

Allowed: "hybrid retrieval", "lexical + vector", "Postgres full-text search",
"`ts_rank_cd`", "reciprocal rank fusion".

The rule is enforced by data, not by style review: the `retrieve` trace row
records the lexical ranker it actually ran (`postgres_fts`), and the dashboard
renders that value rather than a hardcoded label. If the ranker is ever swapped,
the UI changes with it or the trace rows disagree with the screen.

## Consequences

- `ts_rank_cd` will rank a chunk highly for matching a common word many times.
  The reranker is the mitigation, not the lexical score, which is one more
  reason fusion must not leak a magnitude downstream.
- `websearch_to_tsquery` ANDs unquoted terms, so a prose question of a dozen
  words will often return nothing from the lexical half. That is `retrieve`'s
  problem to solve — from the terms `plan_query` extracts, not by loosening the
  parser — and the golden set is where it gets measured. Zero lexical hits is a
  normal outcome and must not be treated as an error.
- The generated column roughly doubles per-chunk text storage and makes ingest
  writes more expensive. Acceptable at this scale, and it keeps the tsvector
  from being recomputed per query.
- The `english` config is hardcoded corpus-wide. A non-English document is
  stemmed wrongly and stays retrievable only by the vector half. Per-collection
  text search config is a later decision, and it would need the same
  invalidate-on-change treatment as the embedding model.
- Changing the lexical ranker later is cheap by construction: RRF consumes
  ranks, so a swap touches one SQL block, one migration and one trace label —
  not the fusion math and no threshold. Revisit if the corpus grows enough for
  IDF to separate anything measurable on the golden set, or if the deployment
  target stops being managed Postgres. Check `pg_search`'s licence terms at that
  point; they were not evaluated here.
- The naive baseline stays vector-only. The benchmark compares against
  single-shot paid-API retrieval, and giving the baseline a lexical half it
  never had would flatter the optimized path's recall story.
