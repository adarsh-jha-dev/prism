# Design review — 2026-09-08

Review of the artifacts in this directory, done before any code was written.
Findings are grouped by severity. `CLAUDE.md` records which of these have since
been resolved by decision.

Sources reviewed:
- `Tenant Management Ecosystem-*.png` (ER, `string` PKs)
- `Tenant-Centric API Key-*.png` (ER, `bigint` PKs)
- `RAG Query Processing-*.png` (LangGraph flow)
- `UI Mockups.pdf` (10 screens: Overview, Design system, Query console, Trace
  viewer, Cost & performance, Eval regression, Robustness, Collections & keys)

---

## Blockers — break a stated requirement

### B1. The semantic cache has no tenant/collection scoping column
A cross-tenant cache hit is a data leak, per the project constraints. But
`CACHE_ENTRIES` is `(id, query_embedding, answer, hit_count, created_at)` in
both diagrams. The Tenant Management ER draws a `caches_for` edge from
`COLLECTIONS` without a corresponding attribute; the Tenant-Centric ER has no
edge to `COLLECTIONS` at all, only `memoizes` -> `QUERIES`.

A drawn relationship is not a predicate. Needs `collection_id NOT NULL`, and the
scope must be inside the ANN `WHERE` clause — a post-filter still lets a
neighbour from another tenant consume the top-k slot.

### B2. `CHUNKS.embedding vector(768)` contradicts the Collections screen
The mockup shows four embedders across six collections: `text-embedding-3-large`
(3072), `multimodal-vision-v2`, `bge-m3` (1024), `text-embedding-3-large`. One
fixed-width column cannot hold all three dimensionalities.

Worse: **pgvector caps HNSW indexes at 2000 dimensions.** `text-embedding-3-large`
at 3072 dims cannot be HNSW-indexed as `vector` at all — it needs `halfvec`
(4000-dim index cap) or Matryoshka truncation. `COLLECTIONS` also has no
`embedding_model` column despite the UI showing one per collection.

RESOLVED: one model, one dim, project-wide (768). See CLAUDE.md.

### B3. The flow diagram has an infinite loop and a missing terminal state
Spec says the retrieval loop terminates as "no relevant evidence". The diagram
routes exhausted retrieval -> Web Search MCP -> "Re-evaluate" -> Relevance Grader
-> FAIL -> Max Retries -> ... with no exit edge. There is no `END: No relevant
evidence` node; the only refusal terminal hangs off the groundedness loop.

Given that correct refusal is the primary correctness criterion, the missing
refusal path is the wrong thing to have missing.

### B4. Ollama Cloud's cost is unrepresentable
`QUERY_TRACES` has `input_tokens`, `output_tokens`, `cost_usd`. Ollama Cloud is
metered in GPU-time windows, not tokens. No `billing_unit` or `gpu_ms` column
exists, so that provider's cost cannot be recorded — and cost-per-query is half
the headline benchmark.

There is also no `providers` or `model_pricing` table anywhere. `cost_usd` is a
computed number with no recorded price basis; when prices change, historical
comparisons stop being reproducible and cannot be recomputed.

### B5. The headline benchmark has no data model
"Naive baseline vs optimized, p95 and cost-per-query under concurrent load" is
the deliverable, and the Cost & performance screen has a panel for it (-51.5%,
-47.4%). No table holds benchmark runs, concurrency level, or baseline config.

Separately, `EVAL_RUNS` stores `accuracy`, `groundedness`, `refusal_rate` but not
p95 or cost — though the Eval regression screen shows a p95 column per commit.

### B6. Correct refusal cannot be distinguished from wrong refusal
`GOLDEN_QUESTIONS` is `(id, question, expected_answer)`. No `is_unanswerable`
flag, no `collection_id`, no expected citations. Refusal rate is meaningless
without knowing which questions *should* be refused. The metric the whole thesis
rests on is not computable from this schema.

### B7. Web search contradicts the robustness claims
The Robustness screen states "no network egress from generation nodes" and "no
tool call reached an external host". The flow has a web-search MCP fallback.

Also `QUERY_CITATIONS.chunk_id` is an FK to `CHUNKS` — a web result cannot be
cited without either a polymorphic citation source or ingesting results as
ephemeral chunks. Neither is designed.

---

## Inconsistencies between artifacts

| # | Conflict |
|---|---|
| I1 | ER diagrams disagree on PK type: `string` vs `bigint`. RESOLVED: UUIDv7. |
| I2 | `GOLDEN_QUESTIONS` exists only in the Tenant-Centric ER, though the other references `golden_question_id`. |
| I3 | `ROBUSTNESS_RUNS` has `tenant_id` in one ER, not the other. |
| I4 | Mockups show `plan_query`, `embed_multimodal`, `rerank` nodes; the flow diagram has none of them. The two artifacts describe different graphs. |
| I5 | Flow pins the relevance grader to "Local Ollama"; mockups run `grade_docs` on `llama-3.3-70b · groq` (paid). |
| I6 | Retry limit is "max 2" (spec), "Yes, 2 Attempts" (flow), `MAX RETRIES 3` / `retry 1/3` (Query console). Also ambiguous whether 2 means retries or total attempts. CANONICAL: max 2 retries = 3 attempts, per loop, counted independently. |
| I7 | Provider rosters differ three ways: spec (5), flow cascade (4), mockups (7). RESOLVED: the four in CLAUDE.md. |
| I8 | Model names in mockups are stale (`claude-sonnet-4.5`, `gpt-4.1`, `gemini-2.5-pro`). |
| I9 | Golden set size is "2,412" on three screens, "2,400" on another. |

---

## Underspecified

**Cache**
- No write-cache node exists anywhere in the graph. The cache is read but never
  populated. Open decision: **are refusals cached?** It saves money and locks in
  a wrong abstention.
- `CACHE_ENTRIES` missing `embedding_model` (a model swap silently poisons the
  cache), `expires_at`, an invalidation link to document versions, and the
  citations — a cached answer returned without citations breaks the "every claim
  carries a citation" promise.
- `hit_count` alone cannot produce the "48.6% cache hit rate" chart; that needs
  hit events with timestamps, or a rollup, plus miss counts from `QUERIES`.

**Tracing**
- `QUERY_TRACES` cannot render its own mockup. Missing: `input_json`/`output_json`
  (the Trace viewer shows a node-output inspector), `started_at` (a waterfall
  needs offsets, not just durations), `attempt` index (UI shows `retrieve ·
  attempt 1` / `attempt 2`), a status for the "2 pruned" greyed nodes, `error`.
- `verdict` is `pass|fail|skip`, but `plan_query`, `retrieve`, `rerank` and
  `generate` produce no verdict.
- Highest-volume table in the system, with no partitioning or retention plan.

**Queries**
- `was_refused boolean` cannot express three terminal states plus two refusal
  causes. Needs `status` + `refusal_reason`.
- No `cache_entry_id` — cannot tell which cached answer served a cached query.
- `retry_count` is one integer for two independent loops.
- `total_cost_usd decimal` unqualified; should be `numeric(12,6)` or micro-dollar
  bigint.
- No `thread_id`. LangGraph's Postgres checkpointer owns its own tables keyed by
  `thread_id`; a nullable per-trace `checkpoint_ref` is not enough to replay or
  fork a whole query.

**Retrieval**
- "Hybrid BM25 + HNSW" has no full-text column — no `tsvector`, no GIN index.
  Postgres FTS is `ts_rank`/`ts_rank_cd`, **not** BM25. Either adopt ParadeDB /
  `pg_search`, or stop claiming BM25 in the UI copy.
  RESOLVED: Postgres FTS, and nothing says BM25. See ADR 0010.
- No fusion (RRF) configuration. RESOLVED: RRF, `k = 60`, ranks only (ADR 0010).
- `rerank` appears in mockups and the eval commit log but is absent from the flow
  diagram and the schema.

**Multimodal and citations**
- `QUERY_CITATIONS` has no character offsets, but the Query console highlights
  specific sentences inside a chunk.
- No bounding-box storage, despite "the parsed bounding box for a table or
  figure". `CHUNKS.metadata jsonb` could hold it but nothing specifies the shape.

**Tenancy**
- `API_KEYS.rate_limit_rpm` is RPM, but the UI shows *quotas* (`341k / 500k
  queries`) — a different thing. No usage-metering table exists at all, despite
  metering being a stated constraint.
- Missing `key_prefix` — without an indexed prefix you must hash every row to
  look up a key.
- Missing `scopes` (read vs ingest vs admin), `last_used_at`, `expires_at`, and
  the per-key `abstention_threshold` override the UI advertises.
- Isolation is FK convention only. Postgres RLS is the thing worth demonstrating
  here, and it affects every query and session setup — cheaper to decide early.

**Documents and collections**
- `DOCUMENTS` has no `content_hash` (no dedup, no re-ingest detection, no cache
  invalidation key), no version, no `size_bytes`, no error field — though the UI
  shows "41 docs failed OCR, quarantined".
- `status` is a bare string with no enum; UI implies `ready|ingesting|partial|
  cold|quarantined`.
- `COLLECTIONS` has no status or size column, though the UI shows both.

**Eval and robustness**
- `EVAL_RUNS` does not version the golden set. Comparing accuracy across commits
  is invalid if the question set changed underneath.
- `EVAL_RESULTS` has no score, latency, cost, or link to a real trace — but the
  UI offers "Open 42 failing traces".
- `ROBUSTNESS_RUNS` stores only aggregates, while the mockup shows per-payload
  categories, before/after mitigation bars, and finding IDs (PRISM-041).
  No per-payload results table and no findings table.

**Graph control flow**
- No error, timeout, or circuit-breaker edges anywhere, despite the circuit
  breaker being a stated constraint.
- No budget-exceeded terminal — the router picks "the cheapest provider meeting
  the budget", with no edge for when none does.
- No query-embedding node, though both the cache check and retrieval need one.
- Cache check is drawn before any tenant scoping, implying a global lookup.
- Ollama Cloud's concurrency cap of 1 has no queue or semaphore in the design.
  Under concurrent-load benchmarking it will dominate p95.

**Cosmetic**
- "Write Trace &" is truncated in every trace box in the flow diagram.
- No `updated_at` on any table; no soft deletes.
- "SOC 2 in progress" on the Overview screen — a claim worth dropping on a
  portfolio project.

---

## What the design gets right

- Three terminal states are clean and complete.
- Per-node trace rows carrying provider, model, cost and latency is exactly the
  right grain for the benchmark.
- Commit-linked eval runs is real regression discipline, not decoration.
- Per-collection abstention threshold is a good primitive to have identified
  this early.
