# Prism

Self-corrective multimodal RAG platform. Portfolio project demonstrating
**infrastructure and inference-cost optimization**, not RAG novelty.

The headline deliverable is a benchmark: **naive baseline vs optimized, measured
on p95 latency and cost-per-query under concurrent load.** The RAG is the least
interesting part. The engineering is in the correction loop, cost-aware provider
routing, caching, tracing, and eval discipline. When a tradeoff arises, favor the
one that makes the benchmark honest and reproducible.

## The one rule that overrides the others

**Correct refusal is the primary correctness criterion, not a fallback.** Prism
must say "insufficient evidence" rather than answer ungrounded. A confident wrong
answer is a worse outcome than no answer. Never add a code path that degrades to
"answer anyway" — including on timeout, provider failure, or budget exhaustion.

## Stack

| Layer | Choice |
|---|---|
| Orchestration | Python 3.12, FastAPI, LangGraph |
| Dashboard | Next.js (App Router), TypeScript, shadcn/ui, Recharts |
| Storage | Postgres 16 + pgvector (HNSW) |
| Cache / queue | Redis 7 |
| Local inference | Ollama — all dev and CI, no paid calls |
| Python tooling | uv, ruff, mypy, pytest |
| Node tooling | pnpm |
| Migrations | Alembic |
| CI | GitHub Actions |

The dashboard is a **thin client over REST/WebSocket**. No business logic in the
web app — if the dashboard needs a computed number, the API computes it.

## Layout

```
services/api/     Python: FastAPI + LangGraph orchestration service
apps/web/         Next.js dashboard
docs/design/      Source design artifacts (ER diagrams, flow, UI mockups)
docs/decisions/   ADRs for choices that are expensive to reverse
scripts/          Dev helpers
```

Python and TypeScript stay cleanly separated — no shared build tool, no shared
lockfile. `Makefile` at the root is the only cross-language entry point.

## Core behavior — the LangGraph StateGraph

Node names are canonical — they appear in trace rows and in the operator
dashboard, so they must match exactly.

```
query
  -> semantic cache check      (scoped by collection; hit => END: cached)
  -> plan_query                decompose, detect modality, pick collections
  -> embed_query               nomic-embed-text, 768-dim
  -> retrieve                  hybrid BM25 + HNSW over pgvector
  -> grade_docs                local grader: can this set support an answer?
       pass -> rerank
       fail -> rewrite_query, retry retrieval
               attempts exhausted (3) -> abstain
                                      -> END: refused (no_relevant_evidence)
  -> rerank                    bge-reranker-v2-m3, in-process, floor 0.44
  -> cost-aware router         cheapest provider meeting cost/latency budget
  -> generate                  local first, escalate under budget
  -> verify_grounding          re-check the answer against its own citations
       pass (>= tau) -> END: answered (with citations)
       fail          -> regenerate via router with stricter constraints
                        attempts exhausted (3) -> abstain
                                               -> END: refused (insufficient_evidence)
```

**There is no web-search fallback.** An earlier design had one; it was dropped
because it contradicts "no network egress from generation nodes" and because a
web result cannot satisfy `query_citations.chunk_id`. Do not reintroduce it
without resolving both.

`abstain` is a real node with a policy model (`policy · tau`), not an implicit
edge. It writes a trace row like any other node — a refusal must be as
inspectable as an answer.

**Three terminal states: `cached`, `answered`, `refused`.** Refusal carries a
reason (`no_relevant_evidence` | `insufficient_evidence`) — a boolean cannot
represent this.

Every node writes a trace row and a LangGraph checkpoint. Any query must be
replayable and forkable from any node.

## Constraints that shape the design

**Providers.** One interface, with a circuit breaker. Cheap-first cascade;
escalate only on grading failure.

| Lane | Concurrency | Role |
|---|---|---|
| `ollama` (self-hosted) | 8 | graders, rewrite, embeddings, most generation |
| `ollama-cloud` | **1 — pinned** | GPU-time metered, not tokens |
| `gemini` | 4 | multimodal ingestion, escalated generation |
| `openai` | 2 | final escalation, used sparingly |
| `in-process` | 8 | pgvector, rerank — not a provider, bound by worker count |

Ollama Cloud's cap of 1 is a serialization point. It needs an explicit semaphore
and a short timeout, or it will dominate p95 under the benchmark's concurrent
load. Because it bills GPU-time, cost records need a `billing_unit` — token
counts cannot express its cost.

**Model roster** (local-first; every node but escalated generation is $0):

| Node | Model |
|---|---|
| `plan_query`, `rewrite_query` | `qwen2.5:14b` · ollama local |
| `embed_query` | `nomic-embed-text` · 768-dim |
| `grade_docs`, `verify_grounding` | `llama3.1:8b` · ollama local |
| `rerank` | `bge-reranker-v2-m3` · in-process |
| `generate` | `qwen2.5:32b` local → `gemini-2.5-flash` → openai |

**Policy constants** — all live in `Settings`, never hardcoded at a call site:

| | |
|---|---|
| Abstention threshold (tau) | **0.58** |
| Max attempts per loop | **3** (total attempts, not 3 on top of one) |
| Rerank score floor | 0.44 |
| Per-query cost budget | **$0.0050** |
| Per-query latency budget | **6s** |

The two correction loops count attempts **independently**. When no provider
meets both budgets, that is a refusal — never a silent overspend.

**Testing.** Paid providers are record-and-replay fixtures. **CI must never make
a paid call.** No API keys in CI. A test that reaches a paid endpoint is a bug.

**Multi-tenancy.** API keys scoped to collections; per-key rate limiting and
usage metering. **The semantic cache MUST be scoped by collection** — a
cross-tenant cache hit is a data leak. Scoping is a `WHERE` predicate inside the
ANN query, never a post-filter.

**Deployment.** Designed for AWS (SQS + spot GPU workers, autoscaling on queue
depth, not CPU). Everything must run locally on free tiers during development.

## Settled decisions

- **IDs: UUIDv7 everywhere.** Time-ordered for index locality, non-enumerable
  across tenant boundaries. (The two ER diagrams disagreed — `string` vs
  `bigint`; this supersedes both.)
- **Embeddings: one model, one dimension, project-wide.** `nomic-embed-text`
  via Ollama, 768-dim, matching the ER diagram. `collections.embedding_model`
  exists so a future swap is *detectable* — mixing models in one index is
  silently wrong, and a model swap invalidates every cache entry.
  Confirmed by the design project: `embed_query` reports `"dims": 768`.
  The mockups' per-collection embedders are aspirational, not the schema.
  Note: pgvector caps HNSW at 2000 dims. A 3072-dim model would need `halfvec`.
- **Providers: the four above.** The mockups' 7-provider mix (anthropic, groq,
  huggingface, openrouter, meta) is illustrative only. "ACS Research" is dropped.
- **Phase 0 database scope: the five core tables only.** One migration enabling
  pgvector and creating `tenants`, `collections`, `documents`, `chunks`
  (768-dim vector + HNSW) and `api_keys`. The HNSW index on `chunks` doubles as
  the round-trip proof, so there is no throwaway smoke table.
  Everything else — `query_traces`, `query_citations`, `cache_entries`,
  `eval_runs`, `eval_results`, `golden_questions`, `robustness_runs` — waits
  until the contradictions in `docs/design/REVIEW.md` are resolved. Each is
  blocked on one of them.
- **`chunks` carries `collection_id`** denormalized from `documents`. Tenant
  scoping has to be a `WHERE` predicate inside the ANN query; joining out to
  `documents` to find the collection would defeat the HNSW index.

## Design sources, in precedence order

1. **This file.**
2. **The Claude Design project** — "Prism operator dashboard design",
   `203a7186-e772-4c18-adde-ed33a7125dba`, read via the `DesignSync` MCP
   (needs `/design-login`). This is a **later revision than the PDFs** and wins
   over them wherever they disagree.
3. `docs/design/*.pdf|png` — the original ER diagrams, flow chart and mockups.
   Superseded in the specifics below; still the only source for the ER schema.

The design project defines nine dashboard surfaces:
`landing · design-system · query · trace · queue · cost · eval · robust · coll`.
**"Queue & scaling" is new** and does not appear in the PDFs — queue depth
against a scale-up threshold of 8, workers 2→8, autoscaling keyed on depth not
CPU, and a load test at 1/8/32/64 concurrent. It is the surface that makes the
headline benchmark visible.

Target numbers are deliberately **honest portfolio scale**, not inflated: a
30-query golden set, 72 injection payloads across 5 categories, ~612 queries/24h.
Headline claim is −95% cost and −34% p95 vs a single-shot paid-API baseline;
$0.00019 mean cost/query; 2.71s p95; 1/72 injections still succeeding. If a
number in the PDFs looks bigger (2,412-query golden set, 1,840 payloads), it is
stale — do not quote it.

## Known design gaps — do not build over these

`docs/design/REVIEW.md` holds the full list. Still open:

- `CACHE_ENTRIES` has no `collection_id` in either ER diagram, and no
  `embedding_model`, `expires_at`, or invalidation link. The design project's
  stack panel says Redis provides "per-collection cache scope", so the intent is
  explicit — the column still has to exist and be a `WHERE` predicate.
- Nothing models providers, models, or pricing — so `cost_usd` has no price
  basis and historical costs are unreproducible. This also leaves Ollama Cloud's
  GPU-time billing unrepresentable.
- The benchmark (the headline deliverable) has no data model at all.
- `GOLDEN_QUESTIONS` has no `is_unanswerable` flag, so refusal rate — the
  primary metric — cannot be computed correctly.
- The UI says "BM25" but the schema has no full-text column; Postgres FTS is
  `ts_rank`, not BM25. Either adopt ParadeDB / `pg_search` or change the copy.
- `QUERY_CITATIONS` has no character offsets, but the query console highlights
  an exact span inside a chunk (`pre` / `hit` / `post`).

Closed by the design revision — **do not re-raise**: the retrieval-side refusal
terminal now exists (`abstain`); web search is gone; `plan_query`, `embed_query`
and `rerank` are in the graph; graders are local; the roster is the four
providers; retry canon is 3 attempts.

## Working agreements

- **Ask before making architectural choices not already specified here.**
- When iterating on existing files, show diffs — not full rewrites.
- Design artifacts in `docs/design/` are the source of truth for intent, but
  they contain known contradictions. This file wins over them.
- Record expensive-to-reverse choices as an ADR in `docs/decisions/`.

## Phase plan

- **Phase 0 (current)** — monorepo layout, Docker Compose (Postgres+pgvector,
  Redis, api, web), core schema migration, FastAPI health endpoints, a Next.js
  page rendering dependency status, CI. `docker compose up` working end to end
  **before any AI code**. Definition of done: fresh clone, one command, three
  green statuses.
- Phase 1 — schema and migrations, ingestion, tenancy and API keys.
- Phase 2 — retrieval, then the LangGraph correction loop.
- Phase 3 — router, cache, tracing.
- Phase 4 — eval harness, benchmark, dashboard.

**Do not build the graph before Phase 2.**

## Commands

```
make up          # docker compose up -d --build --wait
make down        # stop and remove volumes
make install     # uv sync + pnpm install
make api         # run FastAPI locally with reload
make web         # run Next.js dev server
make test        # pytest, unit only — no network, no database, no paid call
make test-all    # adds integration tests (needs `make up`)
make lint        # ruff + mypy + tsc
make fmt         # apply formatting fixes
make migrate     # alembic upgrade head
```
