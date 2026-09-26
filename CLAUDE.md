# Prism

Self-corrective multimodal RAG platform. The deliverable is a benchmark: **naive
baseline vs optimized, on p95 latency and cost-per-query under concurrent load.**
The engineering is in the correction loop, cost-aware routing, caching, tracing
and eval discipline — not in RAG novelty.

## The one rule that overrides the others

**Correct refusal is the primary correctness criterion, not a fallback.** Prism
must say "insufficient evidence" rather than answer ungrounded. Never add a code
path that degrades to "answer anyway" — including on timeout, provider failure,
or budget exhaustion.

## Stack

Python 3.12 · FastAPI · LangGraph · Postgres 16 + pgvector (HNSW) · Redis 7 ·
Ollama for all dev and CI inference · uv, ruff, mypy, pytest · Alembic ·
GitHub Actions. Dashboard: Next.js (App Router), TypeScript, shadcn/ui,
Recharts, pnpm.

```
services/api/     FastAPI + LangGraph orchestration service
apps/web/         Next.js dashboard
docs/design/      Design artifacts (ER diagrams, flow, UI mockups)
docs/decisions/   ADRs
scripts/          Dev helpers
```

Python and TypeScript stay separated — no shared build tool or lockfile.
`Makefile` is the only cross-language entry point. The dashboard is a thin
client over REST/WebSocket: if it needs a computed number, the API computes it.

## The graph

Node names are canonical — they appear in trace rows and the dashboard, so they
must match exactly.

```
query
  -> semantic cache check      (scoped by collection; hit => END: cached)
  -> plan_query
  -> embed_query
  -> retrieve                  hybrid Postgres FTS + HNSW, fused by RRF
  -> rerank                    cross-encoder over the fused pool, cut to k by
                               the floor (ADR 0020: before grade_docs, not after)
  -> grade_docs                judges only what the floor kept
       pass -> cost-aware router
       fail -> rewrite_query, retry retrieval
               attempts exhausted -> abstain -> END: refused (no_relevant_evidence)
  -> cost-aware router         cheapest provider meeting cost/latency budget
  -> generate                  local first, escalate under budget
  -> verify_grounding
       pass (>= tau) -> END: answered (with citations)
       fail          -> regenerate with stricter constraints
                        attempts exhausted -> abstain -> END: refused (insufficient_evidence)
```

- Three terminal states: `cached`, `answered`, `refused`. Refusal carries a
  reason (`no_relevant_evidence` | `insufficient_evidence`) — not a boolean.
- `abstain` is a real node with a policy model, not an implicit edge. A refusal
  must be as inspectable as an answer.
- Every node writes a trace row and a checkpoint. Any query must be replayable
  and forkable from any node.
- The two correction loops count attempts **independently**.
- **No web-search fallback.** It contradicts "no network egress from generation
  nodes" and cannot satisfy `query_citations.chunk_id`.

## Providers and models

One interface, with a circuit breaker. Cheap-first cascade; escalate only on
grading failure.

| Lane | Concurrency | Role |
|---|---|---|
| `ollama` (self-hosted) | 8 | graders, rewrite, embeddings, vision ingestion, most generation |
| `ollama-cloud` | **1 — pinned** | GPU-time metered, not tokens |
| `gemini` | 4 | multimodal ingestion, escalated generation |
| `openai` | 2 | final escalation, used sparingly |
| `in-process` | 8 | pgvector, rerank — bound by worker count |

Ollama Cloud's cap of 1 is a serialization point: explicit semaphore, short
timeout, and a `billing_unit` on cost records — tokens cannot express GPU-time.

| Node | Model |
|---|---|
| `plan_query`, `rewrite_query` | `qwen2.5:14b` · ollama local |
| `embed_query` | `nomic-embed-text` · 768-dim |
| `grade_docs`, `verify_grounding` | `llama3.1:8b` · ollama local |
| `rerank` | `bge-reranker-v2-m3` · in-process |
| `generate` | `qwen2.5:32b` local → `gemini-3.6-flash` → openai |
| vision ingestion (figures, tables) | `qwen2.5vl:7b` local → `gemini-3.6-flash` |

## Policy constants

All live in `Settings`, never hardcoded at a call site.

tau **0.58** · max attempts per loop **3** (total, not 3 on top of one) · rerank
floor **0.01** · cost budget **$0.0050/query** · latency budget **6s/query**.

When no provider meets both budgets, that is a refusal — never a silent
overspend.

`rerank` runs **before** `grade_docs` (ADR 0020). A set the floor empties
re-enters the retrieval loop through `grade_docs`' own empty path, so one pass of
the loop stays one attempt whichever gate fails it.

`abstention_threshold` (tau) applies **only** to the grader's groundedness score
at `verify_grounding`. It must never be compared against a cosine similarity:
the two are different quantities that happen to share a 0-1 scale, and on the
golden set similarity does not separate answerable from unanswerable at any
threshold (`eval/README.md`). Retrieval decides relevance at `grade_docs` and
`rerank_score_floor`; tau decides groundedness.

## Settled decisions

- **CI must never make a paid call.** Paid providers are record-and-replay
  fixtures; no API keys in CI. A test reaching a paid endpoint is a bug.
- **Tenant scoping is a `WHERE` predicate inside the ANN query, never a
  post-filter** — including the semantic cache, where a cross-tenant hit is a
  data leak. `chunks` carries `collection_id` denormalized so this stays inside
  the HNSW index.
- **UUIDv7 everywhere.**
- **One embedding model project-wide:** `nomic-embed-text`, 768-dim. Mixing
  models in one index is silently wrong; a swap invalidates every cache entry.
  pgvector caps HNSW at 2000 dims.
- **The four providers above.** The mockups' 7-provider mix is illustrative.
- **Five core tables so far:** `tenants`, `collections`, `documents`, `chunks`,
  `api_keys`. Everything else is blocked on an open design gap.
- Designed for AWS (SQS + spot GPU workers, autoscaling on queue depth, not
  CPU); must run locally on free tiers.

## Design sources, in precedence order

1. **This file.**
2. **The Claude Design project** — "Prism operator dashboard design",
   `203a7186-e772-4c18-adde-ed33a7125dba`, via the `DesignSync` MCP (needs
   `/design-login`). Later revision than the PDFs; wins over them.
3. `docs/design/*.pdf|png` — still the only source for the ER schema.

Target numbers are **honest portfolio scale**: 31-query golden set, 72 injection
payloads, ~612 queries/24h, −95% cost and −34% p95 vs a single-shot paid-API
baseline, $0.00019 mean cost/query, 2.71s p95, 1/72 injections succeeding.
Bigger numbers in the PDFs are stale — do not quote them.

## Open design gaps — do not build over these

`docs/design/REVIEW.md` holds the list: cache scoping columns, the benchmark's
data model, `GOLDEN_QUESTIONS.is_unanswerable`, and citation character offsets.

Closed by the design revision — **do not re-raise**: `abstain` exists; web
search is gone; `plan_query`, `embed_query`, `rerank` are in the graph; graders
are local; the roster is the four providers; retry canon is 3.

Closed by ADR — **do not re-raise**: the lexical half is Postgres FTS, not BM25,
and nothing user-visible may claim BM25 (ADR 0010); rerank runs before
`grade_docs` (ADR 0020); rerank is an in-process int8
cross-encoder (0011); the query/trace/citation model, its partitioning stance and
"refusals are not cached" (0012); `cost_usd` has a price basis in an
effective-dated `model_pricing` (0013).

## Working agreements

- **Ask before making architectural choices not already specified here.**
- When iterating on existing files, show diffs — not full rewrites.
- Record expensive-to-reverse choices as an ADR in `docs/decisions/`.
- Design artifacts express intent but contain known contradictions. This file
  wins over them.

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
