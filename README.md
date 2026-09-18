# Prism

Self-corrective multimodal RAG platform. A portfolio project about
**infrastructure and inference-cost optimization**, not RAG novelty.

The headline deliverable is a benchmark: naive baseline vs optimized, measured
on p95 latency and cost-per-query under concurrent load. The interesting
engineering is the correction loop, cost-aware provider routing, caching,
tracing and eval discipline.

## Ports

Postgres and Redis sit off their default ports so Prism never collides with
something already running locally.

| Service | Host port |
|---|---|
| web | 3000 |
| api | 8000 |
| postgres | 5433 |
| redis | 6380 |
| ollama (host) | 11434 |

## Developing

Running the app directly on the host, against compose's Postgres and Redis:

```bash
cp .env.example .env
make install               # uv sync + pnpm install
make up                    # dependencies
make api                   # FastAPI with reload, port 8000
make web                   # Next.js dev server, port 3000
```

| | |
|---|---|
| `make test` | unit tests — no network, no database, no paid call |
| `make test-all` | adds integration tests (needs `make up`) |
| `make lint` | ruff + mypy + tsc |
| `make fmt` | apply formatting fixes |
| `make migrate` | `alembic upgrade head` |
| `make down` | stop everything, remove volumes |

`make help` lists them all.

**CI never makes a paid call.** There are no API keys in CI, and paid providers
are exercised through the record/replay fixtures in
`services/api/tests/fixtures/` — re-recorded locally with `make record-fixtures`.
A test that reaches a paid endpoint is a bug.

Generation and the graders run locally, so the graph needs its models pulled:

    ollama pull qwen2.5:32b      # generate
    ollama pull llama3.1:8b      # grade_docs, verify_grounding
    ollama pull qwen2.5:14b      # plan_query, rewrite_query

Ingestion also parses figures and tables with a vision model (`VISION_ENABLED`).
The default lane is local and free and needs the model pulled:

    ollama pull qwen2.5vl:7b

`VISION_LANE=gemini` is the paid escalation. See ADRs `0004` and `0005` in
`docs/decisions/`.

## Layout

```
services/api/     Python: FastAPI + LangGraph orchestration
apps/web/         Next.js dashboard — a thin client over REST/WebSocket
docs/design/      Source design artifacts (ER diagrams, flow, UI mockups)
docs/decisions/   ADRs for choices that are expensive to reverse
```

Python and TypeScript stay cleanly separated: no shared build tool, no shared
lockfile. The root `Makefile` is the only cross-language entry point, and the
dashboard holds no business logic — if it needs a computed number, the API
computes it.

## Roadmap

| Phase | |
|---|---|
| **0** | Monorepo, compose, health endpoints, core schema, CI |
| 1 | Ingestion, tenancy, API keys |
| 2 | Retrieval, then the LangGraph correction loop |
| 3 | Cost-aware router, semantic cache, tracing |
| 4 | Eval harness, benchmark, operator dashboard |

`CLAUDE.md` holds the architecture in full: the graph, the policy constants, the
provider lanes and the settled decisions. `docs/design/REVIEW.md` tracks the
known contradictions in the design artifacts that later phases have to resolve.
