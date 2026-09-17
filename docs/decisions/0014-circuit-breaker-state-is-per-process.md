# 0014 — Circuit-breaker state is per-process, and breaker scope follows failure scope

- **Status:** accepted
- **Date:** 2026-09-17
- **Relates to:** [0004](0004-vision-parsing-for-figures-and-tables.md),
  [0008](0008-rate-limiting-and-usage-metering.md),
  [0011](0011-rerank-runs-in-process.md),
  [0013](0013-cost-basis-and-model-pricing.md)
- **Partly closes:** the `REVIEW.md` graph-control-flow gap "no error, timeout, or
  circuit-breaker edges anywhere, despite the circuit breaker being a stated
  constraint" — the breaker, not the edges

## Context

The provider registry puts one interface in front of four lanes and guards each
call with a per-lane semaphore and a per-lane circuit breaker. The semaphore's
state is necessarily in-process — it is an `asyncio` primitive. The breaker's
state is not: consecutive-failure counts and a cooldown clock could equally live
in Redis, which `db.py` already exposes and ADR 0008 already made a dependency of
every authenticated request.

The choice is expensive to reverse, and not because swapping a store is hard.
What spreads is the *call-site contract*: which exception a rejected call raises,
whether that exception is distinguishable from a failed one, what a node does
with it, and what a trace row records. Once every node in the graph calls the
registry, changing that contract is a change everywhere. The storage behind it is
the cheap half.

Phase 2 wires exactly one lane — `ollama`, local, free, on the same host as the
worker. The other three are defined with their caps and billing units and raise
`NotImplementedError`.

## The case for shared state in Redis

It is a real case, and it gets stronger every time a worker is added.

- **N workers, N failure budgets.** Each process must independently burn
  `threshold` doomed calls before it learns a lane is down. Detection cost scales
  with the fleet while the fact being detected is one fact.
- **N half-open probes.** At cooldown expiry every worker probes independently,
  so a recovering provider is greeted by the whole fleet at once — precisely when
  it is least able to absorb it. Shared state makes that one probe.
- **Spot workers churn.** The AWS target is SQS with spot GPU workers scaling on
  queue depth. A worker that starts mid-outage starts with a closed breaker and
  relearns by failing. Under churn the fleet's aggregate memory approaches zero.
- **The constraints that matter most are account-scoped.** An OpenAI 429 is a
  fact about the account, not about the process that happened to receive it.
  Ollama Cloud's GPU-time meter and its cap of 1 are properties of the
  subscription. A per-process view of an account-wide constraint is structurally
  incomplete, and no threshold tuning fixes that.

## The case against, which is the case for per-process

**1. The Redis-down stance for a shared breaker is per-process.**

ADR 0008 set two opposite stances deliberately: the limiter is a control and
fails closed, metering is a record and fails open. A shared breaker looks like a
control, but neither stance survives contact with it. Failing closed means a
Redis outage refuses every query in the fleet — converting a counter outage into
a total outage, with the refusal reason being neither of the two the graph is
allowed to emit. Failing open means a Redis outage silently deletes the breaker,
which is the failure mode the breaker exists to prevent, arriving unannounced.

The only tolerable third stance is *fall back to this process's own view* — and
that is the per-process breaker, in full. Shared state is therefore a layer over
a local breaker, never a replacement for one. Whatever we decide about Redis, the
in-process implementation gets built either way; choosing shared now means
building both now.

(The query path happens to be unreachable without Redis anyway, since the limiter
already refuses at auth. That is not a general answer: the ingestion pipeline's
paid vision calls and the eval runner both reach providers without passing the
limiter.)

**2. One storage stance cannot be right for all four lanes.**

`ollama` is host-local — a model server beside the worker, sharing its GPU pool
and its failure modes. Sharing that lane's breaker is not merely unnecessary, it
is wrong: one worker with a sick sidecar would open the `ollama` lane for every
worker whose Ollama is perfectly healthy, and the local lane carries the graders,
the rewriter, the embedder and most generation. A false-positive open on the free
local lane refuses queries that had evidence and a working model — the most
expensive mistake available under the rule that correct refusal is the primary
correctness criterion. Refusals are the product here; a wrong refusal is a wrong
answer, and this one would be fleet-wide.

The lanes where shared state is right are the three that are account-scoped. The
lane that exists today is the one where it is wrong.

**3. The per-process cost is bounded, and smaller than "useless at scale" implies.**

A consecutive-failure breaker with a cooldown converges. The waste is
`N × threshold` doomed calls *per outage episode*, each bounded by the lane's own
timeout — not a continuous tax. At the honest scale this project is built for
(~612 queries/24h, a handful of workers) that is tens of calls per outage, most
of them fast errors on lanes whose error responses are unpriced. Per-instance
breakers are also the norm rather than a compromise: Hystrix, resilience4j and
Polly all keep state per process, because part of what a breaker measures is
*this* process's connection pool, DNS and event loop, and that part does not
generalise to a neighbour.

**4. The deliverable's numbers do not depend on this.**

The benchmark runs in one process. Whatever the fleet does in production, the
p95 and cost-per-query figures that are the actual deliverable are produced under
a single-process breaker.

**5. Redis stays off the provider call path.**

A shared breaker adds a network round trip before and after every provider call —
and a query makes many: plan, embed, grade per attempt, rewrite, rerank, generate,
verify. The absolute latency is small. The new failure mode is not: it puts a
second network dependency inside the guard that exists to contain the first one's
failures.

## Decision

**Per-process, behind a seam narrow enough that the storage is not the part that
spreads.**

The breaker is reached through two operations — ask whether a lane may be called,
and record what happened — with an in-memory implementation. Nodes see the error
types and the log events, never the store. A Redis-backed implementation later
satisfies the same two operations and changes no call site.

**Breaker scope follows the scope of the failure it observes, not the lane's
name.** Recorded now so the revisit is a criterion rather than a mood:

- Host-local lanes (`ollama`, and the in-process lane if it ever grows one) stay
  per-process permanently. Sharing them is a correctness bug, not a scaling win.
- Account-scoped lanes (`ollama_cloud`, `gemini`, `openai`) move to shared state
  when both conditions hold: those lanes are wired, and more than one worker runs
  them. Both land in Phase 3, not before.
- When they move, shared state layers over the local breaker: local evidence can
  open a lane locally, shared evidence can open a locally-closed lane, and Redis
  being unavailable degrades to the local view. No stance where a store outage
  either refuses everything or erases the breaker.

**A rejected call is not a failed call, and a queued call is neither.** Only
evidence about the provider counts toward opening: transport errors, timeouts,
5xx, 429. A wait that expires on our own lane semaphore does not — that is this
project's serialization point, not the provider's health, and counting it would
let load open a lane that is merely busy. Ollama Cloud, capped at 1, would
otherwise open itself under exactly the concurrent load the benchmark is built to
measure. This is the same reasoning ADR 0013 uses to keep local queue wait out of
`gpu_ms`: the bottleneck is ours, and it must not be charged twice.

Threshold and cooldown live in `Settings` like every other policy constant, one
value across lanes until a lane earns its own.

## Consequences

- Detection costs `N × threshold` doomed calls per outage episode, and recovery
  costs N half-open probes. Both are accepted, both are bounded by the lane
  timeout, and both grow linearly with a worker count that is currently one.
- A restarting worker forgets. A crash-looping worker rediscovers an outage every
  boot, and the breaker cannot damp that — a restart budget can, and is not this
  ADR's problem.
- **The concurrency cap has the same distributed hole, and it is the larger one.**
  `ollama_cloud`'s cap of 1 is structural within a process and cannot be
  structural across processes: N workers means N concurrent cloud calls against a
  limit that is external and metered in GPU-time. Phase 3 must either confine the
  cloud lane to a single worker or give it a distributed lease. Recording it here
  so that "the cap of 1 is enforced" is never read as more than it is.
- Nothing new is added to the generation path. Redis remains a dependency of auth
  and the future semantic cache, not of a provider call.
- The `REVIEW.md` gap is only half closed: the registry breaks circuits, the graph
  still has no edges that render a broken one. The terminal state for "no provider
  available" is a refusal, and which reason it carries is a Phase 3 question.

## Alternatives considered

- **Shared in Redis now.** Rejected for Phase 2 on the two grounds above: it is
  wrong for the only wired lane, and it requires the per-process breaker anyway as
  its own degraded path. Revisited by the criterion above, not discarded.
- **No breaker until Phase 3, semaphores only.** This is ADR 0004's current state
  for the vision lane, and it is why repeated failures across documents go
  untracked. The registry is the place that fixes it; deferring again means the
  call-site contract is settled by whoever writes the first paid lane.
- **Error-rate window instead of consecutive failures.** A percentage over a
  window is better behaved under partial failure, and needs enough traffic for a
  rate to mean anything. At ~612 queries/24h a window is mostly empty, and an
  empty window's rate is an opinion. Consecutive failures is the honest statistic
  at this volume.
