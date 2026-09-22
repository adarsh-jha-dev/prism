# 0020 — `rerank` runs before `grade_docs`, over the fused pool

- **Status:** accepted
- **Date:** 2026-09-22
- **Relates to:** [0010](0010-full-text-is-postgres-fts.md),
  [0013](0013-cost-basis-and-model-pricing.md),
  [0017](0017-what-the-graph-state-carries.md),
  [0019](0019-the-retrieval-correction-loop.md)
- **Amends:** [0011](0011-rerank-runs-in-process.md), on where the node sits in
  the graph; the in-process runtime, the int8 quantization and the floor's
  meaning are untouched
- **Amends:** [0019](0019-the-retrieval-correction-loop.md), Decision 3, on how
  many passages one grading call sees

## Context

`CLAUDE.md` lists `grade_docs` before `rerank`, and the graph is wired that way:
`retrieve` returns `retrieval_top_k` fused hits, `grade_docs` drops the ones its
verdict fails, `rerank` reorders what is left and applies the floor.

That ordering has two costs. The reranker never sees a pool wider than the set
`retrieve` already cut to `k`, so its only job is reordering. And it runs after
the one LLM call in the retrieval loop, so the grader pays for every candidate
retrieval found, on every pass, whether or not the cross-encoder would have kept
it.

The alternative is the order the eval harness already runs: `retrieve` returns
the fused pool at `max(k, rerank_candidate_k)`, `rerank` scores it and cuts to
`k`, and `grade_docs` judges only the survivors.

This is a cheap moment to settle it. `rerank` is still a stub, so nothing has
been built on either answer.

## Measurements

Both on the same machine (Apple M5, CPU, int8), warm.

**The grader is linear in the number of passages, on input and output alike.**
`llama3.1:8b`, `_GRADER_SYSTEM`, 1200-character passages, median of three:

| passages | median | input tokens | output tokens |
|---|---|---|---|
| 2 | 1.50s | 562 | 36 |
| 4 | 2.53s | 1,008 | 61 |
| 6 | 3.57s | 1,454 | 85 |
| 10 | 6.59s | 2,346 | 153 |

Roughly **0.63s per passage**. The schema returns one verdict per passage, so
output tokens scale with the set too — this is not prefill that a longer context
window makes free.

**The reranker is linear in candidates**, at roughly 0.2s each: 1.94s p50 for 10,
7.26s for 30 (`eval/README.md`).

**The floor leaves very little.** Counting survivors per question in
`runs/baseline-2026-09-16-rerank.json` — 31 questions, pool of 10, floor 0.44:

| | mean | median | max | emptied |
|---|---|---|---|---|
| all 31 | 1.19 | 1 | 6 | 14/31 |
| 25 answerable | 1.44 | 1 | 6 | 9/25 |
| 6 unanswerable | 0.17 | 0 | 1 | 5/6 |

## The two orders

### Grader cost per pass

Order A grades 10 passages on every pass: **6.6s**, unconditionally.

Order B pays 2.0s to score the pool, then grades `m` survivors: **2.0 + 0.63·m**.
The two are equal at `m ≈ 7.3`. The observed maximum on the golden set is 6, and
the median is 1 — so **Order B is cheaper on every question in the set**, and on
the 14 of 31 the floor empties it makes no grader call at all, falling into
`grade_docs`' existing `no_candidates` path.

Against a 6s per-query latency budget across up to three passes, Order A spends
the whole budget on grading a single pass.

**This is latency, not dollars.** Migration 0008 prices `llama3.1:8b` at zero,
and `CLAUDE.md` keeps graders local, so grader tokens cost nothing and cutting
them saves nothing on cost-per-query. The claim this ordering serves is the
−34% p95 half of the benchmark, not the −95% cost half. Worth stating plainly,
because "cutting grader input tokens" reads like a cost argument and is not one.

### Are the floor and the grader redundant?

They are two gates on relevance, and in either order the second only ever sees
what the first kept — so the strictest wins and the set reaching `generate` is
the same intersection either way. At today's constants
(`retrieval_top_k = rerank_candidate_k = 10`) the pool and `k` are the same
number, so **both orders hand `generate` exactly the same evidence.** Today this
is purely a cost decision.

They are not redundant in kind. The floor is absolute and per-passage; the
grader is a set-level judgement and can rank a passage against what else came
back. But the evidence on which one separates better runs one way:
`eval/README.md` measures the cross-encoder separating answerable from
unanswerable (5/6 refused at 0.44, 18/25 kept), and measures that cosine cannot
do it at any threshold. `doc_relevance_threshold` at 0.5 is uncalibrated and
ADR 0019 says so. Order A spends 6.6s on the unmeasured gate to feed the
measured one.

The honest cost of Order B, and it is a real one: with a median of 1 survivor,
the grader usually judges a single passage. ADR 0019 Decision 3 argued for one
batched call partly because "the model sees the candidates together and can
judge relative to what else was retrieved". Under Order B that comparison
mostly disappears. The batching rule still holds — it is still one call, one
provider, one meter — but its second justification weakens, which is why this
ADR amends 0019 rather than merely citing it.

Neither order rescues a gutted question. `eval/README.md` records that at 0.44
the floor removes more evidence than it refuses; the floor kills those chunks in
either order, so that cost is order-invariant and is not an argument for A.

### What the pinned baseline measures

`baseline-2026-09-16-rerank.json` was recorded by `eval/runner.py`, which
retrieves `max(k, rerank_candidate_k)` fused hits and reranks them to `k`. There
is no grader anywhere in the harness.

- Under **Order B**, that baseline is a proper prefix of the graph's pipeline:
  the same two calls in the same order over the same pool. Its ordering numbers
  and its post-floor recall stay a valid upper bound on what the graph retrieves.
- Under **Order A**, the graph's reranker scores a set no baseline has ever
  handed it — one an LLM already filtered. The recorded 0.36 recall-after-floor
  and 0.511 MRR describe a pipeline the graph does not run, and no pinned
  baseline measures the one it does.

Closing that gap under Order A means putting a grader into the eval harness.
`eval/README.md` is explicit that changing the retriever voids every comparison,
so that re-opens all three pinned baselines. Order B needs no eval change and
keeps them.

### The pool is not actually wide yet

`rerank_candidate_k` and `retrieval_top_k` are both 10, so `max(k,
rerank_candidate_k)` is 10 and Order B's "wide pool" is the same 10 chunks
Order A retrieves. The widening is available, not taken: the README prices 30
candidates at 7.26s p50, over the whole query budget, and ADR 0011 already names
a sidecar service as the answer when the pool has to grow. Order B is the
ordering that can spend a wider pool the day one is affordable; it does not
deliver one now, and this ADR does not change either constant.

## Decision

**Order B.** `retrieve` returns the fused pool at `max(k, rerank_candidate_k)`,
`rerank` scores it and cuts to `k`, `grade_docs` judges the survivors.

Three measured reasons, in order of weight:

1. It is cheaper per pass on every question in the golden set, and free of a
   grader call on the 14 of 31 the floor empties — against a latency budget that
   one Order A pass already exhausts.
2. It is the pipeline the pinned baseline measures. Order A makes
   `baseline-2026-09-16-rerank` a record of something the graph does not do, and
   fixing that re-opens every pinned baseline.
3. It puts the measured relevance gate before the uncalibrated one, and the
   evidence reaching `generate` is unchanged at today's constants.

Accepted against: `CLAUDE.md`'s node order, which this contradicts and which
needs updating with this ADR; and ADR 0019's set-comparison rationale for
batched grading, which weakens to a median of one passage.

### The fallback records `fallback`, not `status = 'error'`

ADR 0011 said a degraded rerank writes `status = 'error'`. It cannot: the node
catches `RerankError` and returns a result, and migration 0009 ties
`status = 'error'` to a non-NULL `error`. A row that both succeeded and claims to
have errored would also make every "did this node fail" query wrong.

The degradation is recorded as `fallback: true` in the row's `output_json`. The
exclusion rule ADR 0011 actually cares about is unchanged and now enforceable in
one predicate: a query with any node row carrying `fallback: true`, or any row
with `status = 'error'`, is excluded from eval and benchmark aggregates.

## Consequences

- `CLAUDE.md`'s graph listing moves `rerank` above `grade_docs`. Node names are
  unchanged, so trace rows and the dashboard are unaffected except in order.
- `retrieve` retrieves at `max(k, rerank_candidate_k)` rather than `k`. That is
  a pinned run parameter, so it belongs in `SearchParams` rather than at the
  call site.
- The conditional edge planned after `rerank` changes shape. `grade_docs` stays
  the node that closes an attempt, and a set the floor emptied reaches its
  existing `no_candidates` path — so the floor's loop re-entry (ADR 0011) needs
  no second counter and no third refusal reason, and `abstain` still reports
  `no_relevant_evidence` for an empty set.
- `grade_docs`' trace input shrinks to the post-floor set. A reader of the
  waterfall sees what the floor discarded on the `rerank` row above it, which is
  what `Reranked.ranked` is for.
- The rerank fallback on `RerankError` (ADR 0011) now precedes grading, so a
  degraded run grades the full fused pool in fusion order. It is excluded from
  eval either way.
- The node loads the weights itself before scoring. Nothing else did: `score`
  raises rather than loading lazily, `get_reranker` cannot await, and only the
  eval CLI calls `load`. Without it every graph query fell back to fusion order
  and said so on one trace row nobody was reading. ADR 0011's startup load, and
  the `/health/deps` readiness it belongs to, is still owed — until it lands the
  first query after boot pays a cold load inside its latency budget.
- Revisit together with `rerank_candidate_k`: the ordering only pays for itself
  in retrieval quality once the pool is wider than `k`, and that is a latency
  decision this ADR does not take.
