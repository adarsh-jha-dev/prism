# 0011 — Rerank runs in-process, because the other two options are different features

- **Status:** accepted
- **Date:** 2026-09-14
- **Relates to:** [0010](0010-full-text-is-postgres-fts.md)

## Context

`rerank` sits between `grade_docs` and the cost-aware router. `CLAUDE.md` already
pins the model (`bge-reranker-v2-m3`), the lane (`in-process`, cap 8) and the
policy constant (`rerank_score_floor` 0.44). What it does not pin is how the
model runs, and the three candidates are not three implementations of one thing:

- **In-process** — the cross-encoder loaded into the API process.
- **Ollama-served** — the reranker behind the same HTTP surface as every other
  local model.
- **Fusion-only** — no cross-encoder; keep the RRF ordering from ADR 0010 and let
  `grade_docs` carry relevance on its own.

Nothing here is installed yet: no `torch`, no `onnxruntime`, no reranker weights.
That makes this the cheap moment to decide, and it makes the dependency cost part
of the decision rather than a sunk one.

## Decision

**In-process, as a quantized cross-encoder loaded once per process.**

### Ollama cannot serve a cross-encoder

Ollama's API is generate, chat and embed. There is no rerank endpoint and no way
to get a query-document pair score out of it. "Ollama-served rerank" therefore
does not mean `bge-reranker-v2-m3` behind HTTP — it means prompting
`llama3.1:8b` to rate relevance, which is a different algorithm wearing the same
node name.

That substitution fails on three counts. It is nondeterministic, so the same
candidate set reorders between runs and eval regressions stop being attributable.
Its output is prose that has to be parsed into a number, and that number is not
calibrated to anything, so `rerank_score_floor` would be comparing 0.44 against a
digit an LLM chose — the same category error as comparing tau to a cosine
similarity, which `CLAUDE.md` already forbids. And it puts a generation call
inside the retrieval correction loop, which already spends grader calls per
attempt, against a 6s budget.

### Fusion-only deletes a policy constant

RRF produces ranks, and ADR 0010 forbids a fused rank from reaching a threshold —
it is an ordering, not a measurement. With no cross-encoder there is no magnitude
for `rerank_score_floor` to apply to, so the floor would have to go.

The floor is what lets the retrieval loop refuse on *retrieved, but nothing good
enough* rather than only on *retrieved nothing*. Without it,
`no_relevant_evidence` rests entirely on one local LLM's pass/fail over the whole
candidate set. The cheapest option is the one that narrows the refusal path, and
the refusal path is the primary correctness criterion.

### The floor applies to the sigmoid, not the logit

`bge-reranker-v2-m3` emits an unbounded relevance logit. 0.44 is only meaningful
after a sigmoid. Compared against a raw logit the floor admits every candidate
with a positive score, and the constant silently stops doing work while still
appearing in `Settings` and on the dashboard. The normalization is part of the
constant's definition, not a formatting detail.

### The runtime is part of the constant too

The backbone is XLM-RoBERTa-large, roughly 560M parameters: fp32 weights are on
the order of 2GB resident, and a large cross-encoder on CPU is slow enough that
ten candidates would dominate the latency budget on their own. Int8 ONNX Runtime
cuts the footprint to roughly a quarter of that and is several times faster on
CPU, which is the difference between "runs on the free-tier local target" and
"does not".

So: **ONNX Runtime, int8**. Quantization moves the score distribution, which
means 0.44 is calibrated against a specific model revision *and* a specific
quantization. Changing either re-opens the constant and it must be re-fit on the
golden set — a runtime swap is an eval change, not a deployment detail.

### Rerank's parallelism is intra-request

One query against k candidates is one batch and one forward pass. Running eight
of those concurrently on the same CPU does not raise throughput; it multiplies
latency for all eight. The benchmark measures p95 under concurrent load, so this
is the term that would dominate it.

The model is therefore loaded once per process and guarded by its own small
semaphore, and the forward pass runs in a worker thread so it does not block the
event loop. The `in-process` lane cap of 8 bounds in-flight lane work overall
(pgvector included); it is neither eight copies of the model nor eight concurrent
forward passes. Weights load at startup and are reported by `/health/deps`: a
cold model load inside the first request would spend the whole 6s budget on
warm-up.

### An empty set after the floor re-enters the retrieval loop

If every candidate scores below 0.44 there is nothing to generate from. That
routes to `rewrite_query` and counts against the **retrieval** loop's attempts,
exhausting to `abstain` with `no_relevant_evidence` — the same terminal as a
`grade_docs` failure, because it is the same finding arrived at more precisely.
Without this edge the floor has no defined behaviour at its own limit.

### Failure is degradation, and degradation is not a data point

An in-process model has no network to break, so there is no circuit breaker; it
fails by failing to load, or by OOM. When it is unavailable the query proceeds on
fusion order with the floor not applied, and the trace row records
`status = 'error'`.

This is not the forbidden "answer anyway" path. That rule protects the
groundedness gate, and the groundedness gate is untouched: tau still decides at
`verify_grounding` and `abstain` still fires. Rerank buys precision and cost, not
groundedness. Dropping tau, or letting the router overspend, would be the
violation.

What it does break is measurement. A run that silently skipped rerank is not
measuring the optimized path, so eval and benchmark aggregates **exclude** queries
with any degraded node rather than averaging them in. A degraded query is a
failed observation, not a cheap one.

## Consequences

- New dependencies: `onnxruntime` and a tokenizer, plus a ~600MB weights
  artifact. It must not be downloaded per CI run — unit tests use a stub
  reranker with a fixed score map (`make test` is offline by contract), and the
  real model is exercised only in integration tests.
- One model copy per API process, so the uvicorn worker count is now a memory
  decision rather than a throughput knob. Worth stating in the compose file where
  someone will otherwise raise it.
- Cold start grows by the model load. Liveness stays green during it;
  readiness does not.
- `Settings` gains the rerank runtime knobs — model revision, quantization,
  candidate cap, batch size, semaphore, timeout. `rerank_score_floor` stays where
  it is, now documented as coupled to the first two.
- Rerank has no provider row and no price: its trace row carries
  `billing_unit = 'none'`, a real `duration_ms`, and `cost_usd` of zero. The
  latency is the cost (ADR 0012).
- The reranker is the mitigation for `ts_rank_cd` ranking a chunk highly on a
  repeated common word (ADR 0010). Fusion-only would have left that unmitigated,
  which is a second reason it was not viable.
- Revisit if a rerank endpoint appears in Ollama, or if the candidate set grows
  past what one CPU forward pass can hold inside the latency budget. The second
  is the likelier trigger, and its answer is a sidecar service, not a change of
  algorithm.
