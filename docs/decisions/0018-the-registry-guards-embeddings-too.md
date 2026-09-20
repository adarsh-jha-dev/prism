# 0018 — The registry guards every model call, embeddings included

- **Status:** accepted
- **Date:** 2026-09-20
- **Relates to:** [0013](0013-cost-basis-and-model-pricing.md),
  [0016](0016-how-a-node-reports-to-the-trace-writer.md),
  [0017](0017-what-the-graph-state-carries.md)
- **Amends:** [0014](0014-circuit-breaker-state-is-per-process.md), on which
  errors count as evidence about a provider

## Context

`ProviderRegistry` exists so that no model call escapes its lane's semaphore and
circuit breaker. It was built around `ChatProvider`: `_guarded` resolves a chat
provider from the lane's `factory`, and `complete` and `structured` are the only
ways in.

`embed_query` is a model call on the `ollama` lane. Routed around the registry —
straight to `get_embedding_provider()`, as ingestion and eval do — it is
unguarded: it takes no slot, so it does not count against
`concurrency_ollama_local`, and it neither reads nor trips the lane's breaker.
The result is a lane whose cap is enforced for generation and not for embedding,
sharing one Ollama process, and a breaker that can sit closed through an outage
that every embedding call is observing.

Ingestion is the reason it was built the other way, and that reason still holds:
ingestion embeds thousands of chunks outside any query budget and must not
contend for the query path's slots. So the question is not whether
`EmbeddingProvider` should move behind the registry wholesale. It is how a graph
node gets a guarded embedding while ingestion keeps an unguarded one.

## Decision

### The guard is separated from the chat provider

`_guarded` takes a zero-argument coroutine returning `(value, Usage)` rather than
one that receives a `ChatProvider`. `complete` and `structured` resolve their
provider first — preserving the order that makes `LaneNotImplemented` fire before
anything queues — and close over it. The guards themselves know nothing about
what they are guarding.

`registry.embed(lane, texts, provider=...)` joins them: same breaker check, same
bounded wait on the same semaphore, same `LaneResult` carrying the provider's own
`Usage` unpriced. A lane needs no chat `factory` to serve it, which is correct —
the embedding model is not the lane's chat model and never was.

`prism.embeddings` keeps its direct interface. Ingestion and eval go on calling
`get_embedding_provider()`, and the graph goes through the registry. The split is
by caller, not by provider, and it is the split that was already implied by
`embed_metered` existing for the graph alone (`embeddings/base.py`).

### `EmbeddingError` counts toward the breaker

ADR 0014 says only `ChatError` is evidence about a provider, because at the time
a lane served nothing else. The reasoning was never about chat: `LaneBusy` and
`LaneUnavailable` are our own queue and our own breaker, while a `ChatError`
means a call reached the provider and the provider failed it. An `EmbeddingError`
raised from `embed_metered` means exactly the same thing about the same Ollama
process — a transport failure, a timeout, an HTTP status, a model that is not
pulled.

So the failure set becomes `ChatError | EmbeddingError`, named once as
`PROVIDER_ERRORS` in `providers/base.py` rather than spelled out at the catch
site, so a third provider kind cannot be added without deciding this question
again.

One case is deliberately included: a width mismatch (`nomic-embed-text` returning
other than `embedding_dim`) raises `EmbeddingError` and trips the breaker like
any other. It is not transient and the breaker will not fix it, but it is a true
statement that this lane cannot currently serve embeddings, and a lane that opens
fast on it is better than one that keeps a misconfigured model in rotation. The
error text says which dimensions disagreed either way.

Not included: `ValueError` from a blank or empty input. That is the caller's bug
and never reaches the provider.

## Consequences

- `embed_query` contends for the same eight slots as local generation. That is
  the point — they are the same Ollama process — and it makes
  `concurrency_ollama_local` mean what it says under concurrent load, which is
  the regime the benchmark measures.
- A local Ollama outage now opens the `ollama` lane from whichever node notices
  first. Under the current straight-line graph, `embed_query` is usually first,
  since it precedes every generation call.
- `prism.providers` gains an import of `prism.embeddings`. The dependency runs
  one way — embeddings know nothing about lanes — and matches how
  `prism.providers` already depends on `prism.chat`.
- Ingestion's embedding path is still unguarded, so a large ingest can saturate
  Ollama while the query lane's slots sit free. Rate limiting ingestion against
  the query path is a real gap, and it belongs with the queue-depth autoscaling
  work, not here.
