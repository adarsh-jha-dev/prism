# 0017 — What the graph's state carries: the query embedding, and references to candidates

- **Status:** accepted
- **Date:** 2026-09-20
- **Relates to:** [0010](0010-full-text-is-postgres-fts.md),
  [0012](0012-query-trace-and-citation-model.md),
  [0015](0015-checkpointer-tables-and-connection.md),
  [0016](0016-how-a-node-reports-to-the-trace-writer.md)

## Context

`embed_query` produces 768 floats and `retrieve` needs them. `retrieve` produces
a candidate set and `rerank` and `generate` need it. Neither is a meter or a
payload, so ADR 0016 does not cover them: they are state, or they are re-fetched.

`state.py` says every value in `GraphState` is serialized into every checkpoint.
That is the right instinct and the wrong arithmetic, and the difference decides
this. LangGraph's Postgres saver splits a checkpoint in two
(`checkpoint/postgres/aio.py::aput`, `base.py::_dump_blobs`):

- **Primitives** — `None`, `str`, `int`, `float`, `bool` — are inlined into
  `checkpoints.checkpoint` as jsonb. Those *are* copied into every checkpoint row.
- **Everything else** goes to `checkpoint_blobs`, keyed
  `(thread_id, checkpoint_ns, channel, version)`, and is written **only when that
  channel's version changes**. A checkpoint that does not touch a channel
  references the existing blob.

So a list in state costs one blob **per write**, not one per checkpoint. The
rewrite loop's repeated checkpointing multiplies the scalars, not the vectors.
What it does multiply is any channel the loop itself rewrites — which is exactly
the embedding and the candidate set, once per retrieval attempt, at most
`max_attempts` times.

Measured with the saver's own serializer (`JsonPlusSerializer`, ormsgpack):

| value | bytes per write |
|---|---|
| query embedding, 768 float64 | 6,915 |
| 10 candidate references (ids, ranks, fused positions) | 1,311 |
| 10 hydrated chunks at `chunk_size_chars = 1200` | 14,001 |

## Decision

### 1. State carries the query embedding; `retrieve` does not re-embed

~6.9 KB per write, at most three writes per thread — the loop re-embeds because
it re-asks a rewritten question, not because it checkpointed again. Roughly
21 KB worst case, against a `checkpoint_blobs` row that already exists.

Re-embedding inside `retrieve` is cheaper in bytes and wrong three ways:

1. **It misattributes the meter.** `record_usage()` may be called once per node
   execution (ADR 0016), and a trace row has one `provider` column. If `retrieve`
   embeds, the embedding's meter lands on `retrieve`'s row and `embed_query`'s row
   reads as a node that called nothing. That is ADR 0016's failure 1, landing on
   the one node whose entire purpose is the call.
2. **It breaks replay.** `CLAUDE.md` requires any query to be replayable and
   forkable from any node. A fork resumed at `retrieve` must re-enter with the
   vector the original run used. Re-embedding makes the fork a fresh model call
   against a model that may have been re-pulled since, and a fork that retrieves
   different evidence is not a replay of anything.
3. **It empties the node.** `embed_query` would produce nothing any other node
   reads, which makes it a timing row rather than a step.

Consequence: `hybrid_search` gains an optional `query_vector` parameter, so
`retrieve` hands it the vector instead of the function embedding internally.
`search.py` — the benchmark's naive baseline — is not touched.

### 2. State carries candidate references, not hydrated chunks

1.3 KB against 14 KB per retrieval attempt; ~4 KB against ~42 KB across three.
Size is the smaller half of the argument.

The larger half is isolation. Chunk text in a checkpoint blob is tenant document
text sitting in a table that carries no tenant column and cannot be placed under
RLS (ADR 0015). Retrieval's entire stance is that scope is a predicate inside the
scan, never a filter over its results (`search.py`, ADR 0006); copying the rows
that scan returned into an unscopable table hands back what the predicate bought.

ADR 0012 already made this call for trace payloads — references, not copies — and
the reasoning transfers whole. State has the second cost as well, so the rule
holds a fortiori.

`rerank` and `generate` hydrate by chunk id under the tenant predicate. Ten rows
by primary key is not a measurable fraction of a 6 s budget, and both nodes are
already inside a database round trip.

A fork whose chunks were deleted or re-ingested since hydrates short. That is
correct, not a defect: state is not an archive. ADR 0012 put the archive where it
belongs — `query_citations.chunk_ref` and `cited_content`, snapshotted at answer
time, so what was cited survives the chunk that was cited.

### 3. Reducers: none of the new keys gets one

`monotonic` is for counters, and neither an embedding nor a candidate set is one:
`max()` over two vectors is meaningless and over two candidate lists is
lexicographic nonsense. That much is settled.

The keys do **not** need a custom reducer either. A key declared without
`Annotated` gets LangGraph's `LastValue` channel, which stores the last value
received — replacement, which is precisely what the rewrite loop wants — and
raises `InvalidUpdateError` if two writes arrive in the same step
(`channels/last_value.py`). The loop's writes are in different steps, so
replacement applies; a future fan-out writing `candidates` from two branches at
once is a bug, and `LastValue` is the only option that says so out loud.

Writing an explicit `replace(current, incoming) -> incoming` would document the
intent and silence that error, which is the wrong trade in a codebase whose
governing rule is that a wrong result must never be produced quietly. The intent
goes in a comment at the declaration instead.

| key | channel | why |
|---|---|---|
| `query_embedding` | `LastValue` | rewritten per retrieval attempt, replaced |
| `candidates` | `LastValue` | the loop replaces the set; it never accumulates |
| `retrieval_attempts`, `grounding_attempts`, `sequence` | `monotonic` | unchanged — counters |

## Consequences

- `GraphState` gains non-primitive keys, so `checkpoint_blobs` starts carrying
  real rows. Retention (ADR 0012) already sweeps checkpoints with traces per
  `thread_id`, and this is the commit that makes that sweep worth something.
- `hybrid_search` has two ways to get a vector. The parameter is optional and eval
  keeps calling it without one, so the baselines in `eval/README.md` stay
  comparable.
- `rerank` and `generate` each owe a hydration query, tenant-scoped, and each owes
  a test that a chunk deleted between retrieval and generation degrades to a
  shorter candidate set rather than a crash or a silent substitution.
- Trace payloads and state now hold the same shape for the same reason. A
  `retrieve` trace row's `output_json` is very nearly the `candidates` channel,
  which makes the inspector's panel and the fork's input verifiably the same
  facts.
