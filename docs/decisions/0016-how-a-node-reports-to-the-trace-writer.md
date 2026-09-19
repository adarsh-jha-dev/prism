# 0016 — A node reports usage and payloads through a context object, not its return value

- **Status:** accepted
- **Date:** 2026-09-19
- **Relates to:** [0012](0012-query-trace-and-citation-model.md),
  [0015](0015-checkpointer-tables-and-connection.md)
- **Amends:** [0013](0013-cost-basis-and-model-pricing.md), on which rows a
  query total must price

## Context

`traced` writes one `query_traces` row per node execution. Today a node returns
a state update and nothing else, so the row gets timing and status only. The
next commit makes `embed_query` call a provider and `retrieve` return chunks, and
their rows need a meter (`provider`, `model`, `billing_unit`, tokens or `gpu_ms`)
and payloads (`input_json`, `output_json`: references, not copies, per ADR 0012).

Neither is state. Everything in `GraphState` is serialized into every checkpoint
(`state.py`), and a meter or a payload has no business being replayed into a fork.
The question is how that data gets from a node to the writer. The interesting
difference between the options is what happens when a node gets it wrong.

Two failures matter, and they are not equally bad:

1. **A node forgets to report.** The row writes `billing_unit = 'none'`,
   `provider` NULL. The cost is missing from the total, and the row looks exactly
   like a node that called nothing. This one is silent.
2. **A node calls a provider and then raises** (a paid completion that fails
   validation, a timeout after the provider has accepted the request). The money
   was spent. If the meter goes down with the exception, the error row reads as
   free, and the query total is too low. That is the error ADR 0013 names as the
   one this project cannot afford: a cost figure that fails toward "cheaper".

## Options

### (a) A reserved key in the update dict, popped by `traced`

`return {"retrieval_attempts": 1, "__trace__": TraceMeta(...)}`

- **Forgetting:** silent, as with every option.
- **Raising:** the meter is lost, because the return never happens.
- **Its own failure:** a node that is called without `traced` in front of it (a
  test, a refactor, a second decorator stacked in the wrong order) hands the key
  to LangGraph. LangGraph drops update keys that are not state channels without
  any error (`pregel/_algo.py`), so the meter disappears with no signal. If the key
  were declared in `GraphState` to get it type-checked, it would become a channel
  and be checkpointed, which is exactly what must not happen. The update dict is
  `dict[str, Any]`, so the metadata stays untyped either way.

### (b) A mutable context object passed to the node

`async def embed_query(state: GraphState, trace: TraceContext) -> dict[str, Any]`,
where the node calls `trace.record_usage(u)`, `trace.record_input(...)` and
`trace.record_output(...)`.
`traced` creates the context, calls the node, and writes the row from it.

- **Forgetting:** silent.
- **Raising:** the meter survives. The context outlives the node's frame, so
  whatever was recorded before the exception is written on the error row. This is
  the only option where that works without extra code.
- **Its own failure:** a node can report twice, or report a meter and then call a
  second provider. The row has one `provider` column, so `record_usage()` raises on
  a second call rather than overwriting. That turns a quiet misattribution into a
  loud one.

### (c) The node returns `(update, trace_meta)`

- **Forgetting:** the type checker catches it. `Node` becomes
  `Callable[[GraphState], Awaitable[tuple[dict[str, Any], TraceMeta]]]`, so a node
  that returns a bare dict fails `make lint`. A node that calls nothing has to
  write `TraceMeta.none()`. This is the strongest guarantee of the three against
  failure 1, although it only forces the node to return *some* `TraceMeta`, not
  a correct one. `return update, TraceMeta.none()` on a node that did call a
  provider still type-checks.
- **Raising:** the meter is lost, as in (a). Recovering it would take a
  `TraceMeta` attached to the exception, which means every provider error path
  has to carry one, and any path that doesn't loses it silently.

## Decision

**(b), a context object.** The deciding factor is failure 2, not failure 1.

Failure 1 is silent in all three designs. (c) catches the node that forgets to
return anything, but not the node that returns the wrong thing, and every
provider-calling node is going to need a test asserting its meter anyway. Those
tests will be written, because the benchmark depends on them. Failure 2 is
different. It happens at runtime, under load, on the error path, and only (b)
handles it by construction. It is also the failure that flatters the headline
number, which is the case ADR 0013 exists to prevent.

Rules that come with it:

- `TraceContext` is created by `traced`, one per node execution, and is never
  stored in state, a checkpoint, or anything that outlives the row write.
- `record_usage()` may be called at most once per execution. A second call raises.
- Payloads go through the context as JSON-ready values. The writer serializes,
  applies the size cap from `Settings`, and sets `input_truncated` /
  `output_truncated`. A node never truncates its own payload.
- The error row is written from the same context as the success row, so an
  errored node that already reported a meter gets priced like any other row.

### Rejected: a context variable the providers write into

This would fix failure 1: providers would record their `Usage` into a
`contextvars.ContextVar` that `traced` sets, so a node could not forget. It is
rejected for now for two reasons. It makes `prism.chat` and `prism.embeddings`
depend on the graph's tracing, and those providers are also called by ingestion
and eval, outside any trace. It also makes the meter an ambient side effect of an
`await`, and in a codebase where cost correctness is half the deliverable that is
harder to review than an explicit call. If per-node meter tests turn out not to be
enough, this can be added on top of (b) without changing any node signature.

## Which rows the query total needs priced

ADR 0013 says one unpriced trace row makes `queries.total_cost_usd` NULL, and
that `price_id IS NOT NULL` holds for every row. The second half cannot hold:
a node that called nothing, like a stub now or `abstain` later, has no provider
and no model to look a price row up by. If it counted as unpriced, every query
total would be NULL.

So the rule 0013 states applies to rows that name a provider:

| `provider` | `price_id` | Meaning | Effect on the total |
|---|---|---|---|
| NULL | NULL | no call was made | contributes nothing |
| set | set | priced, `$0` included | summed |
| set | NULL | a call we could not price | total is NULL |

`billing_unit = 'none'` with a provider is still priced: the in-process
reranker has a `$0` row under `'none'`. A missing meter is priceable only against
a zero rate. Zero times an unreported count is still zero, but any other rate
times an unknown count is unknown.

The total is computed in `run.py` in the same `UPDATE` that finalizes the
query, from the trace rows already committed. It is not maintained by a trigger:
a trigger would take a row lock on `queries` for every insert into the
highest-volume table, and this is the only writer. A run that raises before it
finalizes leaves the total NULL. That is incomplete, which is accurate, and not
zero, which would not be. Estimated rows are included in the total. Keeping them
out of the headline figure is the benchmark's job (ADR 0013), not the total's.

## Consequences

- Every node signature gains a second parameter, including stubs. `traced`
  keeps presenting LangGraph with `(state) -> update`, so `graph.py` does not
  change.
- Unit tests can call a node directly with a `TraceContext` they construct, and
  then assert on what it recorded, with no database involved.
- A node that is run without `traced` fails immediately with a missing argument,
  instead of silently losing its meter as it would under (a).
- Failure 1 stays possible. The mitigation is one test per provider-calling node
  asserting the row's `provider`, `model` and meters, not a structural guarantee.
