# 0023 — How a query streams: the trace writer publishes its rows, and a disconnect stops the stream, not the run

- **Status:** accepted
- **Date:** 2026-09-25
- **Relates to:** [0011](0011-rerank-runs-in-process.md),
  [0012](0012-query-trace-and-citation-model.md),
  [0016](0016-how-a-node-reports-to-the-trace-writer.md),
  [0017](0017-what-the-graph-state-carries.md),
  [0021](0021-how-generate-binds-an-answer-to-its-evidence.md),
  [0022](0022-the-grounding-loop.md)

## Context

`POST /collections/{id}/query` is the first route that runs the graph, and the
first place a caller outside the test suite sees a node. A local generation takes
tens of seconds, so the route streams: the dashboard's waterfall wants nodes as
they finish, not one body after the run.

Three things had to be decided before the route could be written, and all three
are expensive to reverse — the event shape becomes a wire contract, and the
disconnect policy decides what ends up in `queries`.

## Decision 1 — a node event is the trace row, published by the writer

`write_trace` publishes a `NodeEvent` to an optional channel held in a
`ContextVar`. `run_query` takes the channel and subscribes it for the duration of
the graph run; a run with no subscriber pays nothing.

**Why not `astream(stream_mode="updates")`.** LangGraph's own stream carries state
updates, and a node's meter, verdict, duration and error are deliberately not
state (ADR 0016). Building a waterfall from updates means re-deriving all of it in
the route, from channels that do not carry it — a second implementation of the
trace writer's knowledge, in a place that can drift from the rows the dashboard
also reads.

**The decisive case is failure.** `traced` writes a row with `status = 'error'`
and re-raises; `astream` emits nothing at all for a node that raised. The error
event this route has to send once the status code is spent is exactly the thing
only the trace writer can see.

**Publishing is a record, not a control** — the stance ADR 0008 took for metering
and `link_checkpoints` already takes. It happens after the insert commits, so an
event never describes a row that is not there, and a closed channel drops the
event rather than failing a node.

The channel is unbounded. A run writes one event per node execution and
`max_attempts` bounds that, so the producer is bounded by the graph.

## Decision 2 — finalization returns its result; nothing reads it back

`finalize` now returns an `Outcome` — status, reason, answer and the ranked
citations it wrote — and `QueryRun` carries it.

Finalization happens after `ainvoke` returns and writes no trace row, so it
cannot reach the stream through the channel. The alternative was for the route to
re-read `queries` and `query_citations` after the run, which makes the response a
second observation of a row we had just written and could see changed. `finalize`
already holds the exact ranked list.

Citation filenames come back from the transaction's existing chunk lookup rather
than from state: a filename is re-fetchable by id (ADR 0017), and a citation
whose chunk was deleted between `generate` and finalization reports none, which is
the same fact `chunk_id IS NULL` already records.

## Decision 3 — a client disconnect stops the stream, never the run

The run is detached from the response. A disconnect stops relaying events and
logs `query.abandoned`; the run keeps its lane slot, finalizes and stays forkable.

**Why not cancel, when cancelling frees a GPU slot.** `queries.status` has three
terminal values and no `cancelled`, and `mint_query` inserts the row as
`refused` / `no_relevant_evidence`. A cancelled run therefore leaves a row that is
*indistinguishable from a genuine refusal* — and correct refusal rate is the
headline number this project reports. Wasted compute is a cost; a corrupted
refusal rate is a wrong measurement. Cancelling mid-node also contradicts
`CLAUDE.md`'s requirement that any query be replayable and forkable from any node.

No cap on detached runs is added. The `ollama` lane's semaphore already bounds the
contended resource at 8, and the run terminates on its own under the lane
timeouts and `max_attempts`.

**What Phase 4 changes.** With a queue, the route stops owning the run: it
enqueues and streams the events for a `query_id` from wherever the worker
publishes them, and a disconnect means only "stop following this query" — which is
what detaching already does. That is the reason to detach now rather than cancel:
Phase 4 keeps this shape, whereas cancel-on-disconnect is behaviour it would have
to remove. Real cancellation becomes possible there, because a worker can mark a
row — a fourth status, its own migration, and its own ADR.

## Decision 4 — one route, negotiated on `Accept`

`Accept: text/event-stream` streams `node` events and closes with a `result`
event; anything else returns that same result as one JSON body.

Two routes would duplicate the scope check, the question bounds and the 404 /
409 / 503 mapping, and the duplicate is where they drift. The payloads are not two
shapes: the JSON body *is* the terminal event's data, built once. SSE-only would
make every non-browser consumer parse an event stream to read one object.

The cost is that OpenAPI describes the JSON variant well and the stream only as a
declared `text/event-stream` response; the route's docstring carries the rest.

## Consequences

- The stream's `node` frame is a wire contract on `query_traces`' columns. A
  column the dashboard renders is added to both or neither.
- `finalize`'s signature changed from a tuple to `Outcome`; its one direct caller
  in the tests moved with it.
- `thread_id` is in no response. `run.py` mints it and never accepts it from a
  client, and returning it invites exactly that.
- A failure after the first byte cannot set a status code, so the SSE variant
  answers 200 with an `error` event and a query row left `refused` with partial
  traces. That is `run.py`'s documented failure direction, and it is also what a
  run excluded by ADR 0011's degradation rule looks like to eval.
