# 0019 — The retrieval correction loop: a separate retrieval query, re-entry at `plan_query`, one grading call, and no pruned rows yet

- **Status:** accepted
- **Date:** 2026-09-21
- **Relates to:** [0010](0010-full-text-is-postgres-fts.md),
  [0012](0012-query-trace-and-citation-model.md),
  [0016](0016-how-a-node-reports-to-the-trace-writer.md),
  [0017](0017-what-the-graph-state-carries.md)
- **Amends:** [0012](0012-query-trace-and-citation-model.md), on when `pruned`
  trace rows get written; [0017](0017-what-the-graph-state-carries.md), on which
  execution of `plan_query` pins `search_params`

## Context

The graph is a straight line. This commit gives it its first conditional edges:
`grade_docs` either passes to `rerank` or fails to `rewrite_query`, and
`rewrite_query` re-enters retrieval until `max_attempts` is spent, at which point
the run refuses `no_relevant_evidence`.

Four things have to be settled before any of that can be built, and each one is
a place where the obvious implementation is quietly wrong.

## Decision 1 — State carries a `retrieval_query`, distinct from `question`

`embed_query` and `retrieve` both read `state["question"]` today, so a rewrite
has nowhere to land: the loop would re-embed the same text, re-plan the same
terms, and re-run an identical search three times before refusing. The retry
would be pure latency.

`GraphState` gains **`retrieval_query: str`**, seeded to `question` at mint in
`run.py`. From then on:

| key | written by | read by |
|---|---|---|
| `question` | mint, once, never again | `grade_docs`, `generate`, `verify_grounding`, the cache key |
| `retrieval_query` | mint, then `rewrite_query` | `plan_query`, `embed_query`, `retrieve` |

The split is not bookkeeping. It is what stops a rewrite from certifying itself.
`rewrite_query` runs because `grade_docs` failed, which means the model that
wrote the new query is being asked to fix its own upstream failure. If grading
then judged relevance against the rewritten text, a rewriter that drifted — "what
is the chinchilla provisioning ratio" becoming "harbour berth scheduling" —
would retrieve harbour documents and grade them relevant, because against its own
query they *are*. The loop would terminate with a confident answer to a question
nobody asked. Grading and generation judge against what the user submitted; only
the search moves.

So `question` is immutable after mint, and the test for it is not a code-review
convention but an assertion at the end of every run.

Channel: `LastValue`, like `search_terms` and `candidates` (ADR 0017 §3). The
loop replaces it once per attempt and never accumulates, and two writes in one
step is a bug we want loud.

Naming: `retrieval_query`, not `search_query`. The `search_*` keys are the
parameters `plan_query` resolved for the searcher; this is the text the loop
rewrites, and pairing it with `retrieval_attempts` makes the loop's channels read
as one group. `search_query` would also collide in the reader's head with
`search.py`, which is the naive baseline and takes no part in any of this.

The semantic cache, when it arrives, keys on `question`. A cache scoped by the
rewriter's output would be a cache of our own intermediate guesses.

## Decision 2 — The loop re-enters at `plan_query`

`rewrite_query` emits query text only. The edge is
`rewrite_query → plan_query → embed_query → retrieve → grade_docs`.

The alternative — `rewrite_query` emitting terms itself and looping to
`embed_query` — saves one local 14b call per retry, and that is a real saving
against a 6 s budget. It is rejected because of what it costs instead.

`_PLANNER_SYSTEM` is not a generic "extract keywords" prompt. It is tuned to one
specific fact: `websearch_to_tsquery` ANDs its terms, so fewer and rarer terms
retrieve more (ADR 0010). Every rule in it — no synonyms, no words absent from
the question, no stopwords, `planner_max_terms` as a recall knob — exists to stop
the lexical half ANDing itself to zero rows. Copying that into the rewriter's
prompt puts the same hard-won constraint in two places, and the two will diverge
the first time either is tuned. The failure that follows is not a crash: it is
attempt 2 retrieving under different lexical rules than attempt 1, which makes
the attempts non-comparable and quietly invalidates the one measurement this
project exists to publish.

It also keeps each node's contract to one sentence. `plan_query` is text → terms.
`embed_query` is text → vector. `rewrite_query` is text → text. A rewriter that
also extracted terms would be one structured call doing two jobs, and the second
job is the one that gets done badly.

### The pin, and the condition attached to it

ADR 0017 pins `search_params` at `plan_query` so a fork retrieves at the width
the original run used. Re-entering `plan_query` breaks that: a second execution
re-reads `Settings`, and a fork taken after a parameter change would silently
retrieve at a width the original run never used.

So `plan_query` pins **only when `retrieval_attempts == 0`**, and on any later
execution returns no `search_params` key at all. `LastValue` leaves an unwritten
channel alone, so the pinned value survives the loop untouched.

The counter that guard reads is incremented by `grade_docs` — the node that
closes an attempt by rendering its verdict. An attempt is a retrieval that was
graded, so the count is only true once grading has happened; incrementing at
`retrieve` would make `grade_docs` read a counter one ahead of every node that
preceded it in the same pass, and incrementing at `rewrite_query` would leave
`queries.retrieval_attempts` at 0 for a run whose first retrieval succeeded.
Every node in the loop, including `grade_docs`, takes `state[counter] + 1` as its
trace `attempt`, read before the node body runs, which makes the numbering
uniform and the sequence contiguous across passes.

Accepted consequence: a fork resumed **at `plan_query` on the first attempt**
does re-pin from current `Settings`. That is what forking that node means — you
asked to re-plan — and every fork downstream of it, including every fork
mid-loop, keeps the original width.

Rejected: seeding `search_params` as `None` at mint so "unpinned" is
representable and `plan_query` pins on `None`. It closes the fork-at-`plan_query`
case, and it makes `search_params` optional for every reader forever, to defend
a case where re-planning is the thing being asked for.

## Decision 3 — One structured call grades the whole candidate set

Confirmed, as expected, and the invariant in ADR 0016 stands unamended.

Per-chunk grading is not merely inconvenient under `record_usage()`'s
one-call rule — it contradicts the row. `query_traces` has one `provider`, one
`model`, one `billing_unit` and one set of meters per row, and eval addresses
rows by `(query_id, node_name, attempt)`. Ten calls under one node execution have
to become either ten rows sharing a node name and attempt, which makes that
address ambiguous and the waterfall unreadable, or one row with a summed meter,
which throws away which call cost what — the thing ADR 0013 says must stay
re-derivable.

The budget settles it independently. Ten candidates × a local 8b grader, up to
three attempts, is thirty serialized grader calls inside a 6 s per-query budget,
on a lane with eight slots shared with embedding and generation. There is no
version of that which fits.

And grading the set in one call is better grading anyway: the model sees the
candidates together and can judge relative to what else was retrieved, which is
the judgement `grade_docs` is actually for.

Two rules come with it:

- **Verdicts are matched by an explicit label, never by position.** The prompt
  numbers the candidates `1..N` and the schema returns that number with each
  verdict. UUIDs get mangled by models; positional lists get shifted by a model
  that returns nine verdicts for ten chunks, and a shifted list assigns chunk 3's
  verdict to chunk 4 with no error anywhere. Labels outside `1..N` are dropped.
- **A candidate with no verdict fails.** Absence of a judgement is not evidence
  of relevance, and the governing rule of this project is that the unsure path
  refuses rather than proceeds.

The set passes if at least one candidate passes, and `grade_docs` writes the
passing subset back to `candidates`. One relevant chunk is enough to attempt an
answer; `verify_grounding` is the real gate, and a chunk the grader called
irrelevant has no business reaching `rerank` or being citable.

Honest cost of batching: one malformed structured response loses every verdict
rather than one. That is a failure into refusal, not into an answer, so it fails
in the direction this system is allowed to fail. Context is not a concern at this
size — ten chunks at `chunk_size_chars = 1200` is roughly 3k tokens against
`llama3.1:8b`'s window.

## Decision 4 — `pruned` rows are deferred, not written at finalization

ADR 0012 introduced `status = 'pruned'` so the viewer could grey out branches not
taken. LangGraph never executes a pruned node, so no wrapper runs and no row
exists. Writing them at finalization is rejected for now, on three grounds.

**They have no honest `sequence`.** `sequence` is `UNIQUE (query_id, sequence)`
and is what orders the waterfall, precisely because `started_at` ties under clock
resolution (ADR 0012, and ADR 0002 before it). A node that never ran has no
position in time. Appending pruned rows after the run puts them at sequences
beyond every real row, so the viewer renders "not taken" at the far right of the
waterfall — the one place they visually do not belong. Interleaving them instead
means synthesising an ordering for events that never happened.

**Their content is derivable.** A pruned row carries no meter, no payload, no
duration and no error. Everything it would say is the node set minus the node
names present, which the viewer computes at read time from the rows it already
fetched. Storing a derived fact in the highest-volume table in the system is the
wrong side of the arithmetic ADR 0012 did for retention and partitioning: four to
six extra rows per query, at ~612 queries/24h, carrying nothing.

**The topology is not finished.** There is one conditional edge today. The cache
check, the grounding loop and the cost-aware router each add more, and each
changes what "not taken" means. A writer built against today's shape gets
rewritten twice before the graph is complete.

The `pruned` value stays in the CHECK constraint. It costs nothing, and it has a
future meaning that is not this one: a node that **started and was abandoned** —
a cancelled parallel branch — is genuinely pruned, has a real `started_at`, and
is not derivable from anything. That is the case worth keeping the status for.

Revisit when a node can be abandoned mid-flight, or when the viewer's derived
greying proves insufficient in use.

## A note on naming the grading threshold

Not one of the four questions, but a policy constant is expensive to rename once
traces and the dashboard carry it, so it is settled here.

`grade_docs` gets **`doc_relevance_threshold`**, defaulting to `0.5`.

The pair has to be unconfusable on sight, because `CLAUDE.md` forbids exactly one
mistake: `abstention_threshold` is the groundedness score at `verify_grounding`
and must never be compared against anything else. The names now carry the
quantity, not the mechanism — `doc_relevance_threshold` scores relevance at
`grade_docs`, `abstention_threshold` scores groundedness at `verify_grounding` —
so a call site comparing the wrong one reads wrong rather than merely being wrong.

`0.5` is uncalibrated and marked as such: it is "the grader says more likely
relevant than not", not a measured operating point. It is calibrated against the
31-query golden set, on its own, and it is never derived from tau. Sharing a 0-1
scale is the coincidence that makes this whole class of bug possible.

## Consequences

- `GraphState` gains `retrieval_query`, so checkpoints carry it and a fork
  mid-loop resumes searching for what that attempt was searching for. It is a
  primitive, so it is inlined into every checkpoint rather than blobbed
  (ADR 0017) — a few hundred bytes per checkpoint, not per write.
- `plan_query` and `embed_query` stop reading `question`. Their trace `input_json`
  changes shape accordingly, and on attempt 2 a reader can see the planner's terms
  change because the query changed rather than because the planner did.
- `plan_query` now behaves differently on its first execution than on its later
  ones. That is a branch on state inside a node, which is worth one test of its
  own: two passes of the loop, one pinned `search_params`.
- `grade_docs` prunes `candidates` to the passing subset, so `rerank` and
  `generate` never see a chunk the grader rejected, and `query_citations` cannot
  cite one.
- The viewer owes a derivation for "not taken" rather than a query for it. That
  is a dashboard change, and it is the cheaper one.
- Exhausted retrieval refuses directly. The web-search MCP fallback that sits
  between exhaustion and refusal in the design is out of scope for Phase 2 and
  stays out: `CLAUDE.md` has no web-search node, and one could not satisfy
  `query_citations.chunk_id` if it existed.
