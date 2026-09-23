# 0021 — How `generate` binds an answer to its evidence: labelled passages, a validated citation list, and cited text in state

- **Status:** accepted
- **Date:** 2026-09-23
- **Relates to:** [0012](0012-query-trace-and-citation-model.md),
  [0016](0016-how-a-node-reports-to-the-trace-writer.md),
  [0019](0019-the-retrieval-correction-loop.md),
  [0020](0020-rerank-runs-before-grade-docs.md)
- **Amends:** [0017](0017-what-the-graph-state-carries.md), on the one channel
  that carries chunk text rather than a reference

## Context

`generate` is the first node that produces an answer, and an answer is only
worth anything if it is bound to the evidence it came from. Three questions have
to be settled before the node can be written, and each has a plausible answer
that is quietly wrong.

`verify_grounding` is still a stub, so nothing here finalizes as `answered`. That
is what makes this the right commit to settle them: the binding, the validation
and the storage shape are decided while no answer can yet escape.

## Decision 1 — Labelled passages, inline markers, and a structured citation list

The prompt numbers the passages `[1]`, `[2]`, … exactly as `grade_docs` does. The
model returns `{answer, citations: [{label}]}`: prose carrying inline `[n]`
markers, plus the list of labels keyed the same way.

The label shape is shared with the grader rather than reinvented —
`_numbered_passages` builds both prompts — because that shape is already proven
against a local model and its label-to-chunk mapping is already the one function
that drops what was never sent.

**What this costs.** Binding is at passage granularity, not clause. A marker's
position says where the model believes the evidence lands, but we store no
character offsets — those are a known gap in `REVIEW.md` — so `query_citations`
can say *this answer cited chunk X* and never *this clause came from chunk X*.
`verify_grounding` will judge the whole answer against the whole cited set, so a
mid-sentence marker that is right about the passage and wrong about the clause it
sits in is invisible to tau. This decision does not introduce that limit; it
declines to fix it, and the citations table is already shaped to take offsets as
two nullable columns against `cited_content` when it is fixed.

Rejected:

- **Structured spans** — `{claim, quote, label}` — require verbatim quote
  reproduction. Models at this size paraphrase, which leaves two options: store an
  unverified quote, which is worse than no quote because it looks like evidence,
  or fuzzy-match it back to the passage, which is post-hoc matching under another
  name. It also fights the deliverable: the answer is prose, not a list of claims
  to be reassembled into some.
- **Post-hoc matching** — embed the answer's sentences, match them to chunks —
  re-introduces cosine similarity as an evidence signal. `CLAUDE.md` is explicit
  that similarity and groundedness are different quantities sharing a 0-1 scale,
  and that similarity does not separate answerable from unanswerable at any
  threshold. It also makes the fabricated-citation test meaningless: every
  sentence matches something, so nothing is ever fabricated.

### The citation list carries no `min_length`, unlike the grader's

`RelevanceVerdicts` needs `min_length=1` because without it the cheapest
completion that validates is `{"verdicts": []}`, and a grader always has
passages to score. Generation does not: a model that correctly reports the
passages do not cover the question has nothing to cite, and forcing a citation
onto it would manufacture the binding this node exists to establish. An answer
that cites nothing is handled as an outcome below, not prevented by a schema.

## Decision 2 — A label that was never shown is dropped, never a failed generation

Every label the answer cites must exist in the set actually put in front of the
model. A fabricated chunk id is one of the injection categories the project
tests for, so this validation is a mitigation rather than tidiness.

The response is reconciled against the shown labels in two directions, and the
asymmetry is deliberate:

| case | outcome | why |
|---|---|---|
| listed label not in the shown set | dropped from the citation list | fabricated; it must never reach a row |
| in-range marker in the prose, absent from the list | admitted as a citation | the passage was shown, so admitting it adds evidence and fabricates nothing |
| listed label with no marker in the prose | kept as a citation | saves an answer whose prose came back clean but unmarked |

A bracketed number **outside** the shown range is not treated as a marker at all.
With five passages, an answer that legitimately contains `[20]` keeps it as
prose. The consequence is that a fabricated `[7]` stays in the answer text as a
bracket resolving to nothing, and that is accepted: the prose is never rewritten,
so `queries.final_answer` is exactly what the model produced, the citation row
never exists, and the dropped label is named on the trace row. An audit trail
that edits its subject is not one.

The resulting invariant, which is what the tests assert: **every marker in the
answer resolves to exactly one citation row, and none resolves to zero.**

**Dropping rather than failing** is the choice, because `generate` has an outcome
and no judgement — migration 0009 made `verdict` nullable precisely so this node
writes none, and rejecting an answer is `verify_grounding`'s job. Dropping is not
leniency, because of what it leaves behind: if validation empties the citation
set, there is no evidence binding, so there is no answer to carry forward and the
run refuses. The same holds for an empty or whitespace answer. Neither is an
error — the provider call succeeded and the schema validated — so the trace row
takes `status = 'ok'` with the reason on its payload, as `grade_docs` does for an
empty candidate set. `'error'` would claim a failure that did not happen, and
0009's `error_present_check` would demand error text we do not have.

A generation that produces nothing also **clears** `answer` and `citations` in
state rather than leaving them. Once the grounding loop exists, a second attempt
that grounds nothing must not leave the first attempt's answer standing for
finalization to persist.

## Decision 3 — Citations become rows once, at finalization

They are held in state until the query finalizes, and written in the same
transaction as the `queries` update that sets `status`, `final_answer` and
`citation_count`. The alternative — write per attempt and replace — was rejected
on three counts:

1. `queries_citation_count_check` is transactional with the counter. Rows written
   while the query still reads `refused / citation_count = 0` contradict their
   parent for the length of the run, and permanently if the process dies there.
   ADR 0012's governing rule is that the constraint enforces this, not the writer.
2. `query_citations_rank_key UNIQUE (query_id, rank)` makes write-per-attempt
   really delete-then-insert per attempt.
3. A run that ends refused then needs no cleanup path at all. Refusals carry zero
   citation rows by construction, which is what ADR 0012 requires.

**A rejected attempt stays inspectable on its trace row**, which is the better
place for it. `generate` writes one row per execution, carrying the labels it
used, the chunk ids behind them and the labels validation dropped — a permanent
per-attempt record. Write-per-attempt would put a rejected attempt into
`query_citations` and then delete it, making it inspectable for one node's
duration, in the table that is supposed to record what was *actually* cited.

## Decision 4 — State carries the cited text. This is an exception to ADR 0017 §2

`cited_content` is `NOT NULL` and ADR 0012 justifies it as a snapshot that
survives the chunk it cites. A snapshot re-derived at write time is not a
snapshot, and re-hydrating at finalization takes it *after* the window in which
the thing being snapshotted can change:

- the chunk is re-ingested between `generate` and finalization — `cited_content`
  then holds text the model was never shown, attributed to an answer that never
  saw it. A false audit trail is worse than a missing one, and it is invisible.
- the chunk is deleted in that window — `hydrate_chunks` returns short by design,
  `cited_content` cannot be filled, and a correct answer becomes a write failure.

So `GraphState` gains a `citations` channel holding each cited passage's text as
the model was shown it. Size is not the argument on either side: ADR 0017
measured ten hydrated chunks at 14,001 bytes per write, the cited set is at most
`k` and usually smaller, and the channel is written once per generation attempt —
under 42 KB across a exhausted loop.

**The honest cost is isolation.** ADR 0017 §2 refused chunk text in state because
`checkpoint_blobs` carries no tenant column and cannot be placed under RLS
(ADR 0015), and copying scanned rows into an unscopable table hands back what the
tenant predicate bought. That argument is still true, and this is a deliberate
exception to it with a stated boundary:

- only text the model actually cited, never the candidate pool;
- only after generation, so nothing upstream of it widens;
- replaced per attempt, never accumulated.

`candidates` remains references, and nothing here reopens it.

## Decision 5 — The third hydration of the same pass is not cached

`rerank`, `grade_docs` and `generate` each hydrate under the tenant predicate in
one pass. They are not the same read three times: the sets shrink — the fused
pool, then what the floor kept, then what grading kept — and each is a
primary-key lookup of at most ten rows with one join, single-digit milliseconds
against a 6 s budget dominated by a 32b local generation.

A cache would cost more than it saves. Nodes take `(state, trace)`, so it lives
in a contextvar — invisible to replay — or in a state channel, which is
Decision 4's problem for the full pool rather than the cited set. It would also
mask behaviour that is deliberate and tested: hydrating short is how `rerank` and
`grade_docs` detect a chunk deleted or re-ingested mid-run, and a cache would
serve the stale text instead.

Revisit when profiling puts hydration in the p95's top costs. Measure first.

## Consequences

- `generate` calls `structured`, not `complete`. `chat/base.py` claimed
  `generate` was `complete`'s only caller; that is amended, and `complete` now
  has no caller in the graph — it is the naive baseline's shape. The reason
  `generate` is structured is that the citation list has to validate, and a
  citation list parsed out of prose afterwards is post-hoc matching again.
- `generate`'s lane is pinned to the local one and marked as the Phase 3 router's
  decision. Nothing here chooses a provider on cost.
- Finalization owes the citation write, the `final_answer` write and the
  `citation_count` in one transaction. None of it exists yet, and none of it can
  run until `verify_grounding` passes an answer.
- `grounding_attempts` is untouched by this node. The gate increments it, as
  `grade_docs` does for the retrieval loop.
- A run whose every citation was fabricated refuses `insufficient_evidence` —
  candidates survived retrieval, and generation is what did not complete. That is
  the correct reading, and it is already what `abstain` derives.
