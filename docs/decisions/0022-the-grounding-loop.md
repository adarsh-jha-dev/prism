# 0022 — The grounding loop: tau from the collection, sentence spans scored by their minimum, judged against what the answer cited

- **Status:** accepted
- **Date:** 2026-09-24
- **Relates to:** [0012](0012-query-trace-and-citation-model.md),
  [0013](0013-cost-basis-and-model-pricing.md),
  [0016](0016-how-a-node-reports-to-the-trace-writer.md),
  [0017](0017-what-the-graph-state-carries.md),
  [0019](0019-the-retrieval-correction-loop.md),
  [0021](0021-how-generate-binds-an-answer-to-its-evidence.md)

## Context

`verify_grounding` is the gate, and this is the commit that opens the first path
to `status = 'answered'`. Everything before it could only refuse, which is why
every decision here is expensive to reverse: once a query can finalize, the
shape of the judgement is what the benchmark measures and what the dashboard
renders.

`abstention_threshold` has existed on `collections` since migration 0001 and
nothing has ever read it. `query_traces.verdict` has been nullable since
migration 0009 "for grading nodes" and nothing has ever written it. Both are
closed here.

## Decision 1 — tau is pinned into `SearchParams`, from the collection

`SearchParams` gains `abstention_threshold`, resolved at `plan_query` from
`collections.abstention_threshold` and falling back to `Settings` only when the
collection is not visible to the tenant.

**Why `SearchParams` and not a channel of its own.** The name reads like widths,
but the type already carries `max_attempts` and `doc_relevance_threshold`, and
its docstring already says "to retrieve **and correct with**". One rule holds for
all of them — policy is pinned once per run, `Settings` is never re-read
mid-run — and a second channel would mean two rules and, eventually, a fork that
honours one and not the other.

**Why `plan_query` and not mint.** ADR 0019 put the pin there, under the
`retrieval_attempts == 0` guard, precisely so the rewrite loop cannot re-pin.
Reading tau anywhere else would leave a fork taken at `plan_query` re-pinning
four parameters and inheriting the fifth, which is the kind of split no one
remembers a year later. The cost is one keyed read under the tenant predicate,
in a node that already makes a 14b call.

**What the fallback is for.** The column is `NOT NULL DEFAULT`, so there is no
null to fall back from. The fallback covers exactly one case: the collection row
is not readable under this tenant. That is close to impossible — `queries` holds
a composite foreign key to `collections` — so it is a defensive default with a
warning on it, not a supported configuration.

**What tau is compared against, and nothing else.** The aggregate groundedness
score that `verify_grounding`'s own call produces. Not a cosine similarity, not
a rerank score. `CLAUDE.md` forbids that comparison and ADR 0019 named the two
thresholds after their quantities so a wrong call site reads wrong. This is the
only node that reads tau.

## Decision 2 — A span is a sentence we split; the aggregate is their minimum

### The split is ours, not the model's

The answer is split on `.!?` followed by whitespace and the start of the next
sentence — a capital, a quote, a bracket. Not before a digit: `0.58` and
`approx. 20` are one claim, and a system whose answers are mostly numbers cannot
afford a splitter that turns a decimal point into a claim boundary. A fragment
under 24 characters is merged into its neighbour rather than judged alone.

Rejected: **model-chosen claims**. This is ADR 0021's structured-spans argument
one node later. A claim list is a paraphrase of the answer, there is no way to
align it back to the text we hold, and a claim the model quietly omits from its
own list is a claim that never gets judged — which is silent, and is the
fabrication path.

Rejected: **marker-delimited claims**, splitting at `[n]`. An uncited sentence
would then not be a span at all, and an uncited sentence is the most likely one
to be fabricated. The gate cannot be blind to exactly the text it exists to
catch.

Accepted cost: a mis-split sentence is judged as a unit. That is conservative —
it can refuse a sound answer, never admit an unsound one — and it is
deterministic, so a trace row can be read against the answer it judged.

### The aggregate is the minimum, and we derive it

The model returns a score per span. The node takes the **minimum** and compares
that to tau, so `pass` holds exactly when every span scored at least tau.

A **mean** was rejected because it lets one fabricated sentence hide behind four
sound ones. Prompt injection is the failure this project measures with 72
payloads, and its signature is one inserted claim in otherwise faithful prose.
An aggregation that averages that away is the wrong instrument for the thing
being instrumented.

The aggregate is **derived here, not returned by the model** — `grade_docs`'
rule, for `grade_docs`' reason: a model that reports both parts and a total can
report a total that contradicts its parts, and there is no way to tell which is
wrong. With min, `groundedness >= tau` and `unsupported_spans == 0` are the same
statement, so the trace row cannot disagree with itself.

A span with no verdict scores nothing and fails, as a candidate with no verdict
fails at `grade_docs`. Absence of a judgement is not evidence of groundedness.

**Honest cost: min is length-sensitive.** A ten-sentence answer has ten chances
to dip under tau, so longer answers refuse more often. That biases toward
refusal, which is the direction this system is allowed to fail. It also means
tau and this aggregation are one calibration, not two: **0.58 is inherited from
`Settings` and is uncalibrated for groundedness**, exactly as
`doc_relevance_threshold`'s 0.5 is uncalibrated for relevance. Calibrating it
against the 31-query golden set is its own piece of work, and it is never
derived from either of the other two thresholds.

## Decision 3 — The answer is judged against the citations it made

Not against every candidate that survived grading.

1. It is the stricter reading, and it is the invariant ADR 0021 established: an
   answer is bound to what it cited.
2. Judging against all survivors forgives a correct claim that cited the wrong
   passage — and that wrong passage is what `query_citations` then persists as
   the truth of where the answer came from. Forgiving it at the gate is how the
   audit trail becomes false.
3. **It needs no database read.** ADR 0021 Decision 4 already put the cited text
   in state as the model was shown it. Judging against survivors would mean a
   fourth hydration of the same pass, and — worse — judging the answer against
   text that may have changed since `generate` produced it.

The evidence is labelled with the same `[n]` labels `generate` showed, so a
marker inside a span still resolves to the passage the claim points at. Claims
are numbered `(n)` so the two schemes cannot collide in an 8b model's reading.

### An answer with no citations is failed before the call

It follows directly: with citations as the evidence, an answer that cited
nothing has nothing to be judged against, and a verifier handed no passages
invents a verdict. So the node fails it without calling anything, and the run
loops or refuses.

This is also what keeps `queries_citation_count_check` satisfiable honestly. The
constraint forbids an answered row with zero citations; the way to satisfy it is
never to invent a citation, but to refuse. The reason is
**`insufficient_evidence`**, not `no_relevant_evidence`: candidates survived
grading, so retrieval did its job, and it is the binding between answer and
evidence that is missing. Booking it as `no_relevant_evidence` would file a
generation failure under retrieval in the one metric this project publishes.
This is the derivation `abstain` already performs.

## Decision 4 — A regeneration is told what was unsupported

`GraphState` gains **`unsupported_spans: list[str]`**, written by
`verify_grounding` on every execution — the failing spans on a fail, empty on a
pass, so it can never be stale — and read by `generate`, which appends them to
its prompt with an instruction to drop what the passages do not support.

Without this, "regenerate under stricter constraints" is the same call with the
same inputs and a different random seed. That is the objection ADR 0019 raised
against re-embedding an unrewritten query: the retry would be pure latency
against a 6 s budget.

This channel carries the model's own output, not corpus text, and every span is
a substring of `answer`, which is already in state. It opens nothing that
ADR 0021 Decision 4 did not already open, and it is replaced per attempt rather
than accumulated.

**Provider escalation is explicitly not this.** `generate` stays pinned to the
local lane; choosing a provider on cost is the Phase 3 router's decision.

## Decision 5 — The gate writes `status`; the edge only reads it

`verify_grounding` writes `status = 'answered'` on a pass, and
`after_verify_grounding` routes on it — the same arrangement as `grade_docs`
writing the surviving candidates and `after_grade_docs` routing on them. Edges
decide without writing a trace row, so the node that made the judgement is the
node that recorded it.

There is **no `finalize` node**. `CLAUDE.md`'s graph has three terminals and no
such node, and the transactional write needs `latency_ms`, which is measured
around `ainvoke` in `run.py`. The pass edge goes to `END` and `run.py` finalizes,
as it already did for refusals.

A run that dies between the gate and the write leaves `queries` as it was
inserted — refused, with a reason, and `total_cost_usd` NULL. It fails closed.

## Decision 6 — `rank` is assigned at write time, from the score

`query_citations.rank` is `UNIQUE (query_id, rank)` and checked `>= 1`, and it is
the order the dashboard lists sources in. It is assigned at finalization by
`rerank_score` descending, densely from 1 — never inherited from the order the
model returned its citation list in.

`rerank`'s fallback leaves a candidate unscored (ADR 0011). Unscored sorts last
and keeps its relative order rather than sorting as zero, which would rank a
chunk nothing scored above one that scored 0.45.

`chunk_ref` takes the cited id unconditionally; `chunk_id` takes it only if the
chunk still exists, resolved in the finalizing transaction. That is migration
0009's stated semantics, and without the resolution a chunk deleted between
`generate` and finalization would fail the foreign key and lose a sound answer.

## Decision 7 — `verdict` is written, by both grading nodes

`TraceContext` gains `record_verdict`, and the writer gains the column.
`verify_grounding` writes `pass`/`fail`; `grade_docs` writes it too, including on
its two paths that make no call — an empty candidate set and a set that hydrates
to nothing are both judgements, and a NULL there would be indistinguishable from
a node that has no judgement to make.

Every other node keeps writing NULL, which is what migration 0009 made the column
nullable for.

## Consequences

- The span texts go on the `verify_grounding` trace row. This is the one place
  answer text reaches a trace payload, and it is deliberate: a rejected attempt
  never reaches `queries.final_answer`, so this row is the only record of what
  the run refused to say. A refusal must be as inspectable as an answer, and
  ADR 0021 already made the trace row the home for a rejected attempt.
- `generate` records the **count** of unsupported spans it was given, not their
  text. The texts are on the previous attempt's `verify_grounding` row, which
  eval already addresses by `(query_id, node_name, attempt)`.
- `run.py`'s finalization grows `final_answer`, `citation_count` and the
  `query_citations` insert, in the transaction that already wrote `status` and
  `total_cost_usd`.
- The two loops still count independently. A run that spends its whole retrieval
  budget arrives at `generate` with all three generations unspent, and neither
  counter is readable from the other.
- The graph now has a cycle that can execute `generate` three times. Nothing
  else changes: no routes, no cache, no escalation, and no web-search fallback.
