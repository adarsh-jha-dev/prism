# Golden-set evaluation

Retrieval quality against the naive retriever, measured **before** the
correction loop exists. That number is what makes Phase 2's contribution a
measurement rather than an assertion.

    make reranker-fetch # download the pinned reranker weights (~570MB, once)
    make eval-ingest    # verify the corpus, ingest it into `prism-eval`
    make eval           # the golden set against retrieve + rerank
    make eval-hybrid    # the same set against retrieval alone
    make eval-vector    # the same set against the naive baseline retriever

All need `make up` and a local Ollama with `nomic-embed-text` pulled. None can
reach a paid provider: retrieval embeds on the `ollama` lane, rerank runs
in-process, and nothing here generates.

`make eval` runs 31 questions through one reranker slot, so the per-query
`rerank_timeout_s` is a batch's queue wait rather than a query's latency. Raise
it for a run (`RERANK_TIMEOUT_S=600`) — a timeout mid-run fails the run rather
than scoring a query that skipped rerank.

## Files

| | |
|---|---|
| `corpus.yaml` | Committed manifest — filename, sha256, source. The PDFs it names are **not** committed; they live in `corpus/`, which is gitignored. |
| `golden.yaml` | Committed question set. |
| `corpus/` | The PDFs. Local only. |
| `runs/baseline-*.json` | Committed. The pinned baselines the optimized system is measured against. |
| `runs/` | Every other report from `make eval`. Local only. |

Adding a paper: drop the PDF in `corpus/`, `shasum -a 256` it, add the entry to
`corpus.yaml`. `make eval-ingest` verifies every digest before writing anything,
so a corpus that has drifted fails the run instead of quietly moving the
benchmark. It is re-runnable — a document already ingested under the same digest
is skipped rather than duplicated.

## What the report says

`recall@k` is the fraction of a question's relevant pages found in the top k.
`hit@k` is whether *any* of them was. They are different numbers and both are
reported: a question with four relevant pages cannot exceed 0.25 recall@1
however good the ranking, which reads as failure and is not. `ceiling` is the
best recall@k the question set allows at that k.

Relevance is judged per `(document, page)` and never per chunk id — chunk ids
are UUIDv7 minted at ingest, so they change on every re-ingest and on any change
to `chunk_size_chars`.

`quote_found` is computed against the retrieved chunks, not against the gold
page, so it cannot tell you a label is wrong: a `false` reads identically
whether the gold page was missed or was retrieved through a different chunk of
that page. Labels are checked by `tests/test_eval_golden_corpus.py`, which
matches every quote against the text of the page it is filed under.

The **refusal calibration** block covers the unanswerable questions. They score
no recall; what they measure is how high a similarity-only threshold would have
to sit to refuse all of them, and how many clear tau today. Correct refusal is
this project's primary correctness criterion, so the set carries these from the
start.

### Cosine similarity cannot decide refusal

On `baseline-2026-09-11.json` the two classes overlap almost completely:

| | top-1 similarity |
|---|---|
| 6 unanswerable | 0.593 - 0.740, **all six above tau (0.58)** |
| 25 answerable | 0.509 - 0.805 |

The overlap band 0.593-0.740 holds 16 of the 25 answerable questions. Only one
answerable question (`gq-018`, 0.509) scores below every unanswerable one, and
only 8 clear the highest. A threshold set high enough to refuse all six
unanswerable would refuse 17 of 25 answerable questions with it; one low enough
to answer them refuses nothing. **No similarity threshold separates the two
classes on this set.**

That is the measured case for `grade_docs` being load-bearing rather than an
optimisation. Retrieval returns something confident for a question the corpus
cannot answer — that is what a nearest-neighbour index does — so the decision to
refuse has to come from a model reading the passages, not from the distance that
retrieved them. It is also why `abstention_threshold` is a groundedness
threshold only, and never compared against a retrieval score.

## Comparability

A recall number is evidence of an improvement only against a comparable run.
Every JSON report records the retriever, the embedding model, chunk size and
overlap, and the corpus size alongside the metrics. **Change any of those and the comparison is
void**, whatever the numbers do — re-record the baseline rather than comparing
across the change.

It does **not** yet record whether vision ingestion was enabled. A corpus
ingested with `VISION_ENABLED=true` carries extra figure and table chunks, so its
chunk count and its retrieval differ from one ingested without — and the report
cannot tell you which you are looking at. Both pinned baselines were recorded
with vision off, at 482 chunks; check that number before trusting a comparison.

That applies to embedding task prefixes in particular. Retrieval currently
embeds both documents and queries with no prefix (`query_prefix: null` in the
report). Adding `search_document:` / `search_query:` re-embeds the corpus and
invalidates every earlier run.

## Pinned baselines

| | recorded | retriever | recall@10 | mrr | notes |
|---|---|---|---|---|---|
| `runs/baseline-2026-09-11.json` | 2026-09-11 | vector (schema 1) | 0.68 | 0.461 | Naive retrieval, before any correction loop. 31 questions, 482 chunks over 7 documents. |
| `runs/baseline-2026-09-14-hybrid.json` | 2026-09-14 | hybrid | 0.68 | 0.461 | Hybrid FTS + vector, RRF k=60. **Identical to the vector baseline at every k** — see below. |
| `runs/baseline-2026-09-16-rerank.json` | 2026-09-16 | rerank | 0.36 | 0.360 | Rerank over 10 fused candidates, floor 0.44. Recall **after** the floor; the ordering alone reaches 0.68 recall@10 and 0.511 MRR. See below — the floor, not the cross-encoder, is what moves this number. |

Recall is not comparable across retrievers, so schema 2 records which one ran. A
schema-1 report predates the hybrid retriever and is vector-only by construction.

### Hybrid retrieval currently measures zero, and the reason is query construction

`baseline-2026-09-14-hybrid.json` matches `baseline-2026-09-11.json` at every k,
on MRR to six decimal places, and in every per-question retrieval list. That is
not a coincidence and it is not a bug in fusion: **the lexical half returns
nothing for 30 of the 31 questions**, so RRF is fusing one list with an empty one
and reproducing the vector ordering exactly.

The cause is `websearch_to_tsquery`, which ANDs unquoted terms. The golden
questions are prose — they parse to 6-12 stemmed lexemes — and no single
1200-character chunk contains all of them:

| | |
|---|---|
| questions with zero lexical hits | 30 / 31 |
| the exception (`gq-020`) | 1 chunk matched |
| terms ANDed per question | 6-12 |

ADR 0010 predicted this and assigned the fix: fewer, better terms from
`plan_query`, never a looser parser. That node now extracts them, but the eval
runner calls `hybrid_search` directly rather than through the graph, so `terms`
is still unset on every run pinned below and the raw question is parsed. Feeding
the planner's terms to eval re-opens these baselines and is its own commit.

The lexical half is not weak — it is unfed. Ranking the same index with OR
semantics as a one-off diagnostic (not a proposed change, and not committed)
puts the **lexical half alone at hit@10 = 64%**, against the vector baseline's
72%. A retriever that independently finds the gold page in two thirds of cases
is worth fusing; it just has to be asked a question it can answer.

So this baseline pins the point the measurement starts from, not an improvement.
The delta to watch for is the one `plan_query` unlocks.

### Rerank measures two things, and they move in opposite directions

`baseline-2026-09-16-rerank.json` reports recall on what the floor keeps, since
that is all generation would see, and reports the ordering beside it. The two
are worth reading separately:

| | recall@1 | recall@10 | hit@10 | MRR |
|---|---|---|---|---|
| hybrid (2026-09-14) | 0.30 | 0.68 | 0.72 | 0.461 |
| rerank, ordering only | **0.40** | 0.68 | 0.72 | **0.511** |
| rerank, after the floor | 0.36 | 0.36 | 0.36 | 0.360 |

The cross-encoder ranks better than fusion: it moves a relevant chunk into first
place on ten more percentage points of the set and lifts MRR by 0.05. It cannot
raise recall@10 here, because with `rerank_candidate_k` at 10 the pool it scores
*is* the top 10 — reordering a list cannot add a page to it.

Scoring deeper does add pages, and costs more than the query budget allows.

### The candidate cap is a latency decision, and 30 does not fit

Measured serially and warm on an Apple M5 CPU, int8, one query at a time:

| candidates | p50 | p95 | recall@10 (ordering) | MRR |
|---|---|---|---|---|
| 10 | 1.94s | 2.27s | 0.68 | 0.511 |
| 20 | 4.24s | 4.72s | — | — |
| 30 | 7.26s | 7.99s | **0.76** | 0.517 |

Roughly 0.2s per candidate, linear, and thread count is not the problem: the
default already uses every performance core, and forcing 12 threads made it
slower. At 30 candidates rerank alone exceeds the whole 6s per-query budget
before `generate` has run, so the default is **10**. The +0.08 recall@10 that 30
buys is real and currently unaffordable — this is the revisit ADR 0011 names,
and its answer there is a sidecar service rather than a different algorithm.

### At 0.44 the floor removes more evidence than it refuses

Sweeping the floor over one scored run (30 candidates, k=10) gives the whole
trade-off. "Gutted" counts answerable questions whose gold page the ranking
*did* retrieve and the floor then removed entirely:

| floor | recall@10 | answerable emptied | unanswerable refused | gutted |
|---|---|---|---|---|
| 0.00 | 0.76 | 0/25 | 0/6 | 0/20 |
| 0.05 | 0.62 | 4/25 | 1/6 | 4/20 |
| 0.20 | 0.58 | 5/25 | 2/6 | 5/20 |
| 0.30 | 0.54 | 6/25 | 4/6 | 6/20 |
| **0.44** | **0.40** | **7/25** | **5/6** | **9/20** |
| 0.60 | 0.34 | 9/25 | 6/6 | 11/20 |

No value is free: this cross-encoder scores many chunks that *do* contain the
answer below 0.44, so every floor high enough to refuse the unanswerable
questions also discards real evidence. The constant stays at its `CLAUDE.md`
value and this table is the evidence for a re-fit, which ADR 0011 already makes
a condition of changing the revision or the quantization — the same argument
applies to a chunk size the floor was never fitted against.

Two things keep this from being the silent failure it looks like. An emptied
question re-enters the retrieval loop rather than answering ungrounded, and a
kept-but-gutted one still has to pass `verify_grounding`. Neither node exists
yet, so the cost is visible here and nowhere else.

### The rerank score is a real refusal signal, where cosine was not

The vector baseline established that no similarity threshold separates
answerable from unanswerable on this set. The cross-encoder does, by top-1 score
(30 candidates):

| threshold | answerable kept | unanswerable refused |
|---|---|---|
| 0.20 | 20/25 | 2/6 |
| 0.44 | 18/25 | 5/6 |
| 0.60 | 16/25 | 6/6 |

At 0.60 it refuses every unanswerable question while keeping two thirds of the
answerable ones. Cosine could not refuse all six at any threshold without
refusing 17 of 25 answerable questions with them. That is the measured case for
ADR 0011's claim that the floor is what lets retrieval refuse on *retrieved, but
nothing good enough* — and it stays a retrieval decision. It is never compared
against tau, which the report enforces: a rerank run's calibration block is
scored against `rerank_score_floor`.

`make eval` writes `runs/latest.json`, which stays ignored. Pinning a baseline
means copying one to `runs/baseline-<date>.json` and committing it — the report
carries no timestamp of its own, so the filename is the record.

A pinned baseline must have been recorded against the question set committed
beside it, or it is measuring something the repo no longer contains.
`tests/test_eval_baseline.py` enforces that: it checks every pinned file covers
exactly the committed ids and was scored against the committed labels. Change
golden.yaml and that test fails until the baseline is re-recorded.
