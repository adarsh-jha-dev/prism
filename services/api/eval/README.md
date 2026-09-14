# Golden-set evaluation

Retrieval quality against the naive retriever, measured **before** the
correction loop exists. That number is what makes Phase 2's contribution a
measurement rather than an assertion.

    make eval-ingest    # verify the corpus, ingest it into `prism-eval`
    make eval           # run the golden set against hybrid retrieval
    make eval-vector    # the same set against the naive baseline retriever

Both need `make up` and a local Ollama with `nomic-embed-text` pulled. Neither
can reach a paid provider: retrieval embeds on the `ollama` lane and nothing
here generates.

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
`plan_query`, never a looser parser. That node does not exist yet, so
`hybrid_search(terms=...)` is unset on every eval run and the raw question is
parsed.

The lexical half is not weak — it is unfed. Ranking the same index with OR
semantics as a one-off diagnostic (not a proposed change, and not committed)
puts the **lexical half alone at hit@10 = 64%**, against the vector baseline's
72%. A retriever that independently finds the gold page in two thirds of cases
is worth fusing; it just has to be asked a question it can answer.

So this baseline pins the point the measurement starts from, not an improvement.
The delta to watch for is the one `plan_query` unlocks.

`make eval` writes `runs/latest.json`, which stays ignored. Pinning a baseline
means copying one to `runs/baseline-<date>.json` and committing it — the report
carries no timestamp of its own, so the filename is the record.

A pinned baseline must have been recorded against the question set committed
beside it, or it is measuring something the repo no longer contains.
`tests/test_eval_baseline.py` enforces that: it checks every pinned file covers
exactly the committed ids and was scored against the committed labels. Change
golden.yaml and that test fails until the baseline is re-recorded.
