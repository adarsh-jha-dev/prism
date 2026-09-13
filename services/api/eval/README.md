# Golden-set evaluation

Retrieval quality against the naive retriever, measured **before** the
correction loop exists. That number is what makes Phase 2's contribution a
measurement rather than an assertion.

    make eval-ingest    # verify the corpus, ingest it into `prism-eval`
    make eval           # run the golden set, report recall@k

Both need `make up` and a local Ollama with `nomic-embed-text` pulled. Neither
can reach a paid provider: retrieval embeds on the `ollama` lane and nothing
here generates.

## Files

| | |
|---|---|
| `corpus.yaml` | Committed manifest — filename, sha256, source. The PDFs it names are **not** committed; they live in `corpus/`, which is gitignored. |
| `golden.yaml` | Committed question set. |
| `corpus/` | The PDFs. Local only. |
| `runs/` | JSON reports from `make eval`. Local only. |

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

## Comparability

A recall number is evidence of an improvement only against a comparable run.
Every JSON report records the embedding model, chunk size and overlap, and the
corpus size alongside the metrics. **Change any of those and the comparison is
void**, whatever the numbers do — re-record the baseline rather than comparing
across the change.

That applies to embedding task prefixes in particular. Retrieval currently
embeds both documents and queries with no prefix (`query_prefix: null` in the
report). Adding `search_document:` / `search_query:` re-embeds the corpus and
invalidates every earlier run.
