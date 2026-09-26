# 0024 — `rerank_score_floor` is re-fitted to 0.01

- **Status:** accepted
- **Date:** 2026-09-26
- **Relates to:** [0019](0019-the-retrieval-correction-loop.md),
  [0020](0020-rerank-runs-before-grade-docs.md)
- **Amends:** [0011](0011-rerank-runs-in-process.md), on the floor's value; its
  meaning, the model revision and the quantization are untouched

## Context

0.44 was never fitted. It kept its `CLAUDE.md` value through every measurement
that showed it cost more than it bought, and `baseline-2026-09-26-answers.json`
is the end-to-end confirmation: recall@10 0.68 before the floor and 0.36 after,
9 of 25 answerable questions emptied, `grade_docs` handed zero candidates on 42
of 70 calls, and 12 of 20 false refusals `no_relevant_evidence`.

Since ADR 0020 the floor is no longer the last relevance gate. `grade_docs`
judges what it keeps, so the floor's job narrows to two things: remove what is
plainly irrelevant before the grader pays 0.63s a passage for it, and refuse
early when the whole pool is plainly irrelevant.

## Measurement

One scoring pass over the golden set at today's constants
(`rerank_candidate_k = retrieval_top_k = 10`), floors applied to it afterwards.
The 0.44 row reproduces `baseline-2026-09-16-rerank.json` exactly. "Gutted" counts
answerable questions whose gold page the ordering retrieved and the floor removed.

| floor | recall@10 | MRR | answerable emptied | unanswerable emptied | gutted | kept, answerable mean |
|---|---|---|---|---|---|---|
| 0.00 | 0.68 | 0.511 | 0/25 | 0/6 | 0/18 | 10.0 |
| **0.01** | **0.66** | **0.471** | **2/25** | **2/6** | **1/18** | **6.5** |
| 0.02 | 0.58 | 0.453 | 2/25 | 2/6 | 3/18 | 5.5 |
| 0.05 | 0.54 | 0.447 | 5/25 | 2/6 | 4/18 | 4.3 |
| 0.20 | 0.54 | 0.447 | 6/25 | 3/6 | 4/18 | 2.7 |
| 0.30 | 0.50 | 0.407 | 8/25 | 4/6 | 5/18 | 2.2 |
| 0.44 | 0.36 | 0.360 | 9/25 | 5/6 | 9/18 | 1.4 |
| 0.60 | 0.32 | 0.320 | 10/25 | 6/6 | 10/18 | 0.9 |

This cross-encoder puts many gold chunks close to zero: 0.016 and 0.019 on two
questions, 0.045 and 0.024 on others. Any floor above 0.01 starts removing them
faster than it removes anything unanswerable.

## Decision

**0.01.** It is the knee: against no floor it costs 0.02 recall@10 and one gutted
question, whose gold chunk the cross-encoder scored 0.006. For that it cuts the
grader's input from 10 passages to a mean of 6.5 on answerable questions and 3.5
on unanswerable ones, and refuses 2 of 6 unanswerable questions with no LLM call.
Every step above it costs recall and refuses nothing more until 0.15.

## Consequences

- **The floor stops doing most of the refusing.** At 0.44 it emptied 5 of 6
  unanswerable questions, and 5 of the baseline's 6 correct refusals were
  `no_relevant_evidence`. At 0.01 it empties 2, so four unanswerable questions
  now reach `grade_docs` at an uncalibrated `doc_relevance_threshold` and, past
  it, `verify_grounding`. Correct refusal can fall from 6/6. That is measured by
  the next `make eval-answer`, not assumed here.
- Pinned as `runs/baseline-2026-09-26-floor.json`, which is retrieval only. The
  answer baseline stays at 0.44 until it is re-recorded.
- Fitted on 31 questions with gold chunks clustered just above it, so it is
  brittle to the model revision, the quantization, the chunk size and
  `rerank_candidate_k`. Changing any of them re-opens it, as ADR 0011 already
  requires.
