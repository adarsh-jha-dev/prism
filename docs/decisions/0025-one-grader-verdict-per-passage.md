# 0025 — The grader returns exactly one verdict per passage

- **Status:** accepted
- **Date:** 2026-09-26
- **Relates to:** [0019](0019-the-retrieval-correction-loop.md),
  [0024](0024-rerank-floor-refit.md)

## Context

`RelevanceVerdicts.verdicts` carried `min_length=1` and no upper bound. The first
answer run at floor 0.01 (local, unpinned) showed both ends of that failing:

- **Too few.** 134 of 177 graded passages got no verdict, and 18 of 27
  multi-passage calls scored passage [1] only. An unscored passage fails, so
  `generate` was usually shown one passage, cited nothing, and 46 of 54
  `verify_grounding` rows recorded 0.0 without a model call.
- **Too many.** 6 of 31 runs degraded on `grade_docs` stopping at `max_tokens`
  mid-JSON — the same failure that cost `gq-024` in the answer baseline, now more
  often because the floor passes ~6 passages instead of ~1.

At floor 0.44 the first was masked: the median survivor count was already 1.

## Decision

`grade_docs` builds its schema per call: `minItems = maxItems = n` and
`1 <= label <= n`, for the `n` passages sent. The name stays `RelevanceVerdicts`.
A reply that violates it is a `ChatError`, which degrades the run rather than
silently failing passages.

Out of scope: `GroundingVerdicts` has the same unbounded shape. An omitted span
already fails toward refusal, so it is left for its own measurement.

## Consequences

- Unmeasured. No answer run has been recorded with this change; the next
  `make eval-answer` is its measurement, against the floor-0.01 run above.
- Duplicate labels still validate, and a duplicated label leaves another
  passage unscored. The next run's unscored count says whether that matters.
