"""Golden-set evaluation: retrieval quality, measured before the correction loop.

The point of running this now is that Phase 2 needs something to have improved
*on*. A recall@k taken against the naive retriever is the only thing that makes
the correction loop's contribution a measurement rather than an assertion.
"""

from prism.eval.golden import (
    Corpus,
    CorpusDocument,
    GoldenQuestion,
    GoldenSet,
    GoldenSetError,
    PageRef,
    load_corpus,
    load_golden_set,
)
from prism.eval.metrics import QuestionResult, hit_at_k, recall_at_k, reciprocal_rank

__all__ = [
    "Corpus",
    "CorpusDocument",
    "GoldenQuestion",
    "GoldenSet",
    "GoldenSetError",
    "PageRef",
    "QuestionResult",
    "hit_at_k",
    "load_corpus",
    "load_golden_set",
    "recall_at_k",
    "reciprocal_rank",
]
