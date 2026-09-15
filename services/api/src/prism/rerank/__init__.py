"""Query and passages in, relevance scores out."""

from functools import lru_cache

from prism.rerank.base import Reranker, RerankError
from prism.rerank.onnx import OnnxReranker

__all__ = ["OnnxReranker", "RerankError", "Reranker", "get_reranker"]


@lru_cache
def get_reranker() -> OnnxReranker:
    """One model copy per process."""
    return OnnxReranker()
