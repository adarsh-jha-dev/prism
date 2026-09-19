"""Text-to-vector, and nothing else."""

from functools import lru_cache

from prism.embeddings.base import Embedded, EmbeddingError, EmbeddingProvider
from prism.embeddings.ollama import OllamaEmbeddingProvider

__all__ = [
    "Embedded",
    "EmbeddingError",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "get_embedding_provider",
]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return OllamaEmbeddingProvider()
