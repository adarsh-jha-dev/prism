"""The embedding provider interface."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["EmbeddingError", "EmbeddingProvider"]


class EmbeddingError(RuntimeError):
    """An embedding could not be produced.

    Never caught and softened into a zero vector or a partial batch: a wrong
    vector silently retrieves wrong evidence, which is worse than no answer.
    """


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str:
        """Model identifier, as recorded in `collections.embedding_model`."""
        ...

    @property
    def dim(self) -> int:
        """Vector width. Must match the `chunks.embedding` column."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch, in order. Raises EmbeddingError rather than degrade."""
        ...

    async def embed_one(self, text: str) -> list[float]: ...
