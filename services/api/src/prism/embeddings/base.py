"""The embedding provider interface.

`embed_metered` exists for the graph, whose trace rows need a meter. Ingestion
and retrieval keep `embed` and do not pay for a Usage they never read.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from prism.chat.base import Usage

__all__ = ["Embedded", "EmbeddingError", "EmbeddingProvider"]


class EmbeddingError(RuntimeError):
    """An embedding could not be produced.

    Never caught and softened into a zero vector or a partial batch: a wrong
    vector silently retrieves wrong evidence, which is worse than no answer.
    """


@dataclass(frozen=True)
class Embedded:
    vectors: list[list[float]]
    usage: Usage


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

    async def embed_metered(self, texts: Sequence[str]) -> Embedded:
        """`embed`, with the provider's own meter reading."""
        ...
