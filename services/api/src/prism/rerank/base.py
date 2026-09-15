"""The reranker interface."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["RerankError", "Reranker"]


class RerankError(RuntimeError):
    """No scores were produced. Never a partial or default-filled list."""


@runtime_checkable
class Reranker(Protocol):
    @property
    def model(self) -> str: ...

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to `query`, in order, as a sigmoid in [0, 1]."""
        ...
