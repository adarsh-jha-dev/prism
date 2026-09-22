"""A reranker with a fixed score per passage, so no weights and no model."""

from collections.abc import Sequence

from prism.rerank import RerankError


class StubReranker:
    model = "bge-reranker-v2-m3"

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        *,
        default: float | None = None,
        fail: bool = False,
    ) -> None:
        self._scores = scores or {}
        self._default = default
        self._fail = fail
        self.calls: list[list[str]] = []
        # What it was asked to judge against. Reranking reads `question`, and a
        # test that only checked the passages could not see that.
        self.queries: list[str] = []

    async def load(self) -> None:
        """Nothing to load. Present because callers must not special-case a stub."""

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append(list(passages))
        self.queries.append(query)
        if self._fail:
            raise RerankError("stub refusing to score")
        if self._default is None:
            return [self._scores[p] for p in passages]
        return [self._scores.get(p, self._default) for p in passages]
