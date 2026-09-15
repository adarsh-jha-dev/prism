"""A reranker with a fixed score per passage, so no weights and no model."""

from collections.abc import Sequence

from prism.rerank import RerankError


class StubReranker:
    model = "bge-reranker-v2-m3"

    def __init__(self, scores: dict[str, float], *, fail: bool = False) -> None:
        self._scores = scores
        self._fail = fail
        self.calls: list[list[str]] = []

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append(list(passages))
        if self._fail:
            raise RerankError("stub refusing to score")
        return [self._scores[p] for p in passages]
