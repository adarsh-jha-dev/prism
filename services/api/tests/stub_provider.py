"""A deterministic embedding provider, shared by the ingestion and API tests.

Same text in, same vector out, no Ollama: what these tests check is the wiring —
the rows written, the statuses, what survives a failure — not the vectors.
"""

import hashlib
from collections.abc import Sequence

from prism.embeddings import EmbeddingError

DIM = 768
MODEL = "nomic-embed-text"


def vector_for(content: str) -> list[float]:
    digest = hashlib.sha256(content.encode()).digest()
    vector = [0.0] * DIM
    vector[int.from_bytes(digest[:4], "big") % DIM] = 1.0
    return vector


class StubProvider:
    model = MODEL
    dim = DIM

    def __init__(self, fail_on: str | None = None) -> None:
        self.batches: list[list[str]] = []
        self._fail_on = fail_on

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self._fail_on is not None and any(self._fail_on in t for t in texts):
            raise EmbeddingError("stub refusing to embed")
        return [vector_for(t) for t in texts]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
