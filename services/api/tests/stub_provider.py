"""A deterministic embedding provider, shared by the ingestion and API tests.

Same text in, same vector out, no Ollama: what these tests check is the wiring —
the rows written, the statuses, what survives a failure — not the vectors.
"""

import hashlib
import math
from collections.abc import Sequence

from prism.chat import Usage
from prism.embeddings import Embedded, EmbeddingError

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

    async def embed_metered(self, texts: Sequence[str]) -> Embedded:
        vectors = await self.embed(texts)
        usage = Usage(
            model=MODEL,
            provider="ollama",
            billing_unit="tokens",
            input_tokens=sum(len(t.split()) for t in texts),
            output_tokens=0,
            gpu_ms=None,
            duration_ms=1,
            cost_basis="metered",
        )
        return Embedded(vectors=vectors, usage=usage)


def graded(similarity: float) -> list[float]:
    """A unit vector whose cosine similarity with QUERY_VECTOR is exactly this."""
    vector = [0.0] * DIM
    vector[0] = similarity
    vector[1] = math.sqrt(max(0.0, 1.0 - similarity**2))
    return vector


QUERY_VECTOR = graded(1.0)


class QueryProvider(StubProvider):
    """Embeds any query to QUERY_VECTOR, and records what it was asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    async def embed_one(self, text: str) -> list[float]:
        self.queries.append(text)
        return QUERY_VECTOR
