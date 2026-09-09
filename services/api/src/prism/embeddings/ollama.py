"""Local Ollama embeddings — nomic-embed-text, 768-dim.

Self-hosted and free, on the `ollama` lane. There is deliberately no paid
embedding path: a model swap invalidates every vector in the index, so the
choice is a schema concern, not a routing one.
"""

from collections.abc import Sequence
from typing import Any

import httpx

from prism.config import Settings, get_settings
from prism.embeddings.base import EmbeddingError

__all__ = ["OllamaEmbeddingProvider"]


class OllamaEmbeddingProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._model = resolved.embedding_model
        self._dim = resolved.embedding_dim
        self._url = f"{resolved.ollama_base_url.rstrip('/')}/api/embed"
        self._timeout = resolved.embed_timeout_s
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            response = await self._client.post(self._url, json=payload, timeout=self._timeout)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        if any(not text.strip() for text in items):
            raise EmbeddingError("refusing to embed a blank string")

        try:
            payload = await self._post({"model": self._model, "input": items})
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"ollama embed failed: {type(exc).__name__}: {exc}") from exc

        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(items):
            got = len(vectors) if isinstance(vectors, list) else type(vectors).__name__
            raise EmbeddingError(f"asked for {len(items)} embeddings, got {got}")

        # A width mismatch means the pulled model is not the configured one.
        # Mixing widths in one index is silently wrong, so it fails loudly here.
        for vector in vectors:
            if len(vector) != self._dim:
                raise EmbeddingError(
                    f"{self._model} returned {len(vector)} dims, expected {self._dim} — "
                    "chunks.embedding and the pulled model disagree"
                )
        return [[float(x) for x in vector] for vector in vectors]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
