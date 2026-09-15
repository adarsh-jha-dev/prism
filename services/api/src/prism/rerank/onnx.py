"""bge-reranker-v2-m3 as an int8 ONNX cross-encoder, in-process (ADR 0011).

Loaded once per process and never inside a request: `load()` is explicit, and
`score()` on an unloaded model raises rather than spend the latency budget on
warm-up. The forward pass runs in a worker thread behind its own semaphore.
"""

import asyncio
from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from prism.config import Settings, get_settings
from prism.rerank.base import RerankError
from prism.rerank.weights import artifact_paths

__all__ = ["OnnxReranker", "sigmoid"]


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


class OnnxReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self._settings = resolved
        self._model = resolved.reranker_model
        self._batch_size = resolved.rerank_batch_size
        self._timeout_s = resolved.rerank_timeout_s
        self._slots = asyncio.Semaphore(resolved.rerank_concurrency)
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._load_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._model

    @property
    def loaded(self) -> bool:
        return self._session is not None

    async def load(self) -> None:
        async with self._load_lock:
            if self._session is None:
                await asyncio.to_thread(self._load)

    def _load(self) -> None:
        model_path, tokenizer_path = artifact_paths(self._settings)
        for path in (model_path, tokenizer_path):
            if not path.is_file():
                raise RerankError(f"{path} is missing — run: make reranker-fetch")

        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_truncation(max_length=self._settings.rerank_max_tokens)
            tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(model_path), options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RerankError(f"reranker failed to load: {type(exc).__name__}: {exc}") from exc

        self._tokenizer = tokenizer
        self._session = session

    def _forward(self, query: str, passages: Sequence[str]) -> list[float]:
        if self._session is None or self._tokenizer is None:
            raise RerankError("reranker is not loaded")

        scores: list[float] = []
        for start in range(0, len(passages), self._batch_size):
            batch = passages[start : start + self._batch_size]
            encodings = self._tokenizer.encode_batch([(query, passage) for passage in batch])
            feeds = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            logits = self._session.run(None, feeds)[0]
            scores.extend(float(s) for s in sigmoid(np.asarray(logits).reshape(-1)))

        if len(scores) != len(passages):
            raise RerankError(f"scored {len(scores)} of {len(passages)} passages")
        return scores

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        if not self.loaded:
            raise RerankError("reranker is not loaded")

        def release(job: asyncio.Future[list[float]]) -> None:
            self._slots.release()
            if not job.cancelled():
                job.exception()  # retrieved, so an abandoned job logs nothing

        try:
            async with asyncio.timeout(self._timeout_s):
                await self._slots.acquire()
                job = asyncio.ensure_future(asyncio.to_thread(self._forward, query, list(passages)))
                # The thread cannot be interrupted, so its slot is held until it
                # actually finishes, not until the caller stops waiting.
                job.add_done_callback(release)
                return await asyncio.shield(job)
        except TimeoutError as exc:
            raise RerankError(f"rerank exceeded {self._timeout_s}s") from exc
        except RerankError:
            raise
        except Exception as exc:
            raise RerankError(f"rerank failed: {type(exc).__name__}: {exc}") from exc
