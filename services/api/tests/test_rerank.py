"""Unit: the rerank node and the reranker runtime. No weights, no database."""

import asyncio
import time
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from prism.config import Settings
from prism.rerank import OnnxReranker, Reranker, RerankError
from prism.rerank.onnx import sigmoid
from prism.rerank.weights import artifact_paths, fetch_weights
from prism.retrieval.hybrid import HybridHit
from prism.retrieval.rerank import rerank
from stub_reranker import StubReranker

SETTINGS = Settings(rerank_score_floor=0.44, rerank_candidate_k=30, retrieval_top_k=10)


def hit(content: str, rank: int) -> HybridHit:
    return HybridHit(
        chunk_id=UUID(int=rank),
        document_id=UUID(int=0),
        filename="a.pdf",
        content=content,
        page_number=rank,
        chunk_index=0,
        rank=rank,
        vector_rank=rank,
        lexical_rank=None,
    )


def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubReranker({}), Reranker)
    assert isinstance(OnnxReranker(SETTINGS), Reranker)


async def test_orders_by_score_not_by_fused_rank() -> None:
    stub = StubReranker({"a": 0.2, "b": 0.9, "c": 0.6})
    candidates = [hit("a", 1), hit("b", 2), hit("c", 3)]
    result = await rerank("q", candidates, reranker=stub, settings=SETTINGS)

    assert [h.content for h in result.ranked] == ["b", "c", "a"]
    assert [h.rank for h in result.ranked] == [1, 2, 3]
    assert [h.fused_rank for h in result.ranked] == [2, 3, 1]


async def test_the_floor_keeps_scores_at_or_above_it() -> None:
    stub = StubReranker({"a": 0.9, "b": 0.44, "c": 0.4399})
    candidates = [hit("a", 1), hit("b", 2), hit("c", 3)]
    result = await rerank("q", candidates, reranker=stub, settings=SETTINGS)

    assert [h.content for h in result.hits] == ["a", "b"]
    assert len(result.ranked) == 3


async def test_everything_below_the_floor_leaves_no_hits() -> None:
    """The edge that re-enters the retrieval loop (ADR 0011)."""
    stub = StubReranker({"a": 0.1, "b": 0.3})
    result = await rerank("q", [hit("a", 1), hit("b", 2)], reranker=stub, settings=SETTINGS)

    assert result.hits == ()
    assert [h.score for h in result.ranked] == [0.3, 0.1]


async def test_equal_scores_keep_fused_order() -> None:
    stub = StubReranker({"a": 0.5, "b": 0.5, "c": 0.5})
    candidates = [hit("c", 3), hit("a", 1), hit("b", 2)]
    result = await rerank("q", candidates, reranker=stub, settings=SETTINGS)
    assert [h.fused_rank for h in result.ranked] == [1, 2, 3]


async def test_scores_only_the_candidate_cap_and_keeps_k() -> None:
    candidates = [hit(str(i), i) for i in range(1, 8)]
    stub = StubReranker({str(i): i / 10 for i in range(1, 8)})
    settings = Settings(rerank_candidate_k=5, rerank_score_floor=0.44)

    result = await rerank("q", candidates, k=2, reranker=stub, settings=settings)

    assert stub.calls == [["1", "2", "3", "4", "5"]]
    assert [h.content for h in result.ranked] == ["5", "4"]


async def test_no_candidates_is_no_hits() -> None:
    result = await rerank("q", [], reranker=StubReranker({}), settings=SETTINGS)
    assert result.ranked == () and result.hits == ()


async def test_a_failure_propagates_instead_of_falling_back_to_fusion_order() -> None:
    with pytest.raises(RerankError):
        await rerank("q", [hit("a", 1)], reranker=StubReranker({}, fail=True), settings=SETTINGS)


async def test_a_short_score_list_is_an_error() -> None:
    class Short(StubReranker):
        async def score(self, query: str, passages: object) -> list[float]:
            return [0.9]

    with pytest.raises(RerankError, match="scored 1 of 2"):
        await rerank("q", [hit("a", 1), hit("b", 2)], reranker=Short({}), settings=SETTINGS)


def test_sigmoid_maps_logits_into_the_floor_unit() -> None:
    """The floor applies after this, never to the raw logit."""
    out = sigmoid(np.array([0.0, -5.0, 5.0]))
    assert out[0] == 0.5
    assert out[1] < 0.44 < out[2]


async def test_scoring_before_load_raises() -> None:
    with pytest.raises(RerankError, match="not loaded"):
        await OnnxReranker(SETTINGS).score("q", ["passage"])


async def test_missing_weights_name_the_fetch_target(tmp_path: Path) -> None:
    reranker = OnnxReranker(Settings(reranker_dir=tmp_path))
    with pytest.raises(RerankError, match="make reranker-fetch"):
        await reranker.load()


def test_weights_are_laid_out_by_model_revision_and_quantization(tmp_path: Path) -> None:
    settings = Settings(reranker_dir=tmp_path, reranker_revision="abc")
    model, tokenizer = artifact_paths(settings)
    assert model == tmp_path / "bge-reranker-v2-m3/abc/onnx/model_int8.onnx"
    assert tokenizer == tmp_path / "bge-reranker-v2-m3/abc/tokenizer.json"


def test_an_unpinned_revision_is_refused_before_any_download(tmp_path: Path) -> None:
    with pytest.raises(RerankError, match="re-fit rerank_score_floor"):
        fetch_weights(Settings(reranker_dir=tmp_path, reranker_revision="unpinned"))


class _Loaded(OnnxReranker):
    """Skips the model; `_forward` blocks for as long as the test says."""

    def __init__(self, settings: Settings, delay_s: float) -> None:
        super().__init__(settings)
        self._session = object()  # type: ignore[assignment]
        self.delay_s = delay_s
        self.running = 0
        self.peak = 0

    def _forward(self, query: str, passages: object) -> list[float]:
        self.running += 1
        self.peak = max(self.peak, self.running)
        time.sleep(self.delay_s)
        self.running -= 1
        return [0.5]


async def test_timeout_raises_but_holds_the_slot_until_the_pass_finishes() -> None:
    """An abandoned forward pass still owns the CPU; a second must not overlap it."""
    reranker = _Loaded(Settings(rerank_timeout_s=0.05, rerank_concurrency=1), delay_s=0.3)

    with pytest.raises(RerankError, match="exceeded"):
        await reranker.score("q", ["p"])

    reranker._timeout_s = 5.0
    assert await reranker.score("q", ["p"]) == [0.5]
    assert reranker.peak == 1


async def test_concurrent_calls_share_the_semaphore() -> None:
    reranker = _Loaded(Settings(rerank_timeout_s=5.0, rerank_concurrency=1), delay_s=0.05)
    await asyncio.gather(*(reranker.score("q", ["p"]) for _ in range(4)))
    assert reranker.peak == 1
