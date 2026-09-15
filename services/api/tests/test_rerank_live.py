"""The real int8 cross-encoder. Needs `make reranker-fetch`; no database."""

import asyncio

import pytest

from prism.config import Settings
from prism.rerank import OnnxReranker

pytestmark = [pytest.mark.integration, pytest.mark.reranker]

QUERY = "what is panda?"
RELEVANT = (
    "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply "
    "panda, is a bear species endemic to China."
)


@pytest.fixture(scope="module")
def reranker() -> OnnxReranker:
    """Loaded once for the module: the load alone is seconds."""
    model = OnnxReranker(Settings(rerank_timeout_s=60.0))
    asyncio.run(model.load())
    return model


async def test_separates_relevant_from_irrelevant_across_the_floor(reranker: OnnxReranker) -> None:
    """The model card's own example pair, which should land either side of 0.44."""
    relevant, irrelevant = await reranker.score(QUERY, [RELEVANT, "hi"])
    assert relevant > 0.44 > irrelevant


async def test_scores_are_independent_of_batch_composition(reranker: OnnxReranker) -> None:
    alone = await reranker.score(QUERY, [RELEVANT])
    batched = await reranker.score(QUERY, [RELEVANT, "hi", "bamboo forests of Sichuan"])
    assert batched[0] == pytest.approx(alone[0], abs=1e-3)


async def test_is_deterministic(reranker: OnnxReranker) -> None:
    passages = [RELEVANT, "hi"]
    assert await reranker.score(QUERY, passages) == await reranker.score(QUERY, passages)
