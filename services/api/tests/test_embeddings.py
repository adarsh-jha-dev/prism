"""Unit: the provider contract, with no network and no model."""

import json

import httpx
import pytest

from prism.config import Settings
from prism.embeddings import EmbeddingError, EmbeddingProvider, OllamaEmbeddingProvider

DIM = 768


def _provider(handler: object) -> OllamaEmbeddingProvider:
    settings = Settings(embedding_model="nomic-embed-text", embedding_dim=DIM)
    return OllamaEmbeddingProvider(
        settings=settings,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),  # type: ignore[arg-type]
    )


def _ok(vectors: list[list[float]]) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "nomic-embed-text", "embeddings": vectors})

    return handler


def test_satisfies_the_protocol() -> None:
    assert isinstance(_provider(_ok([])), EmbeddingProvider)


def test_reports_model_and_dim_from_settings() -> None:
    provider = _provider(_ok([]))
    assert (provider.model, provider.dim) == ("nomic-embed-text", DIM)


async def test_embeds_a_batch_in_order() -> None:
    vectors = [[0.1] * DIM, [0.2] * DIM]
    assert await _provider(_ok(vectors)).embed(["a", "b"]) == vectors


async def test_embed_one_unwraps_the_batch() -> None:
    assert await _provider(_ok([[0.5] * DIM])).embed_one("a") == [0.5] * DIM


async def test_empty_batch_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not have reached ollama")

    assert await _provider(handler).embed([]) == []


async def test_posts_the_configured_model_to_the_embed_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"embeddings": [[0.0] * DIM]})

    await _provider(handler).embed(["hello"])
    assert seen["model"] == "nomic-embed-text"
    assert seen["input"] == ["hello"]
    assert str(seen["url"]).endswith("/api/embed")


async def test_blank_input_is_refused() -> None:
    with pytest.raises(EmbeddingError, match="blank"):
        await _provider(_ok([])).embed(["   "])


async def test_wrong_width_is_an_error_not_a_stored_vector() -> None:
    """The failure this guards is silent: a 1024-dim model in a 768-dim column."""
    with pytest.raises(EmbeddingError, match="1024 dims, expected 768"):
        await _provider(_ok([[0.1] * 1024])).embed(["a"])


async def test_short_batch_is_an_error() -> None:
    with pytest.raises(EmbeddingError, match="asked for 2 embeddings, got 1"):
        await _provider(_ok([[0.1] * DIM])).embed(["a", "b"])


async def test_missing_embeddings_key_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "nomic-embed-text"})

    with pytest.raises(EmbeddingError):
        await _provider(handler).embed(["a"])


async def test_http_error_becomes_an_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(EmbeddingError, match="ollama embed failed"):
        await _provider(handler).embed(["a"])


async def test_transport_error_becomes_an_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(EmbeddingError, match="ollama embed failed"):
        await _provider(handler).embed(["a"])


async def test_metered_embed_reports_the_providers_own_token_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"embeddings": [[0.1] * DIM, [0.2] * DIM], "prompt_eval_count": 9}
        )

    embedded = await _provider(handler).embed_metered(["a", "b"])
    assert embedded.vectors == [[0.1] * DIM, [0.2] * DIM]
    usage = embedded.usage
    assert (usage.provider, usage.model, usage.billing_unit) == (
        "ollama",
        "nomic-embed-text",
        "tokens",
    )
    assert (usage.input_tokens, usage.output_tokens, usage.gpu_ms) == (9, 0, None)
    assert usage.cost_basis == "metered"


async def test_a_missing_count_is_left_unreported_not_zeroed() -> None:
    embedded = await _provider(_ok([[0.1] * DIM])).embed_metered(["a"])
    assert embedded.usage.input_tokens is None
