"""Unit: the local vision provider. Stubbed transport, no live model."""

import base64
import json
from typing import Any

import httpx
import pytest

from prism.config import Settings
from prism.vision import VisionError
from prism.vision.ollama import OllamaVisionProvider, _parse_body

PNG = b"\x89PNG\r\n\x1a\npretend this is a page"


def _body(figures: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "model": "qwen2.5vl:7b",
        "message": {"role": "assistant", "content": json.dumps({"figures": figures})},
        "done": True,
        "done_reason": "stop",
        **extra,
    }


def _provider(
    response: dict[str, Any], status: int = 200
) -> tuple[OllamaVisionProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=response)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaVisionProvider(Settings(), client=client), seen


async def test_a_table_page_becomes_a_table_figure() -> None:
    provider, _ = _provider(
        _body([{"kind": "table", "content": "| a | b |", "caption": "Table 1."}])
    )
    (figure,) = await provider.parse_page(PNG)

    assert figure.kind == "table"
    assert figure.caption == "Table 1."


async def test_a_page_with_nothing_on_it_returns_no_figures() -> None:
    provider, _ = _provider(_body([]))
    assert await provider.parse_page(PNG) == []


async def test_the_request_carries_the_page_as_a_base64_image() -> None:
    provider, seen = _provider(_body([]))
    await provider.parse_page(PNG)

    payload = json.loads(seen[0].content)
    (message,) = payload["messages"]
    assert base64.b64decode(message["images"][0]) == PNG
    assert "figures, charts, diagrams, tables" in message["content"]


async def test_the_request_pins_the_schema_temperature_and_no_streaming() -> None:
    provider, seen = _provider(_body([]))
    await provider.parse_page(PNG)

    payload = json.loads(seen[0].content)
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["format"]["properties"]["figures"]["items"]["required"] == [
        "kind",
        "content",
    ]


async def test_the_schema_is_json_schema_not_geminis_dialect() -> None:
    provider, seen = _provider(_body([]))
    await provider.parse_page(PNG)
    assert json.loads(seen[0].content)["format"]["type"] == "object"


async def test_the_configured_model_is_the_one_asked_for() -> None:
    settings = Settings(vision_ollama_model="granite3.2-vision:2b")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_body([]))

    provider = OllamaVisionProvider(
        settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await provider.parse_page(PNG)

    assert provider.model == "granite3.2-vision:2b"
    assert json.loads(seen[0].content)["model"] == "granite3.2-vision:2b"


async def test_a_missing_model_is_a_vision_error_that_says_so() -> None:
    provider, _ = _provider({"error": "model not found"}, status=404)
    with pytest.raises(VisionError, match="is the model pulled"):
        await provider.parse_page(PNG)


async def test_ollama_being_down_is_a_vision_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = OllamaVisionProvider(
        Settings(), client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(VisionError, match="ollama call failed: ConnectError"):
        await provider.parse_page(PNG)


async def test_an_empty_render_is_refused() -> None:
    provider, _ = _provider(_body([]))
    with pytest.raises(VisionError, match="empty page render"):
        await provider.parse_page(b"")


def test_a_truncated_response_is_an_error_not_a_half_table() -> None:
    body = _body([{"kind": "table", "content": "| a |"}], done_reason="length")
    with pytest.raises(VisionError, match="ollama stopped early: length"):
        _parse_body(body)


def test_an_empty_message_is_an_error() -> None:
    with pytest.raises(VisionError, match="empty message"):
        _parse_body({"done_reason": "stop", "message": {"content": "  "}})


def test_an_unknown_kind_is_rejected() -> None:
    with pytest.raises(VisionError, match="has kind 'photo'"):
        _parse_body(_body([{"kind": "photo", "content": "a cat"}]))


def test_non_json_content_is_an_error() -> None:
    body = {"done_reason": "stop", "message": {"content": "Here is the table:"}}
    with pytest.raises(VisionError, match="did not return JSON"):
        _parse_body(body)
