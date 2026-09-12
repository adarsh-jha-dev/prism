"""Unit: the Gemini vision provider. Replayed, never called.

Happy paths replay a cassette; failure paths drive `_parse_body` directly.
"""

import base64
import json
from typing import Any

import httpx
import pytest

from cassette import replay
from prism.config import Settings
from prism.vision import ParsedFigure, VisionError
from prism.vision.gemini import GeminiVisionProvider, _parse_body

PNG = b"\x89PNG\r\n\x1a\npretend this is a page"


def _provider(name: str) -> tuple[GeminiVisionProvider, httpx.AsyncClient]:
    client = replay(name)
    return GeminiVisionProvider(Settings(), client=client), client


def _body(figures: list[dict[str, Any]], **candidate: Any) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": json.dumps({"figures": figures})}]},
                "finishReason": "STOP",
                **candidate,
            }
        ]
    }


async def test_a_table_page_becomes_a_table_figure() -> None:
    provider, _ = _provider("gemini/table_page")
    (figure,) = await provider.parse_page(PNG)

    assert figure.kind == "table"
    assert figure.caption == "Table 1. Latency by region."
    assert figure.content.startswith("| Region |")
    assert "ap-south-1" in figure.content


async def test_a_page_with_nothing_on_it_returns_no_figures() -> None:
    # Not an error: the detector is deliberately looser than the model.
    provider, _ = _provider("gemini/prose_page")
    assert await provider.parse_page(PNG) == []


async def test_the_request_carries_the_page_as_an_inline_png() -> None:
    provider, client = _provider("gemini/prose_page")
    await provider.parse_page(PNG)

    (request,) = client.recorded_requests  # type: ignore[attr-defined]
    payload = json.loads(request.content)
    (text_part, image_part) = payload["contents"][0]["parts"]

    assert "figures, charts, diagrams, tables" in text_part["text"]
    assert image_part["inline_data"]["mime_type"] == "image/png"
    assert base64.b64decode(image_part["inline_data"]["data"]) == PNG


async def test_the_request_pins_temperature_and_a_response_schema() -> None:
    # A re-ingest must not reword figures.
    provider, client = _provider("gemini/prose_page")
    await provider.parse_page(PNG)

    config = json.loads(client.recorded_requests[0].content)["generationConfig"]  # type: ignore[attr-defined]
    assert config["temperature"] == 0.0
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["properties"]["figures"]["items"]["required"] == [
        "kind",
        "content",
    ]


async def test_the_api_key_travels_in_a_header_not_the_url() -> None:
    settings = Settings(gemini_api_key="secret-key-value")
    provider = GeminiVisionProvider(settings, client=replay("gemini/prose_page"))
    # The cassette's URL match asserts no ?key= was appended.
    await provider.parse_page(PNG)

    live = GeminiVisionProvider(settings)
    assert live._headers() == {"x-goog-api-key": "secret-key-value"}
    assert "secret-key-value" not in live._url


async def test_a_missing_key_refuses_before_it_reaches_the_network() -> None:
    provider = GeminiVisionProvider(Settings(gemini_api_key=None))
    with pytest.raises(VisionError, match="GEMINI_API_KEY is unset"):
        await provider.parse_page(PNG)


async def test_an_empty_render_is_refused() -> None:
    provider, _ = _provider("gemini/prose_page")
    with pytest.raises(VisionError, match="empty page render"):
        await provider.parse_page(b"")


async def test_an_http_error_is_a_vision_error_carrying_the_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "quota"}})

    provider = GeminiVisionProvider(
        Settings(), client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(VisionError, match="gemini returned 429"):
        await provider.parse_page(PNG)


async def test_a_transport_failure_is_a_vision_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    provider = GeminiVisionProvider(
        Settings(), client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(VisionError, match="gemini call failed: ConnectError"):
        await provider.parse_page(PNG)


def test_a_truncated_response_is_an_error_not_a_half_table() -> None:
    body = _body([{"kind": "table", "content": "| a | b |"}], finishReason="MAX_TOKENS")
    with pytest.raises(VisionError, match="gemini stopped early: MAX_TOKENS"):
        _parse_body(body)


def test_a_blocked_prompt_is_an_error_not_an_empty_page() -> None:
    with pytest.raises(VisionError, match="no candidates"):
        _parse_body({"promptFeedback": {"blockReason": "SAFETY"}})


def test_non_json_output_is_an_error() -> None:
    body = {"candidates": [{"content": {"parts": [{"text": "Here is the table:"}]}}]}
    with pytest.raises(VisionError, match="did not return JSON"):
        _parse_body(body)


def test_a_missing_figures_list_is_an_error() -> None:
    body = {"candidates": [{"content": {"parts": [{"text": '{"tables": []}'}]}}]}
    with pytest.raises(VisionError, match="expected a 'figures' list"):
        _parse_body(body)


def test_an_unknown_kind_is_rejected_here_not_by_the_check_constraint() -> None:
    with pytest.raises(VisionError, match="has kind 'photo'"):
        _parse_body(_body([{"kind": "photo", "content": "a cat"}]))


def test_text_is_not_a_kind_vision_may_produce() -> None:
    with pytest.raises(VisionError, match="has kind 'text'"):
        _parse_body(_body([{"kind": "text", "content": "body prose"}]))


def test_an_empty_content_figure_is_rejected() -> None:
    with pytest.raises(VisionError, match=r"figure 0 \(figure\) has no content"):
        _parse_body(_body([{"kind": "figure", "content": "   "}]))


def test_a_missing_caption_is_allowed_and_stays_none() -> None:
    (figure,) = _parse_body(_body([{"kind": "figure", "content": "a scatter plot"}]))
    assert figure == ParsedFigure(kind="figure", content="a scatter plot", caption=None)


def test_a_blank_caption_is_normalised_to_none() -> None:
    (figure,) = _parse_body(_body([{"kind": "table", "content": "| a |", "caption": "  "}]))
    assert figure.caption is None


def test_every_figure_on_a_page_survives_in_order() -> None:
    figures = _parse_body(
        _body(
            [
                {"kind": "table", "content": "| a |"},
                {"kind": "figure", "content": "a bar chart"},
                {"kind": "equation", "content": r"E = mc^2"},
            ]
        )
    )
    assert [figure.kind for figure in figures] == ["table", "figure", "equation"]


def test_reasoning_parts_are_not_treated_as_answer_json() -> None:
    # gemini-3.x is a thinking model; a thought part concatenated into the
    # answer would make the JSON unparseable.
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Let me look at the table...", "thought": True},
                        {"text": json.dumps({"figures": [{"kind": "table", "content": "| a |"}]})},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }
    (figure,) = _parse_body(body)
    assert figure.kind == "table"


def test_a_thought_signature_beside_the_answer_is_harmless() -> None:
    # The shape the recorded cassette actually has: one part, text plus signature.
    body = _body([{"kind": "table", "content": "| a |"}])
    body["candidates"][0]["content"]["parts"][0]["thoughtSignature"] = "EqwDCqkD"
    assert len(_parse_body(body)) == 1
