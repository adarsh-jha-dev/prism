"""Gemini multimodal ingestion — the `gemini` lane."""

import base64
from typing import Any

import httpx
import structlog

from prism.config import Settings, get_settings
from prism.vision.base import (
    FIGURE_KINDS,
    PROMPT,
    ParsedFigure,
    VisionError,
    figures_from_json,
)

__all__ = ["GeminiVisionProvider"]

log = structlog.get_logger(__name__)


# Gemini's own schema dialect: uppercase type names, not JSON Schema.
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "figures": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "kind": {"type": "STRING", "enum": sorted(FIGURE_KINDS)},
                    "content": {"type": "STRING"},
                    "caption": {"type": "STRING"},
                },
                "required": ["kind", "content"],
            },
        }
    },
    "required": ["figures"],
}


class GeminiVisionProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._model = resolved.vision_model
        self._url = f"{resolved.gemini_base_url.rstrip('/')}/models/{self._model}:generateContent"
        self._timeout = resolved.vision_timeout_s
        self._api_key = resolved.gemini_api_key
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        # Header rather than ?key=, so the credential stays out of access logs
        # and recorded fixtures.
        if self._api_key:
            return {"x-goog-api-key": self._api_key}
        # An injected client is a replaying fixture and needs no credential.
        if self._client is not None:
            return {}
        raise VisionError("vision is enabled but GEMINI_API_KEY is unset")

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = self._headers()
        if self._client is not None:
            response = await self._client.post(
                self._url, json=payload, headers=headers, timeout=self._timeout
            )
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload, headers=headers)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    async def parse_page(self, png: bytes) -> list[ParsedFigure]:
        if not png:
            raise VisionError("refusing to parse an empty page render")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(png).decode("ascii"),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                # Ingestion is a build step: a re-ingest must not reword figures.
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": _RESPONSE_SCHEMA,
            },
        }

        try:
            body = await self._post(payload)
        except httpx.HTTPStatusError as exc:
            raise VisionError(
                f"gemini returned {exc.response.status_code} for {self._model}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VisionError(f"gemini call failed: {type(exc).__name__}: {exc}") from exc

        self._log_usage(body)
        return _parse_body(body)

    def _log_usage(self, body: dict[str, Any]) -> None:
        # Tokens, not dollars: there is no model_pricing table to price against.
        usage = body.get("usageMetadata")
        if isinstance(usage, dict):
            log.info(
                "vision_page_parsed",
                model=self._model,
                billing_unit="tokens",
                input_tokens=usage.get("promptTokenCount"),
                output_tokens=usage.get("candidatesTokenCount"),
                # Billed, and on a thinking model it dwarfs the visible output.
                thought_tokens=usage.get("thoughtsTokenCount"),
                total_tokens=usage.get("totalTokenCount"),
            )


def _parse_body(body: dict[str, Any]) -> list[ParsedFigure]:
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        # A safety-blocked prompt lands here, with the reason in promptFeedback.
        raise VisionError(f"gemini returned no candidates: {body.get('promptFeedback')}")

    candidate = candidates[0]
    reason = candidate.get("finishReason")
    if reason not in (None, "STOP"):
        # MAX_TOKENS yields truncated JSON — a half-read table.
        raise VisionError(f"gemini stopped early: {reason}")

    parts = candidate.get("content", {}).get("parts")
    if not isinstance(parts, list) or not parts:
        raise VisionError("gemini candidate carried no parts")

    # Thinking models interleave reasoning parts; only the answer parts are JSON.
    raw = "".join(
        part.get("text", "") for part in parts if isinstance(part, dict) and not part.get("thought")
    )
    return figures_from_json(raw)
