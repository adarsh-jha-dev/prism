"""Local vision parsing — the `ollama` lane.

Self-hosted and free, so it is what dev and CI ingest with. Ollama's `format`
takes a JSON Schema and constrains decoding to it, which is the same guarantee
the Gemini path gets from `responseSchema`.
"""

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

__all__ = ["OllamaVisionProvider"]

log = structlog.get_logger(__name__)

_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": sorted(FIGURE_KINDS)},
                    "content": {"type": "string"},
                    "caption": {"type": "string"},
                },
                "required": ["kind", "content"],
            },
        }
    },
    "required": ["figures"],
}


class OllamaVisionProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._model = resolved.vision_ollama_model
        self._url = f"{resolved.ollama_base_url.rstrip('/')}/api/chat"
        self._timeout = resolved.vision_timeout_s
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            response = await self._client.post(self._url, json=payload, timeout=self._timeout)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    async def parse_page(self, png: bytes) -> list[ParsedFigure]:
        if not png:
            raise VisionError("refusing to parse an empty page render")

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [base64.b64encode(png).decode("ascii")],
                }
            ],
            "format": _FORMAT,
            "stream": False,
            # Ingestion is a build step: a re-ingest must not reword figures.
            "options": {"temperature": 0.0},
        }

        try:
            body = await self._post(payload)
        except httpx.HTTPStatusError as exc:
            raise VisionError(
                f"ollama returned {exc.response.status_code} for {self._model} — "
                "is the model pulled?"
            ) from exc
        except httpx.HTTPError as exc:
            raise VisionError(f"ollama call failed: {type(exc).__name__}: {exc}") from exc

        self._log_usage(body)
        return _parse_body(body)

    def _log_usage(self, body: dict[str, Any]) -> None:
        # Recorded for parity with the paid lanes; self-hosted tokens cost nothing.
        log.info(
            "vision_page_parsed",
            model=self._model,
            billing_unit="none",
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
        )


def _parse_body(body: dict[str, Any]) -> list[ParsedFigure]:
    reason = body.get("done_reason")
    if reason not in (None, "stop"):
        # "length" yields truncated JSON — a half-read table.
        raise VisionError(f"ollama stopped early: {reason}")

    message = body.get("message")
    if not isinstance(message, dict):
        raise VisionError(f"ollama returned no message: {body.get('error')}")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise VisionError("ollama returned an empty message")

    return figures_from_json(content)
