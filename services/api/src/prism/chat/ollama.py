"""Local Ollama chat — the `ollama` lane, and the only generation path so far.

Structured output goes through Ollama's `format`, which takes a JSON Schema and
constrains decoding to it, the same mechanism the vision lane uses. Constrained
decoding is not validation, so the response is parsed and validated afterwards
anyway: `format` cannot stop a generation ending early on `num_predict`, which
yields JSON that is schema-shaped and truncated, and it does not decide whether
a verdict's fields make sense together. Both failures raise.

Token counts come from the provider's own `prompt_eval_count` / `eval_count`,
so the meter is metered rather than estimated. `duration_ms` is wall clock,
which is what a trace row's bar measures; Ollama's `total_duration` excludes
the queue wait a caller actually paid for.
"""

import time
from collections.abc import Sequence
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from prism.chat.base import ChatError, Completion, Message, Structured, Usage
from prism.config import Settings, get_settings

__all__ = ["OllamaChatProvider"]

log = structlog.get_logger(__name__)

PROVIDER = "ollama"


class OllamaChatProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._model = resolved.generator_model
        self._url = f"{resolved.ollama_base_url.rstrip('/')}/api/chat"
        self._timeout = resolved.chat_timeout_s
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    async def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        started = time.perf_counter()
        try:
            if self._client is not None:
                response = await self._client.post(self._url, json=payload, timeout=self._timeout)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(self._url, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ChatError(
                f"ollama did not answer within {self._timeout}s for {payload['model']}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ChatError(
                f"ollama returned {exc.response.status_code} for {payload['model']} — "
                "is the model pulled?"
            ) from exc
        except httpx.HTTPError as exc:
            raise ChatError(f"ollama call failed: {type(exc).__name__}: {exc}") from exc

        duration_ms = round((time.perf_counter() - started) * 1000)
        body: dict[str, Any] = response.json()
        return body, duration_ms

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not messages:
            raise ChatError("refusing to call the model with no messages")
        if any(not m.content.strip() for m in messages):
            raise ChatError("refusing to send a blank message")

        # 0.0 by default: graders and the benchmark both need a rerun to agree
        # with the run it is compared against.
        options: dict[str, Any] = {"temperature": 0.0 if temperature is None else temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        if response_format is not None:
            payload["format"] = response_format
        return payload

    def _usage(self, body: dict[str, Any], *, model: str, duration_ms: int) -> Usage:
        return Usage(
            model=model,
            provider=PROVIDER,
            # Self-hosted tokens are counted and priced at zero (ADR 0013).
            billing_unit="tokens",
            input_tokens=_count(body.get("prompt_eval_count")),
            output_tokens=_count(body.get("eval_count")),
            gpu_ms=None,
            duration_ms=duration_ms,
            cost_basis="metered",
        )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
        )
        body, duration_ms = await self._post(payload)
        text = _content(body)
        usage = self._usage(body, model=payload["model"], duration_ms=duration_ms)
        log.info("chat_completed", **_log_fields(usage))
        return Completion(text=text, usage=usage)

    async def structured[T: BaseModel](
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Structured[T]:
        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=schema.model_json_schema(),
        )
        body, duration_ms = await self._post(payload)
        raw = _content(body)

        try:
            value = schema.model_validate_json(raw)
        except ValidationError as exc:
            # Constrained decoding narrows the shape; it does not guarantee it.
            raise ChatError(
                f"{payload['model']} returned JSON that is not a {schema.__name__}: "
                f"{exc.error_count()} error(s): {exc.errors()[0].get('msg')}"
            ) from exc

        usage = self._usage(body, model=payload["model"], duration_ms=duration_ms)
        log.info("chat_completed", schema=schema.__name__, **_log_fields(usage))
        return Structured(value=value, usage=usage)


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _log_fields(usage: Usage) -> dict[str, Any]:
    return {
        "model": usage.model,
        "provider": usage.provider,
        "billing_unit": usage.billing_unit,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "duration_ms": usage.duration_ms,
    }


def _content(body: dict[str, Any]) -> str:
    reason = body.get("done_reason")
    if reason not in (None, "stop"):
        # "length" is a truncated answer, and a truncated verdict is not a verdict.
        raise ChatError(f"ollama stopped early: {reason}")

    message = body.get("message")
    if not isinstance(message, dict):
        raise ChatError(f"ollama returned no message: {body.get('error')}")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ChatError("ollama returned an empty message")
    return content
