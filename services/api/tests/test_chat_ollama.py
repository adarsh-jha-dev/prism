"""Unit: the local chat provider. Stubbed transport, no live model."""

import json
from typing import Any, Literal

import httpx
import pytest
from pydantic import BaseModel, Field

from prism.chat import ChatError, ChatProvider, Message
from prism.chat.ollama import OllamaChatProvider
from prism.config import Settings

ASK = [Message(role="user", content="Does the corpus say how RRF fuses ranks?")]


class Verdict(BaseModel):
    """A grader's reply — the shape grade_docs and verify_grounding return."""

    decision: Literal["pass", "fail"]
    score: float = Field(ge=0.0, le=1.0)


def _body(content: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": "qwen2.5:32b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 413,
        "eval_count": 57,
        **extra,
    }


def _provider(
    response: dict[str, Any] | Exception, status: int = 200
) -> tuple[OllamaChatProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if isinstance(response, Exception):
            raise response
        return httpx.Response(status, json=response)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaChatProvider(Settings(), client=client), seen


def test_satisfies_the_protocol() -> None:
    provider, _ = _provider(_body("hi"))
    assert isinstance(provider, ChatProvider)


def test_defaults_to_the_generator_model() -> None:
    provider, _ = _provider(_body("hi"))
    assert provider.model == Settings().generator_model


async def test_completion_returns_the_models_prose() -> None:
    provider, _ = _provider(_body("RRF sums reciprocal ranks."))
    completion = await provider.complete(ASK)
    assert completion.text == "RRF sums reciprocal ranks."


async def test_a_structured_call_returns_a_validated_model() -> None:
    provider, _ = _provider(_body(json.dumps({"decision": "pass", "score": 0.81})))
    result = await provider.structured(ASK, Verdict)

    assert result.value == Verdict(decision="pass", score=0.81)


async def test_malformed_json_raises_rather_than_reaching_the_caller() -> None:
    provider, _ = _provider(_body('{"decision": "pass", '))
    with pytest.raises(ChatError, match="not a Verdict"):
        await provider.structured(ASK, Verdict)


async def test_json_that_parses_but_fails_the_schema_raises() -> None:
    """Constrained decoding narrows the shape; it does not decide the values."""
    provider, _ = _provider(_body(json.dumps({"decision": "maybe", "score": 4.2})))
    with pytest.raises(ChatError, match="not a Verdict"):
        await provider.structured(ASK, Verdict)


async def test_a_truncated_answer_is_an_error_not_a_short_one() -> None:
    provider, _ = _provider(_body('{"decision": "pa', done_reason="length"))
    with pytest.raises(ChatError, match="stopped early"):
        await provider.structured(ASK, Verdict)


async def test_usage_carries_what_a_trace_row_records() -> None:
    provider, _ = _provider(_body("an answer"))
    usage = (await provider.complete(ASK)).usage

    assert (usage.input_tokens, usage.output_tokens) == (413, 57)
    assert (usage.provider, usage.model) == ("ollama", Settings().generator_model)
    # Local tokens are counted and priced at zero, never left unpriced (ADR 0013).
    assert usage.billing_unit == "tokens"
    assert usage.cost_basis == "metered"
    assert usage.gpu_ms is None
    assert usage.duration_ms >= 0


async def test_missing_token_counts_stay_none_rather_than_zero() -> None:
    """Zero would price as free; None is a row that could not be priced."""
    body = _body("an answer")
    del body["prompt_eval_count"]
    provider, _ = _provider(body)

    usage = (await provider.complete(ASK)).usage
    assert usage.input_tokens is None
    assert usage.output_tokens == 57


async def test_a_timeout_surfaces_as_the_providers_error() -> None:
    provider, _ = _provider(httpx.ReadTimeout("too slow"))
    with pytest.raises(ChatError, match="did not answer within"):
        await provider.complete(ASK)


async def test_a_missing_model_names_itself() -> None:
    provider, _ = _provider({"error": "model not found"}, status=404)
    with pytest.raises(ChatError, match="is the model pulled"):
        await provider.complete(ASK)


async def test_an_empty_message_is_refused_before_the_call() -> None:
    provider, seen = _provider(_body("hi"))
    with pytest.raises(ChatError, match="blank message"):
        await provider.complete([Message(role="user", content="   ")])
    assert seen == []


async def test_no_messages_is_refused_before_the_call() -> None:
    provider, seen = _provider(_body("hi"))
    with pytest.raises(ChatError, match="no messages"):
        await provider.complete([])
    assert seen == []


async def test_a_structured_call_sends_the_schema_as_the_format() -> None:
    provider, seen = _provider(_body(json.dumps({"decision": "fail", "score": 0.1})))
    await provider.structured(ASK, Verdict)

    payload = json.loads(seen[0].content)
    assert payload["format"] == Verdict.model_json_schema()
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0


async def test_a_completion_sends_no_format() -> None:
    provider, seen = _provider(_body("prose"))
    await provider.complete(ASK, temperature=0.4, max_tokens=256)

    payload = json.loads(seen[0].content)
    assert "format" not in payload
    assert payload["options"] == {"temperature": 0.4, "num_predict": 256}


async def test_a_call_can_name_a_different_model() -> None:
    provider, seen = _provider(_body("prose"))
    completion = await provider.complete(ASK, model="llama3.1:8b")

    assert json.loads(seen[0].content)["model"] == "llama3.1:8b"
    assert completion.usage.model == "llama3.1:8b"
