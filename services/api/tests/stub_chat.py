"""A chat provider with scripted replies, so tests need no model.

Mirrors tests/stub_provider.py: what these tests check is the wiring — what the
caller does with a verdict, a refusal or an error — not what a model writes.
"""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ValidationError

from prism.chat import ChatError, Completion, Message, Structured, Usage

MODEL = "qwen2.5:32b"


def usage(model: str = MODEL, *, input_tokens: int = 11, output_tokens: int = 7) -> Usage:
    return Usage(
        model=model,
        provider="ollama",
        billing_unit="tokens",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        gpu_ms=None,
        duration_ms=5,
        cost_basis="metered",
    )


class StubChat:
    """Replies with `text` verbatim; `structured` validates it like the real one."""

    model = MODEL

    def __init__(self, text: str = "an answer", *, fail: bool = False) -> None:
        self.text = text
        self._fail = fail
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        if self._fail:
            raise ChatError("stub refusing to answer")
        return Completion(text=self.text, usage=usage(model or self.model))

    async def structured[T: BaseModel](
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Structured[T]:
        self.calls.append(list(messages))
        if self._fail:
            raise ChatError("stub refusing to answer")
        try:
            value = schema.model_validate_json(self.text)
        except ValidationError as exc:
            raise ChatError(f"stub reply is not a {schema.__name__}") from exc
        return Structured(value=value, usage=usage(model or self.model))


class ScriptedChat:
    """One stub for a whole graph run: a reply per schema, in order.

    The loop calls three different structured schemas, so a single scripted
    reply cannot serve them. A schema mapped to a list is consumed one reply per
    call and the last one repeats, which is what lets a test make attempt 2
    search differently from attempt 1.
    """

    model = MODEL

    def __init__(
        self,
        replies: Mapping[str, str | Sequence[str]],
        *,
        text: str = "an answer",
    ) -> None:
        self.replies = {
            name: [reply] if isinstance(reply, str) else list(reply)
            for name, reply in replies.items()
        }
        self.text = text
        self.calls: list[tuple[str, list[Message]]] = []

    def calls_for(self, schema_name: str) -> list[list[Message]]:
        return [messages for name, messages in self.calls if name == schema_name]

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        self.calls.append(("complete", list(messages)))
        return Completion(text=self.text, usage=usage(model or self.model))

    async def structured[T: BaseModel](
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Structured[T]:
        self.calls.append((schema.__name__, list(messages)))
        queue = self.replies.get(schema.__name__)
        if not queue:
            raise ChatError(f"nothing scripted for {schema.__name__}")
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        try:
            value = schema.model_validate_json(reply)
        except ValidationError as exc:
            raise ChatError(f"scripted reply is not a {schema.__name__}") from exc
        return Structured(value=value, usage=usage(model or self.model))
