"""The chat provider interface, and the usage record every call returns.

Two call shapes, because the graph needs both. `complete` produces prose —
`generate` is its only caller. `structured` returns a parsed, validated model:
`grade_docs`, `rewrite_query` and `verify_grounding` return verdicts, and a
grader that returns prose has failed rather than degraded, so a response that
does not parse or does not validate raises here instead of reaching a caller
that would have to re-check it.

There is deliberately no streaming variant. `generate`'s output is not an answer
until `verify_grounding` passes it, and a failed verification regenerates or
refuses — so streaming provider tokens onward would put text on screen that the
graph has not yet judged, which is the "answer anyway" path `CLAUDE.md` forbids.
The SSE endpoint streams node events and delivers the answer once, after
verification.

`Usage` carries what a trace row needs (ADR 0012) and what pricing needs later
(ADR 0013), including on the local lane where every figure is zero. Traces
accumulate; a column that was never written cannot be backfilled.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

__all__ = [
    "ChatError",
    "ChatProvider",
    "Completion",
    "Message",
    "Role",
    "Structured",
    "Usage",
]

Role = Literal["system", "user", "assistant"]


class ChatError(RuntimeError):
    """A completion could not be produced, or could not be trusted.

    Never softened into an empty answer or a default verdict: a fabricated
    verdict decides a correction loop, and a wrong loop decision is invisible.
    """


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Usage:
    """One call's meter reading, in the units a trace row records.

    `billing_unit` is `tokens` on every lane that counts them, including local
    Ollama, whose price rows exist at zero (ADR 0013): free is priced, and
    unpriced is not free. `cost_basis` is `metered` when the provider reported
    the meter and `estimated` when we timed it ourselves — estimated rows stay
    out of the headline cost figure.
    """

    model: str
    provider: str
    billing_unit: Literal["tokens", "gpu_ms", "none"]
    input_tokens: int | None
    output_tokens: int | None
    gpu_ms: int | None
    duration_ms: int
    cost_basis: Literal["metered", "estimated"]


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage


@dataclass(frozen=True)
class Structured[T: BaseModel]:
    value: T
    usage: Usage


@runtime_checkable
class ChatProvider(Protocol):
    @property
    def model(self) -> str:
        """The model a call uses when it names none."""
        ...

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Free-form prose. Raises ChatError rather than return a partial answer."""
        ...

    async def structured[T: BaseModel](
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Structured[T]:
        """A validated `schema` instance. Raises ChatError if it cannot be one."""
        ...
