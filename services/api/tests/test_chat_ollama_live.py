"""The local chat model, for real. Needs Ollama with the model pulled.

Marked `ollama`, so CI deselects it. What a stubbed transport cannot tell us is
whether a pulled model actually honours `format` — the graders depend on it, and
a model that answers a verdict in prose fails the loop rather than degrading it.

Runs against the grader model: it is the smallest of the three and the one whose
structured replies the correction loops actually read.
"""

import pytest
from pydantic import BaseModel, Field

from prism.chat import Message
from prism.chat.ollama import OllamaChatProvider
from prism.config import Settings, get_settings

pytestmark = [pytest.mark.integration, pytest.mark.ollama]

PASSAGE = (
    "Reciprocal rank fusion scores each document as the sum over retrievers of "
    "1/(k + rank), where k is a constant, and ranks the documents by that sum."
)


class Verdict(BaseModel):
    grounded: bool
    score: float = Field(ge=0.0, le=1.0)


@pytest.fixture
def provider() -> OllamaChatProvider:
    settings = get_settings()
    return OllamaChatProvider(Settings(generator_model=settings.grader_model))


async def test_the_pulled_model_answers_in_the_pinned_schema(
    provider: OllamaChatProvider,
) -> None:
    result = await provider.structured(
        [
            Message(
                role="system",
                content="Judge whether the passage supports the claim. Answer as JSON.",
            ),
            Message(
                role="user",
                content=f"Passage: {PASSAGE}\n\nClaim: RRF sums 1/(k + rank) over retrievers.",
            ),
        ],
        Verdict,
    )

    assert result.value.grounded is True
    assert 0.0 <= result.value.score <= 1.0
    assert result.usage.input_tokens and result.usage.output_tokens
    assert result.usage.duration_ms > 0


async def test_a_completion_comes_back_as_prose(provider: OllamaChatProvider) -> None:
    completion = await provider.complete(
        [Message(role="user", content="In one sentence, what does reciprocal rank fusion do?")],
        max_tokens=60,
    )

    assert completion.text.strip()
    assert completion.usage.billing_unit == "tokens"
