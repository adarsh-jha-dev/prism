"""Unit: lane resolution and the guards around a call. No model, no network.

The concurrency tests use a stub lane, so a cap of 1 is asserted without a
provider behind it.
"""

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import BaseModel
from structlog.testing import capture_logs

from prism.chat import ChatError, ChatProvider, Completion, Message, Structured, Usage
from prism.chat.ollama import OllamaChatProvider
from prism.config import Settings
from prism.embeddings import Embedded, EmbeddingError
from prism.providers import (
    LANE_NAMES,
    Lane,
    LaneBusy,
    LaneNotImplemented,
    LaneRejected,
    LaneUnavailable,
    ProviderRegistry,
    UnknownLane,
    build_lanes,
)

ASK = [Message(role="user", content="Does the corpus say how RRF fuses ranks?")]
UNWIRED = ("ollama_cloud", "gemini", "openai")


class Verdict(BaseModel):
    decision: str


def _usage(model: str = "stub-model") -> Usage:
    return Usage(
        model=model,
        provider="stub",
        billing_unit="gpu_ms",
        input_tokens=None,
        output_tokens=None,
        gpu_ms=17,
        duration_ms=23,
        cost_basis="metered",
    )


class StubLaneProvider:
    """Counts overlapping calls, and fails when told to."""

    model = "stub-model"

    def __init__(self, *, delay: float = 0.0, fail: bool = False, text: str = "ok") -> None:
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.models: list[str | None] = []
        self.delay = delay
        self.fail = fail
        self.text = text

    async def _run(self, model: str | None) -> Usage:
        self.calls += 1
        self.models.append(model)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                raise ChatError("stub lane is unwell")
            return _usage(model or self.model)
        finally:
            self.in_flight -= 1

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        return Completion(text=self.text, usage=await self._run(model))

    async def structured[T: BaseModel](
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Structured[T]:
        usage = await self._run(model)
        return Structured(value=schema.model_validate_json(self.text), usage=usage)


def _stub_registry(
    provider: StubLaneProvider | None = None,
    *,
    concurrency: int = 1,
    queue_timeout_s: float = 5.0,
    threshold: int = 2,
    cooldown_s: float = 30.0,
) -> tuple[ProviderRegistry, StubLaneProvider]:
    behind = provider or StubLaneProvider()
    lane = Lane(
        name="stub",
        billing_unit="gpu_ms",
        concurrency=concurrency,
        queue_timeout_s=queue_timeout_s,
        timeout_s=1.0,
        model="stub-model",
        factory=lambda: behind,
    )
    settings = Settings(breaker_failure_threshold=threshold, breaker_cooldown_s=cooldown_s)
    return ProviderRegistry({"stub": lane}, settings), behind


# --- lanes ------------------------------------------------------------------


def test_the_four_lanes_are_defined() -> None:
    assert ProviderRegistry(settings=Settings()).names == LANE_NAMES


def test_resolves_the_local_lane_to_its_provider() -> None:
    registry = ProviderRegistry(settings=Settings())

    provider = registry.provider("ollama")

    assert isinstance(provider, OllamaChatProvider)
    assert isinstance(provider, ChatProvider)
    assert registry.lane("ollama").model == Settings().generator_model


def test_the_provider_is_built_once() -> None:
    registry = ProviderRegistry(settings=Settings())

    assert registry.provider("ollama") is registry.provider("ollama")


def test_an_unknown_lane_raises_and_names_the_ones_that_exist() -> None:
    registry = ProviderRegistry(settings=Settings())

    with pytest.raises(UnknownLane, match="ollama_cloud"):
        registry.lane("olama")


@pytest.mark.parametrize("name", UNWIRED)
def test_an_unwired_lane_raises_rather_than_falling_back(name: str) -> None:
    """Silent fallback to the free lane would hide a Phase 3 routing bug."""
    registry = ProviderRegistry(settings=Settings())

    with pytest.raises(LaneNotImplemented) as caught:
        registry.provider(name)

    assert isinstance(caught.value, NotImplementedError)
    assert caught.value.lane == name


@pytest.mark.parametrize("name", UNWIRED)
async def test_calling_an_unwired_lane_raises_before_any_guard(name: str) -> None:
    registry = ProviderRegistry(settings=Settings())

    with pytest.raises(NotImplementedError):
        await registry.complete(name, ASK)

    assert registry.breaker(name).state == "closed"


def test_unwired_lanes_still_carry_their_caps_and_units() -> None:
    lanes = build_lanes(Settings())

    assert lanes["ollama_cloud"].concurrency == 1
    assert lanes["ollama_cloud"].billing_unit == "gpu_ms"
    assert lanes["gemini"].concurrency == 4
    assert lanes["openai"].concurrency == 2
    assert lanes["ollama"].billing_unit == "tokens"
    assert [name for name, lane in lanes.items() if lane.implemented] == ["ollama"]


# --- usage ------------------------------------------------------------------


async def test_usage_comes_back_unchanged_with_the_lane_name() -> None:
    registry, provider = _stub_registry()

    result = await registry.complete("stub", ASK)

    assert result.lane == "stub"
    assert result.value == "ok"
    assert result.usage == _usage()
    assert provider.models == ["stub-model"]  # the lane's model, when none is named


async def test_a_structured_call_is_guarded_the_same_way() -> None:
    registry, _ = _stub_registry(StubLaneProvider(text='{"decision": "pass"}'))

    result = await registry.structured("stub", ASK, Verdict)

    assert result.value.decision == "pass"
    assert result.usage.billing_unit == "gpu_ms"


# --- concurrency ------------------------------------------------------------


async def test_a_cap_of_one_serializes_concurrent_callers() -> None:
    registry, provider = _stub_registry(StubLaneProvider(delay=0.02), concurrency=1)

    results = await asyncio.gather(*(registry.complete("stub", ASK) for _ in range(2)))

    # Both served, never at the same time: Ollama Cloud's cap of 1, structurally.
    assert provider.max_in_flight == 1
    assert provider.calls == 2
    assert [r.lane for r in results] == ["stub", "stub"]


async def test_a_wider_cap_lets_callers_overlap() -> None:
    """Without this, a semaphore that never blocks would pass the test above."""
    registry, provider = _stub_registry(StubLaneProvider(delay=0.02), concurrency=4)

    await asyncio.gather(*(registry.complete("stub", ASK) for _ in range(4)))

    assert provider.max_in_flight == 4


async def test_each_lane_has_its_own_semaphore() -> None:
    settings = Settings()
    registry = ProviderRegistry(settings=settings)

    caps = {name: registry.lane(name).concurrency for name in registry.names}

    assert caps["ollama_cloud"] == 1
    assert caps["ollama"] == settings.concurrency_ollama_local


async def test_a_caller_that_cannot_get_a_slot_is_rejected_not_failed() -> None:
    registry, provider = _stub_registry(
        StubLaneProvider(delay=0.05), concurrency=1, queue_timeout_s=0.01
    )

    first = asyncio.create_task(registry.complete("stub", ASK))
    await asyncio.sleep(0)  # let it take the only slot

    with pytest.raises(LaneBusy) as caught:
        await registry.complete("stub", ASK)

    assert caught.value.lane == "stub"
    assert provider.calls == 1  # the rejected caller never reached the provider
    assert registry.breaker("stub").state == "closed"  # a full queue is not a failure
    assert registry.breaker("stub").consecutive_failures == 0
    await first


# --- the breaker ------------------------------------------------------------


async def test_the_breaker_opens_after_n_consecutive_failures() -> None:
    registry, provider = _stub_registry(StubLaneProvider(fail=True), threshold=2)

    for _ in range(2):
        with pytest.raises(ChatError):
            await registry.complete("stub", ASK)

    assert registry.breaker("stub").state == "open"
    assert provider.calls == 2


async def test_an_open_lane_rejects_without_calling_the_provider() -> None:
    registry, provider = _stub_registry(StubLaneProvider(fail=True), threshold=1)
    with pytest.raises(ChatError):
        await registry.complete("stub", ASK)

    with pytest.raises(LaneUnavailable) as caught:
        await registry.complete("stub", ASK)

    assert provider.calls == 1
    assert caught.value.retry_after_s > 0


async def test_a_rejection_is_not_a_failure() -> None:
    """A different type, and never counted as evidence about the provider."""
    registry, _ = _stub_registry(StubLaneProvider(fail=True), threshold=1)
    with pytest.raises(ChatError):
        await registry.complete("stub", ASK)
    before = registry.breaker("stub").consecutive_failures

    for _ in range(3):
        with pytest.raises(LaneRejected) as caught:
            await registry.complete("stub", ASK)

    assert not isinstance(caught.value, ChatError)
    assert registry.breaker("stub").consecutive_failures == before


async def test_the_lane_recovers_after_the_cooldown() -> None:
    provider = StubLaneProvider(fail=True)
    registry, _ = _stub_registry(provider, threshold=1, cooldown_s=0.02)
    with pytest.raises(ChatError):
        await registry.complete("stub", ASK)

    provider.fail = False
    await asyncio.sleep(0.03)
    result = await registry.complete("stub", ASK)

    assert result.value == "ok"
    assert registry.breaker("stub").state == "closed"


async def test_a_probe_still_failing_reopens_the_lane() -> None:
    registry, provider = _stub_registry(StubLaneProvider(fail=True), threshold=1, cooldown_s=0.02)
    with pytest.raises(ChatError):
        await registry.complete("stub", ASK)
    await asyncio.sleep(0.03)

    with pytest.raises(ChatError):
        await registry.complete("stub", ASK)

    assert provider.calls == 2
    assert registry.breaker("stub").state == "open"
    with pytest.raises(LaneUnavailable):
        await registry.complete("stub", ASK)


async def test_logs_tell_a_rejected_call_from_a_failed_one() -> None:
    registry, _ = _stub_registry(StubLaneProvider(fail=True), threshold=1)

    with capture_logs() as logs:
        with pytest.raises(ChatError):
            await registry.complete("stub", ASK)
        with pytest.raises(LaneUnavailable):
            await registry.complete("stub", ASK)

    events = [entry["event"] for entry in logs]
    assert "lane_call_failed" in events
    rejected = next(entry for entry in logs if entry["event"] == "lane_rejected")
    assert rejected["reason"] == "breaker_open"
    assert rejected["lane"] == "stub"


async def test_a_successful_call_logs_the_lane_and_the_meter() -> None:
    registry, _ = _stub_registry()

    with capture_logs() as logs:
        await registry.complete("stub", ASK)

    call = next(entry for entry in logs if entry["event"] == "lane_call")
    assert call["lane"] == "stub"
    assert call["billing_unit"] == "gpu_ms"
    assert call["duration_ms"] == 23


# ------------------------------------------------- embeddings (ADR 0018)


class StubEmbedder:
    """Counts overlapping calls, the way StubLaneProvider does."""

    model = "nomic-embed-text"
    dim = 4

    def __init__(self, *, delay: float = 0.0, fail: bool = False) -> None:
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.delay = delay
        self._fail = fail

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return (await self.embed_metered(texts)).vectors

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def embed_metered(self, texts: Sequence[str]) -> Embedded:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self._fail:
                raise EmbeddingError("stub refusing to embed")
            return Embedded(
                vectors=[[0.0, 0.0, 0.0, 1.0] for _ in texts],
                usage=Usage(
                    model=self.model,
                    provider="ollama",
                    billing_unit="tokens",
                    input_tokens=7,
                    output_tokens=0,
                    gpu_ms=None,
                    duration_ms=3,
                    cost_basis="metered",
                ),
            )
        finally:
            self.in_flight -= 1


async def test_an_embedding_goes_through_the_lane_and_keeps_its_meter() -> None:
    registry, _ = _stub_registry()
    embedder = StubEmbedder()

    result = await registry.embed("stub", ["a question"], provider=embedder)

    assert result.lane == "stub"
    assert result.value == [[0.0, 0.0, 0.0, 1.0]]
    # The provider's own reading, unpriced.
    assert result.usage.model == "nomic-embed-text"
    assert result.usage.input_tokens == 7


async def test_embedding_takes_a_slot_on_the_lane_it_shares_with_generation() -> None:
    """One Ollama process, one cap."""
    registry, _ = _stub_registry(concurrency=1)
    embedder = StubEmbedder(delay=0.05)

    await asyncio.gather(*(registry.embed("stub", ["q"], provider=embedder) for _ in range(3)))

    assert embedder.calls == 3
    assert embedder.max_in_flight == 1


async def test_a_failed_embedding_is_evidence_about_the_provider() -> None:
    """ADR 0014's failure set, widened by ADR 0018."""
    registry, _ = _stub_registry(threshold=1)
    embedder = StubEmbedder(fail=True)

    with pytest.raises(EmbeddingError):
        await registry.embed("stub", ["q"], provider=embedder)

    assert registry.breaker("stub").state == "open"
    # The open lane rejects the next caller before it reaches the provider.
    with pytest.raises(LaneUnavailable):
        await registry.embed("stub", ["q"], provider=embedder)
    assert embedder.calls == 1


async def test_an_embedding_lane_needs_no_chat_provider() -> None:
    """The embedding model is not the lane's chat model."""
    lane = Lane(
        name="bare",
        billing_unit="tokens",
        concurrency=2,
        queue_timeout_s=1.0,
        timeout_s=1.0,
    )
    registry = ProviderRegistry({"bare": lane}, Settings())

    result = await registry.embed("bare", ["q"], provider=StubEmbedder())
    assert result.value

    # The same lane still refuses a chat call, loudly.
    with pytest.raises(LaneNotImplemented):
        await registry.complete("bare", ASK)
