"""The four lanes, as data. Which lane a node should use is the Phase 3 router.

Lane names are the registry's keys and use underscores. The hyphenated
`provider` string on a `Usage` or `model_pricing` row (ADR 0013) is the
provider's own; nothing maps between them yet.
"""

from prism.chat.ollama import OllamaChatProvider
from prism.config import Settings
from prism.providers.base import Lane

__all__ = ["LANE_NAMES", "build_lanes"]

LANE_NAMES = ("ollama", "ollama_cloud", "gemini", "openai")


def build_lanes(settings: Settings) -> dict[str, Lane]:
    return {
        "ollama": Lane(
            name="ollama",
            # Counted and priced at zero (ADR 0013).
            billing_unit="tokens",
            # Bounds calls we hold open, not what Ollama's GPU pool runs at once.
            concurrency=settings.concurrency_ollama_local,
            queue_timeout_s=settings.lane_queue_timeout_s,
            timeout_s=settings.chat_timeout_s,
            model=settings.generator_model,
            factory=lambda: OllamaChatProvider(settings),
        ),
        "ollama_cloud": Lane(
            name="ollama_cloud",
            # GPU-time, not tokens: what the subscription bills.
            billing_unit="gpu_ms",
            # 1 is external, not a tuning knob — the serialization point.
            concurrency=settings.concurrency_ollama_cloud,
            queue_timeout_s=settings.lane_queue_timeout_s,
            timeout_s=settings.ollama_cloud_timeout_s,
            # Cloud tags are not the local ones; Phase 3 picks one.
            model=None,
        ),
        "gemini": Lane(
            name="gemini",
            billing_unit="tokens",
            concurrency=settings.concurrency_gemini,
            queue_timeout_s=settings.lane_queue_timeout_s,
            timeout_s=settings.gemini_chat_timeout_s,
            model=settings.gemini_chat_model,
        ),
        "openai": Lane(
            name="openai",
            billing_unit="tokens",
            concurrency=settings.concurrency_openai,
            queue_timeout_s=settings.lane_queue_timeout_s,
            timeout_s=settings.openai_chat_timeout_s,
            # No model is named for the final escalation yet.
            model=None,
        ),
    }
