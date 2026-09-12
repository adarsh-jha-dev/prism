"""Rendered page in, figures and tables out."""

from functools import lru_cache

from prism.config import Settings, get_settings
from prism.vision.base import (
    PROMPT,
    FigureKind,
    ParsedFigure,
    VisionError,
    VisionProvider,
    figures_from_json,
)
from prism.vision.gemini import GeminiVisionProvider
from prism.vision.ollama import OllamaVisionProvider

__all__ = [
    "PROMPT",
    "FigureKind",
    "GeminiVisionProvider",
    "OllamaVisionProvider",
    "ParsedFigure",
    "VisionError",
    "VisionProvider",
    "figures_from_json",
    "get_vision_provider",
]


def build_vision_provider(settings: Settings) -> VisionProvider:
    if settings.vision_lane == "gemini":
        return GeminiVisionProvider(settings)
    return OllamaVisionProvider(settings)


@lru_cache
def get_vision_provider() -> VisionProvider:
    return build_vision_provider(get_settings())
