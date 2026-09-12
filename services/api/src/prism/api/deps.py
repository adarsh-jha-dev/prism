"""Shared route dependencies.

Thin wrappers over the module-level getters: a route that declares what it needs
can be handed a stub in a test without reaching into another module's globals.
"""

from typing import Annotated

from fastapi import Depends

from prism.config import Settings, get_settings
from prism.embeddings import EmbeddingProvider, get_embedding_provider
from prism.vision import VisionProvider, get_vision_provider

__all__ = [
    "ProviderDep",
    "SettingsDep",
    "VisionDep",
    "embedding_provider",
    "vision_provider",
]


def embedding_provider() -> EmbeddingProvider:
    return get_embedding_provider()


SettingsDep = Annotated[Settings, Depends(get_settings)]


def vision_provider(settings: SettingsDep) -> VisionProvider | None:
    """None when vision is off."""
    return get_vision_provider() if settings.vision_enabled else None


ProviderDep = Annotated[EmbeddingProvider, Depends(embedding_provider)]
VisionDep = Annotated[VisionProvider | None, Depends(vision_provider)]
