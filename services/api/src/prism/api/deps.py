"""Shared route dependencies.

Thin wrappers over the module-level getters: a route that declares what it needs
can be handed a stub in a test without reaching into another module's globals.
"""

from typing import Annotated

from fastapi import Depends

from prism.config import Settings, get_settings
from prism.embeddings import EmbeddingProvider, get_embedding_provider

__all__ = ["ProviderDep", "SettingsDep", "embedding_provider"]


def embedding_provider() -> EmbeddingProvider:
    return get_embedding_provider()


SettingsDep = Annotated[Settings, Depends(get_settings)]
ProviderDep = Annotated[EmbeddingProvider, Depends(embedding_provider)]
