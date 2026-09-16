"""Messages in, prose or a validated verdict out."""

from functools import lru_cache

from prism.chat.base import (
    ChatError,
    ChatProvider,
    Completion,
    Message,
    Role,
    Structured,
    Usage,
)
from prism.chat.ollama import OllamaChatProvider

__all__ = [
    "ChatError",
    "ChatProvider",
    "Completion",
    "Message",
    "OllamaChatProvider",
    "Role",
    "Structured",
    "Usage",
    "get_chat_provider",
]


@lru_cache
def get_chat_provider() -> ChatProvider:
    return OllamaChatProvider()
