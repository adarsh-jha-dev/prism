"""Fixed-size sliding-window chunking.

Character-based, not token-based: a pure function with no tokenizer keeps
chunking deterministic across models, and `nomic-embed-text` truncates at its
own context window regardless. Sizes are therefore character counts.
"""

__all__ = ["chunk_text"]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split `text` into windows of `size` characters overlapping by `overlap`.

    Chunks are returned in order and cover the whole input; every chunk is
    non-empty and no chunk is wholly contained in its predecessor. The final
    chunk is short whenever the text does not divide evenly.

    Raises ValueError if `size` is not positive, or if `overlap` is negative or
    does not leave forward progress (`overlap >= size` never terminates).
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap must be smaller than size, got {overlap} >= {size}")

    step = size - overlap
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        # The tail is already covered; a further window would only repeat it.
        if end == len(text):
            break
        start += step

    return chunks
