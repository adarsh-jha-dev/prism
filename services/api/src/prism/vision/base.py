"""The vision provider interface, and the parts every provider shares."""

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, runtime_checkable

__all__ = [
    "FIGURE_KINDS",
    "PROMPT",
    "FigureKind",
    "ParsedFigure",
    "VisionError",
    "VisionProvider",
    "figures_from_json",
]

# The subset of chunks.chunk_type a model may write. 'text' is excluded:
# extracted prose comes from the text layer.
FigureKind = Literal["figure", "table", "equation"]

FIGURE_KINDS: frozenset[str] = frozenset(("figure", "table", "equation"))

PROMPT = (
    "You are reading one page of a PDF whose text layer has already been "
    "extracted separately.\n\n"
    "Report only content that the text layer cannot carry: figures, charts, "
    "diagrams, tables, and displayed equations.\n\n"
    "Rules:\n"
    "- Do not transcribe body prose, headers, footers, or page numbers.\n"
    "- Render a table as GitHub-flavored Markdown, preserving every row and "
    "column. Do not summarise or truncate it.\n"
    "- Describe a figure or chart so that someone who cannot see it could "
    "answer questions about it: axes, units, series, and the values or trend "
    "it shows.\n"
    "- Render an equation as LaTeX.\n"
    "- Use the printed caption verbatim if there is one, and omit the caption "
    "field if there is not.\n"
    "- A page with no such content returns an empty list. Inventing a figure "
    "is worse than reporting none.\n"
)


class VisionError(RuntimeError):
    """A page could not be parsed. Never softened into "no figures here"."""


@dataclass(frozen=True)
class ParsedFigure:
    """One figure, table or equation. `content` is written by the model, not
    lifted from the page."""

    kind: FigureKind
    content: str
    caption: str | None = None


@runtime_checkable
class VisionProvider(Protocol):
    @property
    def model(self) -> str:
        """Model identifier, as recorded in `chunks.metadata.vision_model`."""
        ...

    async def parse_page(self, png: bytes) -> list[ParsedFigure]:
        """Parse one rendered page. Empty means the page carried no figures."""
        ...


def figures_from_json(raw: str) -> list[ParsedFigure]:
    """Validate a provider's `{"figures": [...]}` payload. Shared by every provider."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisionError(f"model did not return JSON: {exc}") from exc

    figures = parsed.get("figures") if isinstance(parsed, dict) else None
    if not isinstance(figures, list):
        raise VisionError(f"expected a 'figures' list, got {type(figures).__name__}")

    return [_parse_figure(index, item) for index, item in enumerate(figures)]


def _parse_figure(index: int, item: Any) -> ParsedFigure:
    if not isinstance(item, dict):
        raise VisionError(f"figure {index} is {type(item).__name__}, not an object")

    kind = item.get("kind")
    if kind not in FIGURE_KINDS:
        # Caught here rather than by chunks_chunk_type_check a transaction later.
        raise VisionError(
            f"figure {index} has kind {kind!r}, expected one of {sorted(FIGURE_KINDS)}"
        )

    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        raise VisionError(f"figure {index} ({kind}) has no content")

    caption = item.get("caption")
    if caption is not None and not isinstance(caption, str):
        raise VisionError(f"figure {index} has a non-string caption")

    return ParsedFigure(
        kind=cast(FigureKind, kind),
        content=content,
        caption=caption.strip() or None if caption else None,
    )
