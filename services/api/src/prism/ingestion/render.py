"""Figure detection and page rendering.

Detection reads pdfium's page-object inventory rather than the text layer: a
ruled table or vector chart has no extractable text but is made of path objects.
"""

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

from prism.config import Settings

__all__ = ["RenderError", "RenderedPage", "render_figure_pages"]

_OBJ_PATH = pdfium_raw.FPDF_PAGEOBJ_PATH
_OBJ_IMAGE = pdfium_raw.FPDF_PAGEOBJ_IMAGE

_POINTS_PER_INCH = 72.0


class RenderError(RuntimeError):
    """A document's pages could not be inspected or rendered."""


@dataclass(frozen=True)
class RenderedPage:
    page_number: int  # 1-based, matching chunks.page_number
    png: bytes
    images: int
    paths: int


def render_figure_pages(
    source: str | Path | IO[bytes], *, settings: Settings
) -> list[RenderedPage]:
    """Render every page that looks like it carries a figure, table or equation.

    Raises RenderError if the document cannot be opened, or if more pages
    qualify than `vision_max_pages` allows.
    """
    scale = settings.vision_render_dpi / _POINTS_PER_INCH

    try:
        document = pdfium.PdfDocument(_openable(source))
    except pdfium.PdfiumError as exc:
        raise RenderError(f"could not open PDF for rendering: {exc}") from exc

    try:
        flagged = [
            (number, images, paths)
            for number, images, paths in _inventory(document)
            if images >= 1 or paths >= settings.vision_min_path_objects
        ]
        # A hard stop, not a truncation: a partial parse looks complete downstream.
        if len(flagged) > settings.vision_max_pages:
            raise RenderError(
                f"{len(flagged)} pages carry figures, over the per-document ceiling of "
                f"{settings.vision_max_pages} — raise vision_max_pages to ingest it"
            )
        return [
            RenderedPage(
                page_number=number,
                png=_render_png(document[number - 1], scale),
                images=images,
                paths=paths,
            )
            for number, images, paths in flagged
        ]
    except pdfium.PdfiumError as exc:
        raise RenderError(f"could not render PDF: {exc}") from exc
    finally:
        document.close()


def _openable(source: str | Path | IO[bytes]) -> str | Path | bytes:
    if isinstance(source, str | Path):
        return source
    # The text path has already drained the stream.
    source.seek(0)
    return source.read()


def _inventory(document: pdfium.PdfDocument) -> list[tuple[int, int, int]]:
    """Count image and path objects per page, as `(page_number, images, paths)`."""
    counted: list[tuple[int, int, int]] = []
    for index in range(len(document)):
        page = document[index]
        try:
            images = paths = 0
            # Recurses into form XObjects by default.
            for obj in page.get_objects(filter=(_OBJ_IMAGE, _OBJ_PATH)):
                if obj.type == _OBJ_IMAGE:
                    images += 1
                else:
                    paths += 1
            counted.append((index + 1, images, paths))
        finally:
            page.close()
    return counted


def _render_png(page: pdfium.PdfPage, scale: float) -> bytes:
    try:
        bitmap = page.render(scale=scale)
        try:
            return _encode_png(bytes(bitmap.buffer), bitmap.width, bitmap.height, bitmap.stride)
        finally:
            bitmap.close()
    finally:
        page.close()


def _encode_png(bgr: bytes, width: int, height: int, stride: int) -> bytes:
    """Encode pdfium's top-down BGR bitmap as 8-bit truecolour PNG.

    `stride` is not always `width * 3`; the surplus is row padding.
    """
    rows = bytearray()
    for y in range(height):
        row = bgr[y * stride : y * stride + width * 3]
        rgb = bytearray(row)
        # Both slices read the untouched original, so this is a swap.
        rgb[0::3] = row[2::3]
        rgb[2::3] = row[0::3]
        rows.append(0)  # filter: None
        rows.extend(rgb)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", header),
            _chunk(b"IDAT", zlib.compress(bytes(rows), 6)),
            _chunk(b"IEND", b""),
        )
    )


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )
