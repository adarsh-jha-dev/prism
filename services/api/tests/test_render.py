"""Unit: figure detection and page rendering. No network, no model."""

import io
import struct
import zlib

import pytest

from pdf_builder import build_pdf
from prism.config import Settings
from prism.ingestion.render import RenderError, _encode_png, render_figure_pages


def _settings(**kwargs: object) -> Settings:
    # 72dpi keeps the test renders small.
    defaults: dict[str, object] = {"vision_render_dpi": 72, "vision_min_path_objects": 6}
    return Settings(**{**defaults, **kwargs})  # type: ignore[arg-type]


def _render(pdf: bytes, **kwargs: object) -> list[tuple[int, int, int]]:
    pages = render_figure_pages(io.BytesIO(pdf), settings=_settings(**kwargs))
    return [(page.page_number, page.images, page.paths) for page in pages]


def _scanlines(png: bytes) -> tuple[int, int, bytes]:
    """Decode a PNG to (width, height, raw scanlines). Filter 0 throughout."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    chunks: dict[bytes, bytes] = {}
    position = 8
    while position < len(png):
        length = struct.unpack(">I", png[position : position + 4])[0]
        tag = png[position + 4 : position + 8]
        chunks[tag] = png[position + 8 : position + 8 + length]
        position += 12 + length
    width, height, depth, colour, _, _, _ = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    assert (depth, colour) == (8, 2), "8-bit truecolour"
    return width, height, zlib.decompress(chunks[b"IDAT"])


def test_a_page_of_prose_costs_nothing() -> None:
    assert _render(build_pdf([["just words", "and more words"]])) == []


def test_a_page_with_a_raster_image_is_always_sent() -> None:
    # No threshold on images, only on paths.
    assert _render(build_pdf([["caption"]], images={0: 1})) == [(1, 1, 0)]


def test_a_ruled_table_is_sent_even_though_it_has_no_image() -> None:
    # The case page.images cannot see: the table is drawn, not embedded.
    assert _render(build_pdf([["Table 1"]], rules={0: 7})) == [(1, 0, 7)]


def test_a_few_stray_rules_are_not_a_table() -> None:
    assert _render(build_pdf([["heading"]], rules={0: 2})) == []


def test_the_path_threshold_is_the_cost_knob() -> None:
    pdf = build_pdf([["heading"]], rules={0: 3})
    assert _render(pdf, vision_min_path_objects=6) == []
    assert _render(pdf, vision_min_path_objects=3) == [(1, 0, 3)]


def test_only_the_qualifying_pages_come_back_and_in_page_order() -> None:
    pdf = build_pdf(
        [["prose"], ["Table 1"], ["prose"], ["Figure 2"]],
        rules={1: 8},
        images={3: 1},
    )
    assert _render(pdf) == [(2, 0, 8), (4, 1, 0)]


def test_page_numbers_are_1_based_and_match_the_text_path() -> None:
    # Shared with extract_pages; an off-by-one is a wrong citation.
    ((number, _, _),) = _render(build_pdf([["a"], ["b"], ["Table"]], rules={2: 9}))
    assert number == 3


def test_too_many_figure_pages_is_an_error_not_a_truncation() -> None:
    pdf = build_pdf([["Table"]] * 5, rules=dict.fromkeys(range(5), 8))
    with pytest.raises(RenderError, match="5 pages carry figures, over the per-document ceiling"):
        _render(pdf, vision_max_pages=4)


def test_the_ceiling_admits_exactly_its_limit() -> None:
    pdf = build_pdf([["Table"]] * 4, rules=dict.fromkeys(range(4), 8))
    assert len(_render(pdf, vision_max_pages=4)) == 4


def test_an_unreadable_file_is_a_render_error() -> None:
    with pytest.raises(RenderError, match="could not open PDF"):
        _render(b"not a pdf at all")


def test_a_rendered_page_is_a_decodable_png_of_the_right_size() -> None:
    (page,) = render_figure_pages(
        io.BytesIO(build_pdf([["Table 1"]], rules={0: 8})), settings=_settings()
    )
    width, height, scanlines = _scanlines(page.png)
    # 612x792pt at 72dpi.
    assert (width, height) == (612, 792)
    assert len(scanlines) == height * (1 + width * 3)
    assert set(scanlines[0 :: 1 + width * 3]) == {0}, "every scanline uses filter None"


def test_a_rendered_page_is_mostly_white_paper() -> None:
    # A black or empty render passes every structural check above.
    (page,) = render_figure_pages(
        io.BytesIO(build_pdf([["Table 1"]], rules={0: 8})), settings=_settings()
    )
    width, _, scanlines = _scanlines(page.png)
    pixels = bytes(
        byte
        for row in range(0, len(scanlines), 1 + width * 3)
        for byte in scanlines[row + 1 : row + 1 + width * 3]
    )
    assert pixels.count(255) / len(pixels) > 0.9, "paper is white"
    assert any(byte < 128 for byte in pixels), "the rules and text are drawn"


def test_dpi_scales_the_render() -> None:
    pdf = build_pdf([["Table 1"]], rules={0: 8})
    (low,) = render_figure_pages(io.BytesIO(pdf), settings=_settings(vision_render_dpi=72))
    (high,) = render_figure_pages(io.BytesIO(pdf), settings=_settings(vision_render_dpi=144))
    assert _scanlines(high.png)[0] == 2 * _scanlines(low.png)[0]


def test_the_encoder_swaps_bgr_to_rgb() -> None:
    red_then_green = bytes([0, 0, 255, 0, 255, 0])
    _, _, scanlines = _scanlines(_encode_png(red_then_green, 2, 1, 6))
    assert list(scanlines) == [0, 255, 0, 0, 0, 255, 0]


def test_the_encoder_skips_row_padding() -> None:
    padded = bytes([0, 0, 255, 0, 255, 0, 9, 9, 9])
    _, _, scanlines = _scanlines(_encode_png(padded, 2, 1, 9))
    assert list(scanlines) == [0, 255, 0, 0, 0, 255, 0]
