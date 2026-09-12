"""Unit: how parsed figures become chunks. No database, no network, no model."""

import asyncio
import io

import pytest

from pdf_builder import build_pdf
from prism.config import Settings
from prism.ingestion.pipeline import parse_figures, plan_document_chunks
from prism.vision import ParsedFigure, VisionError

TABLE = ParsedFigure(kind="table", content="| a | b |", caption="Table 1.")
CHART = ParsedFigure(kind="figure", content="a bar chart of revenue")


def _settings(**kwargs: object) -> Settings:
    defaults: dict[str, object] = {
        "chunk_size_chars": 10,
        "chunk_overlap_chars": 0,
        "vision_render_dpi": 72,
        "vision_min_path_objects": 6,
    }
    return Settings(**{**defaults, **kwargs})  # type: ignore[arg-type]


class StubVision:
    """Canned figures per page, with call and concurrency counters."""

    model = "gemini-3.6-flash"

    def __init__(
        self,
        figures: list[ParsedFigure] | None = None,
        fail_after: int | None = None,
        delay: float = 0.0,
    ) -> None:
        self.calls = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self._figures = figures if figures is not None else [TABLE]
        self._fail_after = fail_after
        self._delay = delay

    async def parse_page(self, png: bytes) -> list[ParsedFigure]:
        self.calls += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._fail_after is not None and self.calls > self._fail_after:
                raise VisionError("stub refusing to parse")
            return list(self._figures)
        finally:
            self.in_flight -= 1


def test_text_comes_before_the_figures_on_its_page() -> None:
    planned = plan_document_chunks([(1, "abc")], {1: [TABLE]}, _settings())
    assert [(chunk.chunk_type, chunk.content) for chunk in planned] == [
        ("text", "abc"),
        ("table", "| a | b |"),
    ]


def test_reading_order_interleaves_pages_not_sources() -> None:
    # Page 2's table must not sort after page 3's prose.
    planned = plan_document_chunks(
        [(1, "one"), (2, "two"), (3, "three")], {2: [TABLE]}, _settings()
    )
    assert [(chunk.page_number, chunk.chunk_type) for chunk in planned] == [
        (1, "text"),
        (2, "text"),
        (2, "table"),
        (3, "text"),
    ]


def test_a_figure_page_with_no_text_layer_still_contributes() -> None:
    planned = plan_document_chunks([(1, "prose"), (2, "")], {2: [CHART]}, _settings())
    assert [(chunk.page_number, chunk.chunk_type) for chunk in planned] == [
        (1, "text"),
        (2, "figure"),
    ]


def test_several_figures_on_one_page_keep_their_order() -> None:
    planned = plan_document_chunks([(1, "x")], {1: [TABLE, CHART]}, _settings())
    assert [chunk.chunk_type for chunk in planned] == ["text", "table", "figure"]


def test_a_figure_chunk_is_marked_as_model_written() -> None:
    _, figure = plan_document_chunks(
        [(1, "x")], {1: [TABLE]}, _settings(), vision_model="gemini-3.6-flash"
    )
    assert figure.metadata == {
        "source": "vision",
        "vision_model": "gemini-3.6-flash",
        "caption": "Table 1.",
    }


def test_extracted_text_carries_no_source_marker() -> None:
    # The absence is the signal that the content is verbatim.
    (text,) = plan_document_chunks([(1, "x")], {}, _settings())
    assert text.metadata == {}
    assert text.chunk_type == "text"


def test_the_caption_stays_beside_the_content_not_inside_it() -> None:
    _, figure = plan_document_chunks([(1, "x")], {1: [TABLE]}, _settings())
    assert figure.content == "| a | b |"
    assert figure.metadata["caption"] == "Table 1."


def test_a_figure_without_a_caption_records_none() -> None:
    _, figure = plan_document_chunks([(1, "x")], {1: [CHART]}, _settings())
    assert "caption" not in figure.metadata


def test_no_figures_plans_exactly_what_the_text_path_planned() -> None:
    planned = plan_document_chunks([(1, "abcdefghijkl")], {}, _settings())
    assert [chunk.content for chunk in planned] == ["abcdefghij", "kl"]
    assert {chunk.chunk_type for chunk in planned} == {"text"}


async def test_only_flagged_pages_reach_the_provider() -> None:
    pdf = build_pdf([["prose"], ["Table 1"], ["prose"]], rules={1: 8})
    vision = StubVision()
    figures = await parse_figures(io.BytesIO(pdf), provider=vision, settings=_settings())

    assert vision.calls == 1
    assert list(figures) == [2]


async def test_a_document_of_prose_makes_no_call_at_all() -> None:
    vision = StubVision()
    figures = await parse_figures(
        io.BytesIO(build_pdf([["just words"]])), provider=vision, settings=_settings()
    )
    assert (vision.calls, figures) == (0, {})


async def test_pages_the_model_found_nothing_on_are_dropped() -> None:
    pdf = build_pdf([["Table 1"]], rules={0: 8})
    figures = await parse_figures(
        io.BytesIO(pdf), provider=StubVision(figures=[]), settings=_settings()
    )
    assert figures == {}


async def test_concurrency_is_capped_at_the_gemini_lane() -> None:
    pdf = build_pdf([["Table"]] * 6, rules=dict.fromkeys(range(6), 8))
    vision = StubVision(delay=0.01)
    await parse_figures(io.BytesIO(pdf), provider=vision, settings=_settings(concurrency_gemini=2))
    assert vision.calls == 6
    assert vision.peak_in_flight <= 2


async def test_one_failed_page_fails_the_document() -> None:
    pdf = build_pdf([["Table"]] * 3, rules=dict.fromkeys(range(3), 8))
    with pytest.raises(VisionError, match="stub refusing to parse"):
        await parse_figures(
            io.BytesIO(pdf),
            provider=StubVision(fail_after=1),
            settings=_settings(concurrency_gemini=1),
        )


async def test_a_failure_stops_the_pages_still_queued() -> None:
    pdf = build_pdf([["Table"]] * 8, rules=dict.fromkeys(range(8), 8))
    vision = StubVision(fail_after=1, delay=0.01)
    with pytest.raises(VisionError):
        await parse_figures(
            io.BytesIO(pdf), provider=vision, settings=_settings(concurrency_gemini=2)
        )
    assert vision.calls < 8
