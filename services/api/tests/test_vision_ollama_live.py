"""The local vision model, for real. Needs Ollama with the model pulled.

Marked `ollama`, so CI deselects it. What is under test is that the pulled
model can actually read a rendered page and answer in the pinned schema —
the one thing a stubbed transport cannot tell us.
"""

import io

import pytest

from pdf_builder import build_pdf
from prism.config import Settings
from prism.ingestion.render import render_figure_pages
from prism.vision.ollama import OllamaVisionProvider

pytestmark = [pytest.mark.integration, pytest.mark.ollama]

TABLE_PAGE = [
    "Table 1. Latency by region.",
    "",
    "Region        p50 (ms)   p95 (ms)",
    "us-east-1     41         118",
    "eu-west-1     58         172",
    "ap-south-1    96         305",
]


def _render(lines: list[str], rules: int) -> bytes:
    settings = Settings(vision_render_dpi=150, vision_min_path_objects=6)
    (page,) = render_figure_pages(
        io.BytesIO(build_pdf([lines], rules={0: rules})), settings=settings
    )
    return page.png


async def test_the_local_model_reads_a_table_off_a_rendered_page() -> None:
    figures = await OllamaVisionProvider(Settings()).parse_page(_render(TABLE_PAGE, rules=8))

    assert figures, "the model found nothing on a page that is mostly a table"
    table = next((f for f in figures if f.kind == "table"), None)
    assert table is not None, f"expected a table, got {[f.kind for f in figures]}"

    # The values are what a citation would rest on; a plausible-looking table
    # with the wrong numbers is the failure that matters here.
    content = table.content
    assert "us-east-1" in content
    assert "118" in content
    assert "305" in content


async def test_the_local_model_answers_in_the_pinned_schema() -> None:
    """`format` constrains decoding, so this must hold for any page."""
    figures = await OllamaVisionProvider(Settings()).parse_page(_render(TABLE_PAGE, rules=8))
    assert all(f.kind in ("figure", "table", "equation") for f in figures)
    assert all(f.content.strip() for f in figures)
