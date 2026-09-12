"""Record paid-provider cassettes against the live APIs.

Local only, with real keys in `.env`:

    make record-fixtures
"""

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "services" / "api"
FIXTURES = API / "tests" / "fixtures"

sys.path.insert(0, str(API / "src"))
sys.path.insert(0, str(API / "tests"))

from pdf_builder import build_pdf  # noqa: E402
from prism.config import Settings  # noqa: E402
from prism.ingestion.render import render_figure_pages  # noqa: E402
from prism.vision.gemini import GeminiVisionProvider  # noqa: E402

PAGES: dict[str, bytes] = {
    "gemini/table_page": build_pdf(
        [
            [
                "Table 1. Latency by region.",
                "",
                "Region        p50 (ms)   p95 (ms)",
                "us-east-1     41         118",
                "eu-west-1     58         172",
                "ap-south-1    96         305",
            ]
        ],
        rules={0: 8},
    ),
    "gemini/prose_page": build_pdf(
        [["A page of ordinary prose that the detector flagged on a hairline rule."]],
        rules={0: 8},
    ),
}


class Recorder(httpx.AsyncBaseTransport):
    """Passes the call through and keeps what crossed the wire."""

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.exchanges: list[tuple[httpx.Request, httpx.Response]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        await response.aread()
        self.exchanges.append((request, response))
        return response


def _cassette(request: httpx.Request, response: httpx.Response, secret: str) -> dict[str, Any]:
    """Headers are never recorded; this checks the key leaked nowhere else."""
    recorded = json.dumps({"url": str(request.url), "body": response.json()})
    if secret and secret in recorded:
        raise SystemExit(f"refusing to write {request.url}: the API key leaked into it")
    return {
        "provenance": "recorded",
        "request": {"method": request.method, "url": str(request.url)},
        "response": {"status_code": response.status_code, "json": response.json()},
    }


async def record(name: str, pdf: bytes, settings: Settings) -> None:
    pages = render_figure_pages(io.BytesIO(pdf), settings=settings)
    if not pages:
        raise SystemExit(f"{name}: the detector flagged no page — nothing to record")

    recorder = Recorder()
    provider = GeminiVisionProvider(settings, client=httpx.AsyncClient(transport=recorder))
    figures = await provider.parse_page(pages[0].png)

    (request, response) = recorder.exchanges[-1]
    path = FIXTURES / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cassette = _cassette(request, response, settings.gemini_api_key or "")
    path.write_text(json.dumps(cassette, indent=2) + "\n")
    print(f"{name}: {len(figures)} figure(s) -> {path.relative_to(ROOT)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=None, help="cassettes to re-record")
    args = parser.parse_args()

    settings = Settings()
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is unset — nothing to record against")

    for name in args.names or PAGES:
        if name not in PAGES:
            raise SystemExit(f"unknown cassette {name!r}; known: {', '.join(PAGES)}")
        await record(name, PAGES[name], settings)


if __name__ == "__main__":
    asyncio.run(main())
