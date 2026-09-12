"""Replay recorded paid-provider exchanges.

Matching is on method and URL only: the request body carries a base64 page
render, which would churn on every pypdfium2 upgrade.
"""

import json
from pathlib import Path
from typing import Any

import httpx

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict[str, Any]:
    """Load a cassette by path relative to tests/fixtures, without the suffix."""
    path = FIXTURES / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no cassette at {path} — run `make record-fixtures`")
    body: dict[str, Any] = json.loads(path.read_text())
    return body


def replay(name: str) -> httpx.AsyncClient:
    """An AsyncClient that answers the cassette's request and refuses any other."""
    cassette = load(name)
    expected = cassette["request"]
    recorded = cassette["response"]
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method != expected["method"] or str(request.url) != expected["url"]:
            raise AssertionError(
                f"cassette {name} holds {expected['method']} {expected['url']}, "
                f"got {request.method} {request.url}"
            )
        return httpx.Response(
            status_code=recorded["status_code"],
            json=recorded["json"],
            headers={"content-type": "application/json"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client.recorded_requests = seen  # type: ignore[attr-defined]
    return client
