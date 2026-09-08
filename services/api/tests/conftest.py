"""Unit tests are the default. Integration tests are marked and need `make up`.

Paid providers are never called from either tier — they replay from
tests/fixtures/.
"""

import os
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("PRISM_ENV", "test")


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """In-process client — never binds a port."""
    from prism.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
