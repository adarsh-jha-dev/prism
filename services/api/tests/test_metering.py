"""Integration: the limiter and the usage rows. Needs `make up`."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import text

from conftest import Authenticate
from prism.api.deps import embedding_provider, metered_key
from prism.auth import generate_key
from prism.config import Settings, get_settings
from prism.core.ids import uuid7
from prism.db import get_engine, get_redis
from prism.metering import (
    RateLimitExceeded,
    consume,
    record_usage,
    window_start,
)
from stub_provider import DIM, MODEL, QueryProvider

pytestmark = pytest.mark.integration

URL = "/collections/{}/search"
UNKNOWN = "01890000-0000-7000-8000-00000000dead"


@pytest.fixture
async def api_key_id() -> AsyncIterator[tuple[UUID, UUID]]:
    """A real api_keys row, so usage_records has something to reference."""
    tenant_id, key_id = uuid7(), uuid7()
    key = generate_key()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": f"metering-{tenant_id}"},
        )
        await conn.execute(
            text(
                "INSERT INTO api_keys (id, tenant_id, key_hash, name, key_prefix) "
                "VALUES (:id, :tenant_id, :key_hash, 'metering', :key_prefix)"
            ),
            {
                "id": key_id,
                "tenant_id": tenant_id,
                "key_hash": key.key_hash,
                "key_prefix": key.prefix,
            },
        )
    try:
        yield key_id, tenant_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


class TestConsume:
    async def test_counts_down_from_the_limit(self) -> None:
        key_id = uuid7()
        first = await consume(key_id, limit=3)
        second = await consume(key_id, limit=3)

        assert (first.limit, first.remaining) == (3, 2)
        assert second.remaining == 1

    async def test_refuses_past_the_limit(self) -> None:
        key_id = uuid7()
        for _ in range(2):
            await consume(key_id, limit=2)

        with pytest.raises(RateLimitExceeded) as caught:
            await consume(key_id, limit=2)

        assert caught.value.state.remaining == 0
        assert 0 < caught.value.state.reset_seconds <= 60

    async def test_keys_are_counted_separately(self) -> None:
        mine, theirs = uuid7(), uuid7()
        await consume(mine, limit=1)

        # Their first request must not be refused by my having used mine.
        assert (await consume(theirs, limit=1)).remaining == 0

    async def test_a_new_window_starts_fresh(self) -> None:
        key_id = uuid7()
        now = datetime.now(UTC)
        await consume(key_id, limit=1, now=now)

        later = await consume(key_id, limit=1, now=now + timedelta(seconds=61))

        assert later.remaining == 0

    async def test_the_counter_expires(self) -> None:
        key_id = uuid7()
        await consume(key_id, limit=5)
        bucket = int(datetime.now(UTC).timestamp()) // 60

        ttl = await get_redis().ttl(f"ratelimit:{key_id}:{bucket}")
        assert 0 < ttl <= 60


class TestWindowStart:
    def test_floors_to_the_hour(self) -> None:
        moment = datetime(2026, 9, 13, 14, 37, 42, tzinfo=UTC)
        assert window_start(moment) == datetime(2026, 9, 13, 14, 0, 0, tzinfo=UTC)

    def test_is_stable_across_a_window(self) -> None:
        base = datetime(2026, 9, 13, 14, 0, 1, tzinfo=UTC)
        assert window_start(base) == window_start(base + timedelta(minutes=59))


class TestRecordUsage:
    async def test_accumulates_into_one_row_per_hour(self, api_key_id: tuple[UUID, UUID]) -> None:
        key_id, tenant_id = api_key_id
        for _ in range(3):
            await record_usage(key_id, tenant_id)

        async with get_engine().connect() as conn:
            rows = (
                await conn.execute(
                    text("SELECT requests, throttled FROM usage_records WHERE api_key_id = :id"),
                    {"id": key_id},
                )
            ).all()
        assert len(rows) == 1
        assert rows[0].requests == 3
        assert rows[0].throttled == 0

    async def test_a_throttled_request_still_counts(self, api_key_id: tuple[UUID, UUID]) -> None:
        key_id, tenant_id = api_key_id
        await record_usage(key_id, tenant_id)
        await record_usage(key_id, tenant_id, throttled=True)

        async with get_engine().connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT requests, throttled FROM usage_records WHERE api_key_id = :id"),
                    {"id": key_id},
                )
            ).one()
        assert (row.requests, row.throttled) == (2, 1)

    async def test_separate_hours_are_separate_rows(self, api_key_id: tuple[UUID, UUID]) -> None:
        key_id, tenant_id = api_key_id
        now = datetime.now(UTC)
        await record_usage(key_id, tenant_id, now=now)
        await record_usage(key_id, tenant_id, now=now + timedelta(hours=1))

        async with get_engine().connect() as conn:
            count = (
                await conn.execute(
                    text("SELECT count(*) FROM usage_records WHERE api_key_id = :id"),
                    {"id": key_id},
                )
            ).scalar_one()
        assert count == 2

    async def test_it_touches_last_used_at(self, api_key_id: tuple[UUID, UUID]) -> None:
        key_id, tenant_id = api_key_id
        await record_usage(key_id, tenant_id)

        async with get_engine().connect() as conn:
            last_used = (
                await conn.execute(
                    text("SELECT last_used_at FROM api_keys WHERE id = :id"), {"id": key_id}
                )
            ).scalar_one()
        assert last_used is not None

    async def test_an_unwritable_row_does_not_raise(self) -> None:
        """Metering is a record, not a control: it must never fail a request."""
        await record_usage(uuid7(), uuid7())  # no such key — FK violation, swallowed


class TestRouteEnforcement:
    @pytest.fixture
    def configured(self, app: FastAPI) -> Any:
        def settings() -> Settings:
            return Settings(embedding_model=MODEL, embedding_dim=DIM)

        app.dependency_overrides[get_settings] = settings
        app.dependency_overrides[embedding_provider] = QueryProvider
        yield
        app.dependency_overrides.clear()

    async def test_the_limit_headers_are_on_a_successful_response(
        self,
        client: AsyncClient,
        app: FastAPI,
        configured: None,
        authenticate: Authenticate,
        api_key_id: tuple[UUID, UUID],
    ) -> None:
        """A client should be able to back off before being refused."""
        key_id, tenant_id = api_key_id
        authenticate(tenant_id, rate_limit_rpm=5, key_id=key_id)
        app.dependency_overrides.pop(metered_key)

        response = await client.post(URL.format(UNKNOWN), json={"query": "anything"})

        assert response.headers["x-ratelimit-limit"] == "5"
        assert response.headers["x-ratelimit-remaining"] == "4"
        assert 0 < int(response.headers["x-ratelimit-reset"]) <= 60

    async def test_past_the_limit_is_429_with_retry_after(
        self,
        client: AsyncClient,
        app: FastAPI,
        configured: None,
        authenticate: Authenticate,
        api_key_id: tuple[UUID, UUID],
    ) -> None:
        key_id, tenant_id = api_key_id
        authenticate(tenant_id, rate_limit_rpm=2, key_id=key_id)
        app.dependency_overrides.pop(metered_key)

        for _ in range(2):
            await client.post(URL.format(UNKNOWN), json={"query": "anything"})
        refused = await client.post(URL.format(UNKNOWN), json={"query": "anything"})

        assert refused.status_code == 429
        assert refused.headers["x-ratelimit-remaining"] == "0"
        assert 0 < int(refused.headers["retry-after"]) <= 60

    async def test_the_headers_survive_an_error_response(
        self,
        client: AsyncClient,
        app: FastAPI,
        configured: None,
        authenticate: Authenticate,
        api_key_id: tuple[UUID, UUID],
    ) -> None:
        """404 is the common case for a limited client; it must still carry the limit."""
        key_id, tenant_id = api_key_id
        authenticate(tenant_id, rate_limit_rpm=5, key_id=key_id)
        app.dependency_overrides.pop(metered_key)

        response = await client.post(URL.format(UNKNOWN), json={"query": "anything"})

        assert response.status_code == 404
        assert response.headers["x-ratelimit-limit"] == "5"

    async def test_a_throttled_request_is_recorded_as_usage(
        self,
        client: AsyncClient,
        app: FastAPI,
        configured: None,
        authenticate: Authenticate,
        api_key_id: tuple[UUID, UUID],
    ) -> None:
        key_id, tenant_id = api_key_id
        authenticate(tenant_id, rate_limit_rpm=1, key_id=key_id)
        app.dependency_overrides.pop(metered_key)

        await client.post(URL.format(UNKNOWN), json={"query": "anything"})
        await client.post(URL.format(UNKNOWN), json={"query": "anything"})

        async with get_engine().connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT requests, throttled FROM usage_records WHERE api_key_id = :id"),
                    {"id": key_id},
                )
            ).one()
        assert (row.requests, row.throttled) == (2, 1)
