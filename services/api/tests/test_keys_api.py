"""Integration: /keys, driven by real issued keys end to end. Needs `make up`."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import text

from pdf_builder import build_pdf
from prism import tenancy
from prism.api.deps import embedding_provider
from prism.auth import resolve_key
from prism.config import Settings, get_settings
from prism.core.ids import uuid7
from prism.db import get_engine
from stub_provider import DIM, MODEL, StubProvider

pytestmark = pytest.mark.integration

UNKNOWN = "01890000-0000-7000-8000-00000000dead"
LISTED_FIELDS = {
    "id",
    "name",
    "prefix",
    "scopes",
    "created_at",
    "last_used_at",
    "revoked_at",
    "expires_at",
}


@dataclass(frozen=True)
class Tenant:
    id: UUID
    key_id: UUID
    headers: dict[str, str]


def bearer(plaintext: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {plaintext}"}


@pytest.fixture
def configured(app: FastAPI, unauthenticated: None) -> Iterator[None]:
    """Real key resolution and metering: the autouse stub is undone."""
    app.dependency_overrides[get_settings] = lambda: Settings(
        embedding_model=MODEL, embedding_dim=DIM
    )
    app.dependency_overrides[embedding_provider] = StubProvider
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def tenants() -> AsyncIterator[list[Tenant]]:
    made: list[Tenant] = []
    yield made
    if made:
        async with get_engine().begin() as conn:
            await conn.execute(
                text("DELETE FROM tenants WHERE id = ANY(:ids)"), {"ids": [t.id for t in made]}
            )


async def _new_tenant(made: list[Tenant]) -> Tenant:
    row, key = await tenancy.create_tenant(f"keys-{uuid7()}")
    tenant = Tenant(
        id=row.id, key_id=(await resolve_key(key.plaintext)).id, headers=bearer(key.plaintext)
    )
    made.append(tenant)
    return tenant


@pytest.fixture
async def tenant(tenants: list[Tenant]) -> Tenant:
    return await _new_tenant(tenants)


@pytest.fixture
async def other_tenant(tenants: list[Tenant]) -> Tenant:
    return await _new_tenant(tenants)


async def issue(
    client: AsyncClient,
    tenant: Tenant,
    *scopes: str,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"name": "test key", "scopes": list(scopes)}
    if expires_at is not None:
        body["expires_at"] = expires_at.isoformat()
    response = await client.post("/keys", json=body, headers=tenant.headers)
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


async def listed_by_id(client: AsyncClient, tenant: Tenant) -> dict[str, dict[str, Any]]:
    response = await client.get("/keys", headers=tenant.headers)
    assert response.status_code == 200, response.text
    return {row["id"]: row for row in response.json()}


class TestIssue:
    async def test_the_key_works_with_exactly_the_scopes_requested(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        issued = await issue(client, tenant, "read", "ingest")

        assert issued["api_key"].startswith(issued["prefix"])
        assert issued["scopes"] == ["read", "ingest"]
        resolved = await resolve_key(issued["api_key"])
        assert resolved.tenant_id == tenant.id
        assert {s.value for s in resolved.scopes} == {"read", "ingest"}

        listed = await client.get("/keys", headers=bearer(issued["api_key"]))
        assert listed.status_code == 200

    async def test_the_plaintext_is_not_listed_afterwards(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        issued = await issue(client, tenant, "read")
        listed = await client.get("/keys", headers=tenant.headers)
        assert issued["api_key"] not in listed.text

    @pytest.mark.parametrize(
        "body",
        [
            {"name": "k", "scopes": []},
            {"name": "k", "scopes": ["superuser"]},
            {"name": "", "scopes": ["read"]},
            {"name": "k", "scopes": ["read"], "expires_at": "2020-01-01T00:00:00Z"},
            {"name": "k", "scopes": ["read"], "expires_at": "2999-01-01T00:00:00"},
        ],
        ids=["no-scopes", "unknown-scope", "no-name", "past-expiry", "naive-expiry"],
    )
    async def test_a_malformed_request_is_422(
        self, client: AsyncClient, configured: None, tenant: Tenant, body: dict[str, Any]
    ) -> None:
        response = await client.post("/keys", json=body, headers=tenant.headers)
        assert response.status_code == 422


class TestReadScopedKey:
    @pytest.fixture
    async def read_key(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> dict[str, str]:
        return bearer((await issue(client, tenant, "read"))["api_key"])

    async def test_it_can_list(
        self, client: AsyncClient, configured: None, read_key: dict[str, str]
    ) -> None:
        assert (await client.get("/keys", headers=read_key)).status_code == 200

    async def test_it_cannot_ingest(
        self,
        client: AsyncClient,
        configured: None,
        tenant: Tenant,
        read_key: dict[str, str],
    ) -> None:
        collection = await tenancy.create_collection(
            tenant_id=tenant.id, name="docs", embedding_model=MODEL, embedding_dim=DIM
        )

        response = await client.post(
            f"/collections/{collection.id}/documents",
            files={"file": ("doc.pdf", build_pdf([["hello"]]), "application/pdf")},
            headers=read_key,
        )

        assert response.status_code == 403
        assert await tenancy.list_documents(collection_id=collection.id, tenant_id=tenant.id) == []

    async def test_it_cannot_create_a_collection(
        self, client: AsyncClient, configured: None, read_key: dict[str, str]
    ) -> None:
        response = await client.post("/collections", json={"name": "nope"}, headers=read_key)
        assert response.status_code == 403

    async def test_it_cannot_issue_a_key(
        self, client: AsyncClient, configured: None, read_key: dict[str, str]
    ) -> None:
        response = await client.post(
            "/keys", json={"name": "escalate", "scopes": ["admin"]}, headers=read_key
        )
        assert response.status_code == 403

    async def test_it_cannot_revoke_a_key(
        self, client: AsyncClient, configured: None, tenant: Tenant, read_key: dict[str, str]
    ) -> None:
        response = await client.delete(f"/keys/{tenant.key_id}", headers=read_key)
        assert response.status_code == 403
        assert (await client.get("/keys", headers=tenant.headers)).status_code == 200


class TestList:
    async def test_it_never_discloses_the_hash(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        await issue(client, tenant, "read")
        async with get_engine().connect() as conn:
            hashes = (
                (
                    await conn.execute(
                        text("SELECT key_hash FROM api_keys WHERE tenant_id = :t"), {"t": tenant.id}
                    )
                )
                .scalars()
                .all()
            )

        response = await client.get("/keys", headers=tenant.headers)

        assert response.status_code == 200
        assert len(response.json()) == len(hashes) == 2
        assert all(set(row) == LISTED_FIELDS for row in response.json())
        assert "key_hash" not in response.text
        assert not any(h in response.text for h in hashes)

    async def test_it_includes_revoked_keys_with_their_revocation_time(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        issued = await issue(client, tenant, "read")
        await client.delete(f"/keys/{issued['id']}", headers=tenant.headers)

        listed = await listed_by_id(client, tenant)

        assert listed[issued["id"]]["revoked_at"] is not None
        assert listed[str(tenant.key_id)]["revoked_at"] is None
        assert listed[str(tenant.key_id)]["last_used_at"] is not None


class TestRevokeAndExpire:
    async def test_a_revoked_key_401s_on_the_next_request(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        issued = await issue(client, tenant, "read")
        assert (await client.get("/keys", headers=bearer(issued["api_key"]))).status_code == 200

        revoked = await client.delete(f"/keys/{issued['id']}", headers=tenant.headers)

        assert revoked.status_code == 204
        assert (await client.get("/keys", headers=bearer(issued["api_key"]))).status_code == 401

    async def test_revoking_twice_is_204_and_keeps_the_first_time(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        issued = await issue(client, tenant, "read")
        await client.delete(f"/keys/{issued['id']}", headers=tenant.headers)
        first = (await listed_by_id(client, tenant))[issued["id"]]["revoked_at"]

        again = await client.delete(f"/keys/{issued['id']}", headers=tenant.headers)

        assert again.status_code == 204
        assert first is not None
        assert (await listed_by_id(client, tenant))[issued["id"]]["revoked_at"] == first

    async def test_an_expired_key_401s(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        tomorrow = datetime.now(UTC) + timedelta(days=1)
        issued = await issue(client, tenant, "read", expires_at=tomorrow)
        assert (await client.get("/keys", headers=bearer(issued["api_key"]))).status_code == 200

        async with get_engine().begin() as conn:
            await conn.execute(
                text("UPDATE api_keys SET expires_at = now() - interval '1 second' WHERE id = :id"),
                {"id": issued["id"]},
            )

        assert (await client.get("/keys", headers=bearer(issued["api_key"]))).status_code == 401

    async def test_an_unknown_key_is_404(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        assert (await client.delete(f"/keys/{UNKNOWN}", headers=tenant.headers)).status_code == 404

    async def test_revocation_needs_a_key(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        assert (await client.delete(f"/keys/{tenant.key_id}")).status_code == 401


class TestTenantIsolation:
    async def test_a_tenant_cannot_see_anothers_keys(
        self, client: AsyncClient, configured: None, tenant: Tenant, other_tenant: Tenant
    ) -> None:
        issued = await issue(client, tenant, "read")

        listed = (await client.get("/keys", headers=other_tenant.headers)).json()

        ids = {row["id"] for row in listed}
        assert ids == {str(other_tenant.key_id)}
        assert issued["id"] not in ids
        assert str(tenant.key_id) not in ids

    async def test_a_tenant_cannot_revoke_anothers_key(
        self, client: AsyncClient, configured: None, tenant: Tenant, other_tenant: Tenant
    ) -> None:
        """404 rather than 403 — a 403 would confirm the key exists."""
        issued = await issue(client, tenant, "read")

        response = await client.delete(f"/keys/{issued['id']}", headers=other_tenant.headers)

        assert response.status_code == 404
        assert (await client.get("/keys", headers=bearer(issued["api_key"]))).status_code == 200


class TestLockout:
    async def test_the_last_permanent_admin_key_cannot_revoke_itself(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        response = await client.delete(f"/keys/{tenant.key_id}", headers=tenant.headers)

        assert response.status_code == 409
        assert (await client.get("/keys", headers=tenant.headers)).status_code == 200

    async def test_a_key_may_revoke_itself_when_another_permanent_admin_key_remains(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        successor = await issue(client, tenant, "admin")

        response = await client.delete(f"/keys/{tenant.key_id}", headers=tenant.headers)

        assert response.status_code == 204
        assert (await client.get("/keys", headers=tenant.headers)).status_code == 401
        assert (await client.get("/keys", headers=bearer(successor["api_key"]))).status_code == 200

    async def test_an_expiring_admin_key_does_not_count_as_a_survivor(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        expiring = await issue(
            client, tenant, "admin", expires_at=datetime.now(UTC) + timedelta(days=1)
        )

        response = await client.delete(
            f"/keys/{tenant.key_id}", headers=bearer(expiring["api_key"])
        )

        assert response.status_code == 409
        assert (await client.get("/keys", headers=tenant.headers)).status_code == 200

    async def test_a_revoke_in_flight_is_waited_for_not_counted_as_a_survivor(
        self, client: AsyncClient, configured: None, tenant: Tenant
    ) -> None:
        """Two admin keys revoking each other at once must not leave zero."""
        other = await issue(client, tenant, "admin")

        async with get_engine().connect() as conn:
            in_flight = await conn.begin()
            await conn.execute(
                text("SELECT id FROM tenants WHERE id = :id FOR NO KEY UPDATE"), {"id": tenant.id}
            )
            await conn.execute(
                text("UPDATE api_keys SET revoked_at = now() WHERE id = :id"), {"id": other["id"]}
            )
            racing = asyncio.create_task(tenancy.revoke_key(tenant.key_id, tenant_id=tenant.id))
            await asyncio.sleep(0.5)
            await in_flight.commit()

        with pytest.raises(tenancy.LastAdminKeyError):
            await racing
        assert (await client.get("/keys", headers=tenant.headers)).status_code == 200
