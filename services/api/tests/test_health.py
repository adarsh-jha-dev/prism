from httpx import AsyncClient


async def test_health_is_ok_without_dependencies(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_health_deps_reports_every_dependency(client: AsyncClient) -> None:
    response = await client.get("/health/deps")
    assert response.status_code in (200, 503)
    body = response.json()
    assert set(body["checks"]) == {"postgres", "redis", "ollama"}
    assert all("ok" in check for check in body["checks"].values())


async def test_health_deps_refuses_200_when_a_dependency_is_down(
    client: AsyncClient,
) -> None:
    response = await client.get("/health/deps")
    body = response.json()
    if not all(check["ok"] for check in body["checks"].values()):
        assert response.status_code == 503
        assert body["ok"] is False
        assert any("error" in c for c in body["checks"].values() if not c["ok"])
