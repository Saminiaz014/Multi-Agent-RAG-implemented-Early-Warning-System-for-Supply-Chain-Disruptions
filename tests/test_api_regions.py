"""Region management endpoints (Phase 12.3).

``switch`` mutates process-global state in :mod:`src.api.endpoints`, so every
test here restores the starting region afterwards — otherwise a switch would
leak into ``tests/test_api_6agent.py``, which scores against the default region.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.endpoints import app, get_active_region, set_active_region
from src.core.regions import get_region, list_regions


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _restore_active_region():
    """Leave the service on whichever region it started on."""
    original = get_active_region()
    yield
    set_active_region(original)


class TestListAndInspect:
    """The read-only endpoints."""

    def test_list_returns_every_region(self, client: TestClient) -> None:
        data = client.get("/api/regions/list").json()

        assert data["count"] == len(list_regions())
        assert [r["key"] for r in data["regions"]] == list_regions()
        assert data["current_region"] in list_regions()
        assert sum(r["is_active"] for r in data["regions"]) == 1

    def test_list_carries_activation_and_reasons(self, client: TestClient) -> None:
        """The payload explains exclusions rather than silently omitting agents."""
        regions = {r["key"]: r for r in client.get("/api/regions/list").json()["regions"]}
        panama = regions["panama"]

        assert panama["display_name"] == "Panama Canal"
        assert "geopolitical" in panama["passive_agents"]
        assert "hydrological" in panama["passive_reasons"]["geopolitical"]
        assert "routing" in panama["passive_agents"]
        # Longitude is genuinely negative — a sign flip would put it in Africa.
        assert panama["longitude"] < 0

    def test_info_returns_one_region(self, client: TestClient) -> None:
        response = client.get("/api/regions/info/panama")
        assert response.status_code == 200
        assert response.json()["key"] == "panama"

    def test_info_accepts_any_casing(self, client: TestClient) -> None:
        assert client.get("/api/regions/info/HORMUZ").json()["key"] == "hormuz"

    def test_info_404s_on_unknown_region(self, client: TestClient) -> None:
        response = client.get("/api/regions/info/atlantis")
        assert response.status_code == 404
        # The message lists valid keys so a caller can recover in one round trip.
        assert "hormuz" in response.json()["detail"]

    def test_current_matches_the_service_state(self, client: TestClient) -> None:
        data = client.get("/api/regions/current").json()
        assert data["key"] == get_active_region()
        assert data["is_active"] is True

    def test_every_region_reports_routing_passive(self, client: TestClient) -> None:
        """Phase 11's global routing muting, visible through the API."""
        for region in client.get("/api/regions/list").json()["regions"]:
            assert "routing" in region["passive_agents"], region["key"]
            assert "routing" not in region["active_agents"], region["key"]


class TestSwitch:
    """The one endpoint that mutates state."""

    def test_switch_changes_the_active_region(self, client: TestClient) -> None:
        response = client.post("/api/regions/switch", json={"region": "malacca"})
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["region"] == "malacca"
        assert data["display_name"] == "Strait of Malacca"
        assert set(data["active_agents"]) == set(get_region("malacca").active_agents())

        # The change is observable through the other endpoints, not just echoed.
        assert get_active_region() == "malacca"
        assert client.get("/api/regions/current").json()["key"] == "malacca"

    def test_switch_rebuilds_config_against_the_new_region(
        self, client: TestClient
    ) -> None:
        """The point of switching: later scoring uses the new region's config."""
        from src.api.endpoints import _load_config

        client.post("/api/regions/switch", json={"region": "panama"})
        assert _load_config()["_active_region"] == "panama"
        assert _load_config()["agents"]["geopolitical"]["enabled"] is False

        client.post("/api/regions/switch", json={"region": "hormuz"})
        assert _load_config()["_active_region"] == "hormuz"
        assert _load_config()["agents"]["geopolitical"]["enabled"] is True

    def test_switch_to_the_current_region_succeeds(self, client: TestClient) -> None:
        """Idempotent, so a client can switch without checking first."""
        current = get_active_region()
        data = client.post("/api/regions/switch", json={"region": current}).json()
        assert data["success"] is True
        assert data["previous_region"] == current == data["region"]

    def test_switch_400s_on_unknown_region_without_changing_state(
        self, client: TestClient
    ) -> None:
        """A rejected switch must not leave the service half-switched."""
        before = get_active_region()
        response = client.post("/api/regions/switch", json={"region": "atlantis"})

        assert response.status_code == 400
        assert "hormuz" in response.json()["detail"]
        assert get_active_region() == before

    def test_switch_requires_a_region_field(self, client: TestClient) -> None:
        assert client.post("/api/regions/switch", json={}).status_code == 422
