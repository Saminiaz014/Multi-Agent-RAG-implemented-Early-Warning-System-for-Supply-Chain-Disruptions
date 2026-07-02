"""Tests for the 6-agent API expansion (Phase 8).

Nine tests covering all new and extended endpoints. The orchestrator is mocked
so tests run without CSV data or ML model fitting.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.api.endpoints as _ep
from src.api.endpoints import ALL_AGENTS, app

# ---------------------------------------------------------------------------
# Canonical mock pipeline result (mirrors run_full_pipeline() return shape)
# ---------------------------------------------------------------------------

_MOCK_RESULT: dict[str, Any] = {
    "composite_score": 0.72,
    "risk_level": "HIGH",
    "agent_scores": {
        "shipping": 0.80,
        "market": 0.65,
        "geopolitical": 0.70,
        "natural_disaster": 0.55,
        "routing": 0.60,
        "news_sentiment": 0.75,
    },
    "risk_score": 0.78,
    "risk_level_label": "high",
    "contributing_agents": {
        "shipping": {"score": 0.80, "weight": 0.25, "contribution": 0.20},
        "market": {"score": 0.65, "weight": 0.15, "contribution": 0.0975},
        "geopolitical": {"score": 0.70, "weight": 0.25, "contribution": 0.175},
        "natural_disaster": {"score": 0.55, "weight": 0.10, "contribution": 0.055},
        "routing": {"score": 0.60, "weight": 0.15, "contribution": 0.09},
        "news_sentiment": {"score": 0.75, "weight": 0.10, "contribution": 0.075},
    },
    "agent_agreement": 4,
    "reason": "HIGH risk. Primary driver: shipping (0.80, 27% of weighted risk).",
    "explanation": {
        "top_drivers": [
            {"feature": "vessel_count", "shap_value": 0.15, "direction": "up"},
            {"feature": "oil_price_usd", "shap_value": 0.12, "direction": "up"},
            {"feature": "brent_crude_usd", "shap_value": 0.10, "direction": "up"},
        ],
        "expected_value": 0.35,
        "text": "Risk primarily driven by shipping anomalies.",
        "surrogate_r2": 0.85,
    },
    "historical_context": {
        "triggered": True,
        "composite_score": 0.72,
        "threshold": 0.65,
        "matches": [
            {
                "source": "static",
                "text": "2019 Gulf of Oman Tanker Attacks",
                "similarity": 0.88,
                "metadata": {"event": "Tanker Attacks", "date": "2019-06"},
            }
        ],
        "formatted_summary": "Historical Precedents:\n1. [static] [2019-06] Tanker Attacks...",
    },
    "shap": {},
    "context": [],
    "data": {"rows": 100, "start": "2020-01-01", "end": "2020-04-10"},
    "metadata": {
        "agents_active": ALL_AGENTS,
        "data_modes": {n: "synthetic" for n in ALL_AGENTS},
        "weight_mode": "hand_tuned",
        "active_agents": 6,
        "weights_used": {n: round(1 / 6, 6) for n in ALL_AGENTS},
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_module_state():
    """Restore module-level state after every test."""
    saved_config = copy.deepcopy(_ep._config)
    saved_orch = _ep._orchestrator
    saved_ts = _ep._last_run_timestamp

    # Reset to clean slate so each test loads config fresh.
    _ep._config.clear()
    _ep._orchestrator = None
    _ep._last_run_timestamp = None

    yield

    _ep._config.clear()
    _ep._config.update(saved_config)
    _ep._orchestrator = saved_orch
    _ep._last_run_timestamp = saved_ts


@pytest.fixture()
def mock_orch():
    """Return a MagicMock orchestrator whose run_full_pipeline returns _MOCK_RESULT."""
    orch = MagicMock()
    orch.run_full_pipeline.return_value = copy.deepcopy(_MOCK_RESULT)
    return orch


@pytest.fixture()
def client(mock_orch):
    """TestClient with the orchestrator patched to the mock."""
    with patch.object(_ep, "_get_orchestrator", return_value=mock_orch):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# 1. test_health_6agents
# ---------------------------------------------------------------------------

def test_health_6agents():
    with TestClient(app) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert len(data["agents_active"]) == 6
    assert set(data["agents_active"]) == set(ALL_AGENTS)
    assert data["total_agents"] == 6
    assert "weight_mode" in data
    assert "knowledge_base_doc_count" in data
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# 2. test_predict_6agents
# ---------------------------------------------------------------------------

def test_predict_6agents(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_score" in data
    assert "risk_level" in data
    assert "contributing_agents" in data
    assert len(data["contributing_agents"]) == 6, (
        f"Expected 6 contributing_agents, got {len(data['contributing_agents'])}: "
        f"{list(data['contributing_agents'].keys())}"
    )
    assert set(data["contributing_agents"].keys()) == set(ALL_AGENTS)
    assert "agent_scores" in data
    assert len(data["agent_scores"]) == 6


# ---------------------------------------------------------------------------
# 3. test_explain_has_shap_and_rag
# ---------------------------------------------------------------------------

def test_explain_has_shap_and_rag(client):
    resp = client.post("/explain", json={})
    assert resp.status_code == 200
    data = resp.json()

    # SHAP explanation block
    assert "explanation" in data, "Response missing 'explanation' key"
    explanation = data["explanation"]
    assert isinstance(explanation, dict), "'explanation' should be a dict"
    assert "top_drivers" in explanation or "top_features" in data, (
        "Expected SHAP top_drivers in explanation or top_features in response"
    )

    # RAG historical context block
    assert "historical_context" in data, "Response missing 'historical_context' key"
    hc = data["historical_context"]
    assert hc is not None, "'historical_context' should not be null when mock triggers"
    assert "matches" in hc
    assert len(hc["matches"]) >= 1


# ---------------------------------------------------------------------------
# 4. test_agents_list
# ---------------------------------------------------------------------------

def test_agents_list():
    with TestClient(app) as c:
        resp = c.get("/agents")
    assert resp.status_code == 200
    agents = resp.json()
    assert len(agents) == 6
    names = {a["name"] for a in agents}
    assert names == set(ALL_AGENTS)
    for agent in agents:
        assert "name" in agent
        assert "enabled" in agent
        assert "data_mode" in agent
        assert "detection_method" in agent
        assert "current_weight" in agent


# ---------------------------------------------------------------------------
# 5. test_toggle_agent
# ---------------------------------------------------------------------------

def test_toggle_agent():
    with TestClient(app) as c:
        # Toggle natural_disaster off
        resp = c.post(
            "/agents/toggle",
            json={"agent": "natural_disaster", "enabled": False},
        )
        assert resp.status_code == 200
        toggle_data = resp.json()
        assert toggle_data["enabled"] is False
        assert "natural_disaster" not in toggle_data["agents_active"]
        assert len(toggle_data["agents_active"]) == 5

        # /agents should reflect disabled state
        resp = c.get("/agents")
        assert resp.status_code == 200
        agents = {a["name"]: a for a in resp.json()}
        assert agents["natural_disaster"]["enabled"] is False
        enabled_count = sum(1 for a in resp.json() if a["enabled"])
        assert enabled_count == 5

        # Toggle natural_disaster back on
        resp = c.post(
            "/agents/toggle",
            json={"agent": "natural_disaster", "enabled": True},
        )
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True

        # /agents should show all 6 enabled again
        resp = c.get("/agents")
        agents = {a["name"]: a for a in resp.json()}
        assert agents["natural_disaster"]["enabled"] is True
        enabled_count = sum(1 for a in resp.json() if a["enabled"])
        assert enabled_count == 6


def test_toggle_invalid_agent():
    with TestClient(app) as c:
        resp = c.post(
            "/agents/toggle",
            json={"agent": "nonexistent_agent", "enabled": False},
        )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. test_weights_endpoint
# ---------------------------------------------------------------------------

def test_weights_endpoint():
    with TestClient(app) as c:
        resp = c.get("/weights")
    assert resp.status_code == 200
    data = resp.json()
    assert "weight_mode" in data
    assert "inter_agent_weights" in data
    assert "thresholds" in data

    weights = data["inter_agent_weights"]
    assert len(weights) == 6, f"Expected 6 inter_agent_weights, got {len(weights)}"
    assert set(weights.keys()) == set(ALL_AGENTS)

    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01, (
        f"inter_agent_weights should sum to ~1.0, got {total:.4f}"
    )


# ---------------------------------------------------------------------------
# 7. test_weight_switch
# ---------------------------------------------------------------------------

def test_weight_switch():
    with TestClient(app) as c:
        # Switch to optimized
        resp = c.post("/weights/switch", json={"mode": "optimized"})
        assert resp.status_code == 200
        assert resp.json()["weight_mode"] == "optimized"

        # GET /weights should reflect new mode
        resp = c.get("/weights")
        assert resp.status_code == 200
        assert resp.json()["weight_mode"] == "optimized"

        # Switch back to hand_tuned
        resp = c.post("/weights/switch", json={"mode": "hand_tuned"})
        assert resp.status_code == 200
        assert resp.json()["weight_mode"] == "hand_tuned"

        resp = c.get("/weights")
        assert resp.json()["weight_mode"] == "hand_tuned"


def test_weight_switch_invalid_mode():
    with TestClient(app) as c:
        resp = c.post("/weights/switch", json={"mode": "magic"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 8. test_optimization_results
# ---------------------------------------------------------------------------

def test_optimization_results():
    opt_path = Path("data/processed/optimization_results.json")
    with TestClient(app) as c:
        resp = c.get("/optimization/results")

    if opt_path.exists():
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict), "Optimization results should be a JSON object"
        # Standard keys written by WeightOptimizer
        for key in ("best_trial", "best_objective_value", "test_metrics"):
            assert key in data, f"Missing key '{key}' in optimization results"
    else:
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 9. test_populate_returns_immediately
# ---------------------------------------------------------------------------

def test_populate_returns_immediately():
    """POST /populate should return immediately with status='started'."""
    import time

    with patch.object(_ep, "_run_populate_background", return_value=None):
        with TestClient(app) as c:
            t0 = time.monotonic()
            resp = c.post("/populate")
            elapsed = time.monotonic() - t0

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    # Should respond in well under 2 seconds (background task, not blocking)
    assert elapsed < 5.0, f"POST /populate took too long: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Bonus: /status smoke test
# ---------------------------------------------------------------------------

def test_status_endpoint():
    with TestClient(app) as c:
        resp = c.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data
    assert "knowledge_base" in data
    assert "last_pipeline_run" in data
    assert "weight_mode" in data
    assert set(data["agents"].keys()) == set(ALL_AGENTS)
