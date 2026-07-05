"""Smoke tests for the Streamlit dashboard (Phase 9b).

Full UI testing is out of scope for pytest — these verify the module is
importable, its pure helpers behave, and the Cesium fallback path never
raises without a token.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CHOKEPOINTS = ("hormuz", "red_sea", "malacca", "suez")


# ---------------------------------------------------------------------------
# 1 — the dashboard module imports without a Streamlit runtime
# ---------------------------------------------------------------------------


def test_dashboard_imports():
    import src.dashboard.app as app  # noqa: F401

    # Core surface exists.
    assert callable(app.main)
    assert callable(app.build_globe_html)
    assert callable(app.preset_to_range)


# ---------------------------------------------------------------------------
# 2 — monitoring points resolve to valid lat/lon for all 4 chokepoints
# ---------------------------------------------------------------------------


def test_geospatial_data_available():
    from src.dashboard.app import get_monitoring_points

    config = yaml.safe_load(
        (_PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")
    )
    points = get_monitoring_points(config)

    for region in _CHOKEPOINTS:
        assert region in points, f"No monitoring point resolved for {region!r}"
        p = points[region]
        assert p["name"], f"{region}: empty point name"
        assert -90.0 <= p["lat"] <= 90.0, f"{region}: invalid lat {p['lat']}"
        assert -180.0 <= p["lng"] <= 180.0, f"{region}: invalid lng {p['lng']}"


# ---------------------------------------------------------------------------
# 3 — with CESIUM_ION_TOKEN unset, the globe renders the fallback, no raise
# ---------------------------------------------------------------------------


def test_cesium_token_fallback(monkeypatch):
    monkeypatch.delenv("CESIUM_ION_TOKEN", raising=False)
    from src.dashboard.app import (
        GLOBE_FALLBACK_MESSAGE,
        build_globe_html,
        resolve_cesium_token,
    )

    token = resolve_cesium_token()
    assert token == ""

    html = build_globe_html(
        token=token,
        focus={"name": "Strait of Hormuz", "lat": 26.56, "lng": 56.25},
        secondary=[],
        risk_level="high",
        top_driver="vessel count",
        updated="2026-07-05",
    )
    assert "CESIUM_ION_TOKEN" in html, "Fallback should tell the user which env var to set"
    assert GLOBE_FALLBACK_MESSAGE in html
    assert "Cesium.Ion.defaultAccessToken" not in html, "No Cesium boot code without a token"


# ---------------------------------------------------------------------------
# 4 — predict_action is importable and callable from the dashboard module
# ---------------------------------------------------------------------------


def test_predict_action_importable():
    import src.dashboard.app as app

    assert callable(app.predict_action)
    assert app.predict_action("low", None, None) in app.ACTIONS

    # The dashboard's own wrapper names the rule that fired.
    action, rule = app.decide_action("high", "routing", False, None)
    assert action == "reroute"
    assert "reroute" in rule


# ---------------------------------------------------------------------------
# 5 — every preset maps to a valid, non-empty slice of the 365-day dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset,expected_len",
    [
        ("Last 30 days", 30),
        ("Last 90 days", 90),
        ("Last 6 months", 183),
        ("Last year", 365),
    ],
)
def test_analysis_range_mapping(preset, expected_len):
    from src.dashboard.app import preset_to_range

    start, end = preset_to_range(preset)
    assert 0 <= start <= end <= 364, f"{preset}: out-of-bounds range ({start}, {end})"
    assert end - start + 1 == expected_len, f"{preset}: wrong window length"

    # Custom range clamps to bounds and never inverts.
    start, end = preset_to_range("Custom range", custom=(200, 100))
    assert (start, end) == (99, 199)
    start, end = preset_to_range("Custom range", custom=(-5, 9999))
    assert (start, end) == (0, 364)
