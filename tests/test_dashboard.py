"""Smoke tests for the two-page Streamlit dashboard (Phase 9b + redesign).

Full UI testing is out of scope for pytest — these verify the modules import,
the pure helpers behave, the Decision view leaks no raw scores, route
selection stays in sync via session state, and the JPEG export helper
produces valid images. Heavier page renders use Streamlit's AppTest.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CHOKEPOINTS = ("hormuz", "red_sea", "malacca", "suez")
_DECISION_PAGE = str(_PROJECT_ROOT / "src" / "dashboard" / "pages" / "1_Decision_View.py")
_ANALYSIS_PAGE = str(_PROJECT_ROOT / "src" / "dashboard" / "pages" / "2_Analysis_View.py")

#: Raw metric decimals like 0.87 / 12.50 — forbidden on the Decision view.
_RAW_SCORE_RE = re.compile(r"\b\d+\.\d{2,}\b")
#: Technical vocabulary that must not appear on the Decision view.
_FORBIDDEN_TERMS = ("congestion index", "f1", "precision", "recall",
                    "shap value", "anomaly score", "z-score")


def _page_text(at) -> str:
    """Collect all manager-visible text from an AppTest run.

    Injected ``<style>`` blocks are excluded — CSS values like
    ``font-size:1.35rem`` are not user-visible content.
    """
    chunks = [str(m.value) for m in at.markdown
              if not str(m.value).lstrip().startswith("<style")]
    chunks += [str(c.value) for c in getattr(at, "caption", [])]
    chunks += [str(b.label) for b in at.button]
    for r in at.radio:
        chunks += [str(o) for o in r.options]
    chunks += [str(s.label) for s in getattr(at, "selectbox", [])]
    return "\n".join(chunks)


# ===========================================================================
# Original Phase 9b smoke tests (kept — surface re-exported from app.py)
# ===========================================================================


def test_dashboard_imports():
    import src.dashboard.app as app  # noqa: F401

    assert callable(app.main)
    assert callable(app.build_map_html)
    assert callable(app.preset_to_range)
    # Redesign modules import cleanly too.
    from src.dashboard import analysis_view, core, decision_view  # noqa: F401

    assert callable(decision_view.render)
    assert callable(analysis_view.render)


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


def test_map_renders_without_key(monkeypatch):
    """The MapLibre map needs no API key: with MAPTILER_API_KEY unset, the
    keyless OpenFreeMap + AWS-terrarium path renders the full map (no
    placeholder), and no key material appears in the HTML."""
    monkeypatch.delenv("MAPTILER_API_KEY", raising=False)
    from src.dashboard.app import build_map_html, resolve_map_key

    assert resolve_map_key() == ""

    html = build_map_html(
        focus={"name": "Strait of Hormuz", "lat": 26.56, "lng": 56.25},
        secondary=[],
        risk_level="high",
        top_driver="vessel count",
        updated="2026-07-05",
    )
    assert "maplibre-gl" in html, "MapLibre GL JS must load from the CDN"
    assert "openfreemap.org" in html, "Keyless OpenFreeMap style expected without a key"
    assert "terrarium" in html, "Keyless AWS DEM terrain expected without a key"
    assert "setTerrain" in html, "3-D terrain must be enabled"
    assert "fill-extrusion" in html, "3-D building layer must be defined"
    assert '"pitch": 55' in html, "Camera must be pitched (tilted 3-D view)"
    assert "maptiler.com" not in html, "No keyed provider without a key"
    assert "key=" not in html, "No key material should appear in keyless HTML"

    # When a key IS supplied, the MapTiler upgrade path is used — and the
    # key is injected into the rendered HTML only (by design, client-visible).
    html_keyed = build_map_html(
        focus={"name": "Strait of Hormuz", "lat": 26.56, "lng": 56.25},
        secondary=[], risk_level="low", top_driver="x", updated="2026-07-05",
        maptiler_key="test-key-123",
    )
    assert "maptiler.com" in html_keyed and "test-key-123" in html_keyed


def test_predict_action_importable():
    import src.dashboard.app as app

    assert callable(app.predict_action)
    assert app.predict_action("low", None, None) in app.ACTIONS

    action, rule = app.decide_action("high", "routing", False, None)
    assert action == "reroute"
    assert "reroute" in rule


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

    start, end = preset_to_range("Custom range", custom=(200, 100))
    assert (start, end) == (99, 199)
    start, end = preset_to_range("Custom range", custom=(-5, 9999))
    assert (start, end) == (0, 364)


# ===========================================================================
# Redesign tests
# ===========================================================================


def test_decision_view_no_raw_scores():
    """Page 1 renders with status words + action badge only — no raw numbers."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_DECISION_PAGE, default_timeout=900)
    at.run()
    assert not at.exception, f"Decision view raised: {[e.message for e in at.exception]}"

    text = _page_text(at)
    low = text.lower()

    # Status vocabulary is present …
    assert any(w in text for w in ("Critical", "High", "Low")), "No status words rendered"
    # … raw metric decimals are not (day indices / dates are integers, allowed).
    leaks = _RAW_SCORE_RE.findall(text)
    assert not leaks, f"Raw numeric scores leaked into the Decision view: {leaks[:10]}"
    for term in _FORBIDDEN_TERMS:
        assert term not in low, f"Technical term {term!r} leaked into the Decision view"


def test_route_selection_sync():
    """Route selection via the control strip and the status-list buttons both
    drive the shared session state consumed by the map and status panels."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_DECISION_PAGE, default_timeout=900)
    at.run()
    assert not at.exception

    # Default: first route selected.
    assert at.session_state["selected_route"] == "westbound_tss"

    # Control strip (radio) → session state.
    at.radio(key="route_radio").set_value("eastbound_tss")
    at.run()
    assert at.session_state["selected_route"] == "eastbound_tss"

    # Status-list "Select" button → same shared state.
    at.button(key="sel_fujairah_approach").click()
    at.run()
    assert at.session_state["selected_route"] == "fujairah_approach"


def test_decision_badge_valid_action():
    """The decision badge logic always yields a defined action and tolerates
    missing/None historical context."""
    from src.dashboard.core import ACTIONS, decide_action

    cases = [
        ("low", "", False, None),
        ("medium", "market", False, None),
        ("high", "routing", False, None),
        ("high", "geopolitical", False, None),                       # no context
        ("high", "geopolitical", False, {}),                         # empty context
        ("high", "geopolitical", False, {"similarity": 0.9, "action": "escalate"}),
        ("high", "natural_disaster", True, None),                    # sustained
        (None, None, False, None),                                   # everything missing
    ]
    for level, agent, sustained, hist in cases:
        action, rule = decide_action(level, agent or "", sustained, hist)
        assert action in ACTIONS, f"invalid action {action!r} for {level!r}/{agent!r}"
        assert isinstance(rule, str) and rule, "rule text must always be provided"


def test_analysis_view_all_metrics_present():
    """Page 2 renders all 8 evaluation metrics plus the explorer, no errors."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_ANALYSIS_PAGE, default_timeout=1200)
    at.run()
    assert not at.exception, f"Analysis view raised: {[e.message for e in at.exception]}"

    text = _page_text(at)
    for header in (
        "1 · Detection performance",
        "2 · Explainability faithfulness",
        "3 · Agent diversity",
        "4 · Baseline comparison",
        "5 · Weight-optimization impact",
        "6 · RAG retrieval relevance",
        "7 · Generalization check",
        "8 · Decision effectiveness",
        "9 · Interactive time-range explorer",
    ):
        assert header in text, f"Missing Analysis section: {header!r}"


def test_jpeg_export_helper():
    """fig_to_jpeg flattens chart + title + caption into one valid JPEG."""
    import plotly.graph_objects as go

    from src.dashboard.core import fig_to_jpeg

    fig = go.Figure(go.Bar(x=["a", "b", "c"], y=[1, 3, 2]))
    data = fig_to_jpeg(fig, "Sample chart title", "Sample caption for the thesis document.")

    assert isinstance(data, (bytes, bytearray))
    assert data[:3] == b"\xff\xd8\xff", "Output is not a valid JPEG (bad magic bytes)"
    assert len(data) > 10_000, "JPEG suspiciously small — render likely failed"


def test_route_fetchers_tolerate_any_region():
    """Hormuz is the only region with mapped route corridors, and every fetcher
    accepts an arbitrary region key without raising.

    Phase 12 opened the selector to all four registered chokepoints (see
    ``tests/test_dashboard_regions.py``), but ``core._ROUTES`` still holds
    geometry for Hormuz alone — drawing corridors elsewhere would mean
    inventing them. This pins the resulting asymmetry so it stays deliberate:
    every other region falls to the empty-route branch rather than erroring.
    """
    from src.dashboard.core import AVAILABLE_REGIONS, get_news, get_routes, get_vessels

    # Populated region resolves to routes …
    hormuz_routes = get_routes("hormuz")
    assert len(hormuz_routes) >= 2
    assert get_vessels(hormuz_routes[0], day=155), "vessels should generate for a route"

    # … every other selectable region returns empty, never raises.
    for region in set(AVAILABLE_REGIONS.values()) - {"hormuz"}:
        assert get_routes(region) == [], f"{region} unexpectedly has routes"

    # … and unknown / malformed keys are equally safe.
    for region in ("red_sea", "suez", "atlantis", "", None):
        assert isinstance(get_routes(region), list)
        assert isinstance(get_news(region), list)
