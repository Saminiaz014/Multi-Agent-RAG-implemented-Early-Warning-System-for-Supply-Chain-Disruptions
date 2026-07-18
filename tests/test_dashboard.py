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


class _StubRetriever:
    """Deterministic ContextRetriever stand-in (no ChromaDB) whose payloads
    deliberately contain decimals and a similarity score — the breakdown
    builder must scrub them all before anything reaches Page 1."""

    def query_gated(self, current_signals, composite_risk_score,
                    top_k=None, min_similarity=None):
        if composite_risk_score < 0.65:
            return None
        return {
            "triggered": True,
            "matches": [{
                "source": "static",
                "text": ("In June 2019, two oil tankers were attacked near the strait; "
                         "transit dropped 22.5% and congestion peaked at 0.72. "
                         "War-risk insurance premiums spiked."),
                "similarity": 0.8123,
                "metadata": {"event": "Hormuz Tanker Attacks", "date": "2019-06"},
            }],
        }

    def query(self, current_signals, top_k=None):
        return [{
            "id": "hormuz_2019",
            "similarity": 0.7101,
            "metadata": {"event": "Hormuz Tanker Attacks", "date": "2019-06"},
            "text": "In June 2019, two oil tankers were attacked near the strait.",
        }]


def _stub_shap_row() -> dict:
    """A SurrogateShapExplainer.explain()-shaped row with raw feature names."""
    return {"top_drivers": [
        {"feature": "congestion_index", "agent": "shipping", "shap_value": 0.213},
        {"feature": "brent_crude_usd", "agent": "market", "shap_value": 0.101},
        {"feature": "military_activity_index", "agent": "geopolitical", "shap_value": 0.055},
    ]}


def _breakdown_text(bd) -> str:
    """Every manager-visible string a PeakBreakdown carries, joined."""
    parts = [bd.date_label, bd.level_word, bd.precedent_title,
             bd.precedent_summary, bd.narrative]
    parts += [d["label"] for d in bd.drivers] + [d["level"] for d in bd.drivers]
    return "\n".join(parts)


def test_decision_view_no_raw_scores():
    """Page 1 renders with status words + action badge only — no raw numbers.

    Covers both the rendered page and the peak drill-down panel content
    (built by the pure helper the st.dialog renders)."""
    import pandas as pd

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

    # The peak drill-down obeys the same rules: no decimals, no raw feature
    # names, no agent-score vocabulary, no similarity leak.
    from src.dashboard.core import Peak, build_peak_breakdown
    from src.explainability.shap_explainer import ALL_FEATURE_NAMES

    peak = Peak(index=155, date=pd.Timestamp("2025-06-05"), value=0.83)
    bd = build_peak_breakdown(peak, _stub_shap_row(), _StubRetriever(),
                              signals={"shipping": 0.9, "market": 0.7})
    panel = _breakdown_text(bd)
    panel_low = panel.lower()
    leaks = _RAW_SCORE_RE.findall(panel)
    assert not leaks, f"Raw numeric scores leaked into the peak drill-down: {leaks[:10]}"
    for term in _FORBIDDEN_TERMS + ("similarity", "shap"):
        assert term not in panel_low, f"Term {term!r} leaked into the peak drill-down"
    for feat in ALL_FEATURE_NAMES:
        assert feat not in panel, f"Raw feature name {feat!r} leaked into the drill-down"


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


def test_region_selector_hormuz_only():
    """Hormuz is the only offered region, but every fetcher accepts an
    arbitrary region parameter without raising (post-thesis expansion)."""
    from src.dashboard.core import AVAILABLE_REGIONS, get_news, get_routes, get_vessels

    assert list(AVAILABLE_REGIONS.keys()) == ["Strait of Hormuz"]
    assert AVAILABLE_REGIONS["Strait of Hormuz"] == "hormuz"

    # Populated region resolves to routes …
    hormuz_routes = get_routes("hormuz")
    assert len(hormuz_routes) >= 2
    assert get_vessels(hormuz_routes[0], day=155), "vessels should generate for a route"

    # … arbitrary regions are accepted and return empty results, never raise.
    for region in ("red_sea", "malacca", "suez", "atlantis", "", None):
        assert isinstance(get_routes(region), list)
        assert isinstance(get_news(region), list)


def test_read_last_updated(tmp_path):
    """The freshness-marker reader formats ISO-8601 into a decimal-free badge
    string, and degrades to None (badge omitted) when the marker is absent or
    unparseable."""
    from src.dashboard.core import read_last_updated

    marker = tmp_path / "last_updated.txt"
    marker.write_text("2026-07-17T08:00:12+02:00", encoding="utf-8")
    formatted = read_last_updated(marker)
    assert formatted == "17 Jul 2026, 08:00"
    # The badge must never look like a risk score.
    assert not _RAW_SCORE_RE.findall(f"Data refreshed: {formatted}")

    # Trailing whitespace/newline from `date -Iseconds > file` is tolerated.
    marker.write_text("2026-01-03T09:30:00+01:00\n", encoding="utf-8")
    assert read_last_updated(marker) == "03 Jan 2026, 09:30"

    assert read_last_updated(tmp_path / "missing.txt") is None
    junk = tmp_path / "junk.txt"
    junk.write_text("not-a-timestamp", encoding="utf-8")
    assert read_last_updated(junk) is None


# ===========================================================================
# Trend-graph redesign tests (datetime axis, peak markers, drill-down)
# ===========================================================================


def test_detect_high_peaks():
    """Only prominent HIGH-zone local maxima earn markers — sub-threshold
    bumps and low-prominence wiggles on an already-high shoulder do not."""
    import numpy as np
    import pandas as pd

    from src.dashboard.core import detect_high_peaks

    scores = np.full(60, 0.2)
    scores[8:13] = [0.4, 0.7, 0.85, 0.7, 0.4]          # prominent HIGH peak @ 10
    scores[24:27] = [0.35, 0.5, 0.35]                  # below threshold — ignored
    # Prominent HIGH peak @ 40, with a 0.73 wiggle @ 43 on its 0.7 shoulder
    # (prominence 0.03 < 0.05) that must be filtered out.
    scores[38:49] = [0.5, 0.75, 0.9, 0.78, 0.7, 0.73, 0.7, 0.5, 0.35, 0.2, 0.2]
    dates = pd.date_range("2025-01-01", periods=60, freq="D")

    peaks = detect_high_peaks(scores, dates, high_threshold=0.6, min_prominence=0.05)

    assert [p.index for p in peaks] == [10, 40]
    assert [p.date for p in peaks] == [dates[10], dates[40]]
    # Loosening the prominence floor admits the shoulder wiggle — the
    # parameter is honoured.
    loose = detect_high_peaks(scores, dates, high_threshold=0.6, min_prominence=0.01)
    assert 43 in [p.index for p in loose]


def test_peak_breakdown_plain_language():
    """The drill-down uses friendly labels + qualitative words, composes a
    narrative with a historical precedent, and contains no float-like text."""
    import re

    import pandas as pd

    from src.dashboard.core import FEATURE_DISPLAY_NAMES, Peak, build_peak_breakdown

    peak = Peak(index=155, date=pd.Timestamp("2025-06-05"), value=0.83)
    bd = build_peak_breakdown(peak, _stub_shap_row(), _StubRetriever(),
                              signals={"shipping": 0.9, "market": 0.7})

    assert bd.drivers, "breakdown must surface drivers"
    for d in bd.drivers:
        assert d["label"] in FEATURE_DISPLAY_NAMES.values(), f"unfriendly label {d['label']!r}"
        assert d["level"] in ("moderate", "elevated", "strongly elevated")
    # Top driver is the strongest by construction.
    assert bd.drivers[0] == {"label": "vessel congestion", "level": "strongly elevated"}

    assert bd.narrative, "a narrative paragraph is required"
    assert "June 5" in bd.narrative
    assert bd.precedent_title == "Hormuz Tanker Attacks"
    assert bd.precedent_title in bd.narrative

    text = _breakdown_text(bd)
    floats = re.findall(r"\d+\.\d+", text)
    assert not floats, f"float-looking substrings leaked: {floats}"

    # Below the RAG gate the ungated top-1 fallback still supplies a precedent.
    low_peak = Peak(index=100, date=pd.Timestamp("2025-04-11"), value=0.62)
    bd_low = build_peak_breakdown(low_peak, _stub_shap_row(), _StubRetriever(),
                                  signals={"shipping": 0.7})
    assert bd_low.precedent_title == "Hormuz Tanker Attacks"
    assert not re.findall(r"\d+\.\d+", _breakdown_text(bd_low))


def test_datetime_axis_mapping():
    """Day index 0 maps to the synthetic year start, the last index to the
    last calendar date, and month-start tick positions are produced."""
    import numpy as np
    import pandas as pd

    from src.dashboard.core import month_start_ticks, to_datetime_series

    ds = to_datetime_series(pd.Series(np.linspace(0.0, 1.0, 365)))
    assert ds.index[0] == pd.Timestamp("2025-01-01")
    assert ds.index[-1] == pd.Timestamp("2025-12-31")
    assert float(ds.iloc[-1]) == 1.0

    ticks = month_start_ticks(ds.index)
    assert len(ticks) == 12
    assert all(t.day == 1 for t in ticks)
    assert ticks[0] == pd.Timestamp("2025-01-01")
    assert ticks[-1] == pd.Timestamp("2025-12-01")

    # Configurable start date, and pass-through for existing datetime indexes.
    ds2 = to_datetime_series(pd.Series([1.0, 2.0]), start_date="2024-06-30")
    assert list(ds2.index) == [pd.Timestamp("2024-06-30"), pd.Timestamp("2024-07-01")]
    assert to_datetime_series(ds).index.equals(ds.index)
