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

#: Keys of agents.natural_disaster.monitoring_points (the Ambee query points)
#: — deliberately NOT extraction.chokepoints, which is a different structure
#: and was realigned to the region registry in the GDACS/USGS work.
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

    action, rule = app.decide_action("high", "shipping", False, None)
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
    """Route selection via the map (``?route=``) and the status-list buttons
    both drive the shared session state consumed by the map and status panels."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_DECISION_PAGE, default_timeout=900)
    at.run()
    assert not at.exception

    # Default: first route selected.
    assert at.session_state["selected_route"] == "westbound_tss"

    # Map click (?route= query param) → session state.
    at.query_params["route"] = "eastbound_tss"
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
        ("high", "shipping", False, None),
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
    """Every selectable region has corridors and vessels; unknown keys are safe.

    Phase 12 opened the selector to all four registered chokepoints and Phase
    12.5 gave each of them corridor geometry, so the earlier Hormuz-only
    asymmetry this test pinned no longer exists. What still matters is that an
    arbitrary region key never raises — the fetchers are probed freely by the
    map and news panels.
    """
    from src.dashboard.core import AVAILABLE_REGIONS, get_news, get_routes, get_vessels

    for region in AVAILABLE_REGIONS.values():
        routes = get_routes(region)
        assert len(routes) >= 2, f"{region} has too few corridors"
        assert get_vessels(routes[0], day=155, region=region), (
            f"vessels should generate for a {region} route"
        )

    # Unknown / malformed keys return empty results rather than raising.
    for region in ("red_sea", "suez", "atlantis", "", None):
        assert get_routes(region) == []
        assert isinstance(get_news(region), list)


def test_decision_window_query_params():
    """Without query params the window opens on the last year up to the latest
    day; ``?date=`` and ``?window=`` only change those initial values."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_DECISION_PAGE, default_timeout=900)
    at.run()
    assert not at.exception
    defaults = {s.label: s.value for s in at.select_slider}
    assert defaults.get("Window length") == "1 year"
    end_default = next(s.value for s in at.slider if s.label.startswith("Slide the window"))

    at = AppTest.from_file(_DECISION_PAGE, default_timeout=900)
    at.query_params["date"] = "2024-10-01"
    at.query_params["window"] = "7"
    at.run()
    assert not at.exception
    assert {s.label: s.value for s in at.select_slider}["Window length"] == "1 week"
    end = next(s.value for s in at.slider if s.label.startswith("Slide the window"))
    if str(end_default) > "2024-10-01":       # only meaningful on the real series
        assert str(end) == "2024-10-01"


def test_analysis_case_explorer_deep_link():
    """``?view=cases`` renders the real-data explorer alone, without errors."""
    from streamlit.testing.v1 import AppTest

    from src.dashboard.core import live_series_path

    if not live_series_path("bab_el_mandeb").exists():
        pytest.skip("real series not built (scripts/build_live_risk_series.py)")
    at = AppTest.from_file(_ANALYSIS_PAGE, default_timeout=900)
    at.query_params.update(view="cases", region="bab_el_mandeb",
                           start="2023-06-01", end="2025-12-31", date="2024-01-15")
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    text = _page_text(at)
    assert "10 · Real-data case explorer" in text
    assert "1 · Detection performance" not in text


# ------------------------------------------------ issue 16: split date, enabled agents, counts
class _LazyCore:
    """``src.dashboard.core`` imported on first use, as the other tests here do."""

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module("src.dashboard.core"), name)


core = _LazyCore()


def test_the_split_date_text_is_the_fixed_evaluation_split_date() -> None:
    from src.evaluation.window import eval_window

    assert core.split_date_text() == f"{eval_window()['split_date']:%Y-%m-%d}"


def _live_series(n: int = 1500, start: str = "2021-11-01"):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(4)
    ts = pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="D"),
                       "risk_score": rng.uniform(0.1, 0.9, n)})
    raw = pd.DataFrame({"timestamp": ts["timestamp"],
                        "shipping__vessel_count": rng.integers(20, 90, n).astype(float),
                        "market__brent_crude_usd": rng.normal(80, 5, n),
                        "market__freight_rate_index": rng.normal(100, 5, n),
                        "geopolitical__military_activity_index": rng.random(n)})
    return ts, raw


def test_the_live_surrogate_is_fitted_before_the_fixed_split_date_not_on_70_percent() -> None:
    from src.evaluation.window import eval_window

    ts, raw = _live_series()
    bundle = core.fit_live_surrogate(ts, raw, disabled_agents=set())
    split = eval_window()["split_date"]
    expected = int((ts["timestamp"] < split).sum())
    assert expected != int(len(ts) * 0.70)                   # the old cut would differ
    assert bundle["n_train"] == expected
    assert len(bundle["features"]) == len(ts)                # row-aligned with the series


def test_the_live_surrogate_leaves_out_features_of_disabled_agents() -> None:
    import numpy as np

    ts, raw = _live_series()
    bundle = core.fit_live_surrogate(ts, raw, disabled_agents={"market", "geopolitical"})
    f = bundle["features"]
    for name in ("brent_crude_usd", "freight_rate_index", "military_activity_index"):
        assert f[name].isna().all(), name                   # present in the frame, not used
    assert f["vessel_count"].notna().all()
    kept = core.fit_live_surrogate(ts, raw, disabled_agents=set())["features"]
    assert kept["brent_crude_usd"].notna().all()


def test_the_narrative_counts_the_agents_that_reported_not_a_fixed_number() -> None:
    drivers = [{"feature": "vessel_count", "agent": "shipping", "direction": "elevated"}]
    four = core._compose_narrative(drivers, 2, "High", "Panama Canal", n_agents=4)
    assert "two of four detection agents" in four
    default = core._compose_narrative(drivers, 2, "High", "Panama Canal")
    assert f"of {core._NUM_WORDS[len(core.ALL_AGENTS)]} detection agents" in default


def test_the_dashboard_no_longer_states_a_fixed_agent_count_or_a_70_percent_cut() -> None:
    from pathlib import Path

    text = "\n".join(Path(core.__file__).with_name(n).read_text(encoding="utf-8")
                     for n in ("analysis_view.py", "core.py", "decision_view.py"))
    assert "first 70" not in text
    assert "Six-agent" not in text and "Raw 6-agent" not in text
    assert core.FULL_CONFIG_LABEL == f"{len(core.ALL_AGENTS)}-agent"
    assert '("6-agent", set(ALL_AGENTS))' not in text      # the live full-config label is computed


# ------------------------------------------------ issue 16 review: window, order, edge cases
def test_the_live_surrogates_r2_window_stops_at_the_evaluation_end_date() -> None:
    """Live series run past ``end_date`` (the 2026 crisis); R² is over the evaluation window only."""
    from src.evaluation.window import eval_window

    ts, raw = _live_series(n=1800)                           # runs to 2026-10, past end_date
    end = eval_window()["end_date"]
    assert ts["timestamp"].max() > end
    base = core.fit_live_surrogate(ts, raw, disabled_agents=set())["r2_test"]
    wild = ts.copy()
    wild.loc[wild["timestamp"] > end, "risk_score"] += 100.0   # nothing past end_date may count
    assert core.fit_live_surrogate(wild, raw, disabled_agents=set())["r2_test"] == base


def test_the_live_surrogate_trains_on_days_before_the_split_whatever_the_row_order(
        monkeypatch) -> None:
    import pandas as pd

    from src.evaluation.window import eval_window

    seen = {}
    real_fit = __import__("src.explainability.real_surrogate", fromlist=["x"]).fit_surrogate

    def spy(X, y):
        seen["rows"] = len(X)
        seen["max_ts"] = timestamps.loc[X.index].max()
        return real_fit(X, y)

    ts, raw = _live_series()
    order = ts.sample(frac=1.0, random_state=1).index          # shuffled rows, same data
    ts, raw = ts.loc[order].reset_index(drop=True), raw.loc[order].reset_index(drop=True)
    timestamps = ts["timestamp"]
    monkeypatch.setattr("src.explainability.real_surrogate.fit_surrogate", spy)
    bundle = core.fit_live_surrogate(ts, raw, disabled_agents=set())
    split = eval_window()["split_date"]
    assert seen["max_ts"] < split and seen["rows"] == bundle["n_train"] == int(
        (ts["timestamp"] < split).sum())
    assert pd.notna(bundle["r2_test"])


@pytest.mark.parametrize("start, n", [("2021-11-01", 100),      # all before the split
                                      ("2024-01-01", 300)])     # all after it
def test_the_live_surrogate_names_the_split_when_one_side_is_empty(start, n) -> None:
    ts, raw = _live_series(n=n, start=start)
    with pytest.raises(ValueError, match="split date 20"):
        core.fit_live_surrogate(ts, raw, disabled_agents=set())


def test_the_narrative_prints_digits_beyond_six_agents() -> None:
    drivers = [{"feature": "vessel_count", "agent": "shipping", "direction": "elevated"}]
    text = core._compose_narrative(drivers, 8, "High", "Panama Canal", n_agents=9)
    assert "8 of 9 detection agents" in text
    six = core._compose_narrative(drivers, 6, "High", "Panama Canal", n_agents=6)
    assert "six of six detection agents" in six


def test_no_fixed_agent_count_survives_outside_the_stored_results_section() -> None:
    """'6 vs 2 vs 1' / '6 → 2 → 1' belong only to the stored evaluation results (section 3)."""
    import re
    from pathlib import Path

    stored = ("### 3 · Agent diversity", "Agent-diversity value: 6 vs 2 vs 1 agents")
    path = Path(core.__file__).with_name("analysis_view.py")
    stray = [(i, line.strip()) for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
             if re.search(r"\b6\s*(vs|→)", line) and not any(s in line for s in stored)]
    assert not stray, stray


def test_the_end_date_text_is_the_evaluation_end_date() -> None:
    from src.evaluation.window import eval_window

    assert core.end_date_text() == f"{eval_window()['end_date']:%Y-%m-%d}"
