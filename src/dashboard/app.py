"""Supply Chain DSS — Streamlit dashboard (Phase 9b, final deliverable).

Two tabs:

* **Live Monitoring** — risk gauge, 6-agent signal timeline, SHAP explanation
  panel, CesiumJS 3-D geospatial view of the four chokepoints (SRQ4), a
  rule-based decision-support panel (SRQ4 "assess possible actions"), the
  agent status grid, and RAG historical context for the selected day.
* **Analysis** — an interactive, time-range-driven explorer over the Phase 9a
  evaluation metrics (detection trend, lead time, decision effectiveness,
  agent diversity, RAG similarity) with CSV export.

Data model: the full 365-day synthetic test realisation (seed 44 — the same
held-out split every Phase 9a number is reported on) is scored once per weight
mode and cached with ``st.cache_data``; the sidebar day slider and analysis
range selector index into it.  The optional "Run Live" checkbox re-runs the
real pipeline (preferring the FastAPI service when it is up, falling back to
an in-process ``Orchestrator``) instead of using the cache.

All Streamlit *runtime* calls live inside functions so this module can be
imported by pytest without a running Streamlit session; ``main()`` executes
only under ``streamlit run``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# Make ``src`` importable when Streamlit runs this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.decision_effectiveness import (  # noqa: E402
    ACTIONS,
    load_decision_labels,
    predict_action,
    scenario_correct_action,
)

try:  # same .env pattern as the extractors — never hardcode keys
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
_KB_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "disruption_cases.json"
_LABELS_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "decision_labels.json"
_EVAL_RESULTS_PATH = _PROJECT_ROOT / "data" / "processed" / "evaluation_results.json"

ALL_AGENTS: list[str] = [
    "shipping", "market", "geopolitical",
    "natural_disaster", "routing", "news_sentiment",
]

#: Spec palette — risk colors (green/amber/red) are reserved for risk only.
AGENT_COLORS: dict[str, str] = {
    "shipping": "#1f77b4",          # blue
    "market": "#2ca02c",            # green
    "geopolitical": "#d62728",      # red
    "natural_disaster": "#ff7f0e",  # orange
    "routing": "#9467bd",           # purple
    "news_sentiment": "#17becf",    # teal
}

RISK_COLORS = {"low": "#2e9e5b", "medium": "#d9a514", "high": "#d64545"}

#: Injected scenario windows (0-based day indices, inclusive).
SCENARIOS = {
    "A — Moderate Tension": (60, 74),
    "B — Major Blockage": (150, 170),
    "C — Brief Incident": (280, 290),
}

_TOTAL_DAYS = 365

#: Time-range presets for the Analysis tab, mapped onto trailing day-index
#: windows of the 365-day synthetic year (documented in the tab caption).
RANGE_PRESETS = ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "Custom range"]

# Cesium — pinned stable release (monthly cadence; update deliberately).
_CESIUM_VERSION = "1.130"

GLOBE_FALLBACK_MESSAGE = (
    "Set CESIUM_ION_TOKEN in .env to enable the 3D globe. "
    "Get a free token at https://ion.cesium.com/tokens."
)


# ===========================================================================
# Pure helpers (importable + unit-testable without a Streamlit runtime)
# ===========================================================================


def preset_to_range(preset: str, custom: tuple[int, int] | None = None) -> tuple[int, int]:
    """Map an Analysis-tab preset to an inclusive 0-based day-index window.

    The underlying dataset is one 365-day synthetic year, so "Last N days"
    means the trailing N day-indices of that year (e.g. "Last 30 days" =
    days 336–365 one-based). "Custom range" uses the supplied 1-based tuple.
    """
    if preset == "Last 30 days":
        return _TOTAL_DAYS - 30, _TOTAL_DAYS - 1
    if preset == "Last 90 days":
        return _TOTAL_DAYS - 90, _TOTAL_DAYS - 1
    if preset == "Last 6 months":
        return _TOTAL_DAYS - 183, _TOTAL_DAYS - 1
    if preset == "Last year":
        return 0, _TOTAL_DAYS - 1
    if preset == "Custom range" and custom is not None:
        lo = max(1, min(custom)) - 1
        hi = min(_TOTAL_DAYS, max(custom)) - 1
        return lo, hi
    return 0, _TOTAL_DAYS - 1


def get_monitoring_points(config: dict) -> dict[str, dict]:
    """First monitoring point per chokepoint from settings.yaml.

    Returns ``{region: {"name", "lat", "lng"}}`` for hormuz / red_sea /
    malacca / suez — the same block ``DisasterConnector.fetch_api()`` uses.
    """
    points_cfg = (
        config.get("agents", {}).get("natural_disaster", {}).get("monitoring_points", {}) or {}
    )
    out: dict[str, dict] = {}
    for region, points in points_cfg.items():
        if points:
            p = points[0]
            out[region] = {"name": str(p["name"]), "lat": float(p["lat"]), "lng": float(p["lng"])}
    return out


def resolve_cesium_token() -> str:
    """Read the Cesium ion token from the environment (via .env). Never logged."""
    return os.environ.get("CESIUM_ION_TOKEN", "").strip()


_GLOBE_TEMPLATE = Template("""<!DOCTYPE html>
<html><head>
<script src="https://cesium.com/downloads/cesiumjs/releases/$VERSION/Build/Cesium/Cesium.js"></script>
<link href="https://cesium.com/downloads/cesiumjs/releases/$VERSION/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<style>
  html, body, #cesiumContainer { width:100%; height:100%; margin:0; padding:0; overflow:hidden; background:#0b0e14; }
  .flybar { position:absolute; top:10px; left:10px; z-index:10; display:flex; gap:6px; }
  .flybar button {
    background:#141a24; color:#c8d1dc; border:1px solid #2a3442; border-radius:4px;
    padding:5px 12px; font: 12px 'Segoe UI', sans-serif; cursor:pointer; letter-spacing:.4px;
  }
  .flybar button:hover { background:#1d2634; border-color:#3d4b5e; }
  .flybar button.primary { border-color:#d9a514; color:#e8d9a0; }
</style></head>
<body>
<div id="cesiumContainer"></div>
<div class="flybar">$BUTTONS</div>
<script>
Cesium.Ion.defaultAccessToken = "$TOKEN";
const DATA = $DATA;

const viewer = new Cesium.Viewer("cesiumContainer", {
  terrain: Cesium.Terrain.fromWorldTerrain(),                                        // Cesium World Terrain (asset 1)
  baseLayer: Cesium.ImageryLayer.fromProviderAsync(Cesium.IonImageryProvider.fromAssetId(2)),  // Bing Maps Aerial (asset 2)
  animation: false, timeline: false, geocoder: false, homeButton: false,
  sceneModePicker: false, navigationHelpButton: false, baseLayerPicker: false,
  fullscreenButton: true, infoBox: true, selectionIndicator: true,
});
viewer.scene.globe.enableLighting = false;

// Primary marker — current-focus chokepoint, color-coded by composite risk.
const focus = DATA.focus;
viewer.entities.add({
  name: focus.name,
  position: Cesium.Cartesian3.fromDegrees(focus.lng, focus.lat),
  point: { pixelSize: 16, color: Cesium.Color.fromCssColorString(DATA.risk_color),
           outlineColor: Cesium.Color.WHITE, outlineWidth: 2,
           disableDepthTestDistance: Number.POSITIVE_INFINITY },
  label: { text: focus.name, font: "13px 'Segoe UI', sans-serif",
           fillColor: Cesium.Color.fromCssColorString("#e8edf2"),
           pixelOffset: new Cesium.Cartesian2(0, -24),
           disableDepthTestDistance: Number.POSITIVE_INFINITY },
  description: DATA.popup_html,     // click -> Cesium infoBox popup
});

// Secondary, dimmer markers — architecture supports all 4 chokepoints.
for (const p of DATA.secondary) {
  viewer.entities.add({
    name: p.name,
    position: Cesium.Cartesian3.fromDegrees(p.lng, p.lat),
    point: { pixelSize: 9, color: Cesium.Color.fromCssColorString("#5a6b80").withAlpha(0.9),
             outlineColor: Cesium.Color.fromCssColorString("#2a3442"), outlineWidth: 1,
             disableDepthTestDistance: Number.POSITIVE_INFINITY },
    description: "<div style='font-family:sans-serif'>" + p.name +
                 "<br><small>Monitored chokepoint — not the current analytical focus.</small></div>",
  });
}

function flyTo(lng, lat, height) {
  viewer.camera.flyTo({ destination: Cesium.Cartesian3.fromDegrees(lng, lat, height), duration: 2.0 });
}
// Default camera: the Strait of Hormuz at regional zoom, not full-globe.
flyTo(focus.lng, focus.lat, 320000.0);
</script>
</body></html>""")


def build_globe_html(
    token: str,
    focus: dict,
    secondary: list[dict],
    risk_level: str,
    top_driver: str,
    updated: str,
) -> str:
    """Build the CesiumJS iframe HTML, or the fallback placeholder without a token.

    The token is injected into the rendered HTML only — never logged or printed.
    """
    if not token:
        return (
            "<div style=\"height:100%;display:flex;align-items:center;justify-content:center;"
            "background:#10151d;color:#8d99a8;font:14px 'Segoe UI',sans-serif;border:1px dashed #2a3442;"
            f"border-radius:6px;padding:24px;text-align:center;\">{GLOBE_FALLBACK_MESSAGE}</div>"
        )

    popup_html = (
        "<div style='font-family:sans-serif;line-height:1.6'>"
        f"<b>Risk level:</b> {risk_level.upper()}<br>"
        f"<b>Top SHAP driver:</b> {top_driver}<br>"
        f"<b>Last updated:</b> {updated}</div>"
    )
    data = {
        "focus": focus,
        "secondary": secondary,
        "risk_color": RISK_COLORS.get(risk_level, RISK_COLORS["low"]),
        "popup_html": popup_html,
    }
    buttons = "".join(
        f"<button class=\"{'primary' if p['name'] == focus['name'] else ''}\" "
        f"onclick=\"flyTo({p['lng']},{p['lat']},320000)\">{p['label']}</button>"
        for p in [{**focus, "label": "Hormuz"}]
        + [{**s, "label": s["region"].replace("_", " ").title()} for s in secondary]
    )
    return _GLOBE_TEMPLATE.substitute(
        VERSION=_CESIUM_VERSION,
        TOKEN=token,
        DATA=json.dumps(data),
        BUTTONS=buttons,
    )


def classify_level(score: float, high_thr: float, med_thr: float) -> str:
    if score >= high_thr:
        return "high"
    if score >= med_thr:
        return "medium"
    return "low"


def decide_action(
    risk_level: str,
    top_driver_agent: str,
    sustained: bool,
    historical_context: dict | None,
) -> tuple[str, str]:
    """Call the Phase 9a rule-based mapper and name the rule that fired.

    Returns ``(action, rule_text)`` — the rule text is displayed verbatim in
    the decision panel so the recommendation is never a black box.
    """
    drivers = [{"agent": top_driver_agent}] if top_driver_agent else []
    action = predict_action(risk_level, drivers, historical_context, sustained=sustained)

    level = str(risk_level).lower()
    if level == "low":
        rule = "risk_level == low → no_action"
    elif level == "medium":
        rule = "risk_level == medium → monitor"
    elif sustained:
        rule = "risk HIGH for > 5 consecutive days (sustained) → escalate"
    elif top_driver_agent in ("routing", "shipping"):
        rule = f"risk HIGH ∧ top driver agent = {top_driver_agent} (physical) → reroute"
    elif action == "escalate":
        rule = (
            f"risk HIGH ∧ top driver agent = {top_driver_agent} ∧ "
            "historical match > 0.75 similarity to an escalate-labelled case → escalate"
        )
    else:
        rule = "risk HIGH, no physical driver or escalate precedent → monitor (fallback)"
    return action, rule


# ===========================================================================
# Cached data layer
# ===========================================================================


@st.cache_data(show_spinner=False)
def load_app_config() -> dict:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def _layout_to_params(layout: dict) -> dict:
    return {
        "inter_weights": {k: float(v) for k, v in layout["inter_agent_weights"].items()},
        "intra": {k: dict(v) for k, v in layout["intra_agent_weights"].items()},
        "thresholds": {k: float(v) for k, v in layout["thresholds"].items()},
    }


def resolve_mode_params(weight_mode: str) -> dict:
    """Resolve a weight mode to PipelineEvaluator params (hand-tuned fallback)."""
    from src.optimization.weight_config import load_optimized_weights, resolve_active_weights

    config = load_app_config()
    if weight_mode == "optimized":
        opt = load_optimized_weights(config)
        if opt and opt.get("inter_agent_weights"):
            return _layout_to_params(opt)
    return _layout_to_params(resolve_active_weights({**config, "weight_mode": "hand_tuned"}))


@st.cache_resource(show_spinner="Generating train/test splits …")
def _eval_bundle():
    """Singleton DataSplitManager + PipelineEvaluator (splits generated once)."""
    from src.optimization.data_split import DataSplitManager
    from src.optimization.pipeline_evaluator import PipelineEvaluator

    config = load_app_config()
    dm = DataSplitManager(config)
    dm.get_splits()
    evaluator = PipelineEvaluator(dm, config.get("optimization", {}).get("objective_weights"))
    gt = dm.get_ground_truth("test").reset_index(drop=True)
    return {"dm": dm, "evaluator": evaluator, "gt": gt}


@st.cache_data(show_spinner="Scoring the 365-day timeline …")
def compute_timeseries(weight_mode: str) -> pd.DataFrame:
    """Per-agent anomaly scores on the held-out TEST split (seed 44).

    Agents are fitted on the train realisation (seed 42) and scored on test —
    identical to how every Phase 9a evaluation number is produced, so the
    dashboard and the thesis tables agree by construction.
    """
    bundle = _eval_bundle()
    evaluator, dm = bundle["evaluator"], bundle["dm"]
    params = resolve_mode_params(weight_mode)
    splits = dm.get_splits()
    agents = evaluator.build_agents(params)

    frames: dict[str, np.ndarray] = {}
    n = _TOTAL_DAYS
    for name, agent in agents.items():
        try:
            agent.fit(splits["train"][name])
            validated = agent.run_dataframe(splits["test"][name])
            s = validated["anomaly_score"].to_numpy(dtype=float)
            padded = np.zeros(n)
            padded[-len(s):] = s[:n]  # market drops a warm-up row; right-align
            frames[name] = padded
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agent %s failed while scoring: %s", name, exc)
            frames[name] = np.zeros(n)

    df = pd.DataFrame(frames)
    df["is_disruption"] = bundle["gt"].reindex(range(n)).fillna(False).astype(bool).to_numpy()
    df["day"] = np.arange(1, n + 1)  # 1-based, matches the sidebar slider
    return df


def composite_series(ts: pd.DataFrame, params: dict, enabled: set[str]) -> pd.Series:
    """Composite risk from cached agent scores (aggregation only — no refit).

    Mirrors ``PipelineEvaluator._aggregate_daily``: renormalised weighted mean
    over the enabled agents plus the non-linear agreement bonus. Disabling an
    agent zeroes its weight — the pipeline's own toggle semantics.
    """
    inter = params["inter_weights"]
    thr = params["thresholds"]
    cols = [a for a in ALL_AGENTS if a in enabled and inter.get(a, 0.0) > 0]
    if not cols:
        return pd.Series(np.zeros(len(ts)))
    w_total = sum(float(inter[c]) for c in cols)
    base = np.zeros(len(ts))
    for c in cols:
        base += (float(inter[c]) / w_total) * ts[c].to_numpy()
    agreement = (ts[cols].to_numpy() > 0.5).sum(axis=1)
    amp = np.where(
        agreement >= 5, float(thr.get("agreement_bonus_5", 1.25)),
        np.where(agreement >= 3, float(thr.get("agreement_bonus_3", 1.15)), 1.0),
    )
    return pd.Series(np.minimum(base * amp, 1.0))


@st.cache_resource(show_spinner="Training the SHAP surrogate …")
def get_shap_assets():
    """(features_df, risk_scores, trained SurrogateShapExplainer) — cached once."""
    from src.explainability.shap_explainer import (
        SurrogateShapExplainer,
        build_shap_training_data,
    )

    config = load_app_config()
    features_df, risk_scores = build_shap_training_data(config)
    explainer = SurrogateShapExplainer()
    explainer.train_surrogate(features_df, risk_scores)
    return features_df, risk_scores, explainer


@st.cache_resource(show_spinner="Loading the RAG knowledge base …")
def get_retriever():
    from src.rag.context_retriever import ContextRetriever

    config = load_app_config()
    rag_cfg = dict(config.get("rag", {}) or {})
    rag_cfg.setdefault("collection_name", "disruption_cases")
    retriever = ContextRetriever(rag_cfg)
    retriever.build_index(str(_KB_PATH))
    return retriever


@st.cache_data(show_spinner=False)
def load_cases_by_id() -> dict[str, dict]:
    cases = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    return {str(c["id"]): c for c in cases}


@st.cache_data(show_spinner=False)
def load_eval_results() -> dict:
    if _EVAL_RESULTS_PATH.exists():
        return json.loads(_EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def sustained_high_flags(risk: pd.Series, high_thr: float, min_run: int = 5) -> np.ndarray:
    """Boolean per day: risk has been HIGH for > ``min_run`` consecutive days."""
    out = np.zeros(len(risk), dtype=bool)
    run = 0
    for i, v in enumerate(risk.to_numpy()):
        run = run + 1 if v >= high_thr else 0
        out[i] = run > min_run
    return out


def run_live_pipeline(data_modes: dict[str, str]) -> tuple[dict, str]:
    """Run the real pipeline once — FastAPI first, in-process fallback.

    Returns ``(result_dict, source_label)``. The FastAPI attempt keeps the
    dashboard consistent with the Prompt-1 service when it is running, without
    making it a hard dependency.
    """
    try:
        import requests

        base = "http://127.0.0.1:8000"
        if requests.get(f"{base}/health", timeout=1.5).status_code == 200:
            r = requests.post(f"{base}/predict", json={}, timeout=120)
            r.raise_for_status()
            return r.json(), "FastAPI service (127.0.0.1:8000)"
    except Exception:
        pass  # service not running — use the in-process pipeline

    from src.orchestrator import Orchestrator

    config = json.loads(json.dumps(load_app_config()))  # deep copy
    ingestion = config.setdefault("ingestion", {})
    agents_cfg = config.setdefault("agents", {})
    for name, mode in data_modes.items():
        if name in ("shipping", "market"):
            ingestion.setdefault(name, {})["source_mode"] = mode
        else:
            agents_cfg.setdefault(name, {})["data_mode"] = mode
    return Orchestrator(config).run_full_pipeline(), "in-process Orchestrator"


# ===========================================================================
# Analysis-tab computations (cached per selected range)
# ===========================================================================


@st.cache_data(show_spinner="Computing metrics for the selected range …")
def analysis_bundle(start: int, end: int) -> dict:
    """All five range-filtered analyses for days [start, end] (0-based, incl.)."""
    from src.optimization.pipeline_evaluator import PipelineEvaluator

    out: dict = {"start": start, "end": end}
    window = 30  # rolling-metric window (days)

    per_mode: dict[str, dict] = {}
    for mode in ("hand_tuned", "optimized"):
        ts = compute_timeseries(mode)
        params = resolve_mode_params(mode)
        risk = composite_series(ts, params, set(ALL_AGENTS))
        gt = ts["is_disruption"].to_numpy()
        high_thr = params["thresholds"]["risk_high"]
        med_thr = params["thresholds"]["risk_medium"]
        pred_high = risk.to_numpy() >= high_thr
        alert_med = risk.to_numpy() >= med_thr

        # 1 — rolling detection metrics over the full year, sliced to range
        roll = {"day": [], "f1": [], "precision": [], "recall": []}
        for i in range(start, end + 1):
            lo = max(0, i - window + 1)
            yt, pr = gt[lo:i + 1], pred_high[lo:i + 1]
            tp = int((yt & pr).sum()); fp = int((~yt & pr).sum()); fn = int((yt & ~pr).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            roll["day"].append(i + 1)
            roll["precision"].append(p)
            roll["recall"].append(r)
            roll["f1"].append(2 * p * r / (p + r) if p + r else 0.0)

        # 2 — lead time per scenario window intersecting the range
        leads: dict[str, float] = {}
        for label, (s, e) in SCENARIOS.items():
            if e < start or s > end:
                continue
            lead = 0
            for d in range(max(0, s - 5), s + 1):
                if alert_med[d]:
                    lead = s - d
                    break
            leads[label] = float(min(lead, 5))

        # 3 — decision effectiveness: per-day predicted vs correct action
        sustained = sustained_high_flags(risk, high_thr)
        inter = params["inter_weights"]
        contrib = np.column_stack([ts[a].to_numpy() * float(inter.get(a, 0.0)) for a in ALL_AGENTS])
        top_idx = contrib.argmax(axis=1)
        correct_flags = np.zeros(len(risk), dtype=bool)
        high_run = 0
        for i in range(len(risk)):
            level = classify_level(float(risk[i]), high_thr, med_thr)
            high_run = high_run + 1 if level == "high" else 0
            corr, _ = scenario_correct_action(i, high_run_length=high_run)
            pred = predict_action(level, [{"agent": ALL_AGENTS[top_idx[i]]}], None,
                                  sustained=bool(sustained[i]))
            correct_flags[i] = pred == corr
        dec_roll = [float(correct_flags[max(0, i - window + 1):i + 1].mean())
                    for i in range(start, end + 1)]

        # 4 — agent-diversity F1 within range (aggregation-only masking)
        diversity: dict[str, float] = {}
        for cfg_name, active in (("6-agent", set(ALL_AGENTS)),
                                 ("2-agent", {"shipping", "market"}),
                                 ("1-agent", {"shipping"})):
            r_m = composite_series(ts, params, active).to_numpy()[start:end + 1] >= high_thr
            yt = gt[start:end + 1]
            tp = int((yt & r_m).sum()); fp = int((~yt & r_m).sum()); fn = int((yt & ~r_m).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            diversity[cfg_name] = 2 * p * rc / (p + rc) if p + rc else 0.0

        per_mode[mode] = {
            "rolling": roll,
            "leads": leads,
            "decision_rolling": dec_roll,
            "decision_accuracy": float(correct_flags[start:end + 1].mean()),
            "diversity": diversity,
            "range_f1": roll["f1"][-1] if roll["f1"] else 0.0,
        }
    out["modes"] = per_mode

    # 5 — RAG top-1 similarity sampled through the range (stride-capped)
    try:
        retriever = get_retriever()
        ts_ht = compute_timeseries("hand_tuned")
        stride = max(1, (end - start + 1) // 40)
        days, sims = [], []
        for i in range(start, end + 1, stride):
            signals = {a: float(ts_ht[a].iloc[i]) for a in ALL_AGENTS}
            matches = retriever.query(signals, top_k=1)
            days.append(i + 1)
            sims.append(float(matches[0]["similarity"]) if matches else 0.0)
        out["rag"] = {"day": days, "similarity": sims, "stride": stride}
    except Exception as exc:  # pragma: no cover - chromadb/env dependent
        logger.warning("RAG trend unavailable: %s", exc)
        out["rag"] = {"day": [], "similarity": [], "stride": 1}

    _ = PipelineEvaluator  # imported for parity documentation; aggregation mirrors it
    return out


# ===========================================================================
# UI building blocks
# ===========================================================================

_CSS = """
<style>
  /* Analytical, muted-dark look — risk colors reserved for risk indicators. */
  .stApp { background: #0e1117; }
  h1, h2, h3 { color: #dfe6ee; letter-spacing: .3px; }
  h1 { font-size: 1.55rem !important; }
  section[data-testid="stSidebar"] { background: #12161f; border-right: 1px solid #222b38; }
  div[data-testid="stMetric"] {
    background: #141a24; border: 1px solid #222b38; border-radius: 6px; padding: 10px 14px;
  }
  .risk-badge {
    display: inline-block; padding: 6px 18px; border-radius: 4px; font-weight: 700;
    font-size: 1.05rem; letter-spacing: 1.2px; color: #0b0e14;
  }
  .action-chip {
    display: inline-block; padding: 10px 26px; border-radius: 4px; font-weight: 700;
    font-size: 1.15rem; letter-spacing: 1.5px; color: #0b0e14;
  }
  .agent-card {
    background: #141a24; border: 1px solid #222b38; border-radius: 6px;
    padding: 12px 14px; margin-bottom: 10px; font-size: .85rem; color: #aeb9c6;
  }
  .agent-card b { color: #dfe6ee; font-size: .95rem; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
  .tag { background:#1d2634; border:1px solid #2a3442; border-radius:3px;
         padding:1px 7px; font-size:.72rem; color:#8d99a8; margin-left:6px; }
  .driver-card {
    background:#141a24; border:1px solid #222b38; border-left:3px solid #d9a514;
    border-radius:4px; padding:8px 12px; margin-bottom:8px; font-size:.85rem; color:#aeb9c6;
  }
  .rule-box {
    background:#10151d; border:1px dashed #2a3442; border-radius:4px;
    padding:8px 12px; font-family: Consolas, monospace; font-size:.8rem; color:#8d99a8;
  }
</style>
"""


def _badge(text: str, color: str, cls: str = "risk-badge") -> str:
    return f'<span class="{cls}" style="background:{color}">{text}</span>'


def render_overview(risk_score: float, risk_level: str, fired: int, enabled: int,
                    weight_mode: str, high_thr: float, med_thr: float, live_src: str | None):
    """SECTION A — gauge, badges, counts."""
    import plotly.graph_objects as go

    c1, c2 = st.columns([1.1, 2.2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(risk_score, 3),
            number={"font": {"size": 34, "color": "#dfe6ee"}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#556270"},
                "bar": {"color": "#dfe6ee", "thickness": 0.22},
                "bgcolor": "#141a24",
                "steps": [
                    {"range": [0, med_thr], "color": "#1d4a30"},
                    {"range": [med_thr, high_thr], "color": "#4a3d10"},
                    {"range": [high_thr, 1.0], "color": "#4a1d1d"},
                ],
            },
        ))
        fig.update_layout(height=210, margin=dict(l=18, r=18, t=28, b=6),
                          paper_bgcolor="rgba(0,0,0,0)", font={"color": "#aeb9c6"})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        b1, b2, b3, b4 = st.columns(4)
        b1.markdown("**Risk level**<br>" + _badge(risk_level.upper(), RISK_COLORS[risk_level]),
                    unsafe_allow_html=True)
        b2.metric("Agents firing", f"{fired} / {enabled}")
        b3.markdown(
            "**Weight mode**<br>"
            + _badge(weight_mode.replace("_", "-"), "#3d4b5e", "risk-badge"),
            unsafe_allow_html=True,
        )
        b4.metric("Assessed", datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))
        if live_src:
            st.caption(f"Live pipeline result via **{live_src}**.")
        else:
            st.caption("Cached 365-day synthetic test realisation (seed 44) — the same "
                       "held-out split all Phase 9a evaluation numbers are reported on.")


def render_timeline(ts: pd.DataFrame, risk: pd.Series, enabled: set[str], day: int):
    """SECTION B — 6-agent time series + composite + scenario shading."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for name in ALL_AGENTS:
        if name not in enabled:
            continue
        fig.add_trace(go.Scatter(
            x=ts["day"], y=ts[name], name=name.replace("_", " "),
            line=dict(color=AGENT_COLORS[name], width=1.1), opacity=0.75,
        ))
    fig.add_trace(go.Scatter(
        x=ts["day"], y=risk, name="composite risk",
        line=dict(color="#dfe6ee", width=3.0),
    ))
    for label, (s, e) in SCENARIOS.items():
        fig.add_vrect(x0=s + 1, x1=e + 1, fillcolor="#8d99a8", opacity=0.10, line_width=0,
                      annotation_text=label.split(" — ")[0], annotation_font_color="#8d99a8",
                      annotation_position="top left")
    fig.add_vline(x=day, line=dict(color="#d9a514", width=1.4, dash="dot"))
    fig.update_layout(
        height=420, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12, font=dict(size=11)),
        xaxis_title="day of synthetic year", yaxis_title="anomaly / risk score",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_panel(day: int, risk_score: float, risk_level: str, weight_mode: str):
    """SECTION C — waterfall for the selected day + text + top-3 driver cards.

    Returns the top-driver agent name (feeds the decision + globe panels).
    """
    import plotly.graph_objects as go

    features_df, _, explainer = get_shap_assets()
    idx = min(day - 1, len(features_df) - 1)
    shap_result = explainer.explain(features_df.iloc[[idx]])

    vals = shap_result["shap_values"]
    ordered = sorted(vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    feats = [k.replace("_", " ") for k, _ in ordered][::-1]
    v = [x for _, x in ordered][::-1]
    fig = go.Figure(go.Bar(
        x=v, y=feats, orientation="h",
        marker_color=["#d64545" if x > 0 else "#1f77b4" for x in v],
    ))
    fig.add_vline(x=0, line=dict(color="#556270", width=1))
    fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=26, b=10),
                      title=dict(text=f"SHAP waterfall — day {day}", font=dict(size=13)),
                      xaxis_title="SHAP value", font=dict(size=11))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    text = explainer.generate_explanation_text(risk_score, risk_level, weight_mode, shap_result)
    st.markdown(f"<div class='rule-box'>{text}</div>", unsafe_allow_html=True)

    for d in shap_result["top_drivers"]:
        color = AGENT_COLORS.get(d["agent"], "#8d99a8")
        st.markdown(
            f"<div class='driver-card'><b style='color:{color}'>{d['feature'].replace('_', ' ')}</b>"
            f"<span class='tag'>{d['agent'].replace('_', ' ')}</span>"
            f"<span style='float:right;color:#dfe6ee'>{d['shap_value']:+.4f}</span></div>",
            unsafe_allow_html=True,
        )
    return shap_result["top_drivers"][0] if shap_result["top_drivers"] else {"agent": "", "feature": ""}


def render_globe(config: dict, risk_level: str, top_driver: str):
    """SECTION D — CesiumJS 3-D globe (SRQ4 geospatial mapping)."""
    import streamlit.components.v1 as components

    points = get_monitoring_points(config)
    token = resolve_cesium_token()
    hormuz = points.get("hormuz", {"name": "Strait of Hormuz", "lat": 26.56, "lng": 56.25})
    secondary = [
        {**points[r], "region": r} for r in ("red_sea", "malacca", "suez") if r in points
    ]
    html = build_globe_html(
        token=token,
        focus=hormuz,
        secondary=secondary,
        risk_level=risk_level,
        top_driver=top_driver,
        updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    components.html(html, height=430)
    if token:
        st.caption(
            "CesiumJS World Terrain + Bing Aerial. The bright marker is the current focus "
            "(color = composite risk); dim markers are the other monitored chokepoints — "
            "use the buttons on the globe to fly between them. Click a marker for details."
        )


def render_decision_panel(action: str, rule: str, rationale: str):
    """SECTION E — rule-based action recommendation (SRQ4 'assess possible actions')."""
    chip_colors = {"no_action": "#3d4b5e", "monitor": "#d9a514",
                   "reroute": "#d97b14", "escalate": "#d64545"}
    label = action.replace("_", " ").upper()
    st.markdown(_badge(label, chip_colors.get(action, "#3d4b5e"), "action-chip"),
                unsafe_allow_html=True)
    st.markdown(f"**Rationale:** {rationale}")
    st.markdown(f"<div class='rule-box'>rule fired: {rule}</div>", unsafe_allow_html=True)
    st.caption(
        "⚖️ This is a **transparent rule-based recommendation** (see "
        "`src/evaluation/decision_effectiveness.py`), not an ML decision — it supports "
        "human judgement, it does not replace it. Decision-effectiveness accuracy of "
        "these rules on the synthetic year: see the Analysis tab."
    )


def render_agent_grid(ts: pd.DataFrame, day: int, params: dict, enabled: set[str],
                      data_modes: dict[str, str]):
    """SECTION F — 3×2 agent status cards."""
    inter = params["inter_weights"]
    cols = st.columns(3)
    for i, name in enumerate(ALL_AGENTS):
        score = float(ts[name].iloc[day - 1])
        on = name in enabled
        dot = "#2e9e5b" if (on and score <= 0.5) else ("#d64545" if on else "#556270")
        status = "FIRING" if (on and score > 0.5) else ("ok" if on else "disabled")
        with cols[i % 3]:
            st.markdown(
                f"<div class='agent-card'>"
                f"<span class='dot' style='background:{dot}'></span>"
                f"<b>{name.replace('_', ' ')}</b>"
                f"<span class='tag'>{data_modes.get(name, 'synthetic')}</span>"
                f"<span class='tag'>w={float(inter.get(name, 0.0)):.3f}</span><br>"
                f"score <b style='color:{AGENT_COLORS[name]}'>{score:.3f}</b>"
                f" &nbsp;·&nbsp; {status}</div>",
                unsafe_allow_html=True,
            )


def render_rag_context(ts: pd.DataFrame, day: int) -> dict | None:
    """SECTION G — RAG matches table. Returns the top match for the decision panel."""
    try:
        retriever = get_retriever()
    except Exception as exc:  # pragma: no cover
        st.info(f"RAG knowledge base unavailable: {exc}")
        return None

    signals = {a: float(ts[a].iloc[day - 1]) for a in ALL_AGENTS}
    matches = retriever.query(signals, top_k=3)
    if not matches:
        st.info("No historical precedents retrieved for this signal profile.")
        return None

    cases = load_cases_by_id()
    labels = load_decision_labels(_LABELS_PATH)
    rows = []
    for m in matches:
        case = cases.get(m["id"], {})
        agents_raw = m.get("metadata", {}).get("primary_agents", "[]")
        try:
            agents = json.loads(agents_raw) if isinstance(agents_raw, str) else list(agents_raw)
        except (json.JSONDecodeError, TypeError):
            agents = []
        rows.append({
            "Event": m.get("metadata", {}).get("event", m["id"]),
            "Similarity": round(float(m.get("similarity", 0.0)), 3),
            "Relevant agents": ", ".join(agents),
            "Labelled action": labels.get(m["id"], "—"),
            "Impact": (case.get("impact", "") or "")[:160],
            "Lesson": (case.get("lessons", "") or "")[:160],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    top = matches[0]
    return {
        "similarity": float(top.get("similarity", 0.0)),
        "action": labels.get(top["id"], ""),
        "event": top.get("metadata", {}).get("event", top["id"]),
    }


# ===========================================================================
# Analysis tab
# ===========================================================================


def render_analysis_tab():
    """SECTION H — interactive, range-driven evaluation explorer."""
    import plotly.graph_objects as go

    st.subheader("Evaluation explorer")
    st.caption(
        "Presets map onto **trailing day-index windows of the 365-day synthetic year** "
        "(e.g. “Last 30 days” = days 336–365, “Last 6 months” = days 183–365) — the "
        "underlying data is the seed-44 held-out test realisation, not calendar time."
    )

    c1, c2, c3 = st.columns([1.4, 1.6, 1])
    preset = c1.selectbox("Time range", RANGE_PRESETS, index=3)
    custom = None
    if preset == "Custom range":
        custom = c2.slider("Custom day range", 1, _TOTAL_DAYS, (100, 200))
    start, end = preset_to_range(preset, custom)
    c3.markdown(f"<br>**days {start + 1} – {end + 1}**", unsafe_allow_html=True)

    if c3.button("Recompute for this range", type="primary"):
        st.session_state["analysis_range"] = (start, end)
    if "analysis_range" not in st.session_state:
        st.session_state["analysis_range"] = (0, _TOTAL_DAYS - 1)

    a_start, a_end = st.session_state["analysis_range"]
    bundle = analysis_bundle(a_start, a_end)
    modes = bundle["modes"]
    eval_json = load_eval_results()
    baseline_acc = (
        eval_json.get("metric_8_decision_effectiveness", {})
        .get("hand_tuned", {}).get("baseline_accuracy", 0.0)
    )
    mode_colors = {"hand_tuned": "#1f77b4", "optimized": "#ff7f0e"}
    st.markdown(f"**Showing days {a_start + 1} – {a_end + 1}** "
                f"({a_end - a_start + 1} days)")

    # --- 1. Detection performance trend -----------------------------------
    st.markdown("##### 1 · Detection performance over time (30-day rolling)")
    fig = go.Figure()
    for mode, res in modes.items():
        r = res["rolling"]
        fig.add_trace(go.Scatter(x=r["day"], y=r["f1"], name=f"F1 · {mode}",
                                 line=dict(color=mode_colors[mode], width=2)))
        fig.add_trace(go.Scatter(x=r["day"], y=r["precision"], name=f"precision · {mode}",
                                 line=dict(color=mode_colors[mode], width=1, dash="dot"),
                                 opacity=0.55))
        fig.add_trace(go.Scatter(x=r["day"], y=r["recall"], name=f"recall · {mode}",
                                 line=dict(color=mode_colors[mode], width=1, dash="dash"),
                                 opacity=0.55))
    fig.update_layout(height=320, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="day", yaxis_title="score", hovermode="x unified",
                      legend=dict(orientation="h", y=1.15, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    # --- 2. Lead time per scenario -----------------------------------------
    with col_a:
        st.markdown("##### 2 · Early-warning lead time per scenario")
        scen_labels = sorted({k for res in modes.values() for k in res["leads"]})
        if scen_labels:
            fig = go.Figure()
            for mode, res in modes.items():
                fig.add_trace(go.Bar(
                    x=[s.split(" — ")[0] for s in scen_labels],
                    y=[res["leads"].get(s, 0.0) for s in scen_labels],
                    name=mode, marker_color=mode_colors[mode],
                ))
            fig.update_layout(height=300, template="plotly_dark", barmode="group",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#10151d",
                              margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="lead time (days, cap 5)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scenario window falls inside the selected range.")

    # --- 3. Decision effectiveness trend ------------------------------------
    with col_b:
        st.markdown("##### 3 · Decision-effectiveness accuracy (30-day rolling)")
        fig = go.Figure()
        days_axis = list(range(a_start + 1, a_end + 2))
        for mode, res in modes.items():
            fig.add_trace(go.Scatter(x=days_axis, y=res["decision_rolling"],
                                     name=mode, line=dict(color=mode_colors[mode], width=2)))
        if baseline_acc:
            fig.add_hline(y=baseline_acc, line=dict(color="#8d99a8", width=1.4, dash="dash"),
                          annotation_text=f"naive baseline {baseline_acc:.3f}",
                          annotation_font_color="#8d99a8")
        fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="day", yaxis_title="action accuracy")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    # --- 4. Agent diversity --------------------------------------------------
    with col_c:
        st.markdown("##### 4 · Agent diversity (F1 within range)")
        cfgs = ["6-agent", "2-agent", "1-agent"]
        fig = go.Figure()
        for mode, res in modes.items():
            fig.add_trace(go.Bar(x=cfgs, y=[res["diversity"][c] for c in cfgs],
                                 name=mode, marker_color=mode_colors[mode]))
        fig.update_layout(height=300, template="plotly_dark", barmode="group",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#10151d",
                          margin=dict(l=10, r=10, t=10, b=10), yaxis_title="F1")
        st.plotly_chart(fig, use_container_width=True)

    # --- 5. RAG similarity trend ---------------------------------------------
    with col_d:
        st.markdown("##### 5 · RAG top-1 similarity through the range")
        rag = bundle["rag"]
        fig = go.Figure()
        if rag["day"]:
            fig.add_trace(go.Scatter(x=rag["day"], y=rag["similarity"], name="top-1 similarity",
                                     line=dict(color="#17becf", width=2)))
            fig.add_hline(y=0.6, line=dict(color="#8d99a8", width=1.2, dash="dash"),
                          annotation_text="0.60 target", annotation_font_color="#8d99a8")
        fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="day", yaxis_title="cosine similarity")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Sampled every {rag['stride']} day(s) against the static knowledge base.")

    # --- Summary table + CSV export -------------------------------------------
    st.markdown("##### Range summary")
    rows = []
    for mode, res in modes.items():
        rows.append({
            "mode": mode,
            "rolling_f1_at_range_end": round(res["range_f1"], 4),
            "decision_accuracy": round(res["decision_accuracy"], 4),
            "baseline_decision_accuracy": round(baseline_acc, 4),
            **{f"f1_{k}": round(v, 4) for k, v in res["diversity"].items()},
            **{f"lead_{k.split(' — ')[0]}": v for k, v in res["leads"].items()},
        })
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download summary as CSV",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"evaluation_summary_days_{a_start + 1}_{a_end + 1}.csv",
        mime="text/csv",
    )


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    st.set_page_config(
        page_title="Supply Chain DSS — Strait of Hormuz",
        page_icon="🛳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    config = load_app_config()

    # ------------------------------ Sidebar ------------------------------
    with st.sidebar:
        st.markdown("### Supply Chain DSS")
        st.caption("Multi-agent early-warning · Strait of Hormuz")
        day = st.slider("Day of synthetic year", 1, _TOTAL_DAYS, 155)
        weight_mode = st.radio("Weight mode", ["hand_tuned", "optimized"], horizontal=True)
        st.markdown("**Agents**")
        enabled = {a for a in ALL_AGENTS
                   if st.checkbox(a.replace("_", " "), value=True, key=f"tog_{a}")}
        params = resolve_mode_params(weight_mode)
        high_default = float(params["thresholds"]["risk_high"])
        high_thr = st.slider("HIGH-risk threshold", 0.30, 0.95, high_default, 0.01)
        med_thr = float(params["thresholds"]["risk_medium"])
        with st.expander("Per-agent data mode (live runs)"):
            st.caption("The cached timeline always uses the reproducible synthetic split; "
                       "these modes apply when 'Run Live' is checked.")
            data_modes = {a: st.selectbox(a, ["synthetic", "csv", "api"], key=f"dm_{a}")
                          for a in ALL_AGENTS}
        run_live = st.checkbox("Run Live (re-run real pipeline)")

    # ------------------------------ Data ------------------------------
    ts = compute_timeseries(weight_mode)
    risk = composite_series(ts, params, enabled)
    risk_score = float(risk.iloc[day - 1])
    risk_level = classify_level(risk_score, high_thr, med_thr)
    fired = int(sum(1 for a in enabled if float(ts[a].iloc[day - 1]) > 0.5))
    sustained = bool(sustained_high_flags(risk, high_thr)[day - 1])

    live_src = None
    if run_live:
        with st.spinner("Running the live pipeline …"):
            live_result, live_src = run_live_pipeline(data_modes)
        risk_score = float(live_result.get("risk_score", live_result.get("composite_score", 0.0)))
        risk_level = str(live_result.get("risk_level_label",
                                         live_result.get("risk_level", "low"))).lower()
        if risk_level not in RISK_COLORS:
            risk_level = classify_level(risk_score, high_thr, med_thr)

    tab_live, tab_analysis = st.tabs(["🛰  Live Monitoring", "📊  Analysis"])

    # ============================ TAB 1 ============================
    with tab_live:
        st.markdown(f"## Strait of Hormuz — day {day} risk assessment")
        render_overview(risk_score, risk_level, fired, len(enabled),
                        weight_mode, high_thr, med_thr, live_src)

        st.divider()
        col_b, col_c = st.columns([3, 2])   # ~60 / 40 split
        with col_b:
            st.markdown("#### Signal timeline")
            render_timeline(ts, risk, enabled, day)
        with col_c:
            st.markdown("#### Explanation")
            top_driver = render_shap_panel(day, risk_score, risk_level, weight_mode)

        st.divider()
        col_d, col_e = st.columns([3, 2])
        with col_d:
            st.markdown("#### Chokepoint geography")
            render_globe(config, risk_level,
                         top_driver.get("feature", "—").replace("_", " "))
        with col_e:
            st.markdown("#### Recommended action")
            hist_top = None
            try:
                signals = {a: float(ts[a].iloc[day - 1]) for a in ALL_AGENTS}
                matches = get_retriever().query(signals, top_k=1)
                if matches:
                    labels = load_decision_labels(_LABELS_PATH)
                    hist_top = {
                        "similarity": float(matches[0].get("similarity", 0.0)),
                        "action": labels.get(matches[0]["id"], ""),
                        "event": matches[0].get("metadata", {}).get("event", matches[0]["id"]),
                    }
            except Exception as exc:  # pragma: no cover
                logger.warning("RAG lookup for decision panel failed: %s", exc)

            action, rule = decide_action(
                risk_level, top_driver.get("agent", ""), sustained, hist_top,
            )
            rationale = f"{risk_level.upper()} risk"
            if sustained:
                rationale += ", sustained > 5 days"
            if top_driver.get("agent"):
                rationale += f", driven by {top_driver['agent'].replace('_', ' ')} signals"
            if hist_top and hist_top["similarity"] > 0.6:
                rationale += f" — similar to {hist_top['event']} (sim {hist_top['similarity']:.2f})"
            render_decision_panel(action, rule, rationale + ".")

        st.divider()
        st.markdown("#### Agent status")
        render_agent_grid(ts, day, params, enabled, data_modes)

        st.divider()
        st.markdown("#### Historical context (RAG)")
        render_rag_context(ts, day)

    # ============================ TAB 2 ============================
    with tab_analysis:
        render_analysis_tab()


if __name__ == "__main__":
    main()
