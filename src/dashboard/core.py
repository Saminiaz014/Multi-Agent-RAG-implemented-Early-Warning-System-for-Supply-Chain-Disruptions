"""Shared data layer + helpers for the two-page DSS dashboard.

Everything both pages consume lives here: the cached 365-day timeline, the
CesiumJS globe builder (now with routes + vessels), route definitions, the
plain-language status-word mapping used by the Decision view, the news feed,
the LLM/compositional risk narrative, and the JPEG export helper used by every
Analysis-view chart.

Design notes (flagged per the redesign spec):

* **Status words, not scores** — the Decision view maps composite risk onto
  three manager-facing words: ``Critical`` (>= the engine's ``risk_critical``
  threshold), ``High`` (>= ``risk_medium``) and ``Low``.  The numeric score
  never leaves this module on that page.
* **Routes** are presentational scopes over the *same* underlying agent
  signals: each route's trend is the pipeline's own renormalised weighted
  aggregation restricted to the agents most relevant to that corridor
  (exactly the mechanism the Phase 9a diversity ablation uses).  No new
  analytics are invented per route.
* **Vessel markers are representative synthetic records** (deterministic,
  seeded per route+day) used only for this UI: the shipping connector tracks
  daily aggregate counts, not per-vessel AIS tracks (``aisstream.enabled``
  is false).  Vessel types come from the connector's real CSV vessel-type
  columns.
* **LLM narrative** — no LLM client pattern existed anywhere in this codebase,
  so :func:`generate_risk_narrative` optionally calls the Anthropic API when
  ``ANTHROPIC_API_KEY`` is present (and the ``anthropic`` package installed)
  and otherwise composes the paragraph with a single generic algorithm from
  the live SHAP drivers — not per-feature-combination hardcoded strings.
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config_manager import load_config_for_region  # noqa: E402
from src.core.regions import REGIONS as _REGIONS  # noqa: E402
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

__all__ = [
    "ACTIONS", "predict_action", "ALL_AGENTS", "AGENT_COLORS", "RISK_COLORS",
    "STATUS_COLORS", "STATUS_ICONS", "SCENARIOS", "RANGE_PRESETS",
    "MAP_FALLBACK_MESSAGE", "AVAILABLE_REGIONS",
    "preset_to_range", "get_monitoring_points", "resolve_map_key",
    "build_map_html", "classify_level", "status_word", "decide_action",
    "load_app_config", "resolve_mode_params", "compute_timeseries",
    "composite_series", "get_shap_assets", "get_retriever", "load_cases_by_id",
    "load_eval_results", "sustained_high_flags", "analysis_bundle",
    "get_routes", "route_risk_series", "get_vessels", "get_news",
    "generate_risk_narrative", "fig_to_jpeg", "export_button", "select_route",
    "route_status_word", "DASH_CSS",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KB_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "disruption_cases.json"
_LABELS_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "decision_labels.json"
_EVAL_RESULTS_PATH = _PROJECT_ROOT / "data" / "processed" / "evaluation_results.json"
_NEWS_BACKUP_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "live_extracted_backup.json"

ALL_AGENTS: list[str] = [
    "shipping", "market", "geopolitical",
    "natural_disaster", "routing", "news_sentiment",
]

AGENT_COLORS: dict[str, str] = {
    "shipping": "#1f77b4",
    "market": "#2ca02c",
    "geopolitical": "#d62728",
    "natural_disaster": "#ff7f0e",
    "routing": "#9467bd",
    "news_sentiment": "#17becf",
}

RISK_COLORS = {"low": "#2e9e5b", "medium": "#d9a514", "high": "#d64545"}

#: Manager-facing status vocabulary (Decision view). Words only — red/amber/green.
STATUS_COLORS = {"Critical": "#d64545", "High": "#d9a514", "Low": "#2e9e5b"}
STATUS_ICONS = {"Critical": "⚠", "High": "▲", "Low": "✔"}

SCENARIOS = {
    "A — Moderate Tension": (60, 74),
    "B — Major Blockage": (150, 170),
    "C — Brief Incident": (280, 290),
}

_TOTAL_DAYS = 365
RANGE_PRESETS = ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "Custom range"]

#: Region control (Phase 12): ``{display_name: region_key}`` for every region
#: in :mod:`src.core.regions`, so the selector and the registry cannot drift.
#: Ordered as the registry orders them, which puts Hormuz — the default and
#: the thesis's primary chokepoint — first.
#:
#: Note the asymmetry this exposes: all four regions now carry real *pipeline*
#: config (agent activation, AIS bounds, ACLED countries, news keywords), but
#: the presentational layers below — ``_ROUTES`` and the news feed — remain
#: Hormuz-only. Selecting another region gives its real risk scoring with an
#: empty route list; see :func:`get_routes`.
AVAILABLE_REGIONS: dict[str, str] = {
    _cfg.display_name: key for key, _cfg in _REGIONS.items()
}

#: Monitored routes per chokepoint. Coordinates are schematic corridor
#: polylines for visualisation (lng, lat) — not official IMO TSS geometry.
#: ``agents`` scopes the route's trend to the pipeline signals most relevant
#: to that corridor (same masking mechanism as the diversity ablation).
_ROUTES: dict[str, list[dict]] = {
    "hormuz": [
        {
            "key": "westbound_tss",
            "name": "Westbound TSS — inbound to Persian Gulf",
            "agents": ["shipping", "market", "geopolitical", "news_sentiment"],
            "coords": [(57.10, 25.20), (56.85, 25.90), (56.55, 26.45),
                       (56.10, 26.68), (55.45, 26.55), (54.80, 26.30)],
        },
        {
            "key": "eastbound_tss",
            "name": "Eastbound TSS — outbound to Gulf of Oman",
            "agents": ["shipping", "routing", "geopolitical"],
            "coords": [(54.80, 26.15), (55.45, 26.40), (56.05, 26.55),
                       (56.45, 26.30), (56.75, 25.75), (57.00, 25.05)],
        },
        {
            "key": "fujairah_approach",
            "name": "Fujairah approach & anchorage",
            "agents": ["natural_disaster", "shipping", "market"],
            "coords": [(56.45, 24.95), (56.40, 25.15), (56.42, 25.55),
                       (56.55, 25.95), (56.60, 26.20)],
        },
    ],
}

#: Representative destinations for the synthetic vessel records (real ports).
_VESSEL_DESTINATIONS = ["Jebel Ali", "Ras Tanura", "Shuaiba Port", "Fujairah",
                        "Bandar Abbas", "Hamad Port", "Mina Salman"]
#: Real vessel classes from the shipping connector's CSV schema.
_VESSEL_TYPES = ["Tanker", "Container", "Dry Bulk", "General Cargo", "Roll-on/roll-off"]


# ===========================================================================
# Pure helpers
# ===========================================================================


def preset_to_range(preset: str, custom: tuple[int, int] | None = None) -> tuple[int, int]:
    """Map an Analysis preset to an inclusive 0-based day-index window.

    Presets are trailing windows of the 365-day synthetic year (e.g.
    "Last 30 days" = days 336–365 one-based).
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
    """First monitoring point per chokepoint from settings.yaml."""
    points_cfg = (
        config.get("agents", {}).get("natural_disaster", {}).get("monitoring_points", {}) or {}
    )
    out: dict[str, dict] = {}
    for region, points in points_cfg.items():
        if points:
            p = points[0]
            out[region] = {"name": str(p["name"]), "lat": float(p["lat"]), "lng": float(p["lng"])}
    return out


def classify_level(score: float, high_thr: float, med_thr: float) -> str:
    if score >= high_thr:
        return "high"
    if score >= med_thr:
        return "medium"
    return "low"


def status_word(score: float, thresholds: dict) -> str:
    """Map a composite risk score to the Decision view's 3-word vocabulary.

    Critical = at/above the engine's ``risk_critical`` threshold;
    High = at/above ``risk_medium`` (i.e. the engine's medium+high bands,
    collapsed for a non-technical reader); Low otherwise.
    """
    if score >= float(thresholds.get("risk_critical", 0.8)):
        return "Critical"
    if score >= float(thresholds.get("risk_medium", 0.4)):
        return "High"
    return "Low"


def decide_action(
    risk_level: str,
    top_driver_agent: str,
    sustained: bool,
    historical_context: dict | None,
) -> tuple[str, str]:
    """Call the Phase 9a rule mapper and name the rule that fired."""
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
# Cached data layer (shared across both pages)
# ===========================================================================


@st.cache_data(show_spinner=False)
def load_app_config(region: str | None = None) -> dict:
    """Return the merged config for ``region``, cached per region.

    Streamlit keys ``cache_data`` on the arguments, so each region is loaded
    once and a switch back is free.

    Args:
        region: Region key. ``None`` resolves via ``SUPPLY_CHAIN_REGION``,
            then the ``hormuz`` default — matching the CLI.

    Returns:
        Base settings deep-merged with the region overlay, carrying
        ``_active_region``.
    """
    return load_config_for_region(region)


def _layout_to_params(layout: dict) -> dict:
    return {
        "inter_weights": {k: float(v) for k, v in layout["inter_agent_weights"].items()},
        "intra": {k: dict(v) for k, v in layout["intra_agent_weights"].items()},
        "thresholds": {k: float(v) for k, v in layout["thresholds"].items()},
    }


def resolve_mode_params(weight_mode: str) -> dict:
    from src.optimization.weight_config import load_optimized_weights, resolve_active_weights

    config = load_app_config()
    if weight_mode == "optimized":
        opt = load_optimized_weights(config)
        if opt and opt.get("inter_agent_weights"):
            return _layout_to_params(opt)
    return _layout_to_params(resolve_active_weights({**config, "weight_mode": "hand_tuned"}))


@st.cache_resource(show_spinner="Generating train/test splits …")
def _eval_bundle():
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

    Same fit-on-train / score-on-test procedure as every Phase 9a number, so
    the dashboard and the thesis tables agree by construction.
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
            padded[-len(s):] = s[:n]
            frames[name] = padded
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agent %s failed while scoring: %s", name, exc)
            frames[name] = np.zeros(n)

    df = pd.DataFrame(frames)
    df["is_disruption"] = bundle["gt"].reindex(range(n)).fillna(False).astype(bool).to_numpy()
    df["day"] = np.arange(1, n + 1)
    return df


def composite_series(ts: pd.DataFrame, params: dict, enabled: set[str]) -> pd.Series:
    """Composite risk from cached agent scores (mirrors ``_aggregate_daily``)."""
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
    out = np.zeros(len(risk), dtype=bool)
    run = 0
    for i, v in enumerate(risk.to_numpy()):
        run = run + 1 if v >= high_thr else 0
        out[i] = run > min_run
    return out


# ===========================================================================
# Routes, vessels, news (region-parameterized — Hormuz populated for thesis)
# ===========================================================================


def get_routes(region: str) -> list[dict]:
    """Monitored routes for a chokepoint. Any region key is accepted; only
    ``hormuz`` is populated for the thesis (others return an empty list)."""
    return list(_ROUTES.get(str(region or "").lower(), []))


def route_risk_series(ts: pd.DataFrame, params: dict, route: dict) -> pd.Series:
    """A route's risk trend — the pipeline aggregation scoped to the route's
    relevant agent subset. Same underlying signals, no invented analytics."""
    return composite_series(ts, params, set(route.get("agents", ALL_AGENTS)))


def route_status_word(ts: pd.DataFrame, params: dict, route: dict, day: int) -> str:
    series = route_risk_series(ts, params, route)
    return status_word(float(series.iloc[day - 1]), params["thresholds"])


def get_vessels(route: dict, day: int) -> list[dict]:
    """Representative synthetic vessel records along a route — **UI only**.

    Per-vessel AIS is not tracked by this system (the shipping connector
    ingests daily aggregates; ``aisstream.enabled`` is false), so these are
    deterministic, clearly-synthetic placeholders as allowed by the redesign
    spec. Vessel types are the shipping connector's real CSV classes.
    """
    coords = route.get("coords", [])
    if len(coords) < 2:
        return []
    rng = np.random.default_rng((hash(route["key"]) % 2**16) * 1000 + int(day))
    n = int(rng.integers(4, 7))
    vessels = []
    for i in range(n):
        t = (i + rng.uniform(0.2, 0.8)) / n
        seg = min(int(t * (len(coords) - 1)), len(coords) - 2)
        frac = t * (len(coords) - 1) - seg
        lng = coords[seg][0] + frac * (coords[seg + 1][0] - coords[seg][0])
        lat = coords[seg][1] + frac * (coords[seg + 1][1] - coords[seg][1])
        vessels.append({
            "id": f"IMO 9{rng.integers(100000, 999999)}",
            "type": _VESSEL_TYPES[int(rng.integers(0, len(_VESSEL_TYPES)))],
            "destination": _VESSEL_DESTINATIONS[int(rng.integers(0, len(_VESSEL_DESTINATIONS)))],
            "speed_kn": round(float(rng.uniform(8.0, 16.5)), 1),
            "lng": round(lng + float(rng.normal(0, 0.02)), 4),
            "lat": round(lat + float(rng.normal(0, 0.02)), 4),
            "synthetic": True,
        })
    return vessels


@st.cache_data(show_spinner=False)
def get_news(region: str, all_regions: bool = False, limit: int = 6) -> list[dict]:
    """Recent headlines from the live-extracted knowledge base backup.

    Region-parameterized: accepts any region key (returns [] when nothing is
    extracted for it). Reads ``live_extracted_backup.json`` — real NewsAPI /
    SerpAPI articles from the Phase 7 extraction run — instead of burning
    NewsAPI quota on every dashboard rerun.
    """
    if not _NEWS_BACKUP_PATH.exists():
        return []
    try:
        docs = json.loads(_NEWS_BACKUP_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        logger.warning("news backup unreadable: %s", exc)
        return []

    items = []
    for d in docs:
        meta = d.get("metadata", {}) or {}
        if meta.get("source_api") not in ("newsapi", "serpapi"):
            continue
        if not all_regions and meta.get("region") != str(region or "").lower():
            continue
        title = meta.get("title") or ""
        if not title:
            continue
        items.append({
            "title": title,
            "source": meta.get("source_name", ""),
            "date": meta.get("event_date", ""),
            "url": meta.get("url", ""),
            "region": meta.get("region", ""),
        })
    items.sort(key=lambda x: x["date"], reverse=True)
    return items[:limit]


def select_route(route_key: str) -> None:
    """Single authority for route selection — the control strip, the status
    list, and the map click (via ?route= query param) all funnel through here
    so every panel stays in sync via ``st.session_state``."""
    st.session_state["selected_route"] = route_key


# ===========================================================================
# Risk narrative (LLM with compositional fallback)
# ===========================================================================

_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six"]


def _compose_narrative(drivers: list[dict], agreement: int, word: str, region_label: str) -> str:
    """Generic compositional fallback — one algorithm over the live SHAP
    drivers, not hardcoded per-feature-combination strings."""
    phrases = []
    for d in drivers[:3]:
        feat = str(d.get("feature", "")).replace("_", " ")
        agent = str(d.get("agent", "")).replace("_", " ")
        direction = "elevated" if float(d.get("shap_value", 0.0)) >= 0 else "suppressed"
        phrases.append(f"{direction} {feat} ({agent} signals)")
    if not phrases:
        return f"Risk in the {region_label} is currently {word.lower()}."
    joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " combined with " + phrases[-1]
    n_word = _NUM_WORDS[min(agreement, 6)]
    return (
        f"Risk in the {region_label} is {word.lower()} primarily due to {joined}; "
        f"{n_word} of six detection agents currently agree on the disruption signal."
    )


@st.cache_data(show_spinner=False)
def generate_risk_narrative(
    drivers_json: str, agreement: int, word: str, region_label: str
) -> tuple[str, str]:
    """Natural-language risk paragraph for the peak-explanation popup.

    Tries the Anthropic API when ``ANTHROPIC_API_KEY`` is configured (no LLM
    client existed anywhere in this codebase to reuse, so this is the new,
    optional pattern); otherwise falls back to :func:`_compose_narrative`.

    Returns ``(text, source)`` with source in {"llm", "composed"}.
    """
    drivers = json.loads(drivers_json)
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        try:
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=160,
                messages=[{
                    "role": "user",
                    "content": (
                        "You write one short plain-language paragraph (2 sentences max, no "
                        "numbers, no jargon) explaining a supply-chain risk alert to a "
                        "logistics manager. Region: " + region_label + ". Status: " + word +
                        ". Top anomaly drivers from an explainability model: " +
                        json.dumps(drivers[:3]) +
                        f". {agreement} of six detection agents agree. Write only the paragraph."
                    ),
                }],
            )
            return msg.content[0].text.strip(), "llm"
        except Exception as exc:  # pragma: no cover - network/key dependent
            logger.warning("LLM narrative failed, using compositional fallback: %s", exc)
    return _compose_narrative(drivers, agreement, word, region_label), "composed"


# ===========================================================================
# JPEG export (Analysis view)
# ===========================================================================


@st.cache_data(show_spinner=False)
def _fig_json_to_jpeg(fig_json: str, title: str, caption: str) -> bytes:
    """Render a Plotly figure JSON to a print-ready JPEG (title + caption
    flattened into the image, white background for the thesis document)."""
    import plotly.io as pio

    fig = pio.from_json(fig_json)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1a1a1a", size=13),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center",
                   font=dict(size=16, color="#111111")),
        margin=dict(t=70, b=110, l=60, r=40),
    )
    fig.add_annotation(
        text=caption, xref="paper", yref="paper", x=0.5, y=-0.22,
        showarrow=False, font=dict(size=11, color="#444444"), align="center",
    )
    return pio.to_image(fig, format="jpeg", width=1100, height=620, scale=2)


def fig_to_jpeg(fig, title: str, caption: str) -> bytes:
    """Public JPEG export helper (chart + title + caption in one image)."""
    return _fig_json_to_jpeg(fig.to_json(), title, caption)


def export_button(fig, key: str, title: str, caption: str) -> None:
    """One-per-chart 'Export as JPEG' button for the Analysis view."""
    try:
        data = fig_to_jpeg(fig, title, caption)
        st.download_button(
            "⬇ Export as JPEG", data=data,
            file_name=f"{key}.jpg", mime="image/jpeg", key=f"exp_{key}",
        )
    except Exception as exc:  # pragma: no cover - kaleido/env dependent
        st.caption(f"JPEG export unavailable: {exc}")


# ===========================================================================
# MapLibre GL JS 3-D map (routes + vessels + click-to-select sync)
#
# Replaces the CesiumJS globe per the redesign feedback: a pitched (tilted)
# 3-D regional map rather than a full globe. MapLibre GL JS is open-source
# and, with the tile sources chosen here, requires **no API key at all**:
#
#   * base style  — OpenFreeMap "liberty" (free vector tiles, keyless)
#   * 3-D terrain — AWS Open Data terrarium DEM tiles (keyless), via
#                   map.setTerrain()
#   * 3-D buildings — fill-extrusion over the style's own vector building
#                   layer where height data exists (guarded; the strait view
#                   is mostly water, so buildings appear only when zoomed in)
#
# MAPTILER_API_KEY in .env is an *optional* upgrade (MapTiler dark style +
# terrain-RGB DEM); because the default path is keyless, the missing-key
# placeholder the Cesium version needed is no longer necessary — the map
# always renders.
# ===========================================================================

MAP_FALLBACK_MESSAGE = (
    "Map unavailable — MapLibre GL JS failed to load from the CDN. "
    "Check the network connection."
)

_MAP_TEMPLATE = Template("""<!DOCTYPE html>
<html><head>
<script src="https://unpkg.com/maplibre-gl@5/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@5/dist/maplibre-gl.css" rel="stylesheet">
<style>
  html, body, #map { width:100%; height:100%; margin:0; padding:0; overflow:hidden; background:#0b0e14; }
  .flybar { position:absolute; top:10px; left:10px; z-index:10; display:flex; gap:6px; }
  .flybar button {
    background:#141a24; color:#c8d1dc; border:1px solid #2a3442; border-radius:4px;
    padding:5px 12px; font: 12px 'Segoe UI', sans-serif; cursor:pointer; letter-spacing:.4px;
  }
  .flybar button:hover { background:#1d2634; border-color:#3d4b5e; }
  .flybar button.primary { border-color:#d9a514; color:#e8d9a0; }
  .status-chip {
    position:absolute; top:10px; right:10px; z-index:10; padding:5px 14px; border-radius:14px;
    font:700 12px 'Segoe UI', sans-serif; letter-spacing:.8px; color:#0b0e14;
  }
  .maplibregl-popup-content { background:#141a24; color:#c8d1dc; border:1px solid #2a3442;
    font: 12px 'Segoe UI', sans-serif; }
  .maplibregl-popup-tip { border-top-color:#141a24 !important; }
</style></head>
<body>
<div id="map"></div>
<div class="flybar">$BUTTONS</div>
<div class="status-chip" id="chip"></div>
<script>
const DATA = $DATA;
document.getElementById("chip").textContent = DATA.focus.name + " · " + DATA.status_word;
document.getElementById("chip").style.background = DATA.status_color;

if (typeof maplibregl === "undefined") {
  document.getElementById("map").innerHTML =
    "<div style='height:100%;display:flex;align-items:center;justify-content:center;" +
    "color:#8d99a8;font:14px Segoe UI,sans-serif'>$FALLBACK</div>";
} else {

const map = new maplibregl.Map({
  container: "map",
  style: DATA.style_url,
  center: [DATA.camera.lng, DATA.camera.lat],
  zoom: DATA.camera.zoom,
  pitch: DATA.camera.pitch,      // tilted 3-D regional view, not a globe
  bearing: DATA.camera.bearing,
  attributionControl: {compact: true},
});
map.addControl(new maplibregl.NavigationControl({visualizePitch: true}), "bottom-right");

map.on("load", function () {
  // ---- 3-D terrain (keyless AWS terrarium DEM unless MapTiler key given) --
  try {
    map.addSource("dem", DATA.terrain_source);
    map.setTerrain({source: "dem", exaggeration: 1.4});
  } catch (e) { console.warn("terrain unavailable:", e); }

  // ---- 3-D buildings where the vector source carries height data ---------
  try {
    const sources = map.getStyle().sources;
    const vec = Object.keys(sources).find(function (s) { return sources[s].type === "vector"; });
    if (vec) {
      map.addLayer({
        id: "3d-buildings", source: vec, "source-layer": "building",
        type: "fill-extrusion", minzoom: 13,
        paint: {
          "fill-extrusion-color": "#aab4bf",
          "fill-extrusion-height": ["coalesce", ["get", "render_height"], 8],
          "fill-extrusion-base": ["coalesce", ["get", "render_min_height"], 0],
          "fill-extrusion-opacity": 0.75,
        },
      });
    }
  } catch (e) { console.warn("3d buildings unavailable:", e); }

  // ---- monitored routes: line layers colored by status word --------------
  for (const r of DATA.routes) {
    const selected = (r.key === DATA.selected_route);
    map.addSource("route-" + r.key, {type: "geojson", data: {
      type: "Feature", properties: {key: r.key, name: r.name, status: r.status},
      geometry: {type: "LineString", coordinates: r.coords},
    }});
    map.addLayer({
      id: "route-" + r.key, source: "route-" + r.key, type: "line",
      layout: {"line-cap": "round", "line-join": "round"},
      paint: {"line-color": r.color, "line-width": selected ? 7 : 3.5,
              "line-opacity": selected ? 0.95 : 0.45},
    });
    map.on("click", "route-" + r.key, function (e) {
      new maplibregl.Popup().setLngLat(e.lngLat).setHTML(
        "<b>" + r.name + "</b><br>Status: <b>" + r.status + "</b>" +
        "<br><small>Selecting this route in the dashboard…</small>").addTo(map);
      // Map -> app sync: components.html has no return channel to Streamlit,
      // so update the parent URL's ?route= param to trigger a rerun.
      try {
        const url = new URL(window.parent.location.href);
        if (url.searchParams.get("route") !== r.key) {
          url.searchParams.set("route", r.key);
          window.parent.location.href = url.toString();
        }
      } catch (err) { /* cross-origin guard */ }
    });
    map.on("mouseenter", "route-" + r.key, function () { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", "route-" + r.key, function () { map.getCanvas().style.cursor = ""; });
  }

  // ---- synthetic representative vessels on the selected route (UI only) --
  map.addSource("vessels", {type: "geojson", data: {
    type: "FeatureCollection",
    features: DATA.vessels.map(function (v) { return {
      type: "Feature",
      properties: v,
      geometry: {type: "Point", coordinates: [v.lng, v.lat]},
    }; }),
  }});
  map.addLayer({
    id: "vessels", source: "vessels", type: "circle",
    paint: {"circle-radius": 6, "circle-color": "#e8edf2",
            "circle-stroke-color": "#0b0e14", "circle-stroke-width": 1.5},
  });
  map.on("click", "vessels", function (e) {
    const p = e.features[0].properties;
    new maplibregl.Popup().setLngLat(e.lngLat).setHTML(
      "<b>" + p.id + "</b> · " + p.type +
      "<br>Destination: " + p.destination + "<br>Speed: " + p.speed_kn + " kn" +
      "<br><small>Representative synthetic record — per-vessel AIS not tracked.</small>"
    ).addTo(map);
  });
  map.on("mouseenter", "vessels", function () { map.getCanvas().style.cursor = "pointer"; });
  map.on("mouseleave", "vessels", function () { map.getCanvas().style.cursor = ""; });
});

// ---- chokepoint markers (status-colored focus, dim secondaries) ----------
function dot(color, size) {
  const el = document.createElement("div");
  el.style.cssText = "width:" + size + "px;height:" + size + "px;border-radius:50%;" +
    "background:" + color + ";border:2px solid #e8edf2;box-shadow:0 0 6px rgba(0,0,0,.6)";
  return el;
}
new maplibregl.Marker({element: dot(DATA.status_color, 16)})
  .setLngLat([DATA.focus.lng, DATA.focus.lat])
  .setPopup(new maplibregl.Popup().setHTML(DATA.popup_html))
  .addTo(map);
for (const p of DATA.secondary) {
  new maplibregl.Marker({element: dot("#5a6b80", 10)})
    .setLngLat([p.lng, p.lat])
    .setPopup(new maplibregl.Popup().setHTML(
      "<b>" + p.name + "</b><br><small>Monitored chokepoint — not the current focus.</small>"))
    .addTo(map);
}

function flyTo(lng, lat) {
  map.flyTo({center: [lng, lat], zoom: DATA.camera.zoom, pitch: DATA.camera.pitch, duration: 1600});
}
window.flyTo = flyTo;
}
</script>
</body></html>""")


def resolve_map_key() -> str:
    """Optional MapTiler key from the environment (via .env). Never logged.

    Unlike the old Cesium token this is **not required** — the default
    OpenFreeMap + AWS-terrarium path is fully keyless.
    """
    return os.environ.get("MAPTILER_API_KEY", "").strip()


def build_map_html(
    focus: dict,
    secondary: list[dict],
    risk_level: str,
    top_driver: str,
    updated: str,
    routes: list[dict] | None = None,
    vessels: list[dict] | None = None,
    selected_route: str = "",
    route_status: dict[str, str] | None = None,
    status: str | None = None,
    maptiler_key: str | None = None,
) -> str:
    """Build the MapLibre GL JS iframe HTML for the Decision view.

    Renders a pitched 3-D regional map (55° tilt) centred on the focus
    chokepoint with status-colored route lines, vessel markers, terrain and
    conditional 3-D buildings. Works with zero API keys; a MapTiler key (when
    provided) is injected into the rendered HTML only — never logged.
    """
    key = resolve_map_key() if maptiler_key is None else maptiler_key.strip()
    if key:
        style_url = f"https://api.maptiler.com/maps/dataviz-dark/style.json?key={key}"
        terrain_source = {
            "type": "raster-dem",
            "url": f"https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json?key={key}",
        }
    else:
        # Keyless defaults: OpenFreeMap vector style + AWS Open Data DEM.
        style_url = "https://tiles.openfreemap.org/styles/liberty"
        terrain_source = {
            "type": "raster-dem",
            "tiles": ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
            "encoding": "terrarium",
            "tileSize": 256,
            "maxzoom": 13,
        }

    word = status or {"low": "Low", "medium": "High", "high": "Critical"}.get(risk_level, "Low")
    route_status = route_status or {}
    popup_html = (
        "<div style='line-height:1.6'>"
        f"<b>Status:</b> {word}<br>"
        f"<b>Main driver:</b> {top_driver}<br>"
        f"<b>Last updated:</b> {updated}</div>"
    )
    routes_payload = []
    for r in routes or []:
        w = route_status.get(r["key"], "Low")
        routes_payload.append({
            "key": r["key"],
            "name": r["name"],
            "status": w,
            "color": STATUS_COLORS.get(w, STATUS_COLORS["Low"]),
            "coords": [[lng, lat] for lng, lat in r["coords"]],
        })
    data = {
        "style_url": style_url,
        "terrain_source": terrain_source,
        "focus": focus,
        "secondary": secondary,
        "status_word": word,
        "status_color": STATUS_COLORS.get(word, STATUS_COLORS["Low"]),
        "popup_html": popup_html,
        "routes": routes_payload,
        "vessels": vessels or [],
        "selected_route": selected_route,
        # Pitched regional camera on the strait — not a full-globe view.
        "camera": {"lng": focus["lng"], "lat": focus["lat"] - 0.25,
                   "zoom": 7.6, "pitch": 55, "bearing": -12},
    }
    buttons = "".join(
        f"<button class=\"{'primary' if p['name'] == focus['name'] else ''}\" "
        f"onclick=\"flyTo({p['lng']},{p['lat']})\">{p['label']}</button>"
        for p in [{**focus, "label": "Hormuz"}]
        + [{**s, "label": s.get("region", s["name"]).replace("_", " ").title()} for s in secondary]
    )
    return _MAP_TEMPLATE.substitute(
        DATA=json.dumps(data), BUTTONS=buttons, FALLBACK=MAP_FALLBACK_MESSAGE,
    )


# ===========================================================================
# Analysis-tab range computations (Phase 9a parity — unchanged logic)
# ===========================================================================


@st.cache_data(show_spinner="Computing metrics for the selected range …")
def analysis_bundle(start: int, end: int) -> dict:
    """All range-filtered analyses for days [start, end] (0-based, incl.)."""
    out: dict = {"start": start, "end": end}
    window = 30

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
    except Exception as exc:  # pragma: no cover
        logger.warning("RAG trend unavailable: %s", exc)
        out["rag"] = {"day": [], "similarity": [], "stride": 1}
    return out


# ===========================================================================
# Shared CSS
# ===========================================================================

DASH_CSS = """
<style>
  .stApp { background: #0e1117; }
  h1, h2, h3 { color: #dfe6ee; letter-spacing: .3px; }
  h1 { font-size: 1.35rem !important; }
  h2 { font-size: 1.15rem !important; }
  section[data-testid="stSidebar"] { background: #12161f; border-right: 1px solid #222b38; }
  div[data-testid="stMainBlockContainer"] { padding: 0.8rem 1.4rem 0.4rem; }
  header[data-testid="stHeader"] { background: transparent; height: 2rem; }
  .status-pill {
    display:inline-block; padding:4px 14px; border-radius:14px; font-weight:700;
    font-size:.86rem; letter-spacing:.8px; color:#0b0e14;
  }
  .action-chip {
    display:inline-block; padding:9px 24px; border-radius:4px; font-weight:700;
    font-size:1.05rem; letter-spacing:1.4px; color:#0b0e14;
  }
  .route-row {
    background:#141a24; border:1px solid #222b38; border-radius:6px;
    padding:8px 12px; margin-bottom:6px; font-size:.85rem; color:#c8d1dc;
  }
  .route-row.selected { border-color:#d9a514; }
  .news-item {
    background:#141a24; border:1px solid #222b38; border-radius:4px;
    padding:6px 10px; margin-bottom:5px; font-size:.78rem; color:#aeb9c6; line-height:1.35;
  }
  .news-item b { color:#dfe6ee; font-weight:600; }
  .explain-box {
    background:#10151d; border:1px solid #2a3442; border-left:3px solid #d9a514;
    border-radius:4px; padding:10px 14px; font-size:.85rem; color:#c8d1dc; line-height:1.55;
  }
  .rule-box {
    background:#10151d; border:1px dashed #2a3442; border-radius:4px;
    padding:7px 11px; font-family: Consolas, monospace; font-size:.74rem; color:#8d99a8;
  }
</style>
"""
