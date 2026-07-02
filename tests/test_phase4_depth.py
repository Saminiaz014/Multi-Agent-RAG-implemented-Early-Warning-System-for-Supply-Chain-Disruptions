"""Thesis depth tests — Phase 4.

Four tests:

1. test_compare_explanations_scenario_b  — day 155 (geo+shipping disruption);
   both weight modes produce valid SHAP; driver_rank_changes non-empty.
2. test_faithfulness                     — Scenarios A+B+C combined; score > 0.8.
3. test_rag_quality                      — evaluate_retrieval_quality on 3 scenarios;
   overall_relevance > 0.7.
4. test_comparison_plots_saved           — both PNGs exist after generate_comparison_plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aggregation.risk_engine import RiskEngine
from src.explainability.shap_explainer import (
    ALL_FEATURE_NAMES,
    compare_explanations,
    compute_faithfulness,
    generate_comparison_plot,
)
from src.rag.context_retriever import ContextRetriever

# ---------------------------------------------------------------------------
# Shared fixture: 365-day synthetic dataset with planted disruptions
# ---------------------------------------------------------------------------

#  - Scenario A: days   0–139  → background noise (low risk, no anomalies)
#  - Scenario B: days 140–165  → geopolitical + shipping disruption
#  - Scenario C: days 200–220  → natural disaster + routing disruption


def _make_scenarios(n: int = 365, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (features_df, risk_scores, disruption_flags) with clean Scenarios A/B/C.

    Planted anomalies are *large* (far outside the baseline range) so that
    both SHAP faithfulness and the 1.5-σ threshold are reliably triggered.

    Key design constraint: no features are shared between the two disruption
    types, so the surrogate RF must learn two orthogonal patterns.  This
    prevents vessel_count from dominating as a single universal split and
    ensures SHAP credits the correct 3 features per scenario type.

    Risk formula uses 6 features split cleanly across two disruption types:
      Scenario B (geo+shipping): sanctions (0.20), military (0.15),
                                  vessel_count (0.15) — low vessel = high risk
      Scenario C (disaster+routing): earthquake (0.25), tsunami (0.15),
                                      rerouting (0.10)
    Weights sum to 1.0.  Each group's top-3 formula features are all
    anomalous on the matching disruption days, so SHAP faithfulness ≥ 0.8.
    """
    rng = np.random.default_rng(seed)

    # ---- Normal background values (all features independent) ----
    data: dict[str, np.ndarray] = {
        "vessel_count":              rng.uniform(120, 180, n).astype(float),
        "avg_delay_hours":           rng.uniform(2.0, 10.0, n),
        "congestion_index":          rng.uniform(0.05, 0.30, n),
        "brent_crude_usd":           rng.uniform(60.0, 85.0, n),
        "trade_volume_index":        rng.uniform(0.85, 1.15, n),
        "freight_rate_index":        rng.uniform(0.75, 1.25, n),
        "sanctions_severity":        rng.uniform(0.00, 0.15, n),
        "military_activity_index":   rng.uniform(0.00, 0.15, n),
        "diplomatic_incident_score": rng.uniform(0.00, 0.15, n),
        "regime_stability_index":    rng.uniform(0.70, 0.95, n),
        "earthquake_severity":       rng.uniform(0.00, 0.10, n),
        "tsunami_risk":              rng.uniform(0.00, 0.10, n),
        "cyclone_severity":          rng.uniform(0.00, 0.10, n),
        "severe_weather_index":      rng.uniform(0.00, 0.10, n),
        "rerouting_percentage":      rng.uniform(0.00, 0.08, n),
        "avg_route_deviation_km":    rng.uniform(0.0, 80.0, n),
        "transit_volume_ratio":      rng.uniform(0.90, 1.10, n),
        "sentiment_score":           rng.uniform(0.20, 0.80, n),
        "source_consensus":          rng.uniform(0.30, 0.70, n),
        "article_volume":            rng.uniform(10.0, 80.0, n),
    }
    df = pd.DataFrame(data, columns=ALL_FEATURE_NAMES)
    flags = np.zeros(n, dtype=bool)

    # ---- Scenario B: geopolitical + shipping (days 140–165) ----
    # Only set features that appear in the B-formula group; vessel_count is
    # NOT set anomalous for C days so the RF cannot use it as a shared proxy.
    for i in range(140, 166):
        df.at[i, "sanctions_severity"]      = rng.uniform(0.85, 1.00)   # formula w=0.20
        df.at[i, "military_activity_index"] = rng.uniform(0.75, 1.00)   # formula w=0.15
        df.at[i, "vessel_count"]            = rng.uniform(35.0, 65.0)   # formula w=0.15
        flags[i] = True

    # ---- Scenario C: natural disaster + routing (days 200–220) ----
    # vessel_count left at normal baseline so that low-vessel signal stays
    # exclusive to Scenario B, forcing the RF to use earthquake/tsunami/rerouting.
    for i in range(200, 221):
        df.at[i, "earthquake_severity"]  = rng.uniform(0.85, 1.00)      # formula w=0.25
        df.at[i, "tsunami_risk"]         = rng.uniform(0.75, 1.00)      # formula w=0.15
        df.at[i, "rerouting_percentage"] = rng.uniform(0.35, 0.50)      # formula w=0.10
        flags[i] = True

    # ---- Risk scores: strictly derived from the 6 formula features ----
    # B-group and C-group are completely disjoint: no feature appears in both.
    # The RF must learn two orthogonal patterns, so SHAP credit flows cleanly
    # to the genuinely anomalous features on each disruption type.
    risk = (
        0.20 * df["sanctions_severity"]
        + 0.15 * df["military_activity_index"]
        + 0.15 * (1.0 - np.clip(df["vessel_count"] / 180.0, 0.0, 1.0))
        + 0.25 * df["earthquake_severity"]
        + 0.15 * df["tsunami_risk"]
        + 0.10 * np.clip(df["rerouting_percentage"] / 0.50, 0.0, 1.0)
    ).to_numpy()
    risk = np.clip(risk, 0.0, 1.0)

    return df, risk, flags


@pytest.fixture(scope="module")
def scenario_data():
    return _make_scenarios()


# ---------------------------------------------------------------------------
# Risk engines (hand-tuned and optimized) shared across tests
# ---------------------------------------------------------------------------

_HT_CFG = {
    "weights": {
        "shipping": 0.25, "market": 0.15, "geopolitical": 0.25,
        "natural_disaster": 0.10, "routing": 0.15, "news_sentiment": 0.10,
    },
    "thresholds": {"risk_critical": 0.80, "risk_high": 0.60, "risk_medium": 0.40},
}

_OPT_CFG = {
    "weights": {
        "shipping": 0.491, "market": 0.109, "geopolitical": 0.119,
        "natural_disaster": 0.095, "routing": 0.061, "news_sentiment": 0.125,
    },
    "thresholds": {"risk_critical": 0.80, "risk_high": 0.582, "risk_medium": 0.260},
}

_RAG_CFG = {
    "collection_name": "disruption_cases",
    "top_k": 3,
    "composite_threshold": 0.65,
    "min_similarity": 0.30,
    "collections": {
        "static_cases": "disruption_cases",
        "live_context": "live_extracted_context",
    },
}

# ---------------------------------------------------------------------------
# Test 1 — compare_explanations on Scenario B peak day (index 155)
# ---------------------------------------------------------------------------


def test_compare_explanations_scenario_b(scenario_data):
    """Day 155 (peak of Scenario B): both SHAP outputs valid; rank changes exist."""
    df, _, _ = scenario_data

    ht_engine = RiskEngine(_HT_CFG)
    opt_engine = RiskEngine(_OPT_CFG)

    features_row = df.iloc[[155]]
    result = compare_explanations(df, ht_engine, opt_engine, features_row=features_row)

    # --- structure ---
    assert set(result.keys()) >= {"hand_tuned", "optimized", "driver_rank_changes", "weight_mode_impact"}

    # --- hand_tuned block ---
    ht = result["hand_tuned"]
    assert "top_drivers" in ht and "shap_values" in ht and "r2" in ht
    assert len(ht["top_drivers"]) == 3
    assert ht["r2"] > 0.5, f"Hand-tuned surrogate R² = {ht['r2']:.4f} < 0.5"
    for drv in ht["top_drivers"]:
        assert {"feature", "agent", "shap_value"} <= set(drv.keys())

    # --- optimized block ---
    opt = result["optimized"]
    assert len(opt["top_drivers"]) == 3
    assert opt["r2"] > 0.5, f"Optimized surrogate R² = {opt['r2']:.4f} < 0.5"

    # --- rank changes ---
    changes = result["driver_rank_changes"]
    assert len(changes) > 0, (
        "Expected at least one feature rank change between hand-tuned and optimized "
        "weights — the two weight regimes differ significantly."
    )
    for ch in changes[:3]:
        assert {"feature", "ht_rank", "opt_rank", "direction"} <= set(ch.keys())
        assert ch["direction"] in ("up", "down")
        assert ch["ht_rank"] != ch["opt_rank"]

    # --- impact text ---
    assert len(result["weight_mode_impact"]) > 20


# ---------------------------------------------------------------------------
# Test 2 — faithfulness > 0.8 on Scenarios A+B+C combined
# ---------------------------------------------------------------------------


def test_faithfulness(scenario_data):
    """SHAP top-3 features faithfully capture planted anomalies; score > 0.8."""
    df, risk_scores, disruption_flags = scenario_data

    n_disruption = disruption_flags.sum()
    assert n_disruption > 0, "No disruption days in test fixture."

    score = compute_faithfulness(df, risk_scores, disruption_flags)

    assert 0.0 <= score <= 1.0
    assert score > 0.8, (
        f"Faithfulness {score:.3f} did not exceed 0.8.\n"
        f"Disruption days: {n_disruption}. "
        "The surrogate is not reliably identifying the planted anomalous features."
    )


# ---------------------------------------------------------------------------
# Test 3 — RAG retrieval quality > 0.7 overall relevance
# ---------------------------------------------------------------------------

_EVAL_SCENARIOS = [
    {
        "name": "B_geo_shipping",
        "signals": {
            "shipping": 0.85,
            "geopolitical": 0.82,
            "news_sentiment": 0.72,
            "market": 0.60,
            "routing": 0.40,
            "natural_disaster": 0.08,
        },
        "expected_agents": ["shipping", "geopolitical"],
    },
    {
        "name": "C_disaster_routing",
        "signals": {
            "natural_disaster": 0.90,
            "shipping": 0.68,
            "routing": 0.55,
            "market": 0.28,
            "geopolitical": 0.18,
            "news_sentiment": 0.12,
        },
        "expected_agents": ["natural_disaster", "shipping"],
    },
    {
        "name": "routing_vessel_diversion",
        "signals": {
            "routing": 0.88,
            "shipping": 0.76,
            "market": 0.42,
            "geopolitical": 0.22,
            "natural_disaster": 0.08,
            "news_sentiment": 0.10,
        },
        "expected_agents": ["routing", "shipping"],
    },
]


def test_rag_quality():
    """evaluate_retrieval_quality on 3 scenarios; overall_relevance > 0.7."""
    retriever = ContextRetriever(_RAG_CFG)
    retriever.build_index("data/knowledge_base/disruption_cases.json")

    if retriever._collection.count() == 0:
        pytest.skip("Knowledge base is empty — run populate_knowledge_base.py first.")

    result = retriever.evaluate_retrieval_quality(_EVAL_SCENARIOS)

    # --- structure ---
    assert "per_scenario" in result
    assert "overall_relevance" in result
    assert "mean_similarity" in result

    per = result["per_scenario"]
    assert len(per) == 3, f"Expected 3 per-scenario entries, got {len(per)}"

    for item in per:
        assert "scenario" in item
        assert "top_match" in item
        assert "similarity" in item
        assert "relevant" in item
        assert 0.0 <= item["similarity"] <= 1.0

    # --- quality targets ---
    assert result["overall_relevance"] > 0.7, (
        f"RAG overall_relevance = {result['overall_relevance']:.3f} < 0.7.\n"
        f"Per-scenario details:\n"
        + "\n".join(
            f"  {s['scenario']}: top_match={s['top_match']!r}, "
            f"matched={s.get('matched_agents')}, "
            f"expected={s.get('expected_agents')}, "
            f"sim={s['similarity']:.3f}, relevant={s['relevant']}"
            for s in per
        )
    )
    assert result["mean_similarity"] > 0.6, (
        f"RAG mean_similarity = {result['mean_similarity']:.3f} < 0.6."
    )


# ---------------------------------------------------------------------------
# Test 4 — comparison PNGs saved to data/processed/
# ---------------------------------------------------------------------------


def test_comparison_plots_saved(scenario_data):
    """Both comparison PNGs are created by generate_comparison_plot()."""
    df, _, _ = scenario_data

    ht_engine = RiskEngine(_HT_CFG)
    opt_engine = RiskEngine(_OPT_CFG)

    paths = generate_comparison_plot(
        df,
        save_dir="data/processed/",
        risk_engine_ht=ht_engine,
        risk_engine_opt=opt_engine,
    )

    assert len(paths) == 2, (
        f"Expected 2 plot paths, got {len(paths)}: {paths}"
    )

    for path in paths:
        assert Path(path).exists(), f"Expected plot file not found: {path}"
        assert Path(path).stat().st_size > 1_000, f"Plot file looks empty: {path}"

    filenames = {Path(p).name for p in paths}
    assert "shap_comparison_waterfall.png" in filenames
    assert "shap_comparison_importance.png" in filenames
