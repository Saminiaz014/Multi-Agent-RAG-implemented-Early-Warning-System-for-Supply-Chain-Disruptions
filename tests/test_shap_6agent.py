"""Tests for the six-agent SHAP surrogate explainer (Phase 2F.2).

Five tests:

1. test_surrogate_full_features   — train on synthetic data; R² > 0.85.
2. test_explain_scenario_b        — explain a high-anomaly row; top driver is
                                    in the expected agent domain.
3. test_explain_normal_day        — explain a zero-anomaly row; all SHAP values
                                    close to zero.
4. test_explanation_text_mentions_agents — generated text names at least one
                                    of the six agent domains.
5. test_disabled_agent            — train and explain when one agent's columns
                                    are all zeros (disabled-agent simulation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import (
    ALL_FEATURE_NAMES,
    FEATURE_AGENT_MAP,
    SurrogateShapExplainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = 200  # synthetic training rows (fast; real pipeline uses 364)


def _make_features(n: int = _N, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic n×20 feature DataFrame with realistic value ranges."""
    rng = np.random.default_rng(seed)
    data = {
        # Shipping
        "vessel_count": rng.integers(80, 200, n).astype(float),
        "avg_delay_hours": rng.uniform(0, 48, n),
        "congestion_index": rng.uniform(0, 1, n),
        # Market
        "brent_crude_usd": rng.uniform(40, 120, n),
        "trade_volume_index": rng.uniform(0.5, 1.5, n),
        "freight_rate_index": rng.uniform(0.6, 2.0, n),
        # Geopolitical
        "sanctions_severity": rng.uniform(0, 1, n),
        "military_activity_index": rng.uniform(0, 1, n),
        "diplomatic_incident_score": rng.uniform(0, 1, n),
        "regime_stability_index": rng.uniform(0, 1, n),
        # Natural Disaster
        "earthquake_severity": rng.uniform(0, 1, n),
        "tsunami_risk": rng.uniform(0, 1, n),
        "cyclone_severity": rng.uniform(0, 1, n),
        "severe_weather_index": rng.uniform(0, 1, n),
        # Routing
        "rerouting_percentage": rng.uniform(0, 0.5, n),
        "avg_route_deviation_km": rng.uniform(0, 800, n),
        "transit_volume_ratio": rng.uniform(0.5, 1.5, n),
        # News Sentiment
        "sentiment_score": rng.uniform(-1, 1, n),
        "source_consensus": rng.uniform(0, 1, n),
        "article_volume": rng.integers(0, 500, n).astype(float),
    }
    return pd.DataFrame(data, columns=ALL_FEATURE_NAMES)


def _make_risk_scores(features_df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Deterministic risk scores correlated with a few features."""
    rng = np.random.default_rng(seed)
    # Simple linear mix with noise so RF can achieve R² > 0.85 on train set.
    df = features_df
    scores = (
        0.3 * df["congestion_index"]
        + 0.2 * df["sanctions_severity"]
        + 0.2 * df["rerouting_percentage"] * 2
        + 0.15 * (1 - df["sentiment_score"]) / 2
        + 0.15 * df["earthquake_severity"]
        + rng.normal(0, 0.01, len(df))
    )
    return np.clip(scores, 0, 1).to_numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_surrogate_full_features() -> None:
    """Train surrogate on 200-row synthetic data; R² on training set > 0.85."""
    features_df = _make_features()
    risk_scores = _make_risk_scores(features_df)

    explainer = SurrogateShapExplainer()
    assert not explainer.is_trained

    r2 = explainer.train_surrogate(features_df, risk_scores)

    assert explainer.is_trained
    assert r2 > 0.85, f"Surrogate R² {r2:.4f} did not exceed 0.85."
    assert explainer.r2 == pytest.approx(r2)
    assert set(ALL_FEATURE_NAMES) == set(FEATURE_AGENT_MAP.keys())
    assert len(ALL_FEATURE_NAMES) == 20


def test_explain_scenario_b() -> None:
    """Explain a high-anomaly row; top driver must map to an expected agent."""
    features_df = _make_features()
    risk_scores = _make_risk_scores(features_df)

    explainer = SurrogateShapExplainer()
    explainer.train_surrogate(features_df, risk_scores)

    # High-anomaly scenario: max congestion + max sanctions + heavy rerouting.
    scenario = features_df.copy()
    scenario.iloc[0] = 0.0  # baseline
    scenario.at[0, "congestion_index"] = 1.0
    scenario.at[0, "sanctions_severity"] = 1.0
    scenario.at[0, "rerouting_percentage"] = 0.5
    scenario.at[0, "earthquake_severity"] = 1.0
    scenario.at[0, "sentiment_score"] = -1.0

    row = scenario.iloc[[0]]
    result = explainer.explain(row)

    assert "top_drivers" in result
    assert "shap_values" in result
    assert "expected_value" in result
    assert "feature_names" in result

    top_drivers = result["top_drivers"]
    assert len(top_drivers) == 3

    for driver in top_drivers:
        assert "feature" in driver
        assert "agent" in driver
        assert "shap_value" in driver
        assert driver["agent"] in set(FEATURE_AGENT_MAP.values())

    # At least one top driver should belong to an expected high-signal agent.
    top_agents = {d["agent"] for d in top_drivers}
    high_signal_agents = {"shipping", "geopolitical", "routing", "natural_disaster"}
    assert top_agents & high_signal_agents, (
        f"Expected at least one top driver from {high_signal_agents}, got {top_agents}"
    )


def test_explain_normal_day() -> None:
    """Explain an all-zero feature row; SHAP values should be small in magnitude."""
    features_df = _make_features()
    risk_scores = _make_risk_scores(features_df)

    explainer = SurrogateShapExplainer()
    explainer.train_surrogate(features_df, risk_scores)

    zero_row = pd.DataFrame(
        {feat: [0.0] for feat in ALL_FEATURE_NAMES},
        columns=ALL_FEATURE_NAMES,
    )
    result = explainer.explain(zero_row)

    shap_values = result["shap_values"]
    assert isinstance(shap_values, dict)
    assert set(shap_values.keys()) == set(ALL_FEATURE_NAMES)

    # Total absolute SHAP mass on a zero-feature row should be small (< 1.0).
    total_abs = sum(abs(v) for v in shap_values.values())
    assert total_abs < 1.0, (
        f"All-zero row had unexpectedly large SHAP mass: {total_abs:.4f}"
    )


def test_explanation_text_mentions_agents() -> None:
    """generate_explanation_text() must name at least one agent domain."""
    features_df = _make_features()
    risk_scores = _make_risk_scores(features_df)

    explainer = SurrogateShapExplainer()
    explainer.train_surrogate(features_df, risk_scores)

    peak_row = features_df.iloc[[features_df["congestion_index"].argmax()]]
    shap_result = explainer.explain(peak_row)

    text = explainer.generate_explanation_text(
        risk_score=0.75,
        risk_level="high",
        weight_mode="hand_tuned",
        shap_result=shap_result,
    )

    assert isinstance(text, str)
    assert len(text) > 20

    known_agents = {
        "shipping", "market", "geopolitical",
        "natural disaster", "routing", "news sentiment",
    }
    text_lower = text.lower()
    mentioned = [a for a in known_agents if a in text_lower]
    assert mentioned, (
        f"Explanation text did not mention any agent name.\nText: {text}"
    )
    assert "0.75" in text, "Expected risk score in explanation text."
    assert "HIGH" in text or "high" in text.lower()


def test_disabled_agent() -> None:
    """Surrogate trains and explains correctly when one agent's columns are zero.

    Simulates a disabled routing agent by zeroing its three features; the
    remaining 17 features should still produce a valid explanation.
    """
    features_df = _make_features()
    # Zero out routing columns to simulate a disabled routing agent.
    routing_cols = ["rerouting_percentage", "avg_route_deviation_km", "transit_volume_ratio"]
    features_df_partial = features_df.copy()
    features_df_partial[routing_cols] = 0.0

    # Recalculate risk without the routing signal.
    risk_scores = _make_risk_scores(features_df_partial)

    explainer = SurrogateShapExplainer()
    r2 = explainer.train_surrogate(features_df_partial, risk_scores)

    # R² may drop below 0.85 with fewer signals — just verify it trained.
    assert explainer.is_trained
    assert 0.0 <= r2 <= 1.0

    # Explain a test row with zeroed routing features.
    test_row = features_df_partial.iloc[[0]].copy()
    result = explainer.explain(test_row)

    assert "top_drivers" in result
    assert len(result["top_drivers"]) == 3

    # Routing features should have zero or very small SHAP values.
    shap_values = result["shap_values"]
    routing_shap = [abs(shap_values.get(c, 0.0)) for c in routing_cols]
    non_routing_shap = [
        abs(v) for k, v in shap_values.items() if k not in routing_cols
    ]
    if non_routing_shap:
        max_non_routing = max(non_routing_shap)
        assert max(routing_shap) <= max_non_routing + 0.05, (
            "Zeroed routing features had unexpectedly large SHAP values."
        )
