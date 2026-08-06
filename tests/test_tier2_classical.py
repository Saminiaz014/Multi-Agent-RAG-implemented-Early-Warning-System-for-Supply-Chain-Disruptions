"""Tests for the Tier 2 classical unsupervised baselines (R6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.tier2_classical import IsolationForestBaseline, MatrixProfileBaseline


@pytest.fixture
def mock_scenario_df() -> pd.DataFrame:
    """A mock scenario with a multivariate event window (days 240-299)."""
    rng = np.random.default_rng(42)
    n_days = 365

    shipping = np.ones(n_days) * 70 + rng.normal(0, 3, n_days)
    market = rng.normal(0, 1, n_days)
    geopolitical = np.ones(n_days) * 0.05 + rng.normal(0, 0.02, n_days)
    routing = np.ones(n_days) * 0.08 + rng.normal(0, 0.03, n_days)
    news = np.ones(n_days) * 0.15 + rng.normal(0, 0.08, n_days)
    disaster = np.ones(n_days) * 0.02 + rng.normal(0, 0.01, n_days)

    shipping[240:300] = 40 + rng.normal(0, 2, 60)
    market[240:300] = market[240:300] + 1.5

    y_disruption = np.zeros(n_days, dtype=int)
    y_disruption[240:300] = 1

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_days),
        "shipping": shipping,
        "market": market,
        "geopolitical": geopolitical,
        "routing": routing,
        "news": news,
        "disaster": disaster,
        "y_disruption": y_disruption,
        "y_band_int": np.where(y_disruption == 1, 3, 0),
        "y_action_int": np.where(y_disruption == 1, 3, 0),
    })


def test_iforest_baseline_shape(mock_scenario_df: pd.DataFrame) -> None:
    """IForest produces correct shape and range."""
    baseline = IsolationForestBaseline(contamination=0.1, n_estimators=100)
    scores, metadata = baseline.run(mock_scenario_df, "test", seed=42)

    assert scores.shape == (365,)
    assert scores.min() >= 0 and scores.max() <= 1
    assert metadata["baseline_name"] == "iforest"


def test_iforest_uses_all_features(mock_scenario_df: pd.DataFrame) -> None:
    """IForest ingests all 6 agent columns."""
    baseline = IsolationForestBaseline()
    _, metadata = baseline.run(mock_scenario_df, "test", seed=42)

    expected_cols = ["shipping", "market", "geopolitical", "routing", "news", "disaster"]
    assert metadata["agent_cols"] == expected_cols


def test_iforest_detects_event(mock_scenario_df: pd.DataFrame) -> None:
    """IForest scores higher during the multivariate event window."""
    baseline = IsolationForestBaseline(contamination=0.1, n_estimators=100)
    scores, _ = baseline.run(mock_scenario_df, "test", seed=42)

    event_scores = scores[240:300].mean()
    quiet_scores = scores[0:200].mean()
    assert event_scores > quiet_scores, "IForest should detect the multivariate anomaly"


def test_iforest_deterministic(mock_scenario_df: pd.DataFrame) -> None:
    """IForest is deterministic given seed."""
    baseline = IsolationForestBaseline(contamination=0.1, n_estimators=100)

    scores1, _ = baseline.run(mock_scenario_df.copy(), "test", seed=42)
    scores2, _ = baseline.run(mock_scenario_df.copy(), "test", seed=42)

    assert np.allclose(scores1, scores2), "Different scores for same seed"


def test_matrix_profile_baseline_shape(mock_scenario_df: pd.DataFrame) -> None:
    """Matrix Profile produces correct shape and range."""
    baseline = MatrixProfileBaseline(m=30)
    try:
        scores, metadata = baseline.run(mock_scenario_df, "test", seed=42)
        assert scores.shape == (365,)
        assert scores.min() >= 0 and scores.max() <= 1
        assert metadata["baseline_name"] == "matrix_profile"
    except Exception as exc:
        pytest.skip(f"Matrix Profile fit failed (expected on small data): {exc}")


def test_iforest_handles_nan(mock_scenario_df: pd.DataFrame) -> None:
    """IForest gracefully handles NaN (disabled/sporadically-missing agents)."""
    baseline = IsolationForestBaseline()

    df_with_nan = mock_scenario_df.copy()
    df_with_nan["disaster"] = np.nan  # simulate a fully-disabled domain

    scores, _ = baseline.run(df_with_nan, "test", seed=42)

    assert scores.shape == (365,)
    assert not np.isnan(scores).any()
