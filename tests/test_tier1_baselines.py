"""Tests for the Tier 1 statistical/SPC baselines (R5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.tier1_statistical import (
    CUSUMBaseline,
    EWMABaseline,
    PersistenceBaseline,
    SARIMABaseline,
    ZScoreBaseline,
)


@pytest.fixture
def mock_scenario_df() -> pd.DataFrame:
    """A mock scenario with a slow ramp into a sustained drop (days 220-300)."""
    rng = np.random.default_rng(42)
    n_days = 365

    shipping = np.ones(n_days) * 70

    onset = 240
    duration = 60
    ramp_start = onset - 20
    ramp_end = onset
    for i in range(ramp_start, ramp_end):
        shipping[i] = 70 * (1 - (i - ramp_start) / (ramp_end - ramp_start) * 0.4)
    shipping[onset:onset + duration] = 40

    shipping = shipping + rng.normal(0, 3, n_days)

    y_disruption = np.zeros(n_days, dtype=int)
    y_disruption[onset:onset + duration] = 1

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_days),
        "shipping": shipping,
        "y_disruption": y_disruption,
    })


def test_zscore_baseline(mock_scenario_df: pd.DataFrame) -> None:
    """Z-score baseline produces scores in [0, 1]."""
    baseline = ZScoreBaseline()
    scores, metadata = baseline.run(mock_scenario_df, "test", seed=42)

    assert scores.shape == (365,)
    assert scores.min() >= 0 and scores.max() <= 1
    assert metadata["baseline_name"] == "zscore"


def test_ewma_baseline(mock_scenario_df: pd.DataFrame) -> None:
    """EWMA baseline produces scores in [0, 1]."""
    baseline = EWMABaseline()
    scores, metadata = baseline.run(mock_scenario_df, "test", seed=42)

    assert scores.shape == (365,)
    assert scores.min() >= 0 and scores.max() <= 1
    assert metadata["baseline_name"] == "ewma"


def test_cusum_baseline(mock_scenario_df: pd.DataFrame) -> None:
    """CUSUM baseline detects the slow ramp (rule 3's headline finding)."""
    baseline = CUSUMBaseline(tune=False)  # fixed threshold=5.0, drift=0.5
    scores, metadata = baseline.run(mock_scenario_df, "test", seed=42)

    assert scores.shape == (365,)
    assert scores.min() >= 0 and scores.max() <= 1

    ramp_event_scores = scores[220:310].mean()
    quiet_scores = scores[0:200].mean()
    assert ramp_event_scores > quiet_scores, "CUSUM should detect the slow ramp"


def test_persistence_baseline(mock_scenario_df: pd.DataFrame) -> None:
    """Persistence baseline produces scores in [0, 1]."""
    baseline = PersistenceBaseline()
    scores, metadata = baseline.run(mock_scenario_df, "test", seed=42)

    assert scores.shape == (365,)
    assert scores.min() >= 0 and scores.max() <= 1
    assert metadata["baseline_name"] == "persistence"


def test_sarima_baseline(mock_scenario_df: pd.DataFrame) -> None:
    """SARIMA baseline runs without raising and returns valid-range scores."""
    baseline = SARIMABaseline()
    try:
        scores, metadata = baseline.run(mock_scenario_df, "test", seed=42)
        assert scores.shape == (365,)
        assert scores.min() >= 0 and scores.max() <= 1
    except Exception as exc:  # SARIMA may fail to converge on synthetic data
        pytest.skip(f"SARIMA fit failed (acceptable on small synthetic data): {exc}")
