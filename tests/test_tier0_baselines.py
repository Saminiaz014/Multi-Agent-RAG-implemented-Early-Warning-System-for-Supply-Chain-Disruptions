"""Tests for the Tier 0 control baselines and evaluator (R4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.baseline_evaluator import BaselineEvaluator
from src.baselines.tier0_controls import (
    AlwaysAlarmBaseline,
    NeverAlarmBaseline,
    RandomBaseline,
)


@pytest.fixture
def mock_scenario_df() -> pd.DataFrame:
    """A mock scenario DataFrame with a disruption window on days 240-299."""
    rng = np.random.default_rng(42)
    n_days = 365

    y_disruption = np.zeros(n_days, dtype=int)
    y_disruption[240:300] = 1

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_days),
        "shipping": rng.normal(70, 5, n_days),
        "market": rng.normal(0, 1, n_days),
        "y_disruption": y_disruption,
        "y_band_int": np.where(y_disruption == 1, 3, 0),
        "y_action_int": np.where(y_disruption == 1, 3, 0),
    })


def test_random_baseline_shape(mock_scenario_df: pd.DataFrame) -> None:
    """Random baseline produces correct shape and range."""
    baseline = RandomBaseline()
    scores, metadata = baseline.run(mock_scenario_df, "test_scenario", seed=42)

    assert scores.shape == (365,)
    assert scores.min() >= 0 and scores.max() <= 1
    assert metadata["baseline_name"] == "random"


def test_always_alarm_baseline(mock_scenario_df: pd.DataFrame) -> None:
    """Always-alarm baseline is constant 1.0."""
    baseline = AlwaysAlarmBaseline()
    scores, metadata = baseline.run(mock_scenario_df, "test_scenario", seed=42)

    assert (scores == 1.0).all()
    assert metadata["baseline_name"] == "always_alarm"


def test_never_alarm_baseline(mock_scenario_df: pd.DataFrame) -> None:
    """Never-alarm baseline is constant 0.0."""
    baseline = NeverAlarmBaseline()
    scores, metadata = baseline.run(mock_scenario_df, "test_scenario", seed=42)

    assert (scores == 0.0).all()
    assert metadata["baseline_name"] == "never_alarm"


def test_evaluator_computes_metrics(mock_scenario_df: pd.DataFrame) -> None:
    """Evaluator computes D3, D4, D5-D10 metrics."""
    y_true = mock_scenario_df["y_disruption"].to_numpy()

    np_random = np.random.RandomState(42)
    scores = np_random.uniform(0, 1, len(y_true))

    metrics = BaselineEvaluator.evaluate(y_true, scores, "test_scenario", threshold=0.5)

    assert "D3_auc_pr" in metrics
    assert "D4_auc_roc" in metrics
    assert "D5_f1_tau" in metrics
    assert "D6_best_f1" in metrics
    assert "D9_fpr_tau" in metrics

    assert 0 <= metrics["D4_auc_roc"] <= 1
    assert 0 <= metrics["D3_auc_pr"] <= 1


def test_evaluator_never_alarm(mock_scenario_df: pd.DataFrame) -> None:
    """Never-alarm baseline should have zero recall, precision, and FPR."""
    y_true = mock_scenario_df["y_disruption"].to_numpy()
    scores = np.zeros(len(y_true))  # never alarm

    metrics = BaselineEvaluator.evaluate(y_true, scores, "test_scenario", threshold=0.5)

    assert metrics["D8_recall_tau"] == 0.0
    assert metrics["D7_precision_tau"] == 0.0
    assert metrics["D9_fpr_tau"] == 0.0
