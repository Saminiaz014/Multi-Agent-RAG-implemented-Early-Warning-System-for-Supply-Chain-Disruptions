"""Tests for the real-data surrogate pipeline: temporal split, no zero-filling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explainability.real_surrogate import canonical_features, temporal_split
from src.explainability.shap_explainer import ALL_FEATURE_NAMES


def _frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2019-01-01", periods=n, freq="D"),
        "shipping__vessel_count": rng.integers(20, 60, n).astype(float),
        "market__brent_crude_usd": rng.normal(70, 5, n),
        "geopolitical__military_activity_index": rng.random(n),
    })
    return frame.sample(frac=1.0, random_state=0)  # shuffled on purpose


def test_split_is_temporal() -> None:
    train, test = temporal_split(_frame())
    assert len(train) == 70 and len(test) == 30
    assert train["timestamp"].max() < test["timestamp"].min()


def test_missing_sources_are_nan_not_zero() -> None:
    features, missing = canonical_features(_frame())
    assert list(features.columns) == ALL_FEATURE_NAMES
    assert set(missing) == set(ALL_FEATURE_NAMES) - {
        "vessel_count", "brent_crude_usd", "military_activity_index"}
    for name in missing:
        assert features[name].isna().all(), name
        assert not (features[name] == 0).any(), name
