"""The method comparison's fixed evaluation window (settings.yaml ``evaluation``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src.core.config_manager import load_base_config, load_config_for_region
from src.core.regions import list_regions

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_method_comparison", _ROOT / "scripts" / "run_method_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)


def _frame(start: str, end: str) -> pd.DataFrame:
    ts = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"timestamp": ts, "y_true": False})


def test_window_comes_from_settings() -> None:
    window = rmc.eval_window()
    assert window["split_date"] == pd.Timestamp("2023-07-01")
    assert window["end_date"] == pd.Timestamp("2025-11-21")   # ACLED coverage limit


def test_window_is_identical_for_every_region() -> None:
    base = load_base_config()["evaluation"]
    for region in list_regions():
        assert load_config_for_region(region)["evaluation"] == base, region


def test_split_is_by_date_not_by_fraction() -> None:
    train, test = rmc._split(_frame("2019-01-01", "2026-03-01"))
    assert train["timestamp"].max() == pd.Timestamp("2023-06-30")
    assert test["timestamp"].min() == pd.Timestamp("2023-07-01")
    assert test["timestamp"].max() == pd.Timestamp("2025-11-21")   # days after end dropped


def test_split_does_not_depend_on_series_length() -> None:
    _, short = rmc._split(_frame("2019-01-01", "2025-08-30"))
    _, longer = rmc._split(_frame("2019-01-01", "2025-11-21"))
    assert short["timestamp"].min() == longer["timestamp"].min()


def test_split_refuses_a_frame_that_ends_before_the_split() -> None:
    with pytest.raises(ValueError, match="split"):
        rmc._split(_frame("2019-01-01", "2023-05-01"))


def test_split_warns_when_the_frame_ends_before_end_date(caplog) -> None:
    with caplog.at_level("WARNING"):
        rmc._split(_frame("2019-01-01", "2025-11-20"))
    assert "end_date" in caplog.text


def test_bootstrap_settings_come_from_settings() -> None:
    from src.evaluation.window import bootstrap_config
    assert bootstrap_config() == {"block": 30, "n_resamples": 1000, "ci": 0.95, "seed": 42}


def test_evaluate_adds_paired_block_bootstrap_cis() -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    y = ((np.arange(600) // 60) % 3 == 0).astype(int)
    test_scores = y * 0.8 + rng.normal(0, 0.5, 600)
    train_scores = rng.normal(0, 0.5, 400)
    out = rmc._evaluate(y, test_scores, train_scores, n_resamples=200)
    assert out["auc_ci_low"] <= out["auc"] <= out["auc_ci_high"]
    assert out["f1_ci_low"] <= out["f1_ci_high"]
    assert out["alert_rate_ci_low"] <= out["alert_rate"] <= out["alert_rate_ci_high"]
    assert out["bootstrap_valid"] == 200
    again = rmc._evaluate(y, test_scores + 1e-9, train_scores, n_resamples=200)
    assert again["auc_ci_low"] == out["auc_ci_low"]        # same resamples: paired


def test_real_surrogate_uses_the_same_split() -> None:
    from src.explainability.real_surrogate import temporal_split
    frame = _frame("2019-01-01", "2026-01-31")
    a_train, a_test = rmc._split(frame)
    b_train, b_test = temporal_split(frame)
    assert a_train["timestamp"].tolist() == b_train["timestamp"].tolist()
    assert a_test["timestamp"].tolist() == b_test["timestamp"].tolist()
