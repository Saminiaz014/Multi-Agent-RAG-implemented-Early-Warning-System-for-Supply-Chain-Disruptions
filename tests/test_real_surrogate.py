"""Tests for the real-data surrogate pipeline: temporal split, no zero-filling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explainability.real_surrogate import canonical_features, temporal_split
from src.explainability.shap_explainer import ALL_FEATURE_NAMES


def _frame(n: int = 100, start: str = "2019-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="D"),
        "shipping__vessel_count": rng.integers(20, 60, n).astype(float),
        "market__brent_crude_usd": rng.normal(70, 5, n),
        "geopolitical__military_activity_index": rng.random(n),
    })
    return frame.sample(frac=1.0, random_state=0)  # shuffled on purpose


def test_split_is_temporal_at_the_evaluation_split_date() -> None:
    train, test = temporal_split(_frame(start="2023-06-01"))     # shuffled input
    assert train["timestamp"].max() == pd.Timestamp("2023-06-30")
    assert test["timestamp"].min() == pd.Timestamp("2023-07-01")
    assert train["timestamp"].is_monotonic_increasing


def test_missing_sources_are_nan_not_zero() -> None:
    features, missing = canonical_features(_frame())
    assert list(features.columns) == ALL_FEATURE_NAMES
    assert set(missing) == set(ALL_FEATURE_NAMES) - {
        "vessel_count", "brent_crude_usd", "military_activity_index"}
    for name in missing:
        assert features[name].isna().all(), name
        assert not (features[name] == 0).any(), name


def test_prepare_region_uses_only_the_evaluated_days(monkeypatch) -> None:
    """A frame running past evaluation.end_date: features must match the target.

    The fixed split keeps test days up to end_date; three cached frames run a
    day past it. Features built from the whole frame were one row longer than
    the target, and the surrogate fit crashed (875 vs 876).
    """
    import importlib.util
    from pathlib import Path

    from src.evaluation.window import eval_window

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "run_real_surrogate", root / "scripts" / "run_real_surrogate.py")
    rrs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rrs)

    end = eval_window()["end_date"]
    frame = _frame(n=1500, start="2021-11-01").sort_values("timestamp").reset_index(drop=True)
    frame["y_true"] = 0
    assert frame["timestamp"].max() > end                      # runs past end_date
    monkeypatch.setattr(rrs, "load_region_frame", lambda region: (frame, "test"))
    monkeypatch.setattr(rrs, "agent_scores", lambda config, train, test: {
        "shipping": np.linspace(0, 1, len(train) + len(test))})
    d = rrs.prepare_region("hormuz")
    assert len(d["features"]) == len(d["y"]) == len(d["timestamps"])
    assert d["test_dates"][1] == str(end.date())
    assert d["timestamps"].max() == end


def test_features_of_disabled_agents_are_treated_as_missing() -> None:
    """A region disables an agent on evidence grounds; its features leave the surrogate.

    Malacca disables ``market`` yet its frame still carries market columns, and
    ``freight_rate_index`` reached the region's top-5 SHAP features.
    """
    frame = _frame()
    features, missing = canonical_features(frame, disabled_agents={"market"})
    assert features["brent_crude_usd"].isna().all()              # present in the frame
    assert "brent_crude_usd" in missing
    kept, none_missing = canonical_features(frame)                # default: nothing disabled
    assert kept["brent_crude_usd"].notna().all() and "brent_crude_usd" not in none_missing
    # Other agents' features are untouched.
    assert features["vessel_count"].equals(kept["vessel_count"])


def test_prepare_region_drops_the_features_of_the_regions_disabled_agents(monkeypatch) -> None:
    import copy
    import importlib.util
    from pathlib import Path

    from src.core.config_manager import load_config_for_region

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "run_real_surrogate", root / "scripts" / "run_real_surrogate.py")
    rrs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rrs)

    base = load_config_for_region("hormuz")
    base["agents"]["geopolitical"]["enabled"] = True
    off = copy.deepcopy(base)
    off["agents"]["geopolitical"]["enabled"] = False
    configs = {"enabled": base, "disabled": off}
    frame = _frame(n=1500, start="2021-11-01").sort_values("timestamp").reset_index(drop=True)
    frame["y_true"] = 0
    monkeypatch.setattr(rrs, "load_region_frame", lambda region: (frame, "test"))
    monkeypatch.setattr(rrs, "load_config_for_region", lambda region: configs[region])
    monkeypatch.setattr(rrs, "agent_scores", lambda config, train, test: {
        "shipping": np.linspace(0, 1, len(train) + len(test))})
    d = rrs.prepare_region("disabled")
    assert "military_activity_index" in d["missing"]
    assert d["features"]["military_activity_index"].isna().all()
    assert d["features"]["vessel_count"].notna().all()           # enabled agents stay
    kept = rrs.prepare_region("enabled")
    assert not kept["features"]["military_activity_index"].isna().all()
