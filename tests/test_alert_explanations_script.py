"""Issue 13: run_alert_explanations records the exact mean |SHAP| per region."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "run_alert_explanations", _ROOT / "scripts" / "run_alert_explanations.py")
rae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rae)


def test_mean_abs_shap_uses_every_feature_of_every_alert() -> None:
    rows = [{"a": 0.4, "b": -0.3, "c": 0.01}, {"a": -0.2, "b": 0.5, "c": 0.03}]
    got = rae.mean_abs_shap(rows)
    assert list(got) == ["b", "a", "c"]                         # largest first
    assert got["b"] == pytest.approx(0.4) and got["a"] == pytest.approx(0.3)
    assert got["c"] == pytest.approx(0.02)                      # no top-k truncation


def test_mean_abs_shap_without_alerts_is_empty() -> None:
    assert rae.mean_abs_shap([]) == {}


def test_alert_explanations_refuse_a_frame_with_capped_acled_events(monkeypatch) -> None:
    import pandas as pd

    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3), "y_true": 0})
    stale = {"region": "x", "domains_present": {"geopolitical": ["military_activity_index"]}}
    monkeypatch.setattr(rae.harness, "_load", lambda region: (frame, stale))
    with pytest.raises(ValueError, match="geopolitical_source"):
        rae.run_region("x", "real", Path("unused"))


# ------------------------------------------------ issue 16: disabled agents on every path
def _prefixed_frame(days: int = 40):
    import pandas as pd

    return pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=days, freq="D"),
                         "shipping__vessel_count": 50.0,
                         "market__brent_crude_usd": 80.0,
                         "geopolitical__military_activity_index": 0.3})


def test_features_of_disabled_agents_are_dropped_by_the_helper() -> None:
    frame = _prefixed_frame()
    assert {"vessel_count", "brent_crude_usd", "military_activity_index"} <= set(
        rae._features(frame).columns)                          # default: nothing disabled
    out = rae._features(frame, {"market"})
    assert "brent_crude_usd" not in out.columns
    assert {"vessel_count", "military_activity_index", "timestamp"} <= set(out.columns)


def test_the_synthetic_path_never_sees_a_disabled_agents_columns(monkeypatch, tmp_path) -> None:
    """--surrogate synthetic zero-fills absent features, so the columns must be absent."""
    import numpy as np
    import pandas as pd

    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    frame = pd.DataFrame({"timestamp": ts, "y_true": 0, "shipping__vessel_count": 50.0,
                          "market__brent_crude_usd": 80.0})
    manifest = {"region": "x", "domains_present": {"shipping": ["vessel_count"]}}
    config = {"agents": {"market": {"enabled": False}}, "weights": {}}
    seen: list[list[str]] = []

    def fake_scores(tier, cfg, train, test):
        n = len(test)
        if test is train:                                   # the training-window call
            return np.random.default_rng(0).uniform(0.2, 0.6, n), ["shipping"]
        scores = np.full(n, 0.1)                            # below the training 90th percentile
        scores[:5] = 1.0                                    # five alerts in the test window
        return scores, ["shipping"]

    def fake_shap(cfg, features, rows):
        seen.append(list(features.columns))
        return [{} for _ in rows], {"r2_synthetic_in_sample": 0.0}

    monkeypatch.setattr(rae, "load_config_for_region", lambda region: config)
    monkeypatch.setattr(rae.harness, "_load", lambda region: (frame, manifest))
    monkeypatch.setattr(rae.harness, "tier_scores", fake_scores)
    monkeypatch.setattr(rae, "_synthetic_shap", fake_shap)
    monkeypatch.setattr(rae, "explain_alert", lambda *a, **k: {"complete": True})
    monkeypatch.setattr(rae, "write_explanation", lambda *a, **k: None)
    result = rae.run_region("x", "synthetic", tmp_path)
    assert result["alerts"] == 5
    assert seen and "brent_crude_usd" not in seen[0] and "vessel_count" in seen[0]
