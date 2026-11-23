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
