"""Issue 5: risk bands are calm-day quantiles of the composite, training days only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.agents.base_agent import DetectionResult
from src.aggregation.risk_engine import RiskEngine
from src.evaluation.window import eval_window

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "calibrate_risk_bands", _ROOT / "scripts" / "calibrate_risk_bands.py")
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    y = ((ts >= "2023-02-01") & (ts < "2023-03-01")) | ((ts >= "2024-06-01") & (ts < "2024-09-01"))
    return pd.DataFrame({"timestamp": ts, "y_true": y.astype(int)})


def test_calm_days_are_label_negative_unmasked_training_days() -> None:
    frame = _frame()
    masked = frame["timestamp"].isin(pd.to_datetime(["2022-05-05", "2024-01-03"])).to_numpy()
    calm = cal.calm_mask(frame, masked, "train")
    ts = frame["timestamp"]
    split = eval_window()["split_date"]
    assert not calm[(ts >= split).to_numpy()].any()                  # no test day
    assert not calm[frame["y_true"].to_numpy() == 1].any()           # no positive day
    assert not calm[masked].any()                                    # no masked day
    expected = (ts < split) & (frame["y_true"] == 0) & ~pd.Series(masked)
    assert calm.sum() == int(expected.sum())
    every = cal.calm_mask(frame, masked, "all")                      # the old window
    assert every[(ts >= split).to_numpy() & (frame["y_true"] == 0).to_numpy()
                 & ~masked].all()


def test_bands_ignore_test_positive_and_masked_days() -> None:
    frame = _frame()
    masked = frame["timestamp"].isin(pd.to_datetime(["2022-05-05"])).to_numpy()
    rng = np.random.default_rng(0)
    values = rng.uniform(0.2, 0.8, len(frame))
    poisoned = values.copy()
    split = eval_window()["split_date"]
    poisoned[(frame["timestamp"] >= split).to_numpy()] = 5.0
    poisoned[frame["y_true"].to_numpy() == 1] = 5.0
    poisoned[masked] = 5.0
    calm = cal.calm_mask(frame, masked, "train")
    assert cal.bands(values[calm]) == cal.bands(poisoned[calm])


def test_bands_are_p60_p85_p97_rounded_to_two_decimals() -> None:
    values = np.linspace(0.0, 1.0, 1001)
    out = cal.bands(values)
    assert out == {"risk_medium": 0.60, "risk_high": 0.85, "risk_critical": 0.97}
    assert list(out) == ["risk_medium", "risk_high", "risk_critical"]


def test_composite_is_the_risk_engine_score_per_day() -> None:
    config = yaml.safe_load((_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    rng = np.random.default_rng(1)
    scores = {"shipping": rng.uniform(0, 1, 6), "market": rng.uniform(0, 1, 6),
              "geopolitical": None, "news_sentiment": rng.uniform(0, 1, 6)}
    got = cal.composite(scores, config)
    engine = RiskEngine(config)
    for i in range(6):
        outputs = [DetectionResult(agent_name=a, anomaly_scores=np.array([s[i]]),
                                   anomaly_flags=np.array([False]), feature_names=[])
                   for a, s in scores.items() if s is not None]
        assert got[i] == pytest.approx(engine.compute_risk(outputs)["risk_score"], abs=1e-12)


def test_settings_bands_are_ordered() -> None:
    thr = yaml.safe_load((_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))["thresholds"]
    assert 0 < thr["risk_medium"] < thr["risk_high"] < thr["risk_critical"] <= 1


# ------------------------------------------- what each window is scored on
_POLICY = {"gap_fill": {"policy": "forward-fill from the last earlier value"},
           "market_source": {"dating": "first-release date (ALFRED)"},
           "domains_present": {"market": ["brent_crude_usd"]}}


def test_training_days_are_scored_on_a_training_only_history(monkeypatch) -> None:
    """window="train" never reads a test day; positive and masked days are left out.

    The spy scores test, positive and masked days high: any leak moves the values.
    """
    frame = _frame()
    masked_day = "2022-05-05"
    manifest = {"region": "hormuz", "news_source": {"days_masked": [masked_day]}, **_POLICY}
    split = eval_window()["split_date"]
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spy(domain, config, train, days, history=None):
        assert history is None
        seen.append((pd.Timestamp(train["timestamp"].max()), pd.Timestamp(days["timestamp"].max())))
        ts = pd.to_datetime(days["timestamp"])
        loud = (ts >= split) | (days["y_true"] == 1) | (ts == masked_day)
        return np.where(loud.to_numpy(), 0.95, 0.10)

    monkeypatch.setattr(cal.harness, "_load", lambda region: (frame, manifest))
    monkeypatch.setattr(cal.harness, "_agent_scores", spy)
    days = cal.region_composite("hormuz")
    assert all(fit < split for fit, _ in seen)                       # fitted on train only
    train_calls = [scored for _, scored in seen if scored < split]
    assert len(train_calls) == len(cal.harness._TIER_ORDER)          # one per agent
    values = cal.calm_values(days, "train")
    expected = ((frame["timestamp"] < split) & (frame["y_true"] == 0)).sum() - 1
    assert len(values) == expected                                   # masked day out
    assert np.ptp(values) == 0 and values[0] < 0.5                   # only quiet days
    every = cal.calm_values(days, "all")                             # old window: test too
    assert len(every) > len(values) and every.max() > 0.5


def test_tier_shares_cut_at_the_bands() -> None:
    cuts = {"risk_medium": 0.4, "risk_high": 0.6, "risk_critical": 0.8}
    shares = cal.tier_shares(np.array([0.1, 0.4, 0.5, 0.6, 0.79, 0.8, 0.99, 0.2]), cuts)
    assert shares == {"Low": 2 / 8, "Medium": 2 / 8, "High": 2 / 8, "Critical": 2 / 8}
