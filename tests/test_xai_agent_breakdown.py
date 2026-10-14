"""Within-agent breakdown on hand-made rows with known terms, plus the real agents."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation import path_d
from src.explainability.agent_breakdown import (
    BreakdownMismatch,
    breakdown,
    effective_terms,
    present_columns,
)

_SHIP = SimpleNamespace(
    _if_weight=0.7, _zscore_weight=0.3,
    _active_features=lambda: ["vessel_count", "avg_delay_hours", "congestion_index"],
)


def _ship_row(duration: float) -> pd.Series:
    shock = 0.7 * 0.4 + 0.3 * 0.5
    return pd.Series({
        "isolation_score": 0.4, "max_zscore_norm": 0.5, "shock_score": shock,
        "duration_score": duration, "anomaly_score": max(shock, duration),
        "vessel_count_zscore": -1.0, "delay_zscore": 0.2, "congestion_zscore": 0.3,
    })


def test_shipping_shock_wins() -> None:
    out = breakdown("shipping", _SHIP, _ship_row(0.1))
    assert out["branch_winner"] == "shock"
    assert out["terms"]["isolation_forest"] == pytest.approx(0.28)
    assert out["terms"]["max_zscore"] == pytest.approx(0.15)
    assert out["max_z_feature"] == "vessel_count"
    assert set(effective_terms(out)) == {"isolation_forest", "max_zscore"}


def test_shipping_duration_wins() -> None:
    out = breakdown("shipping", _SHIP, _ship_row(0.9))
    assert out["branch_winner"] == "duration"
    assert out["score"] == 0.9
    assert effective_terms(out) == {"duration": 0.9}


def test_mismatch_raises() -> None:
    row = _ship_row(0.1)
    row["anomaly_score"] = 0.99
    with pytest.raises(BreakdownMismatch):
        breakdown("shipping", _SHIP, row)


def test_market_terms_and_clip() -> None:
    agent = SimpleNamespace(
        _z_threshold=2.0,
        _active_features=lambda: ["brent_crude_usd", "trade_volume_index", "freight_rate_index"],
        _feature_weights=lambda: {"brent_crude_usd": 0.4, "trade_volume_index": 0.35,
                                  "freight_rate_index": 0.25},
    )
    row = pd.Series({"oil_zscore": 3.0, "trade_volume_zscore": -2.0, "freight_zscore": 1.0,
                     "anomaly_score": 1.0})
    out = breakdown("market", agent, row)
    assert out["terms"]["brent_crude_usd"] == pytest.approx(0.6)
    assert out["terms"]["trade_volume_index"] == pytest.approx(0.35)
    assert out["pre_score"] == pytest.approx(1.075)
    assert out["branch_winner"] == "clipped"


def test_geopolitical_renormalises_and_inverts_stability() -> None:
    agent = SimpleNamespace(_weights={"sanctions": 0.35, "military": 0.25,
                                      "diplomatic": 0.25, "stability": 0.15})
    row = pd.Series({"military_activity_index": 0.8, "diplomatic_incident_score": 0.4,
                     "regime_stability_index": 0.9})
    used = 0.65
    raw = (0.25 * 0.8 + 0.25 * 0.4 + 0.15 * 0.1) / used
    row["raw_composite"] = raw
    row["anomaly_score"] = 1.0 / (1.0 + np.exp(-6.0 * (raw - 0.5)))
    out = breakdown("geopolitical", agent, row, present=set(row.index))
    assert "sanctions" not in out["terms"]
    assert out["terms"]["stability"] == pytest.approx(0.15 * 0.1 / used)
    s = row["anomaly_score"]
    assert out["sigmoid_slope"] == pytest.approx(6 * s * (1 - s))


def test_disaster_single_event_wins() -> None:
    agent = SimpleNamespace(
        _weights={"earthquake": 0.35, "tsunami": 0.30, "cyclone": 0.20, "severe_weather": 0.15},
        _feature_columns=["earthquake_severity", "tsunami_risk", "cyclone_severity",
                          "severe_weather_index"],
    )
    row = pd.Series({"earthquake_severity": 0.8, "tsunami_risk": 0.0, "cyclone_severity": 0.0,
                     "severe_weather_index": 0.0, "raw_composite": 0.28, "anomaly_score": 0.8})
    out = breakdown("natural_disaster", agent, row)
    assert out["branch_winner"] == "single:earthquake_severity"
    assert effective_terms(out) == {"earthquake_severity": 0.8}


def test_news_skips_missing_consensus() -> None:
    agent = SimpleNamespace(_weights={"sentiment": 0.4, "consensus": 0.25,
                                      "velocity": 0.2, "volume": 0.15})
    row = pd.Series({"neg_sentiment_norm": 0.5, "consensus_norm": None,
                     "velocity_norm": 0.0, "volume_spike_norm": 1.0})
    row["anomaly_score"] = (0.4 * 0.5 + 0.15 * 1.0) / 0.75
    out = breakdown("news_sentiment", agent, row)
    assert "consensus" not in out["terms"]
    assert out["terms"]["volume"] == pytest.approx(0.2)


def test_effective_terms_without_a_breakdown() -> None:
    """Dashboard records carry agent scores only: no terms, no winner."""
    for agent in ("shipping", "natural_disaster", "market"):
        assert effective_terms({"agent": agent, "terms": {}, "branch_winner": None}) == {}


def test_missing_day() -> None:
    out = breakdown("market", None, pd.Series({"anomaly_score": np.nan}))
    assert out["branch_winner"] == "missing"


@pytest.mark.skipif(not path_d.eval_data_available(), reason="data/eval cache not built")
@pytest.mark.parametrize("region", path_d.REGIONS)
def test_real_agents_reproduce_every_test_day(region: str) -> None:
    """Every test-window score of every agent is rebuilt from its own terms."""
    data = path_d.load_region(region)
    for domain, run in data.agents.items():
        present = present_columns(run.test_frame)
        for _, row in run.test_frame.iterrows():
            breakdown(domain, run.agent, row, present)  # raises on mismatch
