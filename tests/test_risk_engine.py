"""Unit tests for the RiskEngine aggregation logic."""

import numpy as np
import pytest

from src.agents.base_agent import DetectionResult
from src.aggregation.risk_engine import RiskEngine, RiskLevel

_CONFIG = {
    "weights": {"shipping": 0.4, "market": 0.3, "geopolitical": 0.3},
    "thresholds": {"risk_critical": 0.8, "risk_high": 0.7, "risk_medium": 0.4},
}


def _make_result(name: str, scores: list[float]) -> DetectionResult:
    arr = np.array(scores)
    return DetectionResult(
        agent_name=name,
        anomaly_scores=arr,
        anomaly_flags=arr > 0.5,
        feature_names=["f1"],
    )


@pytest.fixture()
def engine() -> RiskEngine:
    return RiskEngine(config=_CONFIG)


def test_empty_results_returns_low_risk(engine: RiskEngine) -> None:
    result = engine.aggregate([])
    assert result["composite_score"] == 0.0
    assert result["risk_level"] == RiskLevel.LOW


def test_critical_scores_produce_critical_risk(engine: RiskEngine) -> None:
    results = [
        _make_result("shipping", [0.9, 0.9]),
        _make_result("market", [0.9, 0.9]),
        _make_result("geopolitical", [0.9, 0.9]),
    ]
    out = engine.aggregate(results)
    assert out["risk_level"] == RiskLevel.CRITICAL
    assert out["composite_score"] >= 0.8


def test_high_scores_produce_high_risk(engine: RiskEngine) -> None:
    # Score sits between 0.7 and 0.8 — strictly HIGH, not CRITICAL.
    results = [
        _make_result("shipping", [0.75]),
        _make_result("market", [0.75]),
        _make_result("geopolitical", [0.75]),
    ]
    out = engine.aggregate(results)
    assert out["risk_level"] == RiskLevel.HIGH
    assert 0.7 <= out["composite_score"] < 0.8


def test_low_scores_produce_low_risk(engine: RiskEngine) -> None:
    results = [
        _make_result("shipping", [0.1, 0.1]),
        _make_result("market", [0.1, 0.1]),
        _make_result("geopolitical", [0.1, 0.1]),
    ]
    out = engine.aggregate(results)
    assert out["risk_level"] == RiskLevel.LOW
    assert out["composite_score"] < 0.4


def test_medium_boundary(engine: RiskEngine) -> None:
    # Score ~0.5 should be MEDIUM (between 0.4 and 0.7).
    results = [
        _make_result("shipping", [0.5]),
        _make_result("market", [0.5]),
        _make_result("geopolitical", [0.5]),
    ]
    out = engine.aggregate(results)
    assert out["risk_level"] == RiskLevel.MEDIUM


def test_unknown_agent_is_skipped(engine: RiskEngine) -> None:
    results = [_make_result("unknown_agent", [0.9, 0.9])]
    out = engine.aggregate(results)
    assert out["composite_score"] == 0.0


def test_agent_scores_present_in_output(engine: RiskEngine) -> None:
    results = [_make_result("shipping", [0.8, 0.6])]
    out = engine.aggregate(results)
    assert "shipping" in out["agent_scores"]
    assert np.isclose(out["agent_scores"]["shipping"], 0.7)


# ---------------------------------------------------------------------------
# compute_risk / classify_risk / compute_risk_timeseries (spec API)
# ---------------------------------------------------------------------------

_SIX_CONFIG = {
    "weights": {
        "shipping": 0.25,
        "market": 0.15,
        "geopolitical": 0.25,
        "natural_disaster": 0.10,
        "routing": 0.15,
        "news_sentiment": 0.10,
    },
    "thresholds": {
        "risk_critical": 0.8,
        "risk_high": 0.6,
        "risk_medium": 0.4,
        "risk_low": 0.2,
    },
}

_ALL_SIX = (
    "shipping",
    "market",
    "geopolitical",
    "natural_disaster",
    "routing",
    "news_sentiment",
)


@pytest.fixture()
def engine6() -> RiskEngine:
    return RiskEngine(config=_SIX_CONFIG)


def test_classify_risk_thresholds(engine6: RiskEngine) -> None:
    assert engine6.classify_risk(0.9) == "high"
    assert engine6.classify_risk(0.6) == "high"
    assert engine6.classify_risk(0.5) == "medium"
    assert engine6.classify_risk(0.4) == "medium"
    assert engine6.classify_risk(0.1) == "low"


def test_compute_risk_weighted_calculation(engine6: RiskEngine) -> None:
    # All six agents at a uniform 0.4 score: below the agreement threshold, so
    # no bonus, and normalised weights sum to 1.0 → risk equals the score.
    results = [_make_result(name, [0.4]) for name in _ALL_SIX]
    out = engine6.compute_risk(results)
    assert out["metadata"]["active_agents"] == 6
    assert out["agent_agreement"] == 0
    assert np.isclose(out["risk_score"], 0.4)
    assert np.isclose(sum(out["metadata"]["weights_used"].values()), 1.0)
    # Per-agent contributions reconstruct the base risk.
    total_contrib = sum(
        v["contribution"] for v in out["contributing_agents"].values()
    )
    assert np.isclose(total_contrib, out["risk_score"])


def test_compute_risk_weight_redistribution(engine6: RiskEngine) -> None:
    # Only 3 of 6 agents active → their weights renormalise to sum to 1.0.
    results = [
        _make_result("shipping", [0.4]),
        _make_result("market", [0.4]),
        _make_result("geopolitical", [0.4]),
    ]
    out = engine6.compute_risk(results)
    assert out["metadata"]["active_agents"] == 3
    weights = out["metadata"]["weights_used"]
    assert np.isclose(sum(weights.values()), 1.0)
    # Original ratio shipping:market = 0.25:0.15 is preserved after redistribution.
    assert np.isclose(weights["shipping"] / weights["market"], 0.25 / 0.15)


def test_compute_risk_single_agent_is_full_weight(engine6: RiskEngine) -> None:
    out = engine6.compute_risk([_make_result("market", [0.55])])
    assert out["metadata"]["active_agents"] == 1
    assert np.isclose(out["metadata"]["weights_used"]["market"], 1.0)
    assert np.isclose(out["risk_score"], 0.55)


def test_compute_risk_agreement_bonus_3plus(engine6: RiskEngine) -> None:
    # 3 agents above 0.5, 3 below → x1.15 bonus applies.
    high = [_make_result(n, [0.6]) for n in _ALL_SIX[:3]]
    low = [_make_result(n, [0.2]) for n in _ALL_SIX[3:]]
    out = engine6.compute_risk(high + low)
    assert out["agent_agreement"] == 3
    base = sum(v["contribution"] for v in out["contributing_agents"].values())
    assert np.isclose(out["risk_score"], min(base * 1.15, 1.0))


def test_compute_risk_agreement_bonus_5plus(engine6: RiskEngine) -> None:
    # 5+ agents above 0.5 → larger x1.25 bonus (not stacked with x1.15).
    results = [_make_result(n, [0.6]) for n in _ALL_SIX]
    out = engine6.compute_risk(results)
    assert out["agent_agreement"] == 6
    base = sum(v["contribution"] for v in out["contributing_agents"].values())
    assert np.isclose(out["risk_score"], min(base * 1.25, 1.0))


def test_compute_risk_score_capped_at_one(engine6: RiskEngine) -> None:
    results = [_make_result(n, [0.95]) for n in _ALL_SIX]
    out = engine6.compute_risk(results)
    assert out["risk_score"] <= 1.0


def test_compute_risk_zero_active_agents(engine6: RiskEngine) -> None:
    out = engine6.compute_risk([])
    assert out["risk_score"] == 0.0
    assert out["risk_level"] == "low"
    assert out["metadata"]["active_agents"] == 0


def test_compute_risk_all_zero_scores(engine6: RiskEngine) -> None:
    results = [_make_result(n, [0.0]) for n in _ALL_SIX]
    out = engine6.compute_risk(results)
    assert out["risk_score"] == 0.0
    assert out["risk_level"] == "low"
    assert out["agent_agreement"] == 0


def test_compute_risk_timeseries_scenario_ranking(engine6: RiskEngine) -> None:
    # Normal: all quiet. Scenario A: geopolitical-led, partial firing (medium).
    # Scenario B: all six firing hard (high, highest score of the three).
    normal = [_make_result(n, [0.1]) for n in _ALL_SIX]
    scenario_a = [
        _make_result("geopolitical", [0.6]),
        _make_result("news_sentiment", [0.55]),
        _make_result("shipping", [0.5]),
        _make_result("market", [0.45]),
        _make_result("routing", [0.5]),
        _make_result("natural_disaster", [0.05]),
    ]
    scenario_b = [_make_result(n, [0.85]) for n in _ALL_SIX]

    df = engine6.compute_risk_timeseries(
        {
            "2024-01-01": normal,
            "2024-03-01": scenario_a,
            "2024-06-01": scenario_b,
        }
    )

    assert list(df["timestamp"]) == ["2024-01-01", "2024-03-01", "2024-06-01"]
    by_day = df.set_index("timestamp")
    assert by_day.loc["2024-01-01", "risk_level"] == "low"
    assert by_day.loc["2024-03-01", "risk_level"] == "medium"
    assert by_day.loc["2024-06-01", "risk_level"] == "high"
    # Scenario B (all six firing) must carry the highest risk of the series.
    assert by_day["risk_score"].idxmax() == "2024-06-01"
    assert "shipping_contribution" in df.columns
