"""Unit tests for the RiskEngine aggregation logic."""

import numpy as np
import pytest

from src.agents.base_agent import DetectionResult
from src.aggregation.risk_engine import RiskEngine, RiskLevel

_CONFIG = {
    "weights": {"shipping": 0.4, "market": 0.3, "geopolitical": 0.3},
    "thresholds": {"risk_high": 0.7, "risk_medium": 0.4},
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


def test_high_scores_produce_high_risk(engine: RiskEngine) -> None:
    results = [
        _make_result("shipping", [0.9, 0.9]),
        _make_result("market", [0.9, 0.9]),
        _make_result("geopolitical", [0.9, 0.9]),
    ]
    out = engine.aggregate(results)
    assert out["risk_level"] == RiskLevel.HIGH
    assert out["composite_score"] >= 0.7


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
