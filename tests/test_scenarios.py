"""End-to-end scenario tests for the DSS pipeline.

These tests exercise the orchestrator with synthetic signal data that
mimics plausible Strait of Hormuz disruption scenarios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.base_agent import BaseAgent, DetectionResult
from src.aggregation.risk_engine import RiskLevel
from src.orchestrator import Orchestrator

_CONFIG = {
    "weights": {"shipping": 0.4, "market": 0.3, "geopolitical": 0.3},
    "thresholds": {"risk_critical": 0.8, "risk_high": 0.7, "risk_medium": 0.4},
}


class _ThresholdAgent(BaseAgent):
    """Agent that scores each row as the mean of its numeric features."""

    def fit(self, df: pd.DataFrame) -> None:
        self._is_fitted = True

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        scores = df.mean(axis=1).to_numpy()
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=scores,
            anomaly_flags=scores > 0.5,
            feature_names=list(df.columns),
        )


@pytest.fixture()
def orchestrator() -> Orchestrator:
    return Orchestrator(config=_CONFIG)


def _normal_signals() -> pd.DataFrame:
    """Synthetic data representing a calm, undisrupted period."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "vessel_count": rng.uniform(0.1, 0.3, 30),
            "oil_price_change": rng.uniform(0.05, 0.15, 30),
            "incident_index": rng.uniform(0.0, 0.2, 30),
        }
    )


def _disrupted_signals() -> pd.DataFrame:
    """Synthetic data representing an active disruption event."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "vessel_count": rng.uniform(0.7, 1.0, 30),
            "oil_price_change": rng.uniform(0.75, 0.95, 30),
            "incident_index": rng.uniform(0.8, 1.0, 30),
        }
    )


def test_no_agents_returns_low_risk(orchestrator: Orchestrator) -> None:
    result = orchestrator.run(_normal_signals())
    assert result["risk_level"] == "LOW"
    assert result["composite_score"] == 0.0


def test_normal_signals_produce_low_risk(orchestrator: Orchestrator) -> None:
    orchestrator.register_agent(_ThresholdAgent("shipping", _CONFIG))
    result = orchestrator.run(_normal_signals())
    assert result["risk_level"] in (RiskLevel.LOW, RiskLevel.MEDIUM)


def test_disrupted_signals_produce_critical_risk(orchestrator: Orchestrator) -> None:
    orchestrator.register_agent(_ThresholdAgent("shipping", _CONFIG))
    orchestrator.register_agent(_ThresholdAgent("market", _CONFIG))
    orchestrator.register_agent(_ThresholdAgent("geopolitical", _CONFIG))
    result = orchestrator.run(_disrupted_signals())
    assert result["risk_level"] == RiskLevel.CRITICAL
    assert result["composite_score"] >= 0.8


def test_pipeline_output_has_required_keys(orchestrator: Orchestrator) -> None:
    orchestrator.register_agent(_ThresholdAgent("shipping", _CONFIG))
    result = orchestrator.run(_normal_signals())
    for key in ("composite_score", "risk_level", "agent_scores", "shap", "context"):
        assert key in result
