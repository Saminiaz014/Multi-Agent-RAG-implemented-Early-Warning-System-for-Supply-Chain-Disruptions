"""Tests for silent-agent detection and weight renormalization (R2).

DetectionResult is untouched here — silence is inferred purely from each
agent's own anomaly_scores array, keyed by agent_name.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.base_agent import DetectionResult
from src.aggregation.risk_engine import aggregate_detections
from src.aggregation.silent_agent_tracker import SilentAgentDetector, WeightRenormalizer


def test_silent_agent_detector_mean() -> None:
    """Detect silence based on mean score."""
    detector = SilentAgentDetector(mean_threshold=0.08, max_threshold=0.15)

    # Silent: mean is low, max stays below threshold too.
    scores_silent = np.array([0.03, 0.05, 0.04, 0.06, 0.12, 0.05, 0.04] * 50)
    assert detector.detect_silent(scores_silent) == True

    # Active: mean is well above threshold.
    scores_active = np.array([0.40, 0.50, 0.60, 0.55] * 50)
    assert detector.detect_silent(scores_active) == False


def test_silent_agent_detector_max() -> None:
    """Detect silence based on max score."""
    detector = SilentAgentDetector(mean_threshold=0.08, max_threshold=0.15)

    # Mean is right at/near threshold, but max never exceeds 0.15 — silent.
    scores = np.array([0.07, 0.08, 0.09, 0.12] * 50)
    assert detector.detect_silent(scores) == True


def test_weight_renormalization() -> None:
    """Renormalize weights when agents are silent."""
    weights = {
        "shipping": 0.20,
        "market": 0.15,
        "disaster": 0.10,
        "geopolitical": 0.20,
        "routing": 0.15,
        "news": 0.20,
    }
    silent = {"disaster"}

    renorm = WeightRenormalizer.renormalize(weights, silent)

    assert renorm["disaster"] == 0.0

    expected_sum = sum(w for a, w in weights.items() if a not in silent)
    for agent in renorm:
        if agent not in silent:
            assert renorm[agent] == pytest.approx(weights[agent] / expected_sum)

    assert sum(renorm.values()) == pytest.approx(1.0)


def test_aggregate_detections_with_silent_agent() -> None:
    """Aggregate multi-agent scores with one silent agent."""
    detections = {
        "shipping": DetectionResult(
            agent_name="shipping",
            anomaly_scores=np.array([0.70, 0.75, 0.80, 0.85]),  # active
            anomaly_flags=np.array([0, 0, 1, 1]),
            feature_names=["arrivals"],
            metadata={},
        ),
        "disaster": DetectionResult(
            agent_name="disaster",
            anomaly_scores=np.array([0.02, 0.03, 0.02, 0.04]),  # silent
            anomaly_flags=np.array([0, 0, 0, 0]),
            feature_names=["earthquake_intensity"],
            metadata={},
        ),
    }
    weights = {"shipping": 0.6, "disaster": 0.4}

    score, renorm, silent = aggregate_detections(
        detections, weights, region="hormuz", handle_silent_agents=True
    )

    assert "disaster" in silent
    # Latest shipping score is 0.85; disaster is renormalized to 0 weight.
    assert score == pytest.approx(0.85, abs=0.01)
    assert renorm["disaster"] == 0.0
    assert renorm["shipping"] == pytest.approx(1.0)


def test_aggregate_detections_no_silent_handling() -> None:
    """Aggregate without silent-agent handling (backward compatibility)."""
    detections = {
        "shipping": DetectionResult(
            agent_name="shipping",
            anomaly_scores=np.array([0.70, 0.75, 0.80, 0.85]),
            anomaly_flags=np.array([0, 0, 1, 1]),
            feature_names=["arrivals"],
            metadata={},
        ),
        "disaster": DetectionResult(
            agent_name="disaster",
            anomaly_scores=np.array([0.02, 0.03, 0.02, 0.04]),  # silent
            anomaly_flags=np.array([0, 0, 0, 0]),
            feature_names=["earthquake_intensity"],
            metadata={},
        ),
    }
    weights = {"shipping": 0.6, "disaster": 0.4}

    score, renorm, silent = aggregate_detections(
        detections, weights, region="hormuz", handle_silent_agents=False
    )

    assert len(silent) == 0
    assert renorm == weights
    expected = 0.85 * 0.6 + 0.04 * 0.4
    assert score == pytest.approx(expected, abs=0.01)
