"""Weighted risk aggregation engine.

Combines per-agent anomaly scores into a single composite risk score and
maps it to a human-readable risk level using configurable thresholds.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np

from src.agents.base_agent import DetectionResult

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Categorical risk classification."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskEngine:
    """Aggregate agent outputs into a composite disruption risk score.

    The engine performs a weighted average of per-agent anomaly scores
    using weights defined in ``config["weights"]``. The composite score
    is then bucketed into :class:`RiskLevel` using ``config["thresholds"]``.

    Args:
        config: Top-level application config dict (reads ``weights`` and
            ``thresholds`` sub-keys).
    """

    def __init__(self, config: dict) -> None:
        self.weights: dict[str, float] = config.get("weights", {})
        self.threshold_critical: float = config["thresholds"]["risk_critical"]
        self.threshold_high: float = config["thresholds"]["risk_high"]
        self.threshold_medium: float = config["thresholds"]["risk_medium"]
        self._validate_weights()

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-3):
            logger.warning(
                "Agent weights sum to %.4f — expected 1.0. "
                "Scores will be renormalised.",
                total,
            )

    def aggregate(self, results: list[DetectionResult]) -> dict:
        """Compute a weighted composite risk score from agent results.

        Only agents whose names appear in ``self.weights`` contribute to
        the composite score. Agents with no configured weight are logged
        as warnings and skipped.

        Args:
            results: Detection results from all active agents.

        Returns:
            Dictionary with keys:
                - ``composite_score`` (float): weighted mean in [0, 1].
                - ``risk_level`` (str): one of CRITICAL / HIGH / MEDIUM / LOW.
                - ``agent_scores`` (dict[str, float]): mean score per agent.
        """
        if not results:
            logger.warning("No detection results provided — returning zero risk.")
            return {
                "composite_score": 0.0,
                "risk_level": RiskLevel.LOW,
                "agent_scores": {},
            }

        agent_scores: dict[str, float] = {}
        weighted_sum: float = 0.0
        weight_total: float = 0.0

        for result in results:
            agent_name = result.agent_name
            weight = self.weights.get(agent_name)
            if weight is None:
                logger.warning(
                    "Agent '%s' has no configured weight — skipping.", agent_name
                )
                continue
            mean_score = float(np.mean(result.anomaly_scores))
            agent_scores[agent_name] = mean_score
            weighted_sum += weight * mean_score
            weight_total += weight

        composite = weighted_sum / weight_total if weight_total > 0 else 0.0
        risk_level = self._classify(composite)

        logger.info(
            "Risk aggregation complete | composite=%.4f | level=%s",
            composite,
            risk_level,
        )
        return {
            "composite_score": composite,
            "risk_level": risk_level,
            "agent_scores": agent_scores,
        }

    def _classify(self, score: float) -> RiskLevel:
        """Map a numeric score to a :class:`RiskLevel`.

        Args:
            score: Composite risk score in [0, 1].

        Returns:
            Appropriate :class:`RiskLevel` enum member.
        """
        if score >= self.threshold_critical:
            return RiskLevel.CRITICAL
        if score >= self.threshold_high:
            return RiskLevel.HIGH
        if score >= self.threshold_medium:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
