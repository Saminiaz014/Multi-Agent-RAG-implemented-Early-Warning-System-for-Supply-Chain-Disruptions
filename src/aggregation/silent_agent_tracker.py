"""Detect and track silent agents without modifying DetectionResult.

An agent is *enabled* (it has a configured weight and produced a
:class:`~src.agents.base_agent.DetectionResult`) but can still be
*silent* — its domain has no meaningful signal in this region (e.g. the
disaster agent in a region with no natural-disaster exposure). Silence is
inferred purely from the agent's own ``anomaly_scores`` array, so nothing
about :class:`DetectionResult` itself needs to change.
"""

from __future__ import annotations

import logging

import numpy as np

from src.agents.base_agent import DetectionResult

logger = logging.getLogger(__name__)


class SilentAgentDetector:
    """Identify agents producing no meaningful signal.

    An agent is silent if, over its full ``anomaly_scores`` window:
        - ``mean(anomaly_scores) < mean_threshold``, OR
        - ``max(anomaly_scores) < max_threshold``

    Either condition is sufficient: a flat-but-elevated agent that never
    spikes (mean above threshold, max below it) is still silent, since it
    never produced a single day worth alerting on.
    """

    def __init__(self, mean_threshold: float = 0.08, max_threshold: float = 0.15):
        """
        Args:
            mean_threshold: Agent is silent if ``mean(anomaly_scores)`` is
                below this.
            max_threshold: Agent is silent if ``max(anomaly_scores)`` is
                below this.
        """
        self.mean_threshold = mean_threshold
        self.max_threshold = max_threshold

    def detect_silent(self, anomaly_scores: np.ndarray | None) -> bool:
        """Check whether a single agent's score array is silent.

        Args:
            anomaly_scores: Array of shape ``(n_days,)``.

        Returns:
            True if the agent is silent (or produced no scores at all).
        """
        if anomaly_scores is None or len(anomaly_scores) == 0:
            return True

        mean_score = np.nanmean(anomaly_scores)
        max_score = np.nanmax(anomaly_scores)

        return bool(mean_score < self.mean_threshold or max_score < self.max_threshold)

    def identify_silent_agents(
        self, detections: dict[str, DetectionResult]
    ) -> set[str]:
        """Scan all detections and identify silent agents.

        Args:
            detections: ``{agent_name: DetectionResult}``.

        Returns:
            Set of agent names classified as silent.
        """
        silent_agents: set[str] = set()
        for agent_name, detection in detections.items():
            if self.detect_silent(detection.anomaly_scores):
                silent_agents.add(agent_name)
                logger.debug(
                    "Agent '%s' marked as silent: mean=%.4f, max=%.4f",
                    agent_name,
                    np.nanmean(detection.anomaly_scores),
                    np.nanmax(detection.anomaly_scores),
                )
        return silent_agents


class WeightRenormalizer:
    """Normalize weights when some agents are silent."""

    @staticmethod
    def renormalize(
        original_weights: dict[str, float], silent_agents: set[str]
    ) -> dict[str, float]:
        """Remove silent agents and rescale remaining weights to sum to 1.0.

        Args:
            original_weights: ``{agent_name: weight}``, assumed to sum to ~1.0.
            silent_agents: Agent names to mute.

        Returns:
            ``{agent_name: weight}`` summing to 1.0, with every silent
            agent's weight set to 0.0.
        """
        active_agents = {
            a: w for a, w in original_weights.items() if a not in silent_agents
        }

        if not active_agents:
            logger.warning("All agents are silent; using original weights as fallback")
            return original_weights

        active_weight_sum = sum(active_agents.values())
        if active_weight_sum <= 0:
            logger.warning("Active weight sum is 0; uniform fallback")
            uniform = 1.0 / len(active_agents)
            return {a: (uniform if a in active_agents else 0.0) for a in original_weights}

        return {
            agent_name: (0.0 if agent_name in silent_agents else weight / active_weight_sum)
            for agent_name, weight in original_weights.items()
        }
