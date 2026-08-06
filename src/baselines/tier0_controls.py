"""Tier 0 control baselines — trivial detectors that establish a lower bound.

Without these rows, a results table has no reference point: random should
be beatable by any real detector, and always/never-alarm mark the two
corners of the precision/recall space (perfect recall + worst precision,
and the reverse).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines.baseline_base import BaselineRunner

logger = logging.getLogger(__name__)


class RandomBaseline(BaselineRunner):
    """Uniform random anomaly scores."""

    def __init__(self):
        super().__init__("random")

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """Generate uniform random scores in ``[0, 1]``."""
        np_random = np.random.RandomState(seed)
        anomaly_scores = np_random.uniform(0, 1, len(df))

        metadata = {
            "scenario_id": scenario_id,
            "baseline_name": self.name,
            "seed": seed,
            "method": "uniform random in [0, 1]",
        }
        logger.info(
            "Random baseline: %s seed=%d, mean score=%.4f",
            scenario_id, seed, anomaly_scores.mean(),
        )
        return anomaly_scores, metadata


class AlwaysAlarmBaseline(BaselineRunner):
    """Constant 1.0 (always alarming)."""

    def __init__(self):
        super().__init__("always_alarm")

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """Constant 1.0 for every day."""
        anomaly_scores = np.ones(len(df))
        metadata = {
            "scenario_id": scenario_id,
            "baseline_name": self.name,
            "seed": seed,
            "method": "constant 1.0 (always alarming)",
        }
        logger.info("Always-alarm baseline: %s seed=%d", scenario_id, seed)
        return anomaly_scores, metadata


class NeverAlarmBaseline(BaselineRunner):
    """Constant 0.0 (never alarming)."""

    def __init__(self):
        super().__init__("never_alarm")

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """Constant 0.0 for every day."""
        anomaly_scores = np.zeros(len(df))
        metadata = {
            "scenario_id": scenario_id,
            "baseline_name": self.name,
            "seed": seed,
            "method": "constant 0.0 (never alarming)",
        }
        logger.info("Never-alarm baseline: %s seed=%d", scenario_id, seed)
        return anomaly_scores, metadata
