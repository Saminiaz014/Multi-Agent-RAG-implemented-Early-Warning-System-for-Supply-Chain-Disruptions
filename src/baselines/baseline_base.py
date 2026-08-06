"""Abstract base class for EVAL01 baselines."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaselineRunner(ABC):
    """Abstract baseline detector.

    A baseline consumes one materialized scenario DataFrame (see
    :mod:`src.benchmark.scenario_batch_generator`) and produces a daily
    anomaly score in ``[0, 1]``. Concrete subclasses implement
    :meth:`run`; threshold selection (:meth:`_compute_threshold`) is
    shared so every baseline follows the same pre-declared-threshold
    protocol (see ``scripts/run_tier0_baselines.py``).
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(
        self, df: pd.DataFrame, scenario_id: str, seed: int
    ) -> tuple[np.ndarray, dict]:
        """Run the baseline on a scenario.

        Args:
            df: Materialized scenario DataFrame (365 rows, with
                ``y_disruption``, ``y_band_int``, etc.).
            scenario_id: e.g. ``"hormuz_P_CRIT"``.
            seed: Random seed.

        Returns:
            ``(anomaly_scores, metadata)`` — scores of shape ``(len(df),)``
            and a dict describing how they were produced.
        """

    def _compute_threshold(
        self, anomaly_scores_val: np.ndarray, y_val: np.ndarray, default: float = 0.5
    ) -> float:
        """Compute the F1-optimal threshold on a validation split.

        Args:
            anomaly_scores_val: Scores on the validation split.
            y_val: Binary labels on the validation split.
            default: Threshold returned when no candidate improves on a
                starting F1 of 0.0 (e.g. the validation window has no
                positive days for this scenario).

        Returns:
            The threshold that maximizes F1 on ``(anomaly_scores_val, y_val)``.
        """
        from sklearn.metrics import f1_score

        best_threshold = default
        best_f1 = 0.0

        # Candidate grid excludes 0.0: with the ">=" comparison below, a
        # constant-zero score (e.g. the never_alarm baseline) would match
        # at threshold=0.0 and be scored as if it alarmed on every day —
        # exactly the behavior a "never alarm" control is supposed to rule
        # out. Starting the sweep at a small positive value keeps constant
        # bottom-of-range scores from ever crossing the threshold.
        for threshold in np.linspace(0.01, 1.0, 100):
            y_pred = (anomaly_scores_val >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        return best_threshold


def best_f1_on_threshold_sweep(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Best achievable F1 across a threshold sweep — an oracle upper bound.

    Shared by any tuning routine that needs to compare *candidates* (a
    hyperparameter value, a weight vector, ...) by how well they separate
    a split, independent of which exact threshold gets declared later for
    test-time scoring. Used by ``tier1_statistical``'s EWMA/CUSUM tuning
    and ``ablation_runner``'s Optuna weight search.

    Args:
        scores: Candidate anomaly scores.
        y_true: Binary labels, same shape as ``scores``.

    Returns:
        Max F1 found across a 0.01-1.0 threshold grid (see
        :meth:`BaselineRunner._compute_threshold` for why 0.0 is excluded).
    """
    from sklearn.metrics import f1_score

    best = 0.0
    for tau in np.linspace(0.01, 1.0, 100):
        y_pred = (scores >= tau).astype(int)
        best = max(best, f1_score(y_true, y_pred, zero_division=0))
    return best
