"""Tier 2 classical unsupervised baselines — the TSB-AD standards.

Both baselines are genuinely multivariate: they ingest all 6 agent
signals as features, answering "why not just one model instead of six
agents?" (see architectural rule 1). The CLI runner
(``scripts/run_tier2_baselines.py``) applies the same pre-declared
train(0-200)/val(201-280)/test(281-364) protocol used for Tier 0/1.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.baselines.baseline_base import BaselineRunner

logger = logging.getLogger(__name__)

AGENT_COLS: tuple[str, ...] = (
    "shipping", "market", "geopolitical", "routing", "news", "disaster",
)
_TRAIN_SLICE = slice(0, 201)


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Build the (n_days, 6) feature matrix with NaN handled per rule 4.

    Two distinct NaN sources need different treatment:
      - Sporadic missing days within an active domain (R1's synthetic
        ``missing_data_rate`` — every domain in the real Hormuz scenarios
        has a handful of these). Forward/back-fill carries the last known
        reading, same as the Tier 1 fix in ``tier1_statistical.py``.
        Filling these with a flat 0.0 instead would inject a fake extreme
        reading (e.g. shipping ~70 -> 0) that has nothing to do with the
        scenario and would swamp real anomalies.
      - A domain that's disabled for the whole region (rule 4's example:
        geopolitical in Panama) — entirely NaN, so ffill/bfill leaves it
        untouched. ``fillna(0.0)`` as a second pass catches this case.
    """
    features = df[list(AGENT_COLS)].ffill().bfill().fillna(0.0)
    return features.to_numpy(dtype=float)


class IsolationForestBaseline(BaselineRunner):
    """Isolation Forest over all 6 agent signals (multivariate)."""

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        super().__init__("iforest")
        self.contamination = contamination
        self.n_estimators = n_estimators

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """Fit IForest on the training split, score the full series.

        Args:
            df: DataFrame with the 6 agent-signal columns.
            scenario_id: Scenario name, for logging.
            seed: Seed for ``IsolationForest(random_state=seed)``.

        Returns:
            ``(anomaly_scores, metadata)``.
        """
        X = _prepare_features(df)
        X_train = X[_TRAIN_SLICE]

        iforest = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=seed,
            n_jobs=-1,
        )
        iforest.fit(X_train)

        # score_samples: higher (closer to 0) = normal, more negative =
        # anomalous. Its range isn't fixed to (-1, 0] — it depends on the
        # data — so scale by the *training* score range (analogous to how
        # the Tier 1 baselines derive scale from train-split stats) rather
        # than a blind negate-and-clip, which would crush most of the
        # signal into a narrow band around 0.5.
        train_scores = iforest.score_samples(X_train)
        train_min, train_max = train_scores.min(), train_scores.max()
        raw_scores = iforest.score_samples(X)
        anomaly_scores = np.clip(
            (train_max - raw_scores) / (train_max - train_min + 1e-9), 0, 1
        )

        metadata = {
            "scenario_id": scenario_id,
            "baseline_name": self.name,
            "seed": seed,
            "method": f"Isolation Forest over 6 agent signals, contamination={self.contamination}",
            "agent_cols": list(AGENT_COLS),
            "n_training_samples": len(X_train),
        }
        logger.info(
            "IForest baseline: %s seed=%d, mean score=%.4f, max score=%.4f",
            scenario_id, seed, anomaly_scores.mean(), anomaly_scores.max(),
        )
        return anomaly_scores, metadata


class MatrixProfileBaseline(BaselineRunner):
    """Multi-dimensional Matrix Profile (stumpy's mSTOMP) over all 6 signals.

    Uses ``stumpy.mstump`` — the genuinely multivariate matrix profile,
    not six independent univariate profiles concatenated together — so a
    subsequence is only flagged when its shape is novel across all 6
    dimensions jointly.

    Lookahead note: ``mstump`` only supports self-joins (stumpy has no
    multivariate AB-join primitive), so this profile is computed once over
    the *full* 365-day series rather than fit-on-train/score-on-full like
    the other Tier 2/1 baselines. A subsequence's nearest neighbor can
    therefore fall in the test window — a real, documented deviation from
    rule 2, not a silent one; there is no available multivariate primitive
    in this dependency that avoids it.
    """

    def __init__(self, m: int = 30):
        super().__init__("matrix_profile")
        self.m = m

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """Compute the full-dimensional matrix profile over all 6 signals.

        Args:
            df: DataFrame with the 6 agent-signal columns.
            scenario_id: Scenario name, for logging.
            seed: Unused — stumpy's self-join matrix profile is deterministic.

        Returns:
            ``(anomaly_scores, metadata)``. On failure (e.g. stumpy
            missing, series too short for ``m``), returns all-zero scores
            and records the error in metadata rather than raising.
        """
        try:
            import stumpy
        except ImportError:
            logger.warning("stumpy not found; Matrix Profile baseline will return zeros")
            return np.zeros(len(df)), {
                "scenario_id": scenario_id,
                "baseline_name": self.name,
                "seed": seed,
                "error": "stumpy not installed",
            }

        X = _prepare_features(df)
        n_days = len(df)

        try:
            # mstump wants shape (d, n): one row per dimension.
            mps, _ = stumpy.mstump(X.T, m=self.m)
            # mps[-1] is the full k=6-dimensional profile (all signals must
            # jointly agree for a subsequence to be flagged) — the direct
            # match to rule 1's "single model over all features".
            mp_values = mps[-1]

            anomaly_scores = np.empty(n_days)
            anomaly_scores[:len(mp_values)] = mp_values
            anomaly_scores[len(mp_values):] = mp_values[-1]  # no valid window near series end

            mp_mean = np.nanmean(mp_values)
            mp_std = np.nanstd(mp_values)
            if mp_std > 0:
                anomaly_scores = np.clip((anomaly_scores - mp_mean) / (3 * mp_std), 0, 1)
            else:
                anomaly_scores = np.zeros(n_days)

            metadata = {
                "scenario_id": scenario_id,
                "baseline_name": self.name,
                "seed": seed,
                "method": f"multi-dimensional Matrix Profile (stumpy.mstump), m={self.m}, k=6",
                "agent_cols": list(AGENT_COLS),
                "n_subsequences": len(mp_values),
            }
            logger.info(
                "Matrix Profile baseline: %s seed=%d, mean score=%.4f, max MP=%.4f",
                scenario_id, seed, anomaly_scores.mean(), mp_values.max(),
            )
            return anomaly_scores, metadata

        except Exception as exc:  # noqa: BLE001 — stumpy failures are data-dependent and varied
            logger.warning("Matrix Profile fit failed for %s: %s; returning zeros", scenario_id, exc)
            return np.zeros(n_days), {
                "scenario_id": scenario_id,
                "baseline_name": self.name,
                "seed": seed,
                "error": str(exc),
            }
