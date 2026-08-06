"""Tier 1 statistical / SPC baselines — what a supply-chain manager uses today.

All five baselines work on a single domain (shipping) to answer "why not
just a control chart?" Each produces a full-series anomaly score; the CLI
runner (``scripts/run_tier1_baselines.py``) applies the same pre-declared
train(0-200)/val(201-280)/test(281-364) protocol used for Tier 0
(``scripts/run_tier0_baselines.py``) so results are comparable across tiers.

Where a baseline has a genuinely tunable hyperparameter (EWMA's lambda,
CUSUM's threshold), it is grid-searched for best F1 on the validation
split, per architectural rule 3. SARIMA's (order, seasonal_order) is kept
fixed rather than grid-searched — repeatedly re-fitting a SARIMAX model
per candidate order across 20 scenario/seed combinations is expensive, and
out of scope for this baseline pass.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.baselines.baseline_base import BaselineRunner

logger = logging.getLogger(__name__)

_TRAIN_SLICE = slice(0, 201)
_VAL_SLICE = slice(201, 281)


def _fill_missing(shipping: np.ndarray) -> np.ndarray:
    """Forward/back-fill missing shipping observations.

    R1's synthetic scenarios inject ~2% missing days (``noise.missing_data_rate``)
    to simulate sensor dropout. Left unfilled, a single NaN poisons every
    baseline here: ``np.ndarray.mean()``/``.std()`` are not NaN-aware, so a
    train-slice baseline_mean/std used by EWMA/CUSUM/Persistence would come
    out NaN and corrupt every downstream day, not just the missing one. A
    real control-chart operator wouldn't tolerate a blank reading either —
    they'd carry the last known value forward, which is what this does.
    """
    return pd.Series(shipping).ffill().bfill().to_numpy()


def _best_f1_on_val(scores_val: np.ndarray, y_val: np.ndarray) -> float:
    """Best achievable F1 on a validation slice across a threshold sweep.

    Used to compare hyperparameter candidates by how well they separate
    the validation split, independent of which exact threshold is later
    declared for test-time scoring.
    """
    best = 0.0
    for tau in np.linspace(0.01, 1.0, 50):
        y_pred = (scores_val >= tau).astype(int)
        best = max(best, f1_score(y_val, y_pred, zero_division=0))
    return best


class ZScoreBaseline(BaselineRunner):
    """Rolling z-score on shipping (vessel counts)."""

    def __init__(self, window: int = 30):
        super().__init__("zscore")
        self.window = window

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """Compute a rolling z-score on the shipping domain.

        Args:
            df: DataFrame with a ``shipping`` column (vessel counts).
            scenario_id: Scenario name, for logging.
            seed: Unused — kept for :class:`BaselineRunner` interface parity.

        Returns:
            ``(anomaly_scores, metadata)``.
        """
        shipping = _fill_missing(df["shipping"].to_numpy())

        rolling_mean = pd.Series(shipping).rolling(self.window, min_periods=1).mean().to_numpy()
        rolling_std = pd.Series(shipping).rolling(self.window, min_periods=1).std().to_numpy()
        rolling_std = np.where(np.isnan(rolling_std) | (rolling_std == 0), 1e-6, rolling_std)

        z_scores = (shipping - rolling_mean) / rolling_std
        # Map |z| up to ~3 sigma onto [0, 1].
        anomaly_scores = np.clip((z_scores + 3) / 6, 0, 1)

        metadata = {
            "scenario_id": scenario_id,
            "baseline_name": self.name,
            "seed": seed,
            "method": f"rolling z-score on shipping, window={self.window}",
        }
        logger.info(
            "Z-score baseline: %s seed=%d, mean score=%.4f",
            scenario_id, seed, anomaly_scores.mean(),
        )
        return anomaly_scores, metadata


class EWMABaseline(BaselineRunner):
    """Exponential moving average control chart.

    ``lambda_param`` is grid-searched on the validation split (per rule 3)
    from ``lambda_grid`` unless a single fixed value is requested via
    ``tune=False``.
    """

    _DEFAULT_LAMBDA_GRID: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)

    def __init__(self, lambda_param: float = 0.2, tune: bool = True):
        super().__init__("ewma")
        self.lambda_param = lambda_param
        self.tune = tune

    def _ewma_scores(self, shipping: np.ndarray, lambda_param: float) -> np.ndarray:
        n_days = len(shipping)
        baseline_mean = shipping[_TRAIN_SLICE].mean()
        baseline_std = shipping[_TRAIN_SLICE].std()

        ewma_series = np.zeros(n_days)
        ewma_series[0] = shipping[0]
        for i in range(1, n_days):
            ewma_series[i] = lambda_param * shipping[i] + (1 - lambda_param) * ewma_series[i - 1]

        control_limit = 3 * baseline_std
        deviation = np.abs(ewma_series - baseline_mean)
        return np.clip(deviation / (control_limit + 1e-6), 0, 1)

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """EWMA control chart on shipping, with validation-tuned lambda.

        Args:
            df: DataFrame with ``shipping`` and ``y_disruption`` columns.
            scenario_id: Scenario name, for logging.
            seed: Unused — deterministic given fixed data.

        Returns:
            ``(anomaly_scores, metadata)``; metadata declares the lambda used.
        """
        shipping = _fill_missing(df["shipping"].to_numpy())
        y_true = df["y_disruption"].to_numpy()

        lambda_param = self.lambda_param
        if self.tune:
            y_val = y_true[_VAL_SLICE]
            best_f1 = -1.0
            for candidate in self._DEFAULT_LAMBDA_GRID:
                scores_val = self._ewma_scores(shipping, candidate)[_VAL_SLICE]
                f1 = _best_f1_on_val(scores_val, y_val)
                if f1 > best_f1:
                    best_f1 = f1
                    lambda_param = candidate

        anomaly_scores = self._ewma_scores(shipping, lambda_param)

        metadata = {
            "scenario_id": scenario_id,
            "baseline_name": self.name,
            "seed": seed,
            "method": f"EWMA control chart on shipping, lambda={lambda_param}",
            "lambda_tuned": self.tune,
        }
        logger.info(
            "EWMA baseline: %s seed=%d, lambda=%.2f, mean score=%.4f",
            scenario_id, seed, lambda_param, anomaly_scores.mean(),
        )
        return anomaly_scores, metadata


class CUSUMBaseline(BaselineRunner):
    """CUSUM (cumulative sum control chart) — change detection.

    ``threshold`` is grid-searched on the validation split (per rule 3);
    ``drift`` is kept fixed (not named as tunable by rule 3).
    """

    _DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0, 12.0)

    def __init__(self, threshold: float = 5.0, drift: float = 0.5, tune: bool = True):
        super().__init__("cusum")
        self.threshold = threshold
        self.drift = drift
        self.tune = tune

    def _cusum_max(self, shipping: np.ndarray) -> np.ndarray:
        n_days = len(shipping)
        baseline_mean = shipping[_TRAIN_SLICE].mean()
        baseline_std = shipping[_TRAIN_SLICE].std()
        baseline_std = max(baseline_std, 1e-6)

        cusum_pos = np.zeros(n_days)
        cusum_neg = np.zeros(n_days)
        prev_pos, prev_neg = 0.0, 0.0
        for i in range(n_days):
            delta = (shipping[i] - baseline_mean) / baseline_std
            prev_pos = max(0.0, prev_pos + delta - self.drift)
            prev_neg = max(0.0, prev_neg - delta - self.drift)
            cusum_pos[i] = prev_pos
            cusum_neg[i] = prev_neg
        return np.maximum(cusum_pos, cusum_neg)

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """CUSUM change detector on shipping, with validation-tuned threshold.

        Args:
            df: DataFrame with ``shipping`` and ``y_disruption`` columns.
            scenario_id: Scenario name, for logging.
            seed: Unused — deterministic given fixed data.

        Returns:
            ``(anomaly_scores, metadata)``; metadata declares the threshold used.
        """
        shipping = _fill_missing(df["shipping"].to_numpy())
        y_true = df["y_disruption"].to_numpy()
        cusum_max = self._cusum_max(shipping)

        threshold = self.threshold
        if self.tune:
            y_val = y_true[_VAL_SLICE]
            cusum_val = cusum_max[_VAL_SLICE]
            best_f1 = -1.0
            for candidate in self._DEFAULT_THRESHOLD_GRID:
                scores_val = np.clip(cusum_val / (candidate + 1e-6), 0, 1)
                f1 = _best_f1_on_val(scores_val, y_val)
                if f1 > best_f1:
                    best_f1 = f1
                    threshold = candidate

        anomaly_scores = np.clip(cusum_max / (threshold + 1e-6), 0, 1)

        metadata = {
            "scenario_id": scenario_id,
            "baseline_name": self.name,
            "seed": seed,
            "method": f"CUSUM on shipping, threshold={threshold}, drift={self.drift}",
            "threshold_tuned": self.tune,
        }
        logger.info(
            "CUSUM baseline: %s seed=%d, threshold=%.2f, mean score=%.4f, max cusum=%.4f",
            scenario_id, seed, threshold, anomaly_scores.mean(), cusum_max.max(),
        )
        return anomaly_scores, metadata


class SARIMABaseline(BaselineRunner):
    """Seasonal ARIMA + threshold on residuals.

    ``(order, seasonal_order)`` are fixed, not grid-searched — see module
    docstring.
    """

    def __init__(self, order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 30)):
        super().__init__("sarima")
        self.order = order
        self.seasonal_order = seasonal_order

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """Fit SARIMA on the train split, score by residual magnitude on the full series.

        Args:
            df: DataFrame with a ``shipping`` column.
            scenario_id: Scenario name, for logging.
            seed: Unused — deterministic given fixed data.

        Returns:
            ``(anomaly_scores, metadata)``. On fit failure (e.g. statsmodels
            missing, non-convergence), returns all-zero scores and records
            the error in metadata rather than raising.
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            logger.warning("statsmodels not found; SARIMA baseline will return zeros")
            return np.zeros(len(df)), {
                "scenario_id": scenario_id,
                "baseline_name": self.name,
                "seed": seed,
                "error": "statsmodels not installed",
            }

        shipping_series = pd.Series(_fill_missing(df["shipping"].to_numpy()))
        shipping_train = shipping_series.iloc[_TRAIN_SLICE]

        try:
            model = SARIMAX(
                shipping_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False)

            forecast = results.get_prediction(start=0, end=len(shipping_series) - 1)
            fitted_values = forecast.predicted_mean
            residuals = shipping_series - fitted_values

            residual_std = residuals.iloc[_TRAIN_SLICE].std()
            residual_std = max(residual_std, 1e-6)
            anomaly_scores = np.clip(np.abs(residuals) / (3 * residual_std), 0, 1)

            metadata = {
                "scenario_id": scenario_id,
                "baseline_name": self.name,
                "seed": seed,
                "method": f"SARIMA{self.order}x{self.seasonal_order} on shipping",
            }
            logger.info(
                "SARIMA baseline: %s seed=%d, mean score=%.4f",
                scenario_id, seed, anomaly_scores.mean(),
            )
            return anomaly_scores.to_numpy(), metadata

        except Exception as exc:  # noqa: BLE001 — SARIMA fit failures are data-dependent and varied
            logger.warning("SARIMA fit failed for %s: %s; returning zeros", scenario_id, exc)
            return np.zeros(len(shipping_series)), {
                "scenario_id": scenario_id,
                "baseline_name": self.name,
                "seed": seed,
                "error": str(exc),
            }


class PersistenceBaseline(BaselineRunner):
    """Naive persistence: today's score = yesterday's deviation from baseline."""

    def __init__(self, window: int = 30):
        super().__init__("persistence")
        self.window = window

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """Lagged rolling-mean residual (no same-day information used).

        Args:
            df: DataFrame with a ``shipping`` column.
            scenario_id: Scenario name, for logging.
            seed: Unused — deterministic given fixed data.

        Returns:
            ``(anomaly_scores, metadata)``.
        """
        shipping = _fill_missing(df["shipping"].to_numpy())
        n_days = len(shipping)

        rolling_mean = pd.Series(shipping).rolling(self.window, min_periods=1).mean().to_numpy()
        residuals = np.abs(shipping - rolling_mean)

        lagged = np.zeros(n_days)
        lagged[1:] = residuals[:-1]
        lagged[0] = residuals[0]  # no prior day available; use today's own residual

        residual_std = max(residuals[_TRAIN_SLICE].std(), 1e-6)
        anomaly_scores = np.clip(lagged / (3 * residual_std), 0, 1)

        metadata = {
            "scenario_id": scenario_id,
            "baseline_name": self.name,
            "seed": seed,
            "method": f"persistence (1-day lagged residual), window={self.window}",
        }
        logger.info(
            "Persistence baseline: %s seed=%d, mean score=%.4f",
            scenario_id, seed, anomaly_scores.mean(),
        )
        return anomaly_scores, metadata
