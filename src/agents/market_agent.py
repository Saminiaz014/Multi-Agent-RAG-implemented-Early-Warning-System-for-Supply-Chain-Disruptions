"""Market anomaly detection agent for Brent / trade volume / freight signals.

This agent is the secondary signal source in the multi-agent DSS pipeline.
Market features (oil price, trade throughput, freight rates) react to
physical-flow disruptions with a 1-2 day lag and exhibit strong temporal
autocorrelation, so the detection strategy here differs from
:class:`~src.agents.shipping_agent.ShippingAgent`:

* No Isolation Forest. Instead, **rolling-window Z-scores** (default 30-day
  trailing window) are computed per feature on every call — this adapts to
  slow drift in the underlying time series and avoids the look-ahead bias
  a global StandardScaler would introduce on autocorrelated data.
* The combined anomaly score is a **weighted mean of absolute z-scores**
  (oil 0.40, trade volume 0.35, freight 0.25), reflecting the relative
  diagnostic value each feature has for a Strait of Hormuz disruption.
* Validation requires the **oil price** z-score plus **at least one** of
  the other two features to clear ``z_threshold`` simultaneously — a
  hard-coded asymmetry, since oil is the most direct price-side signal of
  a Hormuz disruption and a move in trade volume or freight without a
  corresponding oil move is far more likely to be ordinary market noise.

The agent is deterministic and stateless across :meth:`run` invocations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent, DetectionResult

logger = logging.getLogger(__name__)


_FEATURE_COLUMNS: tuple[str, ...] = (
    "brent_crude_usd",
    "trade_volume_index",
    "freight_rate_index",
)
_LOCATION: str = "Strait of Hormuz"
_FEATURE_WEIGHTS: dict[str, float] = {
    "brent_crude_usd": 0.40,
    "trade_volume_index": 0.35,
    "freight_rate_index": 0.25,
}
_PERSISTENCE_DAYS: int = 2
_DEFAULT_WINDOW: int = 30
_OIL_FEATURE: str = "brent_crude_usd"


class MarketAgent(BaseAgent):
    """Rolling Z-score anomaly detector for market signals.

    Each row is scored by computing per-feature z-scores against a trailing
    rolling window, taking a feature-weighted mean of their absolute
    values, and normalising the result by ``z_threshold`` so that the
    combined ``anomaly_score`` lies in ``[0, 1]`` (saturating at 1.0 when
    the weighted |z| equals the configured threshold).

    Validation gates the raw flag with two checks:

    1. **Persistence** — the row must sit in a run of ``>= 2`` flagged days.
    2. **Oil-led multi-feature** — the oil-price z-score must exceed
       ``z_threshold`` AND at least one of the trade-volume / freight-rate
       z-scores must also exceed ``z_threshold``. An oil spike on its own,
       or a freight spike with quiet oil, is treated as ordinary noise.

    Configuration (read from ``config['agents']['market']`` in
    ``settings.yaml``):

    ============== ======= =====================================================
    Key            Default Description
    ============== ======= =====================================================
    z_threshold    2.5     Per-feature absolute z-score that counts as elevated
    threshold      0.55    Min combined score to set ``is_anomaly=True``
    window         30      Trailing rolling-window length, in days
    ============== ======= =====================================================

    Args:
        config: Agent-specific configuration block. The keys above are
            consulted; sensible defaults apply for any missing key.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="market", config=dict(config or {}))
        self._feature_columns: list[str] = list(_FEATURE_COLUMNS)
        self._z_threshold: float = float(self.config.get("z_threshold", 2.5))
        self._threshold: float = float(self.config.get("threshold", 0.55))
        self._window: int = int(self.config.get("window", _DEFAULT_WINDOW))

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> None:
        """Validate schema and mark the agent fitted.

        Rolling Z-scores are computed inline during :meth:`preprocess`, so
        no global parameters need to be learned. ``fit`` exists to satisfy
        the :class:`BaseAgent` ABC contract and to fail fast if the
        incoming frame is missing required columns.

        Args:
            df: Historical market frame; must contain the three feature
                columns. ``is_disruption`` and ``timestamp`` are optional.
        """
        self._validate_columns(df)
        self._is_fitted = True
        logger.info(
            "[MarketAgent.fit] schema validated | window=%d | z_threshold=%.2f",
            self._window,
            self._z_threshold,
        )

    # ----------------------------------------------------------- preprocess
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Attach trailing rolling mean and standard deviation per feature.

        Uses ``min_periods=2`` so the very first row is dropped (a
        single-sample std is undefined) but subsequent early rows still
        get a rolling estimate from whatever history is available. The
        trailing window means no look-ahead leakage: each row's rolling
        statistics depend only on itself and prior rows.

        Args:
            data: Raw market frame.

        Returns:
            Frame with the original feature columns preserved, plus
            ``<feature>_rolling_mean`` and ``<feature>_rolling_std`` for
            each of the three features. ``timestamp`` and
            ``is_disruption`` (when present) are carried through.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "MarketAgent.preprocess called before fit() — call fit() first."
            )
        self._validate_columns(data)

        df = data.copy()
        df[self._feature_columns] = df[self._feature_columns].ffill()
        df = df.dropna(subset=self._feature_columns).reset_index(drop=True)

        for col in self._feature_columns:
            roll = df[col].rolling(window=self._window, min_periods=2)
            df[f"{col}_rolling_mean"] = roll.mean()
            df[f"{col}_rolling_std"] = roll.std(ddof=0)

        before = len(df)
        df = df.dropna(
            subset=[f"{c}_rolling_std" for c in self._feature_columns]
        ).reset_index(drop=True)
        logger.info(
            "[MarketAgent.preprocess] rolling stats (window=%d) | %d → %d rows",
            self._window,
            before,
            len(df),
        )
        return df

    # --------------------------------------------------------------- detect
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        """Score each row by feature-weighted absolute z-scores.

        Per-feature ``z = (value - rolling_mean) / rolling_std`` is
        computed first; rows where the rolling std collapses below
        ``1e-9`` get ``z = 0`` for that feature (degenerate flat window).
        The combined score is the configured weighted mean of
        ``|z|``, normalised by ``z_threshold`` and clipped at 1.0 so a
        weighted |z| equal to the threshold saturates the score.

        Args:
            data: Output of :meth:`preprocess`.

        Returns:
            ``data`` with ``oil_zscore``, ``trade_volume_zscore``,
            ``freight_zscore``, ``anomaly_score`` (in ``[0, 1]``), and
            ``is_anomaly`` (``anomaly_score >= threshold``) appended.
        """
        if not self._is_fitted:
            raise RuntimeError("MarketAgent.detect called before fit().")

        out = data.copy()
        weighted = np.zeros(len(out), dtype=float)
        z_columns: dict[str, str] = {
            "brent_crude_usd": "oil_zscore",
            "trade_volume_index": "trade_volume_zscore",
            "freight_rate_index": "freight_zscore",
        }

        for col, z_col in z_columns.items():
            mu = out[f"{col}_rolling_mean"].to_numpy()
            sd = out[f"{col}_rolling_std"].to_numpy()
            sd_safe = np.where(sd > 1e-9, sd, np.nan)
            z = (out[col].to_numpy() - mu) / sd_safe
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            out[z_col] = z
            weighted += _FEATURE_WEIGHTS[col] * np.abs(z)

        out["anomaly_score"] = np.minimum(weighted / self._z_threshold, 1.0)
        out["is_anomaly"] = out["anomaly_score"] >= self._threshold
        logger.info(
            "[MarketAgent.detect] %d rows scored | %d raw anomalies "
            "(threshold=%.2f, z_threshold=%.2f)",
            len(out),
            int(out["is_anomaly"].sum()),
            self._threshold,
            self._z_threshold,
        )
        return out

    # ------------------------------------------------------------- validate
    def validate(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Filter raw flags with persistence + oil-led feature gates.

        A row is marked ``validated=True`` iff:

        1. ``is_anomaly`` is part of a run of length
           ``>= _PERSISTENCE_DAYS`` (suppresses single-day spikes).
        2. ``|oil_zscore| > z_threshold`` (oil is the lead indicator).
        3. At least one of ``|trade_volume_zscore|`` or
           ``|freight_zscore|`` exceeds ``z_threshold`` (corroboration).

        Args:
            signals: Output of :meth:`detect`.

        Returns:
            ``signals`` with ``oil_elevated``, ``other_elevated``, and
            the boolean ``validated`` columns appended.
        """
        s = signals.copy().reset_index(drop=True)
        flags = s["is_anomaly"].astype(bool).to_numpy()

        oil_elevated = (s["oil_zscore"].abs() > self._z_threshold).to_numpy()
        other_elevated = (
            (s["trade_volume_zscore"].abs() > self._z_threshold)
            | (s["freight_zscore"].abs() > self._z_threshold)
        ).to_numpy()

        # Combined per-row gate: anomalous AND oil-led AND corroborated
        combined = flags & oil_elevated & other_elevated

        # Persistence — the combined gate itself must hold for >= 2 days,
        # so validated flags always form runs of length 2+ (no isolated days).
        validated = np.zeros_like(combined)
        n = len(combined)
        i = 0
        while i < n:
            if combined[i]:
                j = i
                while j < n and combined[j]:
                    j += 1
                if (j - i) >= _PERSISTENCE_DAYS:
                    validated[i:j] = True
                i = j
            else:
                i += 1

        s["oil_elevated"] = oil_elevated
        s["other_elevated"] = other_elevated
        s["validated"] = validated
        logger.info(
            "[MarketAgent.validate] %d/%d raw anomalies survived validation",
            int(s["validated"].sum()),
            int(flags.sum()),
        )
        return s

    # ---------------------------------------------------------------- output
    def output(self, validated_signals: pd.DataFrame) -> list[dict[str, Any]]:
        """Group consecutive validated days into market anomaly windows.

        Confidence per window is the mean fraction of features elevated
        per day — oil counts as one feature, plus the trade-volume and
        freight features for two more. Maximum 3, divided by 3 to land in
        ``[0, 1]``.

        Args:
            validated_signals: Output of :meth:`validate`.

        Returns:
            One dictionary per contiguous validated window, conforming to
            the same unified anomaly-report schema used by
            :class:`ShippingAgent.output` but stamped ``"agent": "market"``
            with market-specific signal keys.
        """
        s = validated_signals.reset_index(drop=True)
        flags = s["validated"].astype(bool).to_numpy()

        windows: list[tuple[int, int]] = []
        start: int | None = None
        for i, flag in enumerate(flags):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                windows.append((start, i - 1))
                start = None
        if start is not None:
            windows.append((start, len(flags) - 1))

        results: list[dict[str, Any]] = []
        zt = self._z_threshold
        for start_idx, end_idx in windows:
            w = s.iloc[start_idx : end_idx + 1]
            elevated_per_day = (
                (w["oil_zscore"].abs() > zt).astype(int)
                + (w["trade_volume_zscore"].abs() > zt).astype(int)
                + (w["freight_zscore"].abs() > zt).astype(int)
            )
            confidence = float(elevated_per_day.mean() / 3.0)
            start_ts = w["timestamp"].iloc[0] if "timestamp" in w.columns else None
            end_ts = w["timestamp"].iloc[-1] if "timestamp" in w.columns else None
            results.append(
                {
                    "agent": "market",
                    "anomaly_score": float(w["anomaly_score"].max()),
                    "confidence": confidence,
                    "signals": {
                        "oil_zscore": float(w["oil_zscore"].abs().max()),
                        "trade_volume_zscore": float(
                            w["trade_volume_zscore"].abs().max()
                        ),
                        "freight_zscore": float(w["freight_zscore"].abs().max()),
                    },
                    "start_timestamp": (
                        str(pd.Timestamp(start_ts).date())
                        if start_ts is not None
                        else None
                    ),
                    "end_timestamp": (
                        str(pd.Timestamp(end_ts).date())
                        if end_ts is not None
                        else None
                    ),
                    "location": _LOCATION,
                }
            )

        logger.info(
            "[MarketAgent.output] produced %d anomaly windows", len(results)
        )
        return results

    # ------------------------------------------------------------------ run
    def run(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Run the full preprocess → detect → validate → output pipeline.

        If the agent has not yet been fitted, it auto-fits on ``data``.
        Rolling stats mean ``fit`` is essentially a schema check, so
        auto-fit is safe in production as well as evaluation.

        Args:
            data: Raw market frame.

        Returns:
            List of validated anomaly window reports.
        """
        if not self._is_fitted:
            logger.info("[MarketAgent.run] auto-fitting on input frame")
            self.fit(data)

        logger.info("[MarketAgent.run] step 1/4 preprocess")
        prepped = self.preprocess(data)
        logger.info("[MarketAgent.run] step 2/4 detect")
        scored = self.detect(prepped)
        logger.info("[MarketAgent.run] step 3/4 validate")
        validated = self.validate(scored)
        logger.info("[MarketAgent.run] step 4/4 output")
        reports = self.output(validated)
        logger.info(
            "[MarketAgent.run] complete | %d windows reported", len(reports)
        )
        return reports

    # ----------------------------------------------------- helper utilities
    def run_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the pipeline and return the per-row validated frame.

        Used by evaluation harnesses that need per-row predictions for
        confusion-matrix scoring; :meth:`run` only emits aggregated
        windows.

        Args:
            data: Raw market frame.

        Returns:
            DataFrame from :meth:`validate` — z-scores, anomaly scores,
            and the boolean ``validated`` column.
        """
        if not self._is_fitted:
            self.fit(data)
        prepped = self.preprocess(data)
        scored = self.detect(prepped)
        return self.validate(scored)

    def to_detection_result(self, validated: pd.DataFrame) -> DetectionResult:
        """Adapt the validated frame to the :class:`DetectionResult` contract.

        Lets the agent slot into
        :class:`~src.aggregation.risk_engine.RiskEngine.aggregate` with
        the same signature as :class:`ShippingAgent.to_detection_result`.

        Args:
            validated: Output of :meth:`validate`.

        Returns:
            DetectionResult with the combined anomaly score and validated
            boolean flag.
        """
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=validated["anomaly_score"].to_numpy(dtype=float),
            anomaly_flags=validated["validated"].to_numpy(dtype=bool),
            feature_names=list(self._feature_columns),
            metadata={
                "z_threshold": self._z_threshold,
                "threshold": self._threshold,
                "window": self._window,
            },
        )

    # ---------------------------------------------------------------- guard
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Raise if any of the required feature columns are missing."""
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"MarketAgent: missing required feature columns: {missing}"
            )
