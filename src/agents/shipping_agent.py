"""Shipping anomaly detection agent for the Strait of Hormuz corridor.

This agent is the primary signal source in the multi-agent DSS pipeline. It
ingests preprocessed shipping data (vessel counts, transit delays, corridor
congestion), produces anomaly scores via an Isolation Forest combined with a
per-feature Z-score fallback, validates raw signals to suppress false
positives, and emits structured anomaly reports for downstream aggregation,
explainability, and RAG context retrieval.

The detection logic is deterministic given the same input and configuration
(``random_state=42`` is pinned on the Isolation Forest). The agent is
stateless across :meth:`run` invocations — only the fitted scaler and
Isolation Forest are retained between calls, both produced by :meth:`fit`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.agents.base_agent import BaseAgent, DetectionResult

logger = logging.getLogger(__name__)


_FEATURE_COLUMNS: tuple[str, ...] = (
    "vessel_count",
    "avg_delay_hours",
    "congestion_index",
)
_LOCATION: str = "Strait of Hormuz"
_FEATURE_ELEVATION_Z: float = 1.5
_PERSISTENCE_DAYS: int = 2
_MIN_FEATURES_ELEVATED: int = 2


class ShippingAgent(BaseAgent):
    """Multi-feature anomaly detector for Strait of Hormuz shipping signals.

    The agent combines an Isolation Forest (primary, contamination-driven
    multivariate detector) with a per-feature Z-score fallback (secondary,
    catches univariate outliers the forest may smooth over). Raw flags are
    then validated by requiring (a) ``>= 2`` consecutive anomalous days and
    (b) ``>= 2`` of the three features showing elevated absolute z-scores,
    eliminating single-day noise spikes and lone-feature artefacts.

    Configuration (read from ``config['agents']['shipping']`` in
    ``settings.yaml``):

    ============== ======= =====================================================
    Key            Default Description
    ============== ======= =====================================================
    contamination  0.1     Expected anomaly fraction for the Isolation Forest
    threshold      0.65    Minimum combined score to flag a row anomalous
    z_threshold    3.0     Absolute z-score at which a feature is "extreme"
    ============== ======= =====================================================

    Args:
        config: Agent-specific configuration block. The keys above are
            consulted; sensible defaults apply for any missing key.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="shipping", config=dict(config or {}))
        self._scaler: StandardScaler | None = None
        self._iforest: IsolationForest | None = None
        self._feature_columns: list[str] = list(_FEATURE_COLUMNS)
        self._contamination: float = float(self.config.get("contamination", 0.1))
        self._threshold: float = float(self.config.get("threshold", 0.65))
        self._z_threshold: float = float(self.config.get("z_threshold", 3.0))

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> None:
        """Fit the StandardScaler and Isolation Forest on historical data.

        To avoid leakage, when a ground-truth ``is_disruption`` column is
        present the scaler and forest are fit only on the rows where it is
        ``False`` — i.e. exclusively on normal-period observations. When the
        column is absent the entire frame is used (production mode).

        Args:
            df: Historical feature frame; must contain the three shipping
                feature columns and may optionally contain
                ``is_disruption`` for clean training-window selection.
        """
        self._validate_columns(df)
        train = df.copy()
        if "is_disruption" in train.columns:
            mask = train["is_disruption"].astype(bool)
            train = train.loc[~mask]
            logger.info(
                "[ShippingAgent.fit] using %d non-disruption rows for fit "
                "(filtered out %d disruption rows)",
                len(train),
                int(mask.sum()),
            )
        features = train[self._feature_columns].ffill().dropna()

        self._scaler = StandardScaler().fit(features.to_numpy())
        scaled = self._scaler.transform(features.to_numpy())
        self._iforest = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=200,
        ).fit(scaled)
        self._is_fitted = True
        logger.info(
            "[ShippingAgent.fit] fitted scaler+IsolationForest on %d rows "
            "(contamination=%.3f)",
            len(features),
            self._contamination,
        )

    # ----------------------------------------------------------- preprocess
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select features, fill gaps, and apply the fitted scaler.

        Args:
            data: Raw shipping frame containing at minimum the three
                feature columns; ``timestamp`` is preserved if present.

        Returns:
            Frame with the original timestamp column (if any) plus the
            three feature columns transformed onto the scaler's standard
            space (mean 0, unit variance, in z-score units).
        """
        if not self._is_fitted or self._scaler is None:
            raise RuntimeError(
                "ShippingAgent.preprocess called before fit() — call fit() "
                "with a clean training window first."
            )
        self._validate_columns(data)

        df = data.copy()
        df[self._feature_columns] = df[self._feature_columns].ffill()
        df = df.dropna(subset=self._feature_columns).reset_index(drop=True)

        scaled = self._scaler.transform(df[self._feature_columns].to_numpy())
        out = pd.DataFrame(scaled, columns=self._feature_columns)
        if "timestamp" in df.columns:
            out.insert(0, "timestamp", pd.to_datetime(df["timestamp"]).values)
        if "is_disruption" in df.columns:
            out["is_disruption"] = df["is_disruption"].astype(bool).values
        logger.info(
            "[ShippingAgent.preprocess] scaled %d rows over %d features",
            len(out),
            len(self._feature_columns),
        )
        return out

    # --------------------------------------------------------------- detect
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        """Score each row with Isolation Forest + Z-score fallback.

        The Isolation Forest score is ``-decision_function`` min-max
        normalised into ``[0, 1]`` (1 = most anomalous). The Z-score
        fallback takes the maximum absolute scaled value across the three
        features and clips it at ``z_threshold`` before normalising. The
        combined score is ``0.7 * iforest + 0.3 * max_abs_z``.

        Args:
            data: Output of :meth:`preprocess` — already standardised.

        Returns:
            Frame with the input columns plus per-feature z-scores,
            ``isolation_score``, ``max_zscore_norm``, ``anomaly_score``,
            and the boolean ``is_anomaly`` flag.
        """
        if not self._is_fitted or self._iforest is None:
            raise RuntimeError("ShippingAgent.detect called before fit().")

        scaled = data[self._feature_columns].to_numpy()
        iforest_raw = self._iforest.decision_function(scaled)
        iforest_score = -iforest_raw  # higher = more anomalous
        denom = iforest_score.max() - iforest_score.min()
        iforest_norm = (
            (iforest_score - iforest_score.min()) / denom
            if denom > 1e-9
            else np.zeros_like(iforest_score)
        )

        max_abs_z = np.max(np.abs(scaled), axis=1)
        max_z_norm = np.minimum(max_abs_z / self._z_threshold, 1.0)

        combined = 0.7 * iforest_norm + 0.3 * max_z_norm

        out = data.copy()
        out["vessel_count_zscore"] = scaled[:, 0]
        out["delay_zscore"] = scaled[:, 1]
        out["congestion_zscore"] = scaled[:, 2]
        out["isolation_score"] = iforest_norm
        out["max_zscore_norm"] = max_z_norm
        out["anomaly_score"] = combined
        out["is_anomaly"] = combined >= self._threshold
        logger.info(
            "[ShippingAgent.detect] %d rows scored | %d raw anomalies "
            "(threshold=%.3f)",
            len(out),
            int(out["is_anomaly"].sum()),
            self._threshold,
        )
        return out

    # ------------------------------------------------------------- validate
    def validate(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Filter raw flags to reduce false positives.

        A row is marked ``validated=True`` only when both:

        1. **Persistence** — the anomaly is part of a run of at least
           ``_PERSISTENCE_DAYS`` consecutive flagged rows (single-day
           spikes are dropped).
        2. **Multi-feature** — at least ``_MIN_FEATURES_ELEVATED`` of the
           three features have ``|z| > _FEATURE_ELEVATION_Z`` on that row
           (lone-feature outliers are dropped).

        Args:
            signals: Output of :meth:`detect`.

        Returns:
            ``signals`` with an added boolean ``validated`` column.
        """
        s = signals.copy().reset_index(drop=True)
        is_anom = s["is_anomaly"].astype(bool).to_numpy()

        # Persistence: each True must sit in a run of length >= 2
        persistent = np.zeros_like(is_anom)
        n = len(is_anom)
        i = 0
        while i < n:
            if is_anom[i]:
                j = i
                while j < n and is_anom[j]:
                    j += 1
                if (j - i) >= _PERSISTENCE_DAYS:
                    persistent[i:j] = True
                i = j
            else:
                i += 1

        # Multi-feature elevation
        elevated = (
            (s["vessel_count_zscore"].abs() > _FEATURE_ELEVATION_Z).astype(int)
            + (s["delay_zscore"].abs() > _FEATURE_ELEVATION_Z).astype(int)
            + (s["congestion_zscore"].abs() > _FEATURE_ELEVATION_Z).astype(int)
        )
        feature_check = (elevated >= _MIN_FEATURES_ELEVATED).to_numpy()

        s["features_elevated"] = elevated.values
        s["validated"] = persistent & feature_check
        logger.info(
            "[ShippingAgent.validate] %d/%d raw anomalies survived validation",
            int(s["validated"].sum()),
            int(is_anom.sum()),
        )
        return s

    # ---------------------------------------------------------------- output
    def output(self, validated_signals: pd.DataFrame) -> list[dict[str, Any]]:
        """Group consecutive validated days into anomaly windows.

        Confidence is the mean per-day fraction of features elevated
        (``features_elevated / 3``) inside the window — windows where all
        three features fire on every day get confidence 1.0.

        Args:
            validated_signals: Output of :meth:`validate`.

        Returns:
            List of dictionaries — one per contiguous validated window —
            each conforming to the unified anomaly-report schema used by
            the downstream aggregator.
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
        for start_idx, end_idx in windows:
            w = s.iloc[start_idx : end_idx + 1]
            confidence = float(w["features_elevated"].mean() / 3.0)
            start_ts = w["timestamp"].iloc[0] if "timestamp" in w.columns else None
            end_ts = w["timestamp"].iloc[-1] if "timestamp" in w.columns else None
            results.append(
                {
                    "agent": "shipping",
                    "anomaly_score": float(w["anomaly_score"].max()),
                    "confidence": confidence,
                    "signals": {
                        "vessel_count_zscore": float(
                            w["vessel_count_zscore"].abs().max()
                        ),
                        "delay_zscore": float(w["delay_zscore"].abs().max()),
                        "congestion_zscore": float(
                            w["congestion_zscore"].abs().max()
                        ),
                    },
                    "start_timestamp": (
                        str(pd.Timestamp(start_ts).date()) if start_ts is not None else None
                    ),
                    "end_timestamp": (
                        str(pd.Timestamp(end_ts).date()) if end_ts is not None else None
                    ),
                    "location": _LOCATION,
                }
            )

        logger.info(
            "[ShippingAgent.output] produced %d anomaly windows", len(results)
        )
        return results

    # ------------------------------------------------------------------ run
    def run(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Run the full preprocess → detect → validate → output pipeline.

        If the agent has not yet been fitted, it auto-fits on ``data`` —
        useful for unsupervised one-shot evaluation. In production the
        caller should call :meth:`fit` explicitly with a clean training
        window first.

        Args:
            data: Raw shipping frame.

        Returns:
            List of validated anomaly window reports (see :meth:`output`).
        """
        if not self._is_fitted:
            logger.info("[ShippingAgent.run] auto-fitting on input frame")
            self.fit(data)

        logger.info("[ShippingAgent.run] step 1/4 preprocess")
        scaled = self.preprocess(data)
        logger.info("[ShippingAgent.run] step 2/4 detect")
        scored = self.detect(scaled)
        logger.info("[ShippingAgent.run] step 3/4 validate")
        validated = self.validate(scored)
        logger.info("[ShippingAgent.run] step 4/4 output")
        reports = self.output(validated)
        logger.info(
            "[ShippingAgent.run] complete | %d windows reported", len(reports)
        )
        return reports

    # ----------------------------------------------------- helper utilities
    def run_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the pipeline and return the per-row validated frame.

        Convenience wrapper used by evaluation harnesses that need access
        to per-row predictions for confusion-matrix / TPR / FPR scoring.
        :meth:`run` only emits aggregated windows.

        Args:
            data: Raw shipping frame.

        Returns:
            The DataFrame returned by :meth:`validate` — i.e. detection
            scores plus the boolean ``validated`` column.
        """
        if not self._is_fitted:
            self.fit(data)
        scaled = self.preprocess(data)
        scored = self.detect(scaled)
        return self.validate(scored)

    def to_detection_result(self, validated: pd.DataFrame) -> DetectionResult:
        """Adapt the validated frame to the :class:`DetectionResult` contract.

        Lets the agent slot into the existing
        :class:`~src.aggregation.risk_engine.RiskEngine` aggregation path
        despite :meth:`detect` returning a richer DataFrame.

        Args:
            validated: Output of :meth:`validate`.

        Returns:
            DetectionResult populated with the combined anomaly score and
            the validated flag.
        """
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=validated["anomaly_score"].to_numpy(dtype=float),
            anomaly_flags=validated["validated"].to_numpy(dtype=bool),
            feature_names=list(self._feature_columns),
            metadata={
                "contamination": self._contamination,
                "threshold": self._threshold,
                "z_threshold": self._z_threshold,
            },
        )

    # ---------------------------------------------------------------- guard
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Raise if any of the required feature columns are missing."""
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"ShippingAgent: missing required feature columns: {missing}"
            )
