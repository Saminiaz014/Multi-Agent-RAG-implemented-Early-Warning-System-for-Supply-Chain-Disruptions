"""Vessel-routing anomaly agent.

Hybrid detector: an Isolation Forest trained on a non-disruption
baseline (the *routing baseline model*) supplies the multivariate
signal, augmented with a z-score on ``transit_volume_ratio`` to keep
single-feature rerouting waves detectable. The trained model is
versioned (default ``"hormuz_v1.0"``) so the agent can be retargeted
to other corridors by re-fitting on local data.
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
    "rerouting_percentage",
    "avg_route_deviation_km",
    "transit_volume_ratio",
    "vessels_holding",
    "alternative_route_traffic",
)
_LOCATION: str = "Strait of Hormuz"
_DEFAULT_MODEL_VERSION: str = "hormuz_v1.0"
_DEFAULT_CONTAMINATION: float = 0.08
_DEFAULT_THRESHOLD: float = 0.55
_DEFAULT_MIN_REROUTING_PCT: float = 10.0
_PERSISTENCE_DAYS: int = 2


class RoutingAgent(BaseAgent):
    """Isolation Forest + transit-ratio z-score detector.

    Args:
        config: Reads ``contamination`` (default 0.08), ``threshold``
            (default 0.55), ``min_rerouting_pct`` (default 10),
            ``model_version`` (default ``"hormuz_v1.0"``), ``weights``
            sub-block, optional ``location`` override.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="routing", config=dict(config or {}))
        self._contamination: float = float(
            self.config.get("contamination", _DEFAULT_CONTAMINATION)
        )
        self._threshold: float = float(
            self.config.get("threshold", _DEFAULT_THRESHOLD)
        )
        self._min_rerouting_pct: float = float(
            self.config.get("min_rerouting_pct", _DEFAULT_MIN_REROUTING_PCT)
        )
        weights_cfg = self.config.get("weights") or {}
        self._w_model: float = float(weights_cfg.get("model_score", 0.6))
        self._w_z: float = float(weights_cfg.get("transit_zscore", 0.4))
        self.model_version: str = str(
            self.config.get("model_version", _DEFAULT_MODEL_VERSION)
        )
        self._location: str = str(self.config.get("location") or _LOCATION)
        self._feature_columns = list(_FEATURE_COLUMNS)
        self._scaler: StandardScaler | None = None
        self._iforest: IsolationForest | None = None
        self._transit_mean: float = 0.0
        self._transit_std: float = 1.0

    # ----------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> None:
        """Train the baseline model on non-disruption rows."""
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"RoutingAgent: missing columns: {missing}")
        train = df.copy()
        if "is_disruption" in train.columns:
            mask = train["is_disruption"].astype(bool)
            train = train.loc[~mask]
            logger.info(
                "[RoutingAgent.fit] using %d non-disruption rows for fit "
                "(filtered out %d disruption rows)",
                len(train), int(mask.sum()),
            )
        features = train[self._feature_columns].ffill().dropna()
        self._scaler = StandardScaler().fit(features.to_numpy())
        scaled = self._scaler.transform(features.to_numpy())
        self._iforest = IsolationForest(
            contamination=self._contamination,
            random_state=42, n_estimators=200,
        ).fit(scaled)
        self._transit_mean = float(features["transit_volume_ratio"].mean())
        self._transit_std = float(features["transit_volume_ratio"].std(ddof=0)) or 1.0
        self._is_fitted = True
        logger.info(
            "[RoutingAgent.fit] fitted baseline model v%s on %d rows "
            "(contamination=%.3f)",
            self.model_version, len(features), self._contamination,
        )

    def train_baseline(self, historical_data: pd.DataFrame) -> None:
        """Alias for :meth:`fit` — explicit name for the baseline-training step."""
        self.fit(historical_data)

    # ------------------------------------------------------------ preprocess
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted or self._scaler is None:
            raise RuntimeError("RoutingAgent.preprocess called before fit().")
        df = data.copy().reset_index(drop=True)
        df[self._feature_columns] = df[self._feature_columns].ffill()
        df = df.dropna(subset=self._feature_columns).reset_index(drop=True)
        scaled = self._scaler.transform(df[self._feature_columns].to_numpy())
        out = pd.DataFrame(scaled, columns=self._feature_columns)
        for col in ("timestamp", "is_disruption", "rerouting_percentage", "transit_volume_ratio"):
            if col in df.columns:
                out[col] = df[col].values
        return out

    # --------------------------------------------------------------- detect
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        if not self._is_fitted or self._iforest is None:
            raise RuntimeError("RoutingAgent.detect called before fit().")
        scaled = data[self._feature_columns].to_numpy()
        iforest_raw = -self._iforest.decision_function(scaled)
        denom = iforest_raw.max() - iforest_raw.min()
        iforest_norm = (
            (iforest_raw - iforest_raw.min()) / denom
            if denom > 1e-9 else np.zeros_like(iforest_raw)
        )
        # Transit ratio is in scaled (z-score) space already after preprocess.
        transit_z = np.abs(data["transit_volume_index"].to_numpy()) \
            if "transit_volume_index" in data.columns \
            else np.abs(scaled[:, self._feature_columns.index("transit_volume_ratio")])
        transit_z_norm = np.minimum(transit_z / 3.0, 1.0)

        combined = self._w_model * iforest_norm + self._w_z * transit_z_norm
        out = data.copy()
        out["model_score"] = iforest_norm
        out["transit_zscore_norm"] = transit_z_norm
        out["anomaly_score"] = combined
        out["is_anomaly"] = combined >= self._threshold
        logger.info(
            "[RoutingAgent.detect] %d rows | %d raw flags (threshold=%.2f)",
            len(out), int(out["is_anomaly"].sum()), self._threshold,
        )
        return out

    # ------------------------------------------------------------- validate
    def validate(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Persistence (>= 2 days) + rerouting_percentage >= 10."""
        s = signals.copy().reset_index(drop=True)
        flags = s["is_anomaly"].astype(bool).to_numpy()
        persistent = np.zeros_like(flags)
        n = len(flags)
        i = 0
        while i < n:
            if flags[i]:
                j = i
                while j < n and flags[j]:
                    j += 1
                if (j - i) >= _PERSISTENCE_DAYS:
                    persistent[i:j] = True
                i = j
            else:
                i += 1
        rerouting_ok = (
            (s["rerouting_percentage"] >= self._min_rerouting_pct).to_numpy()
            if "rerouting_percentage" in s.columns
            else np.ones_like(flags)
        )
        s["validated"] = persistent & rerouting_ok
        logger.info(
            "[RoutingAgent.validate] %d/%d raw flags survived "
            "(persistence>=%d, rerouting>=%.0f%%)",
            int(s["validated"].sum()), int(flags.sum()),
            _PERSISTENCE_DAYS, self._min_rerouting_pct,
        )
        return s

    # ---------------------------------------------------------------- output
    def output(self, validated_signals: pd.DataFrame) -> list[dict[str, Any]]:
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
        for s_idx, e_idx in windows:
            w = s.iloc[s_idx:e_idx + 1]
            confidence = float(np.clip(w["anomaly_score"].mean() * 1.1, 0.0, 1.0))
            results.append({
                "agent": "routing",
                "anomaly_score": float(w["anomaly_score"].max()),
                "confidence": confidence,
                "signals": {
                    "rerouting_percentage": float(
                        w["rerouting_percentage"].max()
                        if "rerouting_percentage" in w.columns else 0.0
                    ),
                    "avg_route_deviation_km": float(
                        w["avg_route_deviation_km"].max()
                        if "avg_route_deviation_km" in w.columns else 0.0
                    ),
                    "transit_volume_ratio": float(
                        w["transit_volume_ratio"].min()
                        if "transit_volume_ratio" in w.columns else 0.0
                    ),
                    "vessels_holding": int(
                        w["vessels_holding"].max()
                        if "vessels_holding" in w.columns else 0
                    ),
                    "alternative_route_traffic": float(
                        w["alternative_route_traffic"].max()
                        if "alternative_route_traffic" in w.columns else 0.0
                    ),
                },
                "model_version": self.model_version,
                "start_timestamp": str(pd.Timestamp(w["timestamp"].iloc[0]).date()),
                "end_timestamp": str(pd.Timestamp(w["timestamp"].iloc[-1]).date()),
                "location": self._location,
            })
        logger.info("[RoutingAgent.output] produced %d windows", len(results))
        return results

    # ------------------------------------------------------------------ run
    def run(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        if not self._is_fitted:
            self.fit(data)
        scored = self.detect(self.preprocess(data))
        return self.output(self.validate(scored))

    def run_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            self.fit(data)
        # Merge unscaled raw columns back for downstream reporting.
        scored = self.detect(self.preprocess(data))
        raw = data.copy().reset_index(drop=True)
        if len(scored) == len(raw):
            for col in (
                "rerouting_percentage", "avg_route_deviation_km",
                "transit_volume_ratio", "vessels_holding",
                "alternative_route_traffic",
            ):
                if col in raw.columns and col not in scored.columns:
                    scored[col] = raw[col].values
                elif col in raw.columns:
                    # Replace scaled values with raw for human-readable reports.
                    if col in scored.columns:
                        scored[col] = raw[col].values
            if "timestamp" in raw.columns:
                scored["timestamp"] = raw["timestamp"].values
        return self.validate(scored)

    def to_detection_result(self, validated: pd.DataFrame) -> DetectionResult:
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=validated["anomaly_score"].to_numpy(dtype=float),
            anomaly_flags=validated["validated"].to_numpy(dtype=bool),
            feature_names=list(self._feature_columns),
            metadata={
                "contamination": self._contamination,
                "threshold": self._threshold,
                "model_version": self.model_version,
                "location": self._location,
            },
        )
