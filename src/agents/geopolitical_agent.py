"""Geopolitical risk-scoring agent.

Unlike the shipping and market agents, the geopolitical signal is
categorical/event-based rather than a continuous time series, so this
agent uses **weighted composite scoring** with persistence + breadth
validation rather than statistical anomaly detection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent, DetectionResult

logger = logging.getLogger(__name__)

_FEATURE_COLUMNS: tuple[str, ...] = (
    "sanctions_severity",
    "military_activity_index",
    "diplomatic_incident_score",
    "regime_stability_index",
)
_LOCATION: str = "Strait of Hormuz"
_DEFAULT_WEIGHTS: dict[str, float] = {
    "sanctions": 0.35,
    "military": 0.25,
    "diplomatic": 0.25,
    "stability": 0.15,
}
_PERSISTENCE_DAYS: int = 3
_MIN_FEATURES_ELEVATED: int = 2
_ELEVATION_THRESHOLD: float = 0.4
_ROLLING_WINDOW: int = 14


class GeopoliticalAgent(BaseAgent):
    """Weighted-composite scorer for geopolitical risk.

    Args:
        config: Reads ``threshold`` (default 0.5), ``weights`` sub-block,
            and an optional ``location`` override.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="geopolitical", config=dict(config or {}))
        self._threshold: float = float(self.config.get("threshold", 0.5))
        weights_cfg = self.config.get("weights") or {}
        self._weights: dict[str, float] = {
            "sanctions": float(weights_cfg.get("sanctions", _DEFAULT_WEIGHTS["sanctions"])),
            "military": float(weights_cfg.get("military", _DEFAULT_WEIGHTS["military"])),
            "diplomatic": float(weights_cfg.get("diplomatic", _DEFAULT_WEIGHTS["diplomatic"])),
            "stability": float(weights_cfg.get("stability", _DEFAULT_WEIGHTS["stability"])),
        }
        self._location: str = str(self.config.get("location") or _LOCATION)
        self._feature_columns = list(_FEATURE_COLUMNS)

    def set_weights(
        self,
        sanctions: float,
        military: float,
        diplomatic: float,
        stability: float,
    ) -> None:
        """Override the composite weights, normalised to sum to 1.0.

        Args:
            sanctions: Raw weight on sanctions severity.
            military: Raw weight on military-activity index.
            diplomatic: Raw weight on diplomatic-incident score.
            stability: Raw weight on (inverted) regime-stability index.
        """
        total = sanctions + military + diplomatic + stability
        if total <= 0:
            raise ValueError("GeopoliticalAgent.set_weights: weights must sum to > 0.")
        self._weights = {
            "sanctions": sanctions / total,
            "military": military / total,
            "diplomatic": diplomatic / total,
            "stability": stability / total,
        }

    def set_threshold(self, threshold: float) -> None:
        """Override the composite-score cutoff for flagging a row."""
        self._threshold = float(threshold)

    # ----------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> None:
        """Schema check only; weighted composite has no learned parameters."""
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"GeopoliticalAgent: missing columns: {missing}")
        self._is_fitted = True
        logger.info(
            "[GeopoliticalAgent.fit] schema validated | threshold=%.2f | weights=%s",
            self._threshold, self._weights,
        )

    # ------------------------------------------------------------ preprocess
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add 14-day rolling baselines + per-feature deviations."""
        if not self._is_fitted:
            raise RuntimeError("GeopoliticalAgent.preprocess called before fit().")
        df = data.copy().reset_index(drop=True)
        for col in self._feature_columns:
            rolling = df[col].rolling(window=_ROLLING_WINDOW, min_periods=2).mean()
            df[f"{col}_baseline"] = rolling.fillna(df[col])
            df[f"{col}_deviation"] = df[col] - df[f"{col}_baseline"]
        return df

    # --------------------------------------------------------------- detect
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        """Weighted composite score with sigmoid compression."""
        if not self._is_fitted:
            raise RuntimeError("GeopoliticalAgent.detect called before fit().")
        out = data.copy()
        raw = (
            self._weights["sanctions"] * out["sanctions_severity"]
            + self._weights["military"] * out["military_activity_index"]
            + self._weights["diplomatic"] * out["diplomatic_incident_score"]
            + self._weights["stability"] * (1.0 - out["regime_stability_index"])
        )
        # Sigmoid compression centred at 0.5 with gain 6 so a raw score of
        # 0.5 maps to 0.5 and extremes saturate gracefully.
        anomaly_score = 1.0 / (1.0 + np.exp(-6.0 * (raw - 0.5)))
        out["raw_composite"] = raw.astype(float)
        out["anomaly_score"] = anomaly_score.astype(float)
        out["is_anomaly"] = out["anomaly_score"] >= self._threshold
        logger.info(
            "[GeopoliticalAgent.detect] %d rows | %d raw flags (threshold=%.2f)",
            len(out), int(out["is_anomaly"].sum()), self._threshold,
        )
        return out

    # ------------------------------------------------------------- validate
    def validate(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Persistence (>= 3 days) + breadth (>= 2 of 4 features elevated)."""
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
        elevated = sum(
            (s[c] > _ELEVATION_THRESHOLD).astype(int) for c in (
                "sanctions_severity",
                "military_activity_index",
                "diplomatic_incident_score",
            )
        ) + (s["regime_stability_index"] < (1.0 - _ELEVATION_THRESHOLD)).astype(int)
        breadth_ok = (elevated >= _MIN_FEATURES_ELEVATED).to_numpy()
        s["features_elevated"] = elevated.to_numpy()
        s["validated"] = persistent & breadth_ok
        logger.info(
            "[GeopoliticalAgent.validate] %d/%d raw flags survived",
            int(s["validated"].sum()), int(flags.sum()),
        )
        return s

    # ---------------------------------------------------------------- output
    def output(self, validated_signals: pd.DataFrame) -> list[dict[str, Any]]:
        """Group consecutive validated days into window reports."""
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
            incidents: list[str] = []
            if "flagged_incidents" in w.columns:
                for raw in w["flagged_incidents"]:
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else []
                    except json.JSONDecodeError:
                        parsed = []
                    incidents.extend(parsed)
            confidence = float(w["features_elevated"].mean() / 4.0)
            results.append({
                "agent": "geopolitical",
                "anomaly_score": float(w["anomaly_score"].max()),
                "confidence": confidence,
                "signals": {
                    "sanctions_severity": float(w["sanctions_severity"].max()),
                    "military_activity_index": float(w["military_activity_index"].max()),
                    "diplomatic_incident_score": float(w["diplomatic_incident_score"].max()),
                    "regime_stability_index": float(w["regime_stability_index"].min()),
                },
                "flagged_incidents": sorted(set(incidents)),
                "start_timestamp": str(pd.Timestamp(w["timestamp"].iloc[0]).date()),
                "end_timestamp": str(pd.Timestamp(w["timestamp"].iloc[-1]).date()),
                "location": self._location,
            })
        logger.info("[GeopoliticalAgent.output] produced %d windows", len(results))
        return results

    # ------------------------------------------------------------------ run
    def run(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """preprocess → detect → validate → output."""
        if not self._is_fitted:
            self.fit(data)
        scored = self.detect(self.preprocess(data))
        return self.output(self.validate(scored))

    def run_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the per-row validated frame."""
        if not self._is_fitted:
            self.fit(data)
        return self.validate(self.detect(self.preprocess(data)))

    def to_detection_result(self, validated: pd.DataFrame) -> DetectionResult:
        """Adapt to the `DetectionResult` contract."""
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=validated["anomaly_score"].to_numpy(dtype=float),
            anomaly_flags=validated["validated"].to_numpy(dtype=bool),
            feature_names=list(self._feature_columns),
            metadata={
                "threshold": self._threshold,
                "weights": dict(self._weights),
                "location": self._location,
            },
        )
