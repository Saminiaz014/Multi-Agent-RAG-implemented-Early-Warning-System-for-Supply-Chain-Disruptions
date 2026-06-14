"""Natural-disaster risk-scoring agent.

Uses the same weighted-composite pattern as
:class:`~src.agents.geopolitical_agent.GeopoliticalAgent`, but with
*single-day* validation: a magnitude-6.5 earthquake on day N is itself
the signal, so requiring multi-day persistence would mask it.
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
    "earthquake_severity",
    "tsunami_risk",
    "cyclone_severity",
    "severe_weather_index",
)
_LOCATION: str = "Strait of Hormuz"
_DEFAULT_WEIGHTS: dict[str, float] = {
    "earthquake": 0.35,
    "tsunami": 0.30,
    "cyclone": 0.20,
    "severe_weather": 0.15,
}
_DEFAULT_COMPOSITE_THRESHOLD: float = 0.30
_DEFAULT_SINGLE_EVENT_THRESHOLD: float = 0.40
_MIN_SEVERITY_FOR_FLAG: float = 0.10


class DisasterAgent(BaseAgent):
    """Weighted-composite scorer for natural-disaster risk.

    Args:
        config: Reads ``threshold`` (composite cutoff, default 0.30),
            ``single_event_threshold`` (any single feature crosses this →
            flag, default 0.40), ``weights`` sub-block, optional
            ``location`` override.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="natural_disaster", config=dict(config or {}))
        self._composite_threshold: float = float(
            self.config.get("threshold", _DEFAULT_COMPOSITE_THRESHOLD)
        )
        self._single_event_threshold: float = float(
            self.config.get("single_event_threshold", _DEFAULT_SINGLE_EVENT_THRESHOLD)
        )
        weights_cfg = self.config.get("weights") or {}
        self._weights: dict[str, float] = {
            k: float(weights_cfg.get(k, _DEFAULT_WEIGHTS[k]))
            for k in _DEFAULT_WEIGHTS
        }
        self._location: str = str(self.config.get("location") or _LOCATION)
        self._feature_columns = list(_FEATURE_COLUMNS)

    def set_weights(
        self,
        earthquake: float,
        tsunami: float,
        cyclone: float,
        weather: float,
    ) -> None:
        """Override the composite weights, normalised to sum to 1.0.

        Args:
            earthquake: Raw weight on earthquake severity.
            tsunami: Raw weight on tsunami risk.
            cyclone: Raw weight on cyclone severity.
            weather: Raw weight on the severe-weather index.
        """
        total = earthquake + tsunami + cyclone + weather
        if total <= 0:
            raise ValueError("DisasterAgent.set_weights: weights must sum to > 0.")
        self._weights = {
            "earthquake": earthquake / total,
            "tsunami": tsunami / total,
            "cyclone": cyclone / total,
            "severe_weather": weather / total,
        }

    def set_threshold(
        self,
        threshold: float,
        single_event_threshold: float | None = None,
    ) -> None:
        """Override the composite cutoff (and optionally single-event cutoff)."""
        self._composite_threshold = float(threshold)
        if single_event_threshold is not None:
            self._single_event_threshold = float(single_event_threshold)

    # ----------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> None:
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"DisasterAgent: missing columns: {missing}")
        self._is_fitted = True
        logger.info(
            "[DisasterAgent.fit] schema validated | composite_threshold=%.2f | "
            "single_event_threshold=%.2f",
            self._composite_threshold, self._single_event_threshold,
        )

    # ------------------------------------------------------------ preprocess
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Disasters are sparse — no rolling baseline; pass through."""
        if not self._is_fitted:
            raise RuntimeError("DisasterAgent.preprocess called before fit().")
        return data.copy().reset_index(drop=True)

    # --------------------------------------------------------------- detect
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        """Weighted composite + single-event override."""
        if not self._is_fitted:
            raise RuntimeError("DisasterAgent.detect called before fit().")
        out = data.copy()
        composite = (
            self._weights["earthquake"] * out["earthquake_severity"]
            + self._weights["tsunami"] * out["tsunami_risk"]
            + self._weights["cyclone"] * out["cyclone_severity"]
            + self._weights["severe_weather"] * out["severe_weather_index"]
        )
        max_single = out[list(self._feature_columns)].max(axis=1)
        out["raw_composite"] = composite.astype(float)
        out["max_single_severity"] = max_single.astype(float)
        out["anomaly_score"] = np.clip(
            np.maximum(composite, max_single), 0.0, 1.0
        ).astype(float)
        out["is_anomaly"] = (
            (composite >= self._composite_threshold)
            | (max_single >= self._single_event_threshold)
        )
        logger.info(
            "[DisasterAgent.detect] %d rows | %d raw flags "
            "(composite>=%.2f or single>=%.2f)",
            len(out), int(out["is_anomaly"].sum()),
            self._composite_threshold, self._single_event_threshold,
        )
        return out

    # ------------------------------------------------------------- validate
    def validate(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Single-day acceptable; suppress only sub-minimum severities."""
        s = signals.copy().reset_index(drop=True)
        flags = s["is_anomaly"].astype(bool).to_numpy()
        passes_severity = (s["max_single_severity"] >= _MIN_SEVERITY_FOR_FLAG).to_numpy()
        s["validated"] = flags & passes_severity
        logger.info(
            "[DisasterAgent.validate] %d/%d raw flags survived "
            "(min_severity=%.2f)",
            int(s["validated"].sum()), int(flags.sum()), _MIN_SEVERITY_FOR_FLAG,
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
            events: list[str] = []
            if "active_events" in w.columns:
                for raw in w["active_events"]:
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else []
                    except json.JSONDecodeError:
                        parsed = []
                    events.extend(parsed)
            # Confidence is the share of features over a notional 0.30 cutoff.
            elevated_share = float(
                (w[list(self._feature_columns)] > 0.30).sum(axis=1).mean()
                / len(self._feature_columns)
            )
            confidence = min(1.0, 0.5 + elevated_share)
            results.append({
                "agent": "natural_disaster",
                "anomaly_score": float(w["anomaly_score"].max()),
                "confidence": confidence,
                "signals": {
                    "earthquake_severity": float(w["earthquake_severity"].max()),
                    "tsunami_risk": float(w["tsunami_risk"].max()),
                    "cyclone_severity": float(w["cyclone_severity"].max()),
                    "severe_weather_index": float(w["severe_weather_index"].max()),
                },
                "active_events": sorted(set(events)),
                "start_timestamp": str(pd.Timestamp(w["timestamp"].iloc[0]).date()),
                "end_timestamp": str(pd.Timestamp(w["timestamp"].iloc[-1]).date()),
                "location": self._location,
            })
        logger.info("[DisasterAgent.output] produced %d windows", len(results))
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
        return self.validate(self.detect(self.preprocess(data)))

    def to_detection_result(self, validated: pd.DataFrame) -> DetectionResult:
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=validated["anomaly_score"].to_numpy(dtype=float),
            anomaly_flags=validated["validated"].to_numpy(dtype=bool),
            feature_names=list(self._feature_columns),
            metadata={
                "composite_threshold": self._composite_threshold,
                "single_event_threshold": self._single_event_threshold,
                "weights": dict(self._weights),
                "location": self._location,
            },
        )
