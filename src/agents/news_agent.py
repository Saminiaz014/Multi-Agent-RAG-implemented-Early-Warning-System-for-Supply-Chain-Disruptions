"""News-sentiment risk-scoring agent.

Threshold-based detector with consensus + velocity + volume amplifiers.
Score formula:

    anomaly_score = 0.40 * normalised_negative_sentiment
                  + 0.25 * source_consensus
                  + 0.20 * sentiment_velocity_norm
                  + 0.15 * volume_spike_norm
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent, DetectionResult

logger = logging.getLogger(__name__)

_FEATURE_COLUMNS: tuple[str, ...] = (
    "sentiment_score",
    "sentiment_magnitude",
    "source_consensus",
    "article_volume",
    "recency_weighted_score",
)
_LOCATION: str = "Strait of Hormuz"
_DEFAULT_NEG_THRESHOLD: float = -0.30
_DEFAULT_CONSENSUS_THRESHOLD: float = 0.40
_DEFAULT_VOLUME_MULTIPLIER: float = 2.0
_PERSISTENCE_DAYS: int = 2
_DEFAULT_THRESHOLD: float = 0.40

_DEFAULT_WEIGHTS: dict[str, float] = {
    "sentiment": 0.40, "consensus": 0.25,
    "velocity": 0.20, "volume": 0.15,
}


class NewsAgent(BaseAgent):
    """Sentiment + consensus + velocity + volume threshold detector.

    Args:
        config: Reads ``negative_threshold`` (default -0.30),
            ``consensus_threshold`` (default 0.40),
            ``volume_spike_multiplier`` (default 2.0),
            ``threshold`` (composite score cutoff, default 0.40),
            ``weights`` sub-block, optional ``location`` override.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="news_sentiment", config=dict(config or {}))
        self._neg_threshold: float = float(
            self.config.get("negative_threshold", _DEFAULT_NEG_THRESHOLD)
        )
        self._consensus_threshold: float = float(
            self.config.get("consensus_threshold", _DEFAULT_CONSENSUS_THRESHOLD)
        )
        self._volume_spike_multiplier: float = float(
            self.config.get("volume_spike_multiplier", _DEFAULT_VOLUME_MULTIPLIER)
        )
        self._threshold: float = float(
            self.config.get("threshold", _DEFAULT_THRESHOLD)
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
        sentiment: float,
        consensus: float,
        velocity: float,
        volume: float,
    ) -> None:
        """Override the composite weights, normalised to sum to 1.0.

        Args:
            sentiment: Raw weight on normalised negative sentiment.
            consensus: Raw weight on source consensus.
            velocity: Raw weight on (negative) sentiment velocity.
            volume: Raw weight on the article-volume spike factor.
        """
        total = sentiment + consensus + velocity + volume
        if total <= 0:
            raise ValueError("NewsAgent.set_weights: weights must sum to > 0.")
        self._weights = {
            "sentiment": sentiment / total,
            "consensus": consensus / total,
            "velocity": velocity / total,
            "volume": volume / total,
        }

    def set_threshold(
        self,
        threshold: float | None = None,
        *,
        negative_threshold: float | None = None,
        consensus_threshold: float | None = None,
    ) -> None:
        """Override the composite cutoff and/or the validation gates.

        Args:
            threshold: Composite anomaly-score cutoff.
            negative_threshold: Recency-weighted sentiment must be ``<=`` this
                to validate (more negative = more disruptive).
            consensus_threshold: Source consensus must be ``>=`` this to validate.
        """
        if threshold is not None:
            self._threshold = float(threshold)
        if negative_threshold is not None:
            self._neg_threshold = float(negative_threshold)
        if consensus_threshold is not None:
            self._consensus_threshold = float(consensus_threshold)

    # ----------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> None:
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"NewsAgent: missing columns: {missing}")
        self._is_fitted = True
        logger.info(
            "[NewsAgent.fit] schema validated | threshold=%.2f | weights=%s",
            self._threshold, self._weights,
        )

    # ------------------------------------------------------------ preprocess
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("NewsAgent.preprocess called before fit().")
        df = data.copy().reset_index(drop=True)
        df["sentiment_rolling_7d"] = df["sentiment_score"].rolling(
            window=7, min_periods=2
        ).mean().fillna(df["sentiment_score"])
        df["sentiment_velocity"] = df["sentiment_score"].diff(periods=3).fillna(0.0)
        df["volume_rolling_30d"] = df["article_volume"].rolling(
            window=30, min_periods=5
        ).mean().fillna(df["article_volume"])
        return df

    # --------------------------------------------------------------- detect
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        if not self._is_fitted:
            raise RuntimeError("NewsAgent.detect called before fit().")
        out = data.copy()

        # Each component normalised into [0, 1] before weighting.
        neg_sent = (
            np.clip(-out["recency_weighted_score"], 0.0, 1.0)
        ).to_numpy()
        consensus = np.clip(out["source_consensus"], 0.0, 1.0).to_numpy()
        # Negative velocity = sentiment dropping fast = high risk.
        velocity = np.clip(-out["sentiment_velocity"], 0.0, 1.0).to_numpy()
        baseline_vol = out["volume_rolling_30d"].replace(0.0, 1.0).to_numpy()
        volume_factor = np.clip(
            out["article_volume"].to_numpy() / baseline_vol
            / self._volume_spike_multiplier, 0.0, 1.0,
        )

        weighted = (
            self._weights["sentiment"] * neg_sent
            + self._weights["consensus"] * consensus
            + self._weights["velocity"] * velocity
            + self._weights["volume"] * volume_factor
        )

        out["neg_sentiment_norm"] = neg_sent
        out["consensus_norm"] = consensus
        out["velocity_norm"] = velocity
        out["volume_spike_norm"] = volume_factor
        out["anomaly_score"] = np.clip(weighted, 0.0, 1.0)
        out["is_anomaly"] = out["anomaly_score"] >= self._threshold
        logger.info(
            "[NewsAgent.detect] %d rows | %d raw flags (threshold=%.2f)",
            len(out), int(out["is_anomaly"].sum()), self._threshold,
        )
        return out

    # ------------------------------------------------------------- validate
    def validate(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Persistence (>= 2 days) + sentiment below threshold + consensus high enough."""
        s = signals.copy().reset_index(drop=True)
        flags = s["is_anomaly"].astype(bool).to_numpy()
        sent_ok = (
            s["recency_weighted_score"] <= self._neg_threshold
        ).to_numpy()
        consensus_ok = (
            s["source_consensus"] >= self._consensus_threshold
        ).to_numpy()
        combined = flags & sent_ok & consensus_ok

        persistent = np.zeros_like(combined)
        n = len(combined)
        i = 0
        while i < n:
            if combined[i]:
                j = i
                while j < n and combined[j]:
                    j += 1
                if (j - i) >= _PERSISTENCE_DAYS:
                    persistent[i:j] = True
                i = j
            else:
                i += 1
        s["validated"] = persistent
        logger.info(
            "[NewsAgent.validate] %d/%d raw flags survived "
            "(persistence>=%d, sent<=%.2f, consensus>=%.2f)",
            int(s["validated"].sum()), int(flags.sum()),
            _PERSISTENCE_DAYS, self._neg_threshold, self._consensus_threshold,
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
            narratives = (
                [n for n in w["dominant_narrative"] if isinstance(n, str) and n]
                if "dominant_narrative" in w.columns else []
            )
            confidence = float(
                np.clip(w["consensus_norm"].mean() + 0.1, 0.0, 1.0)
            )
            results.append({
                "agent": "news_sentiment",
                "anomaly_score": float(w["anomaly_score"].max()),
                "confidence": confidence,
                "signals": {
                    "sentiment_score": float(w["sentiment_score"].min()),
                    "sentiment_magnitude": float(w["sentiment_magnitude"].max()),
                    "source_consensus": float(w["source_consensus"].max()),
                    "article_volume": int(w["article_volume"].max()),
                    "sentiment_velocity": float(w["sentiment_velocity"].min()),
                    "recency_weighted_score": float(w["recency_weighted_score"].min()),
                },
                "dominant_narrative": (
                    max(set(narratives), key=narratives.count)
                    if narratives else "No dominant narrative"
                ),
                "start_timestamp": str(pd.Timestamp(w["timestamp"].iloc[0]).date()),
                "end_timestamp": str(pd.Timestamp(w["timestamp"].iloc[-1]).date()),
                "location": self._location,
            })
        logger.info("[NewsAgent.output] produced %d windows", len(results))
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
                "negative_threshold": self._neg_threshold,
                "consensus_threshold": self._consensus_threshold,
                "volume_spike_multiplier": self._volume_spike_multiplier,
                "threshold": self._threshold,
                "weights": dict(self._weights),
                "location": self._location,
            },
        )
