"""News-sentiment risk-scoring agent.

Threshold-based detector with consensus + velocity + volume amplifiers.

Tone is only observed on days with articles. A day with ``article_volume``
0 has no tone (its ``sentiment_score`` of 0 means "no signal", not neutral
coverage), and a day with no news data at all (NaN, e.g. a GDELT outage) has
neither. Tone *means* skip those days: the recency-weighted tone the score
uses (``recency_weighted_score``, built NaN-aware by the connector) and the
diagnostic 7-day mean below. Velocity keeps its original definition, the
3-day change in the daily ``sentiment_score``. A day without any news data
scores NaN, so it is never read as calm.

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
        """Schema check; narrows features to those the source supplies.

        GDELT's timeline endpoints return an aggregate tone per day with no
        per-source breakdown, so ``source_consensus`` cannot be measured from
        them. A source may omit a feature and :meth:`detect` renormalises the
        remaining weights rather than scoring an invented column.

        Raises:
            ValueError: If ``sentiment_score`` is absent — without it there is
                no news signal left to score.
        """
        present = [c for c in _FEATURE_COLUMNS if c in df.columns]
        if "sentiment_score" not in present:
            raise ValueError(
                f"NewsAgent: sentiment_score is required; got {list(df.columns)}."
            )
        self._feature_columns = present
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
        # No articles, no tone: mask before averaging, so quiet days are
        # skipped rather than counted as neutral tone. Diagnostic only: the
        # score reads the connector's recency_weighted_score, the same kind
        # of NaN-aware tone mean.
        tone = df["sentiment_score"].where(df["article_volume"] > 0)
        df["sentiment_rolling_7d"] = tone.rolling(window=7, min_periods=1).mean()
        # Velocity: unchanged daily definition; a day with no news data at
        # all (NaN) gives no velocity signal.
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

        # A day with no news data at all is not scored (NaN), never "calm".
        no_data = (out["sentiment_score"].isna()
                   & out["article_volume"].isna()).to_numpy()

        # Each component normalised into [0, 1] before weighting; a component
        # that cannot be computed on a day contributes no signal.
        neg_sent = np.nan_to_num(
            np.clip(-out["recency_weighted_score"], 0.0, 1.0).to_numpy()
        )
        consensus = (
            np.clip(out["source_consensus"], 0.0, 1.0).to_numpy()
            if "source_consensus" in out.columns
            else None
        )
        # Negative velocity = sentiment dropping fast = high risk.
        velocity = np.clip(-out["sentiment_velocity"], 0.0, 1.0).to_numpy()
        baseline_vol = out["volume_rolling_30d"].replace(0.0, 1.0).to_numpy()
        volume_factor = np.nan_to_num(np.clip(
            out["article_volume"].to_numpy() / baseline_vol
            / self._volume_spike_multiplier, 0.0, 1.0,
        ))

        # Renormalise over the components actually available. consensus is
        # absent for GDELT-sourced data; scoring it as zero would read as
        # "outlets disagree" rather than "not measured", and silently damp
        # every score by its 0.25 weight.
        components = {
            "sentiment": neg_sent,
            "consensus": consensus,
            "velocity": velocity,
            "volume": volume_factor,
        }
        weighted = np.zeros(len(out), dtype=float)
        used = 0.0
        for key, values in components.items():
            if values is None:
                continue
            weighted = weighted + self._weights[key] * np.asarray(values)
            used += self._weights[key]
        if used <= 0:
            raise ValueError("NewsAgent: no scoring components available.")
        if used < 1.0:
            logger.warning(
                "[NewsAgent] scoring without %s; weights renormalised over "
                "%.3f of the original mass.",
                sorted(k for k, v in components.items() if v is None), used,
            )
        weighted = weighted / used

        out["neg_sentiment_norm"] = neg_sent
        out["consensus_norm"] = consensus
        out["velocity_norm"] = velocity
        out["volume_spike_norm"] = volume_factor
        out["anomaly_score"] = np.where(no_data, np.nan, np.clip(weighted, 0.0, 1.0))
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
        # Without a consensus signal the gate cannot be applied; requiring it
        # would suppress every flag rather than leaving the remaining
        # sentiment gate to do the work.
        combined = flags & sent_ok
        if "source_consensus" in s.columns:
            combined = combined & (
                s["source_consensus"] >= self._consensus_threshold
            ).to_numpy()

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
            # Without a consensus measure (GDELT sources), fall back to the
            # window's mean score instead of crashing on a None column.
            has_consensus = "source_consensus" in w.columns
            confidence = float(np.clip(
                (w["consensus_norm"].astype(float).mean() + 0.1)
                if has_consensus else w["anomaly_score"].mean(),
                0.0, 1.0,
            ))
            results.append({
                "agent": "news_sentiment",
                "anomaly_score": float(w["anomaly_score"].max()),
                "confidence": confidence,
                "signals": {
                    "sentiment_score": float(w["sentiment_score"].min()),
                    "sentiment_magnitude": float(w["sentiment_magnitude"].max()),
                    "source_consensus": (
                        float(w["source_consensus"].max()) if has_consensus else None
                    ),
                    # float: GDELT volume is normalised (per 100k events)
                    "article_volume": float(w["article_volume"].max()),
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
