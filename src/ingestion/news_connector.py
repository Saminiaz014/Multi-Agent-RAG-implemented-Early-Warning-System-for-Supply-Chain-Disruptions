"""News-sentiment connector for the Strait of Hormuz corridor.

Synthetic mode generates a per-day sentiment envelope that *leads*
the shipping disruption windows by ``lead_days`` (default 2) — news
breaks before port data reflects the underlying event. CSV mode
loads a pre-aggregated daily-sentiment frame. API mode is stubbed
with the full planned pipeline (NewsAPI / GDELT → VADER + embeddings
+ DBSCAN cluster + source-consensus + recency weighting) documented
in the docstring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.base_connector import BaseConnector

logger = logging.getLogger(__name__)

_DEFAULT_CSV_PATH: str = "data/raw/news_sentiment.csv"
_LOCATION: str = "Strait of Hormuz"
_SOURCE: str = "news_sentiment"
_DEFAULT_LEAD_DAYS: int = 2

_BASELINE_SENTIMENT: float = 0.10
_BASELINE_MAGNITUDE: float = 0.40
_BASELINE_CONSENSUS: float = 0.40
_BASELINE_ARTICLE_VOLUME: float = 10.0
_BASELINE_NOISE = {
    "sentiment": 0.06, "magnitude": 0.06, "consensus": 0.05, "volume": 4.0,
}


@dataclass(frozen=True)
class _NewsScenario:
    name: str
    base_start: int
    base_end: int
    ramp_days: int
    decay_days: int
    sentiment_range: tuple[float, float]
    consensus_range: tuple[float, float]
    volume_range: tuple[float, float]
    dominant_narrative: str


_SCENARIOS: tuple[_NewsScenario, ...] = (
    _NewsScenario(
        name="Moderate Tension",
        base_start=60, base_end=74,
        ramp_days=3, decay_days=10,
        sentiment_range=(-0.50, -0.30),
        consensus_range=(0.60, 0.80),
        volume_range=(30.0, 50.0),
        dominant_narrative="Maritime tensions rising in Gulf corridor",
    ),
    _NewsScenario(
        name="Major Blockage",
        base_start=150, base_end=170,
        ramp_days=4, decay_days=14,
        sentiment_range=(-0.90, -0.60),
        consensus_range=(0.80, 0.95),
        volume_range=(80.0, 150.0),
        dominant_narrative="Major shipping disruption in Strait of Hormuz",
    ),
    _NewsScenario(
        name="Brief Incident",
        base_start=280, base_end=290,
        ramp_days=2, decay_days=7,
        sentiment_range=(-0.30, -0.20),
        consensus_range=(0.50, 0.60),
        volume_range=(20.0, 35.0),
        dominant_narrative="Tanker incident under investigation",
    ),
)


class NewsConnector(BaseConnector):
    """Daily news-sentiment generator for the Hormuz corridor.

    Args:
        config: Reads ``data_mode``, ``csv_path``, ``lead_days``,
            ``location_context`` (for API keyword generation), and an
            ``api`` sub-block.
    """

    LOCATION: str = _LOCATION
    SOURCE: str = _SOURCE
    FEATURE_COLUMNS: tuple[str, ...] = (
        "sentiment_score",
        "sentiment_magnitude",
        "source_consensus",
        "article_volume",
        "recency_weighted_score",
        "composite_news_risk",
    )

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(dict(config or {}))
        self.data_mode: str = str(self.config.get("data_mode", "synthetic")).lower()
        self.csv_path: str = str(self.config.get("csv_path", _DEFAULT_CSV_PATH))
        self.lead_days: int = int(self.config.get("lead_days", _DEFAULT_LEAD_DAYS))
        self.location_context: dict = dict(
            self.config.get("location_context") or {
                "primary_location": "Strait of Hormuz",
                "region": "Persian Gulf",
                "countries": ["Iran", "Oman", "UAE", "Saudi Arabia"],
                "topics": ["shipping", "oil", "tanker", "sanctions", "military", "blockade"],
            }
        )

    # ------------------------------------------------------------------ fetch
    def fetch(self) -> pd.DataFrame:
        if self.data_mode == "synthetic":
            return self.generate_dataset()
        if self.data_mode == "csv":
            return self.load_csv()
        if self.data_mode == "api":
            return self.fetch_api()
        raise ValueError(
            f"Unknown data_mode={self.data_mode!r}; "
            "expected 'synthetic', 'csv', or 'api'."
        )

    # ---------------------------------------------------------- synthetic
    def generate_dataset(
        self, days: int = 365, seed: int = 42
    ) -> pd.DataFrame:
        """Generate a synthetic daily news-sentiment DataFrame."""
        if days <= 0:
            raise ValueError("days must be positive.")
        rng = np.random.default_rng(seed)
        timestamps = pd.date_range("2025-01-01", periods=days, freq="D")

        sentiment = np.clip(
            rng.normal(_BASELINE_SENTIMENT, _BASELINE_NOISE["sentiment"], size=days),
            -1.0, 1.0,
        )
        magnitude = np.clip(
            rng.normal(_BASELINE_MAGNITUDE, _BASELINE_NOISE["magnitude"], size=days),
            0.0, 1.0,
        )
        consensus = np.clip(
            rng.normal(_BASELINE_CONSENSUS, _BASELINE_NOISE["consensus"], size=days),
            0.0, 1.0,
        )
        volume = np.clip(
            rng.normal(_BASELINE_ARTICLE_VOLUME, _BASELINE_NOISE["volume"], size=days),
            0.0, None,
        )
        narratives: list[str] = [""] * days
        is_disruption = np.zeros(days, dtype=bool)

        for scenario in _SCENARIOS:
            self._apply_scenario(
                rng=rng, scenario=scenario,
                sentiment=sentiment, magnitude=magnitude,
                consensus=consensus, volume=volume,
                narratives=narratives,
                is_disruption=is_disruption, total_days=days,
            )

        # Recency-weighted score = sentiment lightly EWM-smoothed; positive
        # spike & negative spike both attenuated by article volume so a
        # single-article positive day doesn't move the needle.
        recency = pd.Series(sentiment).ewm(span=3, adjust=False).mean().to_numpy()
        recency_weighted = recency * np.clip(volume / 30.0, 0.3, 1.5)
        recency_weighted = np.clip(recency_weighted, -1.0, 1.0)

        composite = np.clip(
            np.maximum(0.0, -sentiment) * consensus * magnitude, 0.0, 1.0
        )

        df = pd.DataFrame({
            "timestamp": timestamps,
            "sentiment_score": np.round(sentiment, 4),
            "sentiment_magnitude": np.round(magnitude, 4),
            "source_consensus": np.round(consensus, 4),
            "article_volume": np.round(volume).astype(int),
            "dominant_narrative": narratives,
            "recency_weighted_score": np.round(recency_weighted, 4),
            "composite_news_risk": np.round(composite, 4),
            "is_disruption": is_disruption,
        })
        logger.info(
            "[NewsConnector/synthetic] generated %d rows; "
            "disruption_days=%d; lead_days=%d",
            days, int(is_disruption.sum()), self.lead_days,
        )
        return df

    def load_csv(self, path: str | Path | None = None) -> pd.DataFrame:
        csv_path = Path(path) if path is not None else Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"News CSV not found at {csv_path}. "
                "Provide one or set data_mode='synthetic'."
            )
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "dominant_narrative" in df.columns:
            df["dominant_narrative"] = df["dominant_narrative"].fillna(
                "No dominant narrative"
            )
        if not self.validate(df):
            raise ValueError(
                f"News CSV at {csv_path} failed schema validation."
            )
        return df

    def fetch_api(self) -> pd.DataFrame:
        """Planned NewsAPI / GDELT → VADER + embeddings + cluster pipeline.

        Planned implementation:
            1. Derive keywords from ``self.location_context`` (primary
               location, region, country list, topic list).
            2. Query NewsAPI ``/v2/everything`` and GDELT ``/api/v2/doc``
               for the past 24h; fallback to Reuters / AP / Bloomberg RSS.
            3. VADER sentiment per article (``nltk.sentiment.vader``).
            4. Embed via ``sentence-transformers/all-MiniLM-L6-v2``.
            5. DBSCAN cluster the embeddings; for each cluster compute
               ``source_consensus = unique_sources / total_articles``.
            6. Recency weights: <2d = 1.0, 2-7d = 0.5, >7d = 0.1.
            7. Aggregate per-day → DataFrame with this connector's schema.

        Raises:
            NotImplementedError: Wiring stubbed for thesis scope.
        """
        raise NotImplementedError(
            "API mode not yet implemented. Planned: NewsAPI + GDELT + "
            "VADER + sentence-transformers (see docstring). Set "
            "data_mode='synthetic' or 'csv' in config."
        )

    def validate(self, df: pd.DataFrame) -> bool:
        required = {
            "timestamp",
            "sentiment_score",
            "sentiment_magnitude",
            "source_consensus",
            "article_volume",
            "recency_weighted_score",
            "composite_news_risk",
            "is_disruption",
        }
        missing = required - set(df.columns)
        if missing:
            logger.error("[NewsConnector] missing columns: %s", sorted(missing))
            return False
        if df["timestamp"].isna().any():
            logger.error("[NewsConnector] NaN in timestamp")
            return False
        if not df["sentiment_score"].between(-1.0, 1.0).all():
            logger.error("[NewsConnector] sentiment_score out of [-1, 1]")
            return False
        if not df["sentiment_magnitude"].between(0.0, 1.0).all():
            logger.error("[NewsConnector] sentiment_magnitude out of [0, 1]")
            return False
        if not df["source_consensus"].between(0.0, 1.0).all():
            logger.error("[NewsConnector] source_consensus out of [0, 1]")
            return False
        if (df["article_volume"] < 0).any():
            logger.error("[NewsConnector] negative article_volume")
            return False
        if not df["recency_weighted_score"].between(-1.0, 1.0).all():
            logger.error("[NewsConnector] recency_weighted_score out of [-1, 1]")
            return False
        if not df["composite_news_risk"].between(0.0, 1.0).all():
            logger.error("[NewsConnector] composite_news_risk out of [0, 1]")
            return False
        return True

    def save_raw(
        self, path: str | Path = _DEFAULT_CSV_PATH
    ) -> Path:
        df = self.fetch()
        if not self.validate(df):
            raise ValueError("News data failed validation prior to save.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("[NewsConnector] wrote %d rows to %s", len(df), out)
        return out.resolve()

    def to_signal_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["timestamp"]).isoformat()
            for feature in feature_cols:
                val = row[feature]
                if pd.isna(val):
                    continue
                records.append({
                    "timestamp": ts,
                    "source": self.SOURCE,
                    "feature": feature,
                    "value": float(val),
                    "location": self.LOCATION,
                })
        return records

    # -------------------------------------------------- internal helpers
    def _apply_scenario(
        self, *,
        rng: np.random.Generator,
        scenario: _NewsScenario,
        sentiment: np.ndarray,
        magnitude: np.ndarray,
        consensus: np.ndarray,
        volume: np.ndarray,
        narratives: list[str],
        is_disruption: np.ndarray,
        total_days: int,
    ) -> None:
        """Apply scenario with ``lead_days`` head start over shipping window."""
        start = max(scenario.base_start - self.lead_days, 0)
        end = min(scenario.base_end - self.lead_days + scenario.decay_days, total_days - 1)
        if start >= total_days:
            return

        sentiment_floor = rng.uniform(*scenario.sentiment_range)
        consensus_peak = rng.uniform(*scenario.consensus_range)
        volume_peak = rng.uniform(*scenario.volume_range)
        magnitude_peak = min(0.95, abs(sentiment_floor) + 0.2)

        window_len = end - start + 1
        ramp = min(scenario.ramp_days, max(window_len // 3, 1))
        decay = min(scenario.decay_days, max(window_len - ramp - 1, 1))

        for offset, day_idx in enumerate(range(start, end + 1)):
            if offset < ramp:
                intensity = (offset + 1) / (ramp + 1)
            elif offset >= window_len - decay:
                tail = offset - (window_len - decay)
                intensity = 1.0 - (tail + 1) / (decay + 1)
            else:
                intensity = 1.0
            sentiment[day_idx] = min(
                sentiment[day_idx],
                _BASELINE_SENTIMENT + (sentiment_floor - _BASELINE_SENTIMENT) * intensity,
            )
            magnitude[day_idx] = max(magnitude[day_idx], magnitude_peak * intensity)
            consensus[day_idx] = max(consensus[day_idx], consensus_peak * intensity)
            volume[day_idx] = max(
                volume[day_idx],
                _BASELINE_ARTICLE_VOLUME + (volume_peak - _BASELINE_ARTICLE_VOLUME) * intensity,
            )
            is_disruption[day_idx] = True
            if intensity >= 0.6:
                narratives[day_idx] = scenario.dominant_narrative

        logger.info(
            "[NewsConnector/synthetic] scenario '%s' days %d-%d "
            "(sentiment floor=%.2f, consensus peak=%.2f, lead=%d)",
            scenario.name, start, end, sentiment_floor, consensus_peak, self.lead_days,
        )
