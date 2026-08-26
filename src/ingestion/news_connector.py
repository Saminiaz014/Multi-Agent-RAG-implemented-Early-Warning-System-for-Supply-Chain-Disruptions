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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

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


# --- Live GDELT mode --------------------------------------------------------
_GDELT_URL: str = "https://api.gdeltproject.org/api/v2/doc/doc"
#: GDELT tone is nominally [-100, 100] but sits within about +-10 in practice.
_GDELT_TONE_SCALE: float = 10.0
#: Extra keywords OR'd into the query. Beyond a handful, GDELT's relevance
#: degrades and the series stops tracking the chokepoint.
_GDELT_MAX_CONTEXT_TERMS: int = 5
#: GDELT rate-limits hard: measured at roughly a 50% success rate, with HTTP
#: 429 served as HTML and dropped connections, and a burst of requests gets
#: the caller blocked for minutes. Retries therefore fail *fast* rather than
#: deep -- three short attempts, not five long ones. A 5-minute stall while a
#: throttled service refuses is worse for the pipeline than falling back to
#: synthetic, and hammering it is what earns the block in the first place.
_GDELT_ATTEMPTS: int = 3
_GDELT_BACKOFF_SECONDS: float = 5.0
_GDELT_TIMEOUT: int = 45
#: Trailing window beyond recent_days over which old news still counts.
_RECENCY_TAIL_DAYS: int = 7
#: Matches PortWatch and the hazard connector so all series align.
_API_START_YEAR: int = 2019
_API_END_YEAR: int = 2026

#: NewsAgent weights minus consensus (0.25), which GDELT timelines cannot
#: supply; renormalised so the connector and agent agree on scale.
_API_COMPOSITE_WEIGHTS: dict[str, float] = {
    "sentiment": 0.40, "velocity": 0.20, "volume": 0.15,
}


class NewsConnector(BaseConnector):
    """Daily news-sentiment generator for the Hormuz corridor.

    Args:
        config: Reads ``data_mode``, ``csv_path``, ``lead_days``,
            ``location_context`` (for API keyword generation),
            ``newsapi_keywords`` (explicit override for those keywords), and
            an ``api`` sub-block.
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
        self.newsapi_keywords: list[str] = self._resolve_keywords()

    def _resolve_keywords(self) -> list[str]:
        """Resolve the NewsAPI search keywords for this region.

        An explicit ``newsapi_keywords`` list in config wins. Otherwise the
        keywords are derived from ``location_context`` — which the region
        overlays already set per region — following step 1 of
        :meth:`fetch_api`'s planned pipeline: the primary location, then the
        region, then its topics. Countries are deliberately left out: they are
        the ACLED filter, and as free-text news queries they pull in unrelated
        national coverage.

        Returns:
            Ordered, de-duplicated keyword list. Empty only if
            ``location_context`` carries neither a location nor any topics.
        """
        explicit = self.config.get("newsapi_keywords")
        if explicit:
            keywords = [str(k) for k in explicit]
            logger.info(
                "[NewsConnector] explicit newsapi_keywords: %s", keywords
            )
            return keywords

        ctx = self.location_context
        candidates = [
            ctx.get("primary_location"),
            ctx.get("region"),
            *(ctx.get("topics") or []),
        ]
        seen: dict[str, None] = {}
        for candidate in candidates:
            text = str(candidate).strip() if candidate else ""
            if text:
                seen.setdefault(text, None)
        keywords = list(seen)

        if keywords:
            logger.info(
                "[NewsConnector] keywords derived from location_context "
                "('%s'): %s",
                ctx.get("primary_location", "?"),
                keywords,
            )
        else:
            logger.warning(
                "[NewsConnector] no newsapi_keywords and an empty "
                "location_context — a live NewsAPI query would be unfiltered."
            )
        return keywords

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
            1. Keywords — already resolved into ``self.newsapi_keywords`` at
               construction, either from an explicit config list or derived
               from ``self.location_context``.
            2. Query NewsAPI ``/v2/everything`` and GDELT ``/api/v2/doc``
               for the past 24h; fallback to Reuters / AP / Bloomberg RSS.
            3. VADER sentiment per article (``nltk.sentiment.vader``).
            4. Embed via ``sentence-transformers/all-MiniLM-L6-v2``.
            5. DBSCAN cluster the embeddings; for each cluster compute
               ``source_consensus = unique_sources / total_articles``.
            6. Recency weights: <2d = 1.0, 2-7d = 0.5, >7d = 0.1.
            7. Aggregate per-day → DataFrame with this connector's schema.

        Raises:
            NotImplementedError: Wiring stubbed for thesis scope. The message
                reports the resolved keywords so a region-config gap surfaces
                here rather than as a silently unfiltered query later.
        """
        if not self.newsapi_keywords:
            raise ValueError(
                "No keywords resolved for this region; a live query would be "
                "unfiltered and return world news. Set agents.news_sentiment."
                "location_context, or use data_mode='synthetic'."
            )

        query = self._gdelt_query()
        tone = self._gdelt_timeline("timelinetone", query)
        volume = self._gdelt_timeline("timelinevolraw", query)
        if tone.empty:
            raise ValueError(
                f"GDELT returned no tone series for {query!r}. The service "
                "rate-limits aggressively (HTTP 429); retry, or use "
                "data_mode='synthetic'."
            )

        df = tone.rename(columns={"value": "tone"})
        df = df.merge(volume.rename(columns={"value": "raw_volume"}),
                      on="timestamp", how="left")
        df["raw_volume"] = df["raw_volume"].fillna(0.0)

        # GDELT tone is roughly [-100, 100] but sits within about ±10 in
        # practice, so scaling by that keeps ordinary coverage across the
        # usable range instead of squashed near zero.
        df["sentiment_score"] = (df["tone"] / _GDELT_TONE_SCALE).clip(-1.0, 1.0)
        df["sentiment_magnitude"] = df["sentiment_score"].abs()

        # article_volume ∈ [0, 1] against the series' own p95 — a percentile
        # anchor so one exceptional news day does not flatten the rest.
        anchor = float(df["raw_volume"].quantile(0.95))
        df["article_volume"] = (
            (df["raw_volume"] / anchor).clip(0.0, 1.0)
            if anchor > 0 else 0.0
        )

        # Recency weighting per config: recent days count for more. Applied as
        # a trailing weighted mean so a run of negative days compounds.
        weights_cfg = (self.config.get("api", {}) or {}).get("recency_weights", {})
        df["recency_weighted_score"] = self._recency_weighted(
            df["sentiment_score"], weights_cfg
        )

        # source_consensus is absent, not zero: GDELT's timeline endpoints
        # return an aggregate tone per day with no per-source breakdown, so
        # agreement between outlets cannot be measured from them. NewsAgent
        # renormalises its weights over the features present.
        weights = _API_COMPOSITE_WEIGHTS
        negative = (-df["sentiment_score"]).clip(lower=0.0)
        velocity = df["article_volume"].diff().fillna(0.0).clip(lower=0.0)
        df["composite_news_risk"] = (
            weights["sentiment"] * negative
            + weights["velocity"] * velocity
            + weights["volume"] * df["article_volume"]
        ) / sum(weights.values())

        df["is_disruption"] = df["composite_news_risk"] >= float(
            self.config.get("threshold", 0.40)
        )
        df = df[[
            "timestamp", "sentiment_score", "sentiment_magnitude",
            "article_volume", "recency_weighted_score", "composite_news_risk",
            "is_disruption",
        ]]

        logger.info(
            "[NewsConnector/GDELT] %r -> %d days [%s .. %s] mean_tone=%.2f "
            "disruption_days=%d (no source_consensus: GDELT timelines carry "
            "no per-source breakdown)",
            query, len(df), df["timestamp"].min().date(),
            df["timestamp"].max().date(), df["sentiment_score"].mean(),
            int(df["is_disruption"].sum()),
        )
        return df

    def _gdelt_query(self) -> str:
        """Build a GDELT query from the region's keywords.

        The primary location is quoted as a phrase; the remaining terms are
        OR'd as context. Without the phrase quotes, "Strait of Hormuz" would
        match any article containing "of".
        """
        primary, *rest = self.newsapi_keywords
        query = f'"{primary}"'
        context = " OR ".join(f'"{k}"' for k in rest[:_GDELT_MAX_CONTEXT_TERMS])
        return f"{query} ({context})" if context else query

    def _gdelt_timeline(self, mode: str, query: str) -> pd.DataFrame:
        """Fetch one GDELT daily timeline, retrying around its rate limiting.

        GDELT is free and needs no key, but it rate-limits hard: measured at
        roughly a 50% success rate, with HTTP 429s and dropped connections,
        and 20-40s per response. Retries use a longer backoff than the other
        connectors for that reason.

        A whole multi-year span comes back in one request (2,747 daily points
        for 2019-2026), so a region costs two requests, not one per month.

        Args:
            mode: ``"timelinetone"`` or ``"timelinevolraw"``.
            query: GDELT query string.

        Returns:
            ``timestamp`` and ``value``; empty if every attempt failed.
        """
        api_cfg = self.config.get("api", {}) or {}
        url = api_cfg.get("gdelt_url", _GDELT_URL)
        start = f"{int(api_cfg.get('start_year', _API_START_YEAR))}0101000000"
        # Clamp to now: GDELT has nothing past today, and a future bound is
        # at best ignored and at worst rejected.
        requested_end = pd.Timestamp(
            f"{int(api_cfg.get('end_year', _API_END_YEAR))}-12-31 23:59:59"
        )
        end = min(requested_end, pd.Timestamp.now('UTC').tz_localize(None)).strftime(
            "%Y%m%d%H%M%S"
        )

        for attempt in range(1, _GDELT_ATTEMPTS + 1):
            try:
                response = requests.get(
                    url,
                    params={
                        "query": query, "mode": mode, "format": "json",
                        "startdatetime": start, "enddatetime": end,
                    },
                    timeout=_GDELT_TIMEOUT,
                )
                if "json" not in response.headers.get("content-type", ""):
                    # 429 arrives as an HTML body, not a JSON error.
                    raise RuntimeError(f"HTTP {response.status_code}")
                timeline = response.json().get("timeline") or []
                points = timeline[0].get("data", []) if timeline else []
                if points:
                    return pd.DataFrame({
                        "timestamp": pd.to_datetime(
                            [p["date"] for p in points], format="%Y%m%dT%H%M%SZ"
                        ).normalize(),
                        "value": [float(p["value"]) for p in points],
                    })
                raise RuntimeError("empty timeline")
            except Exception as exc:
                if attempt == _GDELT_ATTEMPTS:
                    logger.warning(
                        "[NewsConnector/GDELT] %s gave up after %d attempts (%s)",
                        mode, _GDELT_ATTEMPTS, exc,
                    )
                    return pd.DataFrame(columns=["timestamp", "value"])
                logger.info(
                    "[NewsConnector/GDELT] %s attempt %d/%d failed (%s) — retrying",
                    mode, attempt, _GDELT_ATTEMPTS, exc,
                )
                time.sleep(_GDELT_BACKOFF_SECONDS * attempt)
        return pd.DataFrame(columns=["timestamp", "value"])

    @staticmethod
    def _recency_weighted(sentiment: pd.Series, weights_cfg: dict) -> pd.Series:
        """Trailing weighted mean of sentiment, recent days weighted higher.

        Uses the ``recency_weights`` block the config already defines for the
        planned pipeline, so live and synthetic modes share one definition of
        "recent".
        """
        fresh_days = int(weights_cfg.get("fresh_days", 2))
        fresh_weight = float(weights_cfg.get("fresh_weight", 1.0))
        recent_days = int(weights_cfg.get("recent_days", 7))
        recent_weight = float(weights_cfg.get("recent_weight", 0.5))
        old_weight = float(weights_cfg.get("old_weight", 0.1))

        weights = [fresh_weight] * fresh_days
        weights += [recent_weight] * max(recent_days - fresh_days, 0)
        weights += [old_weight] * _RECENCY_TAIL_DAYS
        total = sum(weights) or 1.0

        return (
            sentiment.rolling(window=len(weights), min_periods=1)
            .apply(
                lambda w: float(
                    np.dot(w[::-1], weights[: len(w)]) / sum(weights[: len(w)])
                ),
                raw=True,
            )
            .fillna(sentiment)
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
