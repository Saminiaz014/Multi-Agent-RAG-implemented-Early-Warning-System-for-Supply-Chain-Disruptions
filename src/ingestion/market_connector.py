"""Synthetic market-data connector for the Strait of Hormuz corridor.

In production this would pull from oil-futures APIs, port throughput
feeds, and the Baltic Exchange. For thesis purposes it generates a
daily-frequency synthetic dataset temporally aligned with
:class:`ShippingConnector` so the multi-agent system can cross-validate
shipping anomalies against confirming market signals.

Disruptions propagate from shipping to markets with a 1-2 day
information-propagation lag and then mean-revert exponentially toward
baseline so prices/volumes do not snap back instantly when the physical
event ends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.ingestion.base_connector import BaseConnector

logger = logging.getLogger(__name__)


# Baselines tuned to the user-specified ranges in the prompt:
#   Brent: $75-82, freight: 100-120, trade volume: 0-1 normalised.
_BASELINE_BRENT_USD: float = 78.5
_BASELINE_TRADE_VOLUME: float = 0.85
_BASELINE_FREIGHT_INDEX: float = 110.0

_BRENT_NOISE_SD: float = 0.6
_TRADE_NOISE_SD: float = 0.02
_FREIGHT_NOISE_SD: float = 1.5

# Peak response per unit of severity (severity in [0, 1]).
_BRENT_PEAK_PCT: float = 0.20
_TRADE_PEAK_DROP: float = 0.45
_FREIGHT_PEAK_PCT: float = 0.50

# Per-day persistence of the disruption envelope post-window.
# Higher = slower mean reversion. 0.7 implies ~30% decay per day.
_PERSISTENCE: float = 0.70

_DEFAULT_LAG_DAYS: int = 2


@dataclass(frozen=True)
class _DefaultPeriod:
    """Default disruption period aligned with the shipping connector."""

    start_day: int
    end_day: int
    severity: float


# Severities chosen to mirror the shipping connector's three scenarios:
# Moderate Tension, Major Blockage, Brief Incident.
_DEFAULT_DISRUPTION_PERIODS: tuple[_DefaultPeriod, ...] = (
    _DefaultPeriod(start_day=60, end_day=74, severity=0.45),
    _DefaultPeriod(start_day=150, end_day=170, severity=1.00),
    _DefaultPeriod(start_day=280, end_day=290, severity=0.25),
)


class MarketConnector(BaseConnector):
    """Generates synthetic Brent crude / trade volume / freight signals.

    Produces a daily-frequency DataFrame whose disruption windows are
    aligned with — but slightly lagged behind — those produced by
    :class:`ShippingConnector`. Mean-reverting tails extend the response
    past the underlying shipping window so post-disruption dynamics look
    realistic instead of instantly snapping back.
    """

    LOCATION: str = "Strait of Hormuz"
    SOURCE: str = "market"
    FEATURE_COLUMNS: tuple[str, ...] = (
        "brent_crude_usd",
        "trade_volume_index",
        "freight_rate_index",
    )

    def fetch(self) -> pd.DataFrame:
        """Generate the synthetic dataset using values from ``self.config``.

        Reads ``days``, ``seed``, and ``lag_days`` from the connector
        config when present, otherwise uses the defaults.

        Returns:
            Daily-frequency DataFrame ready for downstream agents.
        """
        days = int(self.config.get("days", 365))
        seed = int(self.config.get("seed", 42))
        lag_days = int(self.config.get("lag_days", _DEFAULT_LAG_DAYS))
        return self.generate_dataset(days=days, seed=seed, lag_days=lag_days)

    def generate_dataset(
        self,
        days: int = 365,
        seed: int = 42,
        disruption_periods: Iterable[Sequence[float]] | None = None,
        lag_days: int = _DEFAULT_LAG_DAYS,
    ) -> pd.DataFrame:
        """Generate the daily synthetic market dataset.

        Args:
            days: Number of consecutive days to simulate.
            seed: NumPy random seed for reproducibility.
            disruption_periods: Iterable of ``(start_day, end_day, severity)``
                tuples. ``severity`` is a float in roughly ``[0, 1]`` that
                scales the magnitude of the market response. When ``None``,
                the three default scenarios aligned with
                :class:`ShippingConnector` are used.
            lag_days: Number of days the market response lags the underlying
                shipping disruption to simulate information propagation.

        Returns:
            DataFrame with columns ``timestamp``, ``brent_crude_usd``,
            ``trade_volume_index``, ``freight_rate_index``, and the
            ground-truth label ``is_disruption``.
        """
        if days <= 0:
            raise ValueError("days must be positive.")
        if lag_days < 0:
            raise ValueError("lag_days must be non-negative.")

        rng = np.random.default_rng(seed)
        timestamps = pd.date_range(start="2025-01-01", periods=days, freq="D")
        periods = self._resolve_periods(disruption_periods)

        # `is_disruption` marks the underlying shipping window so this
        # column matches the shipping connector's ground truth exactly,
        # enabling clean cross-agent validation.
        is_disruption = np.zeros(days, dtype=bool)

        # Build the lagged market envelope: peaks `lag_days` after the
        # shipping window opens and decays exponentially after it closes.
        market_envelope = np.zeros(days)
        for start, end, severity in periods:
            for day in range(max(start, 0), min(end, days - 1) + 1):
                is_disruption[day] = True
            self._stamp_envelope(
                envelope=market_envelope,
                start=start + lag_days,
                end=end + lag_days,
                severity=float(severity),
                total_days=days,
            )
        market_envelope = self._apply_persistence(market_envelope, days=days)

        brent = rng.normal(_BASELINE_BRENT_USD, _BRENT_NOISE_SD, size=days)
        trade = rng.normal(_BASELINE_TRADE_VOLUME, _TRADE_NOISE_SD, size=days)
        freight = rng.normal(_BASELINE_FREIGHT_INDEX, _FREIGHT_NOISE_SD, size=days)

        brent *= 1.0 + _BRENT_PEAK_PCT * market_envelope
        trade -= _TRADE_PEAK_DROP * market_envelope
        freight *= 1.0 + _FREIGHT_PEAK_PCT * market_envelope

        trade = np.clip(trade, 0.0, 1.0)
        brent = np.clip(brent, 30.0, 250.0)
        freight = np.clip(freight, 50.0, 400.0)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "brent_crude_usd": np.round(brent, 2),
                "trade_volume_index": np.round(trade, 4),
                "freight_rate_index": np.round(freight, 2),
                "is_disruption": is_disruption,
            }
        )

        self._report_separation(df)
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Run schema and domain checks against the generated dataset.

        Args:
            df: DataFrame returned by :meth:`fetch` or :meth:`generate_dataset`.

        Returns:
            True when all checks pass, False otherwise.
        """
        required = {
            "timestamp",
            "brent_crude_usd",
            "trade_volume_index",
            "freight_rate_index",
            "is_disruption",
        }
        missing = required - set(df.columns)
        if missing:
            logger.error("Missing required columns: %s", sorted(missing))
            return False
        if df.isna().any().any():
            logger.error("NaN values detected in dataset.")
            return False
        if not df["trade_volume_index"].between(0.0, 1.0).all():
            logger.error("trade_volume_index out of [0, 1] range.")
            return False
        if (df["brent_crude_usd"] <= 0).any():
            logger.error("Non-positive brent_crude_usd values detected.")
            return False
        if (df["freight_rate_index"] <= 0).any():
            logger.error("Non-positive freight_rate_index values detected.")
            return False
        return True

    def save_raw(
        self, path: str | Path = "data/raw/market_data.csv"
    ) -> Path:
        """Generate, validate, and persist the dataset as CSV.

        Args:
            path: Destination CSV path. Parent directories are created if
                missing.

        Returns:
            Resolved absolute path of the written file.
        """
        df = self.fetch_and_validate()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Wrote %d rows to %s", len(df), out)
        return out.resolve()

    def to_signal_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Convert a dataset into the unified signal-record schema.

        Emits one record per (timestamp, feature) pair, excluding the
        ground-truth ``is_disruption`` label.

        Args:
            df: DataFrame returned by :meth:`generate_dataset`.

        Returns:
            List of dicts with keys ``timestamp``, ``source``, ``feature``,
            ``value``, ``location`` — JSON-serialisable.
        """
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["timestamp"]).isoformat()
            for feature in self.FEATURE_COLUMNS:
                records.append(
                    {
                        "timestamp": ts,
                        "source": self.SOURCE,
                        "feature": feature,
                        "value": float(row[feature]),
                        "location": self.LOCATION,
                    }
                )
        return records

    @staticmethod
    def _resolve_periods(
        disruption_periods: Iterable[Sequence[float]] | None,
    ) -> list[tuple[int, int, float]]:
        """Normalise the disruption_periods argument into validated tuples."""
        if disruption_periods is None:
            return [
                (p.start_day, p.end_day, p.severity)
                for p in _DEFAULT_DISRUPTION_PERIODS
            ]
        out: list[tuple[int, int, float]] = []
        for tup in disruption_periods:
            tup = tuple(tup)
            if len(tup) != 3:
                raise ValueError(
                    "Each disruption period must be (start_day, end_day, severity)."
                )
            start, end, severity = tup
            if int(start) > int(end):
                raise ValueError("start_day must not exceed end_day.")
            out.append((int(start), int(end), float(severity)))
        return out

    @staticmethod
    def _stamp_envelope(
        envelope: np.ndarray,
        start: int,
        end: int,
        severity: float,
        total_days: int,
    ) -> None:
        """Stamp a triangular ramp/plateau/decay envelope into ``envelope``."""
        if start >= total_days or severity <= 0.0:
            return
        clipped_start = max(start, 0)
        clipped_end = min(end, total_days - 1)
        if clipped_end < clipped_start:
            return

        window_len = clipped_end - clipped_start + 1
        ramp = max(window_len // 4, 1)
        decay = max(window_len // 4, 1)

        for offset, day_idx in enumerate(range(clipped_start, clipped_end + 1)):
            if offset < ramp:
                intensity = (offset + 1) / (ramp + 1)
            elif offset >= window_len - decay:
                tail_offset = offset - (window_len - decay)
                intensity = 1.0 - (tail_offset + 1) / (decay + 1)
            else:
                intensity = 1.0
            envelope[day_idx] = max(envelope[day_idx], severity * intensity)

    @staticmethod
    def _apply_persistence(envelope: np.ndarray, days: int) -> np.ndarray:
        """Carry residual disruption forward for mean-reverting tails.

        Each day's value is at least the raw envelope (so ramp-up is
        unaffected) but no less than ``_PERSISTENCE`` times the previous
        day's smoothed value, producing exponential decay after the
        shipping window closes.
        """
        out = np.zeros_like(envelope)
        if days == 0:
            return out
        out[0] = envelope[0]
        for i in range(1, days):
            out[i] = max(envelope[i], _PERSISTENCE * out[i - 1])
        return out

    @staticmethod
    def _report_separation(df: pd.DataFrame) -> None:
        """Log mean trade-volume index in normal vs disruption windows."""
        normal = df.loc[~df["is_disruption"], "trade_volume_index"].to_numpy(dtype=float)
        disrupted = df.loc[df["is_disruption"], "trade_volume_index"].to_numpy(dtype=float)
        if len(disrupted) == 0 or len(normal) == 0:
            return
        msg = (
            f"[MarketConnector] trade_volume_index — "
            f"normal mean={normal.mean():.3f} (n={len(normal)}), "
            f"disruption mean={disrupted.mean():.3f} (n={len(disrupted)})"
        )
        logger.info(msg)
        print(msg)
