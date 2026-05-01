"""Synthetic AIS-style connector for the Strait of Hormuz corridor.

In production this would pull from AIS feeds or port APIs. For thesis
purposes it generates realistic synthetic data with three injected
disruption scenarios so the detection agents have ground-truth labels
to validate against.
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


_BASELINE_VESSEL_COUNT: float = 70.0
_BASELINE_DELAY_HOURS: float = 5.0
_BASELINE_CONGESTION: float = 0.25
_BASELINE_OIL_PRICE: float = 77.5

_VESSEL_NOISE_SD: float = 4.0
_DELAY_NOISE_SD: float = 1.2
_CONGESTION_NOISE_SD: float = 0.05
_OIL_NOISE_SD: float = 2.0


@dataclass(frozen=True)
class _DisruptionScenario:
    """Configuration for a single injected disruption window."""

    name: str
    start_day: int
    end_day: int
    ramp_days: int
    decay_days: int
    vessel_drop_range: tuple[float, float]
    delay_mult_range: tuple[float, float]
    congestion_range: tuple[float, float]
    oil_pct_range: tuple[float, float]


_SCENARIOS: tuple[_DisruptionScenario, ...] = (
    _DisruptionScenario(
        name="Moderate Tension",
        start_day=60,
        end_day=74,
        ramp_days=3,
        decay_days=5,
        vessel_drop_range=(0.20, 0.35),
        delay_mult_range=(2.0, 3.0),
        congestion_range=(0.50, 0.70),
        oil_pct_range=(0.10, 0.15),
    ),
    _DisruptionScenario(
        name="Major Blockage",
        start_day=150,
        end_day=170,
        ramp_days=3,
        decay_days=5,
        vessel_drop_range=(0.50, 0.70),
        delay_mult_range=(4.0, 6.0),
        congestion_range=(0.70, 0.95),
        oil_pct_range=(0.25, 0.40),
    ),
    _DisruptionScenario(
        name="Brief Incident",
        start_day=280,
        end_day=290,
        ramp_days=2,
        decay_days=3,
        vessel_drop_range=(0.10, 0.20),
        delay_mult_range=(1.5, 2.0),
        congestion_range=(0.40, 0.60),
        oil_pct_range=(0.05, 0.10),
    ),
)


class ShippingConnector(BaseConnector):
    """Generates synthetic Strait of Hormuz vessel-traffic signals.

    Produces a 365-day daily-frequency DataFrame with vessel count,
    average delay, congestion index, and Brent oil price. Three
    disruption scenarios (moderate, major, brief) are injected with
    gradual onset and recovery so detection agents have realistic
    ground-truth labels to evaluate against.
    """

    LOCATION: str = "Strait of Hormuz"
    SOURCE: str = "shipping"
    FEATURE_COLUMNS: tuple[str, ...] = (
        "vessel_count",
        "avg_delay_hours",
        "congestion_index",
        "oil_price_usd",
    )

    def fetch(self) -> pd.DataFrame:
        """Generate the synthetic dataset using values from ``self.config``.

        Reads ``days`` and ``seed`` from the connector config when present,
        otherwise falls back to 365 days with seed 42.

        Returns:
            Daily-frequency DataFrame ready for downstream agents.
        """
        days = int(self.config.get("days", 365))
        seed = int(self.config.get("seed", 42))
        return self.generate_dataset(days=days, seed=seed)

    def generate_dataset(self, days: int = 365, seed: int = 42) -> pd.DataFrame:
        """Generate the daily synthetic Hormuz dataset.

        Args:
            days: Number of consecutive days to simulate.
            seed: NumPy random seed for reproducibility.

        Returns:
            DataFrame with columns ``timestamp``, ``vessel_count``,
            ``avg_delay_hours``, ``congestion_index``, ``oil_price_usd``,
            and the ground-truth label ``is_disruption``.
        """
        if days <= 0:
            raise ValueError("days must be positive.")

        rng = np.random.default_rng(seed)
        timestamps = pd.date_range(start="2025-01-01", periods=days, freq="D")

        vessel = rng.normal(_BASELINE_VESSEL_COUNT, _VESSEL_NOISE_SD, size=days)
        delay = rng.normal(_BASELINE_DELAY_HOURS, _DELAY_NOISE_SD, size=days)
        delay = np.clip(delay, 1.5, None)
        congestion = rng.normal(_BASELINE_CONGESTION, _CONGESTION_NOISE_SD, size=days)
        oil = rng.normal(_BASELINE_OIL_PRICE, _OIL_NOISE_SD, size=days)
        is_disruption = np.zeros(days, dtype=bool)

        for scenario in _SCENARIOS:
            self._apply_scenario(
                rng=rng,
                scenario=scenario,
                vessel=vessel,
                delay=delay,
                congestion=congestion,
                oil=oil,
                is_disruption=is_disruption,
                total_days=days,
            )

        congestion = np.clip(congestion, 0.0, 1.0)
        vessel_int = np.clip(np.round(vessel).astype(int), 0, None)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "vessel_count": vessel_int,
                "avg_delay_hours": np.round(delay, 2),
                "congestion_index": np.round(congestion, 3),
                "oil_price_usd": np.round(oil, 2),
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
            True when all checks pass, False otherwise. Failures are
            logged at ERROR level so callers can diagnose without
            re-running validation.
        """
        required = {
            "timestamp",
            "vessel_count",
            "avg_delay_hours",
            "congestion_index",
            "oil_price_usd",
            "is_disruption",
        }
        missing = required - set(df.columns)
        if missing:
            logger.error("Missing required columns: %s", sorted(missing))
            return False
        if df.isna().any().any():
            logger.error("NaN values detected in dataset.")
            return False
        if (df["vessel_count"] < 0).any():
            logger.error("Negative vessel_count values detected.")
            return False
        if not df["congestion_index"].between(0.0, 1.0).all():
            logger.error("congestion_index out of [0, 1] range.")
            return False
        n_disrupt = int(df["is_disruption"].sum())
        if not 40 <= n_disrupt <= 55:
            logger.error("Disruption day count %d outside expected ~46 band.", n_disrupt)
            return False
        return True

    def save_raw(
        self, path: str | Path = "data/raw/shipping_hormuz.csv"
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

    def _apply_scenario(
        self,
        rng: np.random.Generator,
        scenario: _DisruptionScenario,
        vessel: np.ndarray,
        delay: np.ndarray,
        congestion: np.ndarray,
        oil: np.ndarray,
        is_disruption: np.ndarray,
        total_days: int,
    ) -> None:
        """Mutate per-feature arrays in place to inject one disruption."""
        start = scenario.start_day
        end = min(scenario.end_day, total_days - 1)
        if start >= total_days:
            return

        vessel_drop = rng.uniform(*scenario.vessel_drop_range)
        delay_mult = rng.uniform(*scenario.delay_mult_range)
        target_congestion = rng.uniform(*scenario.congestion_range)
        oil_pct = rng.uniform(*scenario.oil_pct_range)

        window_len = end - start + 1
        ramp = min(scenario.ramp_days, max(window_len // 3, 1))
        decay = min(scenario.decay_days, max(window_len - ramp - 1, 1))

        for offset, day_idx in enumerate(range(start, end + 1)):
            if offset < ramp:
                intensity = (offset + 1) / (ramp + 1)
            elif offset >= window_len - decay:
                tail_offset = offset - (window_len - decay)
                intensity = 1.0 - (tail_offset + 1) / (decay + 1)
            else:
                intensity = 1.0

            vessel[day_idx] *= 1.0 - vessel_drop * intensity
            delay[day_idx] *= 1.0 + (delay_mult - 1.0) * intensity
            congestion[day_idx] = (
                congestion[day_idx] * (1.0 - intensity)
                + target_congestion * intensity
            )
            oil[day_idx] *= 1.0 + oil_pct * intensity
            is_disruption[day_idx] = True

        logger.info(
            "Injected scenario '%s' days %d-%d "
            "(vessel -%.0f%%, delay x%.2f, congestion %.2f, oil +%.0f%%)",
            scenario.name,
            start,
            end,
            vessel_drop * 100,
            delay_mult,
            target_congestion,
            oil_pct * 100,
        )

    @staticmethod
    def _report_separation(df: pd.DataFrame) -> None:
        """Print a Welch-style t-statistic separating disruption vs normal."""
        normal = df.loc[~df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        disrupted = df.loc[df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        if len(disrupted) == 0 or len(normal) == 0:
            return
        diff = normal.mean() - disrupted.mean()
        pooled_se = np.sqrt(
            normal.var(ddof=1) / len(normal)
            + disrupted.var(ddof=1) / len(disrupted)
        )
        t_stat = diff / pooled_se if pooled_se > 0 else float("inf")
        msg = (
            f"[ShippingConnector] vessel_count separation — "
            f"normal mean={normal.mean():.2f} (n={len(normal)}), "
            f"disruption mean={disrupted.mean():.2f} (n={len(disrupted)}), "
            f"Welch t={t_stat:.2f}"
        )
        logger.info(msg)
        print(msg)
