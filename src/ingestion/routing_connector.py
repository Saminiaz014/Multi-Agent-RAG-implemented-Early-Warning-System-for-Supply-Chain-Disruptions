"""Vessel-routing connector for the Strait of Hormuz corridor.

Daily-frequency metrics that quantify *pre-emptive rerouting* — when
operators steer ships around the Strait of Hormuz, the system sees it
in this dataset before port-side congestion bites. The synthetic mode
*leads* the shipping disruption windows by ``lead_days`` (default 2)
so detection harnesses can verify the agent flags early.
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

_DEFAULT_CSV_PATH: str = "data/raw/routing_data.csv"
_LOCATION: str = "Strait of Hormuz"
_SOURCE: str = "routing"
_DEFAULT_LEAD_DAYS: int = 2

_BASELINE_REROUTING_PCT: float = 3.0  # percent
_BASELINE_DEVIATION_KM: float = 0.0
_BASELINE_TRANSIT_RATIO: float = 0.97
_BASELINE_VESSELS_HOLDING: float = 1.5
_BASELINE_ALT_ROUTE: float = 0.15
_BASELINE_NOISE = {
    "rerouting": 1.0, "deviation": 50.0, "transit": 0.02,
    "holding": 0.5, "alt_route": 0.03,
}


@dataclass(frozen=True)
class _RoutingScenario:
    name: str
    base_start: int
    base_end: int
    ramp_days: int
    decay_days: int
    rerouting_range: tuple[float, float]
    deviation_range: tuple[float, float]
    transit_range: tuple[float, float]
    holding_range: tuple[float, float]
    alt_route_range: tuple[float, float]


_SCENARIOS: tuple[_RoutingScenario, ...] = (
    _RoutingScenario(
        name="Moderate Tension",
        base_start=60, base_end=74,
        ramp_days=3, decay_days=6,
        rerouting_range=(15.0, 25.0),
        deviation_range=(2000.0, 4000.0),
        transit_range=(0.70, 0.80),
        holding_range=(4.0, 9.0),
        alt_route_range=(0.30, 0.45),
    ),
    _RoutingScenario(
        name="Major Blockage",
        base_start=150, base_end=170,
        ramp_days=4, decay_days=10,
        rerouting_range=(50.0, 70.0),
        deviation_range=(5000.0, 8000.0),
        transit_range=(0.30, 0.50),
        holding_range=(15.0, 30.0),
        alt_route_range=(0.60, 0.80),
    ),
    _RoutingScenario(
        name="Brief Incident",
        base_start=280, base_end=290,
        ramp_days=2, decay_days=4,
        rerouting_range=(8.0, 15.0),
        deviation_range=(800.0, 2000.0),
        transit_range=(0.85, 0.90),
        holding_range=(2.0, 6.0),
        alt_route_range=(0.20, 0.30),
    ),
)


class RoutingConnector(BaseConnector):
    """Daily AIS-derived routing-behaviour generator for the Hormuz corridor."""

    LOCATION: str = _LOCATION
    SOURCE: str = _SOURCE
    FEATURE_COLUMNS: tuple[str, ...] = (
        "rerouting_percentage",
        "avg_route_deviation_km",
        "transit_volume_ratio",
        "vessels_holding",
        "alternative_route_traffic",
        "composite_routing_risk",
    )

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(dict(config or {}))
        self.data_mode: str = str(self.config.get("data_mode", "synthetic")).lower()
        self.csv_path: str = str(self.config.get("csv_path", _DEFAULT_CSV_PATH))
        self.lead_days: int = int(self.config.get("lead_days", _DEFAULT_LEAD_DAYS))

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
        """Generate the synthetic daily routing-behaviour DataFrame."""
        if days <= 0:
            raise ValueError("days must be positive.")
        rng = np.random.default_rng(seed)
        timestamps = pd.date_range("2025-01-01", periods=days, freq="D")

        rerouting = np.clip(
            rng.normal(_BASELINE_REROUTING_PCT, _BASELINE_NOISE["rerouting"], size=days),
            0.0, 100.0,
        )
        deviation = np.clip(
            rng.normal(_BASELINE_DEVIATION_KM, _BASELINE_NOISE["deviation"], size=days),
            0.0, None,
        )
        transit = np.clip(
            rng.normal(_BASELINE_TRANSIT_RATIO, _BASELINE_NOISE["transit"], size=days),
            0.0, 1.0,
        )
        holding = np.clip(
            rng.normal(_BASELINE_VESSELS_HOLDING, _BASELINE_NOISE["holding"], size=days),
            0.0, None,
        )
        alt_route = np.clip(
            rng.normal(_BASELINE_ALT_ROUTE, _BASELINE_NOISE["alt_route"], size=days),
            0.0, 1.0,
        )
        is_disruption = np.zeros(days, dtype=bool)

        for scenario in _SCENARIOS:
            self._apply_scenario(
                rng=rng, scenario=scenario,
                rerouting=rerouting, deviation=deviation,
                transit=transit, holding=holding, alt_route=alt_route,
                is_disruption=is_disruption, total_days=days,
            )

        composite = self._composite(
            rerouting=rerouting, transit=transit, holding=holding, alt_route=alt_route,
        )

        df = pd.DataFrame({
            "timestamp": timestamps,
            "rerouting_percentage": np.round(rerouting, 2),
            "avg_route_deviation_km": np.round(deviation, 1),
            "transit_volume_ratio": np.round(transit, 4),
            "vessels_holding": np.round(holding).astype(int),
            "alternative_route_traffic": np.round(alt_route, 4),
            "composite_routing_risk": np.round(composite, 4),
            "is_disruption": is_disruption,
        })
        logger.info(
            "[RoutingConnector/synthetic] generated %d rows; "
            "disruption_days=%d; lead_days=%d",
            days, int(is_disruption.sum()), self.lead_days,
        )
        return df

    def load_csv(self, path: str | Path | None = None) -> pd.DataFrame:
        csv_path = Path(path) if path is not None else Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Routing CSV not found at {csv_path}. "
                "Provide one or set data_mode='synthetic'."
            )
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if not self.validate(df):
            raise ValueError(
                f"Routing CSV at {csv_path} failed schema validation."
            )
        return df

    def fetch_api(self) -> pd.DataFrame:
        """Planned Kpler / MarineTraffic AIS integration.

        Planned implementation:
            * Pull raw AIS waypoints for vessels with Persian-Gulf
              origin/destination via the Kpler or MarineTraffic API.
            * For each vessel, compare actual route to the great-circle
              path through Hormuz; vessels diverging ≥ 200 km count as
              rerouted. Aggregate daily.
            * Cache responses to ``data/raw/routing_api_cache.json``.

        Raises:
            NotImplementedError: Wiring stubbed for thesis scope.
        """
        raise NotImplementedError(
            "API mode not yet implemented. Planned: Kpler / MarineTraffic "
            "AIS feed. Set data_mode='synthetic' or 'csv' in config."
        )

    def validate(self, df: pd.DataFrame) -> bool:
        required = {
            "timestamp",
            "rerouting_percentage",
            "avg_route_deviation_km",
            "transit_volume_ratio",
            "vessels_holding",
            "alternative_route_traffic",
            "composite_routing_risk",
            "is_disruption",
        }
        missing = required - set(df.columns)
        if missing:
            logger.error("[RoutingConnector] missing columns: %s", sorted(missing))
            return False
        if df["timestamp"].isna().any():
            logger.error("[RoutingConnector] NaN in timestamp")
            return False
        if not df["rerouting_percentage"].between(0.0, 100.0).all():
            logger.error("[RoutingConnector] rerouting_percentage out of [0, 100]")
            return False
        if not df["transit_volume_ratio"].between(0.0, 1.0).all():
            logger.error("[RoutingConnector] transit_volume_ratio out of [0, 1]")
            return False
        if not df["alternative_route_traffic"].between(0.0, 1.0).all():
            logger.error("[RoutingConnector] alternative_route_traffic out of [0, 1]")
            return False
        if (df["vessels_holding"] < 0).any():
            logger.error("[RoutingConnector] negative vessels_holding")
            return False
        if not df["composite_routing_risk"].between(0.0, 1.0).all():
            logger.error("[RoutingConnector] composite_routing_risk out of [0, 1]")
            return False
        return True

    def save_raw(
        self, path: str | Path = _DEFAULT_CSV_PATH
    ) -> Path:
        df = self.fetch()
        if not self.validate(df):
            raise ValueError("Routing data failed validation prior to save.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("[RoutingConnector] wrote %d rows to %s", len(df), out)
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
        scenario: _RoutingScenario,
        rerouting: np.ndarray,
        deviation: np.ndarray,
        transit: np.ndarray,
        holding: np.ndarray,
        alt_route: np.ndarray,
        is_disruption: np.ndarray,
        total_days: int,
    ) -> None:
        """Shift the scenario window earlier by ``lead_days`` to simulate
        operators rerouting ahead of port-side congestion."""
        start = max(scenario.base_start - self.lead_days, 0)
        end = min(scenario.base_end - self.lead_days + scenario.decay_days, total_days - 1)
        if start >= total_days:
            return

        rerouting_peak = rng.uniform(*scenario.rerouting_range)
        deviation_peak = rng.uniform(*scenario.deviation_range)
        transit_low = rng.uniform(*scenario.transit_range)
        holding_peak = rng.uniform(*scenario.holding_range)
        alt_route_peak = rng.uniform(*scenario.alt_route_range)

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

            rerouting[day_idx] = max(rerouting[day_idx], rerouting_peak * intensity)
            deviation[day_idx] = max(deviation[day_idx], deviation_peak * intensity)
            transit[day_idx] = min(
                transit[day_idx],
                _BASELINE_TRANSIT_RATIO - (_BASELINE_TRANSIT_RATIO - transit_low) * intensity,
            )
            transit[day_idx] = max(transit[day_idx], 0.0)
            holding[day_idx] = max(holding[day_idx], holding_peak * intensity)
            alt_route[day_idx] = max(alt_route[day_idx], alt_route_peak * intensity)
            is_disruption[day_idx] = True

        logger.info(
            "[RoutingConnector/synthetic] scenario '%s' days %d-%d "
            "(rerouting peak=%.1f%%, transit min=%.2f, lead=%d)",
            scenario.name, start, end, rerouting_peak, transit_low, self.lead_days,
        )

    @staticmethod
    def _composite(
        *, rerouting: np.ndarray, transit: np.ndarray,
        holding: np.ndarray, alt_route: np.ndarray,
    ) -> np.ndarray:
        # Normalise rerouting and holding into [0, 1] before averaging.
        rerouting_norm = np.clip(rerouting / 60.0, 0.0, 1.0)
        holding_norm = np.clip(holding / 25.0, 0.0, 1.0)
        risk = (
            0.35 * rerouting_norm
            + 0.30 * (1.0 - transit)
            + 0.20 * alt_route
            + 0.15 * holding_norm
        )
        return np.clip(risk, 0.0, 1.0)
