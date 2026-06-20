"""Geopolitical risk connector for the Strait of Hormuz corridor.

Produces a daily-frequency DataFrame of geopolitical risk indicators:
sanctions activity, military deployments, diplomatic incidents, and
regime stability. Supports three ingestion modes:

- ``"synthetic"``: Generate a 365-day dataset with disruption scenarios
  that *lead* the shipping disruptions by ``lead_days`` (default 3).
- ``"csv"``: Load from a user-supplied CSV at ``csv_path``.
- ``"api"``: Stubbed integration with ACLED + OpenSanctions; raises
  ``NotImplementedError`` and documents the planned wiring.

Unlike :class:`~src.ingestion.ShippingConnector` and
:class:`~src.ingestion.MarketConnector`, geopolitical events are
categorical / event-driven rather than continuous time series, so the
output is best interpreted as a *severity envelope* with a free-text
``flagged_incidents`` column rather than a measurement series.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.base_connector import BaseConnector

logger = logging.getLogger(__name__)


_DEFAULT_CSV_PATH: str = "data/raw/geopolitical_events.csv"
_DEFAULT_LEAD_DAYS: int = 3
_LOCATION: str = "Strait of Hormuz"
_SOURCE: str = "geopolitical"

#Why are we defining baseline levels and noise for the synthetic data generation?
# Because we want the synthetic data to have realistic variability around typical levels 
# of geopolitical risk, with occasional spikes during disruption scenarios. 
# The baselines represent typical "quiet" conditions, while the noise adds day-to-day 
# fluctuations that make the data more lifelike. This allows us to test the agent's ability 
# to detect meaningful elevations in risk above normal background levels, rather than just flagging 
# any non-zero value as a disruption.
_BASELINE_SANCTIONS: float = 0.10
_BASELINE_MILITARY: float = 0.15
_BASELINE_DIPLOMATIC: float = 0.05
_BASELINE_STABILITY: float = 0.80
_BASELINE_NOISE_SD: float = 0.03

_DEFAULT_WEIGHTS: dict[str, float] = {
    "sanctions": 0.35,
    "military": 0.25,
    "diplomatic": 0.25,
    "stability": 0.15,
}


@dataclass(frozen=True)
class _Scenario:
    """One synthetic geopolitical disruption window."""

    name: str
    base_start: int
    base_end: int
    ramp_days: int
    decay_days: int
    sanctions_range: tuple[float, float]
    military_range: tuple[float, float]
    diplomatic_range: tuple[float, float]
    stability_drop: tuple[float, float]
    incident_templates: tuple[str, ...]


_SCENARIOS: tuple[_Scenario, ...] = (
    _Scenario(
        name="Moderate Tension",
        base_start=60, base_end=74,
        ramp_days=4, decay_days=8,
        sanctions_range=(0.40, 0.60),
        military_range=(0.50, 0.70),
        diplomatic_range=(0.30, 0.50),
        stability_drop=(0.05, 0.15),
        incident_templates=(
            "Targeted sanctions package announced against shipping firms",
            "Naval freedom-of-navigation exercise reported in Gulf region",
            "Diplomatic protest filed at the UN over corridor access",
        ),
    ),
    _Scenario(
        name="Major Blockage",
        base_start=150, base_end=170,
        ramp_days=5, decay_days=10,
        sanctions_range=(0.70, 0.90),
        military_range=(0.80, 0.95),
        diplomatic_range=(0.60, 0.80),
        stability_drop=(0.30, 0.50),
        incident_templates=(
            "Comprehensive sanctions package targeting maritime exports",
            "Major naval deployment to Gulf chokepoint reported",
            "Diplomatic mission expelled following corridor incident",
            "Regional ally calls for military de-escalation",
        ),
    ),
    _Scenario(
        name="Brief Incident",
        base_start=280, base_end=290,
        ramp_days=3, decay_days=5,
        sanctions_range=(0.20, 0.40),
        military_range=(0.30, 0.50),
        diplomatic_range=(0.20, 0.30),
        stability_drop=(0.02, 0.08),
        incident_templates=(
            "Maritime incident under investigation",
            "Diplomatic clarification requested by regional partner",
        ),
    ),
)


class GeopoliticalConnector(BaseConnector):
    """Daily geopolitical-risk signal generator for the Hormuz corridor.

    Args:
        config: Connector-specific configuration block. Reads
            ``data_mode``, ``csv_path``, ``lead_days``, ``api`` (sub-block).
    """

    LOCATION: str = _LOCATION
    SOURCE: str = _SOURCE
    FEATURE_COLUMNS: tuple[str, ...] = (
        "sanctions_severity",
        "military_activity_index",
        "diplomatic_incident_score",
        "regime_stability_index",
        "composite_geopolitical_risk",
    )

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(dict(config or {}))
        self.data_mode: str = str(self.config.get("data_mode", "synthetic")).lower()
        self.csv_path: str = str(self.config.get("csv_path", _DEFAULT_CSV_PATH))
        self.lead_days: int = int(self.config.get("lead_days", _DEFAULT_LEAD_DAYS))

    # ------------------------------------------------------------------ fetch
    def fetch(self) -> pd.DataFrame:
        """Route to the configured ingestion mode."""
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
        """Generate a synthetic daily geopolitical-risk DataFrame.

        Args:
            days: Number of days to simulate.
            seed: NumPy seed for reproducibility.

        Returns:
            DataFrame whose disruption scenarios *lead* the shipping
            connector's windows by ``self.lead_days`` so tensions build
            before vessels are affected.
        """
        if days <= 0:
            raise ValueError("days must be positive.")
        rng = np.random.default_rng(seed)
        timestamps = pd.date_range("2025-01-01", periods=days, freq="D")

        sanctions = np.clip(
            rng.normal(_BASELINE_SANCTIONS, _BASELINE_NOISE_SD, size=days), 0.0, 1.0
        )
        military = np.clip(
            rng.normal(_BASELINE_MILITARY, _BASELINE_NOISE_SD, size=days), 0.0, 1.0
        )
        diplomatic = np.clip(
            rng.normal(_BASELINE_DIPLOMATIC, _BASELINE_NOISE_SD, size=days), 0.0, 1.0
        )
        stability = np.clip(
            rng.normal(_BASELINE_STABILITY, _BASELINE_NOISE_SD, size=days), 0.0, 1.0
        )
        is_disruption = np.zeros(days, dtype=bool)
        incidents: list[list[str]] = [[] for _ in range(days)]

        for scenario in _SCENARIOS:
            self._apply_scenario(
                rng=rng,
                scenario=scenario,
                sanctions=sanctions, military=military,
                diplomatic=diplomatic, stability=stability,
                is_disruption=is_disruption, incidents=incidents,
                total_days=days,
            )

        composite = self._composite(
            sanctions=sanctions, military=military,
            diplomatic=diplomatic, stability=stability,
            weights=_DEFAULT_WEIGHTS,
        )

        df = pd.DataFrame({
            "timestamp": timestamps,
            "sanctions_severity": np.round(sanctions, 4),
            "military_activity_index": np.round(military, 4),
            "diplomatic_incident_score": np.round(diplomatic, 4),
            "regime_stability_index": np.round(stability, 4),
            "composite_geopolitical_risk": np.round(composite, 4),
            "flagged_incidents": [json.dumps(items) for items in incidents],
            "is_disruption": is_disruption,
        })
        logger.info(
            "[GeopoliticalConnector/synthetic] generated %d rows; "
            "disruption_days=%d; lead_days=%d",
            days, int(is_disruption.sum()), self.lead_days,
        )
        return df

    # ----------------------------------------------------------- csv mode
    def load_csv(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load a CSV at ``path`` (or ``self.csv_path``) and validate schema."""
        csv_path = Path(path) if path is not None else Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Geopolitical CSV not found at {csv_path}. "
                "Provide one or set data_mode='synthetic' in config."
            )
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if not self.validate(df):
            raise ValueError(
                f"Geopolitical CSV at {csv_path} failed schema validation."
            )
        return df

    # ----------------------------------------------------------- api mode
    def fetch_api(self) -> pd.DataFrame:
        """Planned ACLED + OpenSanctions integration.

        Planned implementation:
            * ACLED (``api.acleddata.com``) — filter Middle-East/Gulf armed
              conflict events; daily-aggregate severity by event_type
              weights → ``military_activity_index``.
            * OpenSanctions API — extract new sanctions designations
              targeting maritime / oil entities → ``sanctions_severity``.
            * Diplomatic incidents derived from event_type tagging plus
              key actor pairs; regime stability inferred from rolling
              event volume.

        Raises:
            NotImplementedError: Always — wiring stubbed for thesis scope.
        """
        raise NotImplementedError(
            "API mode not yet implemented. Planned: ACLED + OpenSanctions "
            "(see docstring). Set data_mode='synthetic' or 'csv' in config."
        )

    # ----------------------------------------------------------- validate
    def validate(self, df: pd.DataFrame) -> bool:
        """Schema + domain checks; returns True iff all checks pass."""
        required = {
            "timestamp",
            "sanctions_severity",
            "military_activity_index",
            "diplomatic_incident_score",
            "regime_stability_index",
            "composite_geopolitical_risk",
            "is_disruption",
        }
        missing = required - set(df.columns)
        if missing:
            logger.error("[GeopoliticalConnector] missing columns: %s", sorted(missing))
            return False
        score_cols = [c for c in required if c not in {"timestamp", "is_disruption"}]
        for c in score_cols:
            if df[c].isna().any():
                logger.error("[GeopoliticalConnector] NaN in %s", c)
                return False
            if not df[c].between(0.0, 1.0).all():
                logger.error("[GeopoliticalConnector] %s out of [0, 1]", c)
                return False
        return True

    # ----------------------------------------------------------- persist
    def save_raw(
        self, path: str | Path = _DEFAULT_CSV_PATH
    ) -> Path:
        """Generate + validate + persist the dataset as CSV."""
        df = self.fetch()
        if not self.validate(df):
            raise ValueError("Geopolitical data failed validation prior to save.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("[GeopoliticalConnector] wrote %d rows to %s", len(df), out)
        return out.resolve()

    def to_signal_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Convert to the unified `{timestamp, source, feature, value, location}` schema."""
        records: list[dict[str, Any]] = []
        feature_cols = [
            c for c in self.FEATURE_COLUMNS if c in df.columns
        ]
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
        self,
        *,
        rng: np.random.Generator,
        scenario: _Scenario,
        sanctions: np.ndarray,
        military: np.ndarray,
        diplomatic: np.ndarray,
        stability: np.ndarray,
        is_disruption: np.ndarray,
        incidents: list[list[str]],
        total_days: int,
    ) -> None:
        """Inject a scenario with ``lead_days`` lead over the shipping window."""
        start = max(scenario.base_start - self.lead_days, 0)
        end = min(scenario.base_end - self.lead_days + scenario.decay_days, total_days - 1)
        if start >= total_days:
            return

        sanctions_peak = rng.uniform(*scenario.sanctions_range)
        military_peak = rng.uniform(*scenario.military_range)
        diplomatic_peak = rng.uniform(*scenario.diplomatic_range)
        stability_dip = rng.uniform(*scenario.stability_drop)

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
            sanctions[day_idx] = max(sanctions[day_idx], sanctions_peak * intensity)
            military[day_idx] = max(military[day_idx], military_peak * intensity)
            diplomatic[day_idx] = max(diplomatic[day_idx], diplomatic_peak * intensity)
            stability[day_idx] = min(stability[day_idx], _BASELINE_STABILITY - stability_dip * intensity)
            stability[day_idx] = max(stability[day_idx], 0.0)
            is_disruption[day_idx] = True
            if intensity >= 0.6 and not incidents[day_idx]:
                incidents[day_idx].append(
                    rng.choice(np.array(scenario.incident_templates))
                )

        logger.info(
            "[GeopoliticalConnector/synthetic] scenario '%s' days %d-%d "
            "(sanctions peak=%.2f, military peak=%.2f, lead=%d)",
            scenario.name, start, end, sanctions_peak, military_peak, self.lead_days,
        )

    @staticmethod
    def _composite(
        *,
        sanctions: np.ndarray,
        military: np.ndarray,
        diplomatic: np.ndarray,
        stability: np.ndarray,
        weights: dict[str, float],
    ) -> np.ndarray:
        """Per-day weighted composite risk score."""
        return np.clip(
            weights["sanctions"] * sanctions
            + weights["military"] * military
            + weights["diplomatic"] * diplomatic
            + weights["stability"] * (1.0 - stability),
            0.0, 1.0,
        )
