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


# --- Live ACLED mode --------------------------------------------------------
#: ACLED event categories mapped onto the three features it can support.
#: sanctions_severity has no ACLED analogue and is deliberately absent.
_ACLED_MILITARY_TYPES: frozenset[str] = frozenset(
    {"Battles", "Explosions/Remote violence"}
)
#: ACLED's category for agreements, arrests and non-violent transfers of
#: territory -- the closest the dataset comes to a diplomatic incident.
_ACLED_DIPLOMATIC_TYPES: frozenset[str] = frozenset({"Strategic developments"})
#: Civil unrest, inverted into regime_stability_index.
_ACLED_UNREST_TYPES: frozenset[str] = frozenset({"Protests", "Riots"})

#: Mirrors GeopoliticalAgent's weights with sanctions (0.35) removed and the
#: remainder renormalised, so connector and agent agree on the scale.
_API_COMPOSITE_WEIGHTS: dict[str, float] = {
    "military": 0.25,
    "diplomatic": 0.25,
    "stability": 0.15,
}
_ACLED_RATE_LIMIT: int = 20


class GeopoliticalConnector(BaseConnector):
    """Daily geopolitical-risk signal generator for the Hormuz corridor.

    Args:
        config: Connector-specific configuration block. Reads
            ``data_mode``, ``csv_path``, ``lead_days``, ``acled_countries``,
            ``api`` (sub-block).
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

        # Countries to pull ACLED events for, projected from the active
        # region's extraction.countries by config_manager. Only the ``api``
        # data_mode consumes this; synthetic and CSV are unaffected by its
        # absence, which is why an empty list is a debug note and not a
        # warning at construction time.
        self.acled_countries: list[str] = [
            str(c) for c in (self.config.get("acled_countries") or [])
        ]
        if self.acled_countries:
            logger.info(
                "[GeopoliticalConnector] region ACLED countries: %s",
                self.acled_countries,
            )
        else:
            logger.debug(
                "[GeopoliticalConnector] no acled_countries configured; "
                "only data_mode='api' needs them."
            )

        # Live-ACLED settings. Defaults match the extraction historical range
        # so the live series lines up with the knowledge base's coverage.
        api_cfg: dict = self.config.get("api", {}) or {}
        self.api_start_year: int = int(api_cfg.get("start_year", 2018))
        self.api_end_year: int = int(api_cfg.get("end_year", 2026))
        #: Per country-year cap. Hitting it means that year is truncated, so
        #: it is logged rather than passed off as a complete year.
        self.api_events_per_year: int = int(api_cfg.get("events_per_year", 5000))
        self.api_disruption_threshold: float = float(
            api_cfg.get("disruption_threshold", 0.5)
        )

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
    @staticmethod
    def _normalised_intensity(counts: pd.Series) -> pd.Series:
        """Scale a daily count series into [0, 1] against its own p95.

        A percentile anchor rather than the maximum: one exceptional day would
        otherwise compress every ordinary day toward zero and flatten the
        signal the agent is looking for. Days above the p95 clip at 1.0.
        """
        anchor = float(counts.quantile(0.95))
        if not np.isfinite(anchor) or anchor <= 0:
            # No events at all, or a degenerate series — genuinely flat, and
            # a zeroed series is the honest representation of that.
            return pd.Series(0.0, index=counts.index)
        return (counts / anchor).clip(0.0, 1.0)

    def fetch_api(self) -> pd.DataFrame:
        """Build a daily geopolitical series from ACLED conflict events.

        Three of the four features come from ACLED event categories, daily
        aggregated over the region's countries and scaled to [0, 1]:

        * ``military_activity_index`` — Battles and Explosions/Remote
          violence, weighted so a fatal event counts more than a bloodless
          one.
        * ``diplomatic_incident_score`` — Strategic developments, ACLED's
          category for agreements, arrests and non-violent transfers of
          territory. The closest genuine analogue to a diplomatic incident
          that the dataset carries.
        * ``regime_stability_index`` — inverse of Protest and Riot intensity;
          higher means calmer.

        ``sanctions_severity`` is **not** produced. ACLED carries no sanctions
        data, OpenSanctions' API is paywalled and publishes current
        designations rather than a time series, and sanctions are discrete
        events, so any daily "severity" curve would be a modelling artefact
        rather than a measurement. :class:`~src.agents.geopolitical_agent
        .GeopoliticalAgent` renormalises its weights over the features present
        instead of scoring an invented column.

        Returns:
            Daily-frequency frame over the configured historical range.

        Raises:
            ValueError: If the region has no ``acled_countries`` configured,
                or ACLED returns nothing for all of them — an empty result
                means a credential or country-name problem, not a peaceful
                decade.
        """
        if not self.acled_countries:
            raise ValueError(
                "No acled_countries configured for this region; a live query "
                "would have no country filter. Set agents.geopolitical."
                "acled_countries, or use data_mode='synthetic'."
            )

        from src.extractors.acled_extractor import ACLEDExtractor

        # The extractor owns ACLED's OAuth lifecycle; re-implementing the
        # token exchange here would be a second thing to keep working.
        extractor = ACLEDExtractor(self._acled_config())
        events: list[dict] = []
        for country in self.acled_countries:
            for year in range(self.api_start_year, self.api_end_year + 1):
                batch = extractor._fetch_events(
                    country, year, limit=self.api_events_per_year
                )
                if len(batch) >= self.api_events_per_year:
                    # Same silent-truncation trap as the extractors: a capped
                    # page is indistinguishable from a complete one.
                    logger.warning(
                        "[GeopoliticalConnector] %s/%d hit the %d-event cap — "
                        "that year is incomplete; raise events_per_year.",
                        country, year, self.api_events_per_year,
                    )
                events.extend(batch)

        if not events:
            raise ValueError(
                f"ACLED returned no events for {self.acled_countries} over "
                f"{self.api_start_year}-{self.api_end_year}. Check "
                "ACLED_USERNAME / ACLED_PASSWORD and the country spellings."
            )

        raw = pd.DataFrame(events)
        raw["timestamp"] = pd.to_datetime(raw["event_date"], errors="coerce")
        raw = raw.dropna(subset=["timestamp"])
        raw["fatalities"] = pd.to_numeric(
            raw.get("fatalities"), errors="coerce"
        ).fillna(0.0)
        # A fatal event weighs more than a bloodless one, with diminishing
        # returns: log1p keeps a single mass-casualty day from dominating.
        raw["weight"] = 1.0 + np.log1p(raw["fatalities"])

        daily_index = pd.date_range(
            raw["timestamp"].min().normalize(),
            raw["timestamp"].max().normalize(),
            freq="D",
        )
        df = pd.DataFrame({"timestamp": daily_index})

        def _daily(types: set[str], weighted: bool) -> pd.Series:
            subset = raw[raw["event_type"].isin(types)]
            grouped = (
                subset.groupby(subset["timestamp"].dt.normalize())["weight"].sum()
                if weighted
                else subset.groupby(subset["timestamp"].dt.normalize()).size()
            )
            return grouped.reindex(daily_index, fill_value=0.0).astype(float)

        military = _daily(_ACLED_MILITARY_TYPES, weighted=True)
        diplomatic = _daily(_ACLED_DIPLOMATIC_TYPES, weighted=False)
        unrest = _daily(_ACLED_UNREST_TYPES, weighted=False)

        df["military_activity_index"] = self._normalised_intensity(military).to_numpy()
        df["diplomatic_incident_score"] = self._normalised_intensity(
            diplomatic
        ).to_numpy()
        df["regime_stability_index"] = (
            1.0 - self._normalised_intensity(unrest)
        ).to_numpy()

        # Mirrors the agent's renormalisation: a weighted mean over the three
        # available features, so the connector's own composite stays on the
        # same scale as a four-feature one.
        weights = _API_COMPOSITE_WEIGHTS
        df["composite_geopolitical_risk"] = (
            weights["military"] * df["military_activity_index"]
            + weights["diplomatic"] * df["diplomatic_incident_score"]
            + weights["stability"] * (1.0 - df["regime_stability_index"])
        ) / sum(weights.values())

        df["flagged_incidents"] = ""
        df["is_disruption"] = (
            df["composite_geopolitical_risk"] >= self.api_disruption_threshold
        )

        logger.info(
            "[GeopoliticalConnector/ACLED] %d events over %s -> %d days "
            "[%s .. %s] disruption_days=%d (no sanctions_severity: ACLED "
            "carries none)",
            len(raw), self.acled_countries, len(df),
            df["timestamp"].min().date(), df["timestamp"].max().date(),
            int(df["is_disruption"].sum()),
        )
        return df

    def _acled_config(self) -> dict:
        """Config shaped as :class:`ACLEDExtractor` expects it.

        The connector receives ``agents.geopolitical``; the extractor wants
        credentials under ``api_keys``. Reading the environment directly here
        would be a second credential path to keep in step.
        """
        api_cfg = self.config.get("api", {}) or {}
        return {
            "api_keys": {
                "acled_username": api_cfg.get("acled_username")
                or "${ACLED_USERNAME}",
                "acled_password": api_cfg.get("acled_password")
                or "${ACLED_PASSWORD}",
            },
            "extraction": {"rate_limits": {"acled": _ACLED_RATE_LIMIT}},
        }

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
