"""Natural-disaster connector for the Strait of Hormuz corridor.

Daily-frequency severity scores for earthquakes, tsunamis, cyclones,
and severe weather, weighted by proximity to the Hormuz chokepoint
(26.5°N, 56.5°E). The signal is **sparse by construction**: most days
sit at near-zero baseline noise, and only Scenario B (the Major
Blockage window in the shipping connector) injects an actual disaster
in the synthetic mode — this is what gives downstream attribution
meaningful contrast (Scenarios A and C are geopolitical/market-only).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.extractors.base_extractor import resolve_env_value
from src.ingestion.base_connector import BaseConnector

logger = logging.getLogger(__name__)

_DEFAULT_CSV_PATH: str = "data/raw/disaster_events.csv"
_LOCATION: str = "Strait of Hormuz"
_SOURCE: str = "natural_disaster"

# Strait of Hormuz center (approx).
_CENTER_LAT: float = 26.5
_CENTER_LON: float = 56.5
_FULL_WEIGHT_RADIUS_KM: float = 500.0
_DECAY_RADIUS_KM: float = 1500.0

_BASELINE_EARTHQUAKE: float = 0.03
_BASELINE_TSUNAMI: float = 0.02
_BASELINE_CYCLONE: float = 0.04
_BASELINE_WEATHER: float = 0.05
_BASELINE_NOISE_SD: float = 0.02

_DEFAULT_WEIGHTS: dict[str, float] = {
    "earthquake": 0.35,
    "tsunami": 0.30,
    "cyclone": 0.20,
    "severe_weather": 0.15,
}

# Scenario B earthquake (lead 2 days before shipping window starts).
_QUAKE_DAY: int = 148
_QUAKE_DECAY_DAYS: int = 7
_QUAKE_PROXIMITY_KM: float = 200.0
_QUAKE_MAGNITUDE: float = 6.5

_MINOR_TREMOR_RATE: float = 0.04  # P(minor tremor on any day)
_MINOR_TREMOR_MAG_RANGE: tuple[float, float] = (2.0, 3.5)

# Ambee categorical -> numerical [0,1] severity mapping (live api mode).
# Ambee's /disasters endpoints expose two categorical fields only — no
# magnitude / wind-speed number — so this is a deliberate approximation,
# unlike the Richter-scale severity used by the synthetic/CSV paths above.
_AMBEE_PROXIMITY_MAP: dict[str, float] = {"Low": 0.20, "Moderate": 0.50, "High Risk": 0.85}
_AMBEE_ALERT_MAP: dict[str, float] = {"Green": 0.10, "Yellow": 0.35, "Orange": 0.65, "Red": 0.90}
_AMBEE_PROXIMITY_WEIGHT: float = 0.6
_AMBEE_ALERT_WEIGHT: float = 0.4

_AMBEE_EVENT_TYPE_TO_FEATURE: dict[str, str] = {
    "EQ": "earthquake_severity",
    "CY": "cyclone_severity",
    "FL": "severe_weather_index",
    "SW": "severe_weather_index",
    "Misc": "severe_weather_index",
}


class DisasterConnector(BaseConnector):
    """Daily natural-disaster severity generator for the Hormuz corridor.

    Args:
        config: Reads ``data_mode``, ``csv_path``, optional
            ``proximity`` overrides, and an ``api`` sub-block.
    """

    LOCATION: str = _LOCATION
    SOURCE: str = _SOURCE
    FEATURE_COLUMNS: tuple[str, ...] = (
        "earthquake_severity",
        "tsunami_risk",
        "cyclone_severity",
        "severe_weather_index",
        "composite_disaster_risk",
    )

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(dict(config or {}))
        self.data_mode: str = str(self.config.get("data_mode", "synthetic")).lower()
        self.csv_path: str = str(self.config.get("csv_path", _DEFAULT_CSV_PATH))
        prox = self.config.get("proximity") or {}
        self.center_lat: float = float(prox.get("center_lat", _CENTER_LAT))
        self.center_lon: float = float(prox.get("center_lon", _CENTER_LON))
        self.full_weight_km: float = float(
            prox.get("full_weight_radius_km", _FULL_WEIGHT_RADIUS_KM)
        )
        self.decay_km: float = float(
            prox.get("decay_radius_km", _DECAY_RADIUS_KM)
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
        """Generate the synthetic daily disaster-severity DataFrame."""
        if days <= 0:
            raise ValueError("days must be positive.")
        rng = np.random.default_rng(seed)
        timestamps = pd.date_range("2025-01-01", periods=days, freq="D")

        earthquake = np.clip(
            rng.normal(_BASELINE_EARTHQUAKE, _BASELINE_NOISE_SD, size=days), 0.0, 1.0
        )
        tsunami = np.clip(
            rng.normal(_BASELINE_TSUNAMI, _BASELINE_NOISE_SD, size=days), 0.0, 1.0
        )
        cyclone = np.clip(
            rng.normal(_BASELINE_CYCLONE, _BASELINE_NOISE_SD, size=days), 0.0, 1.0
        )
        weather = np.clip(
            rng.normal(_BASELINE_WEATHER, _BASELINE_NOISE_SD, size=days), 0.0, 1.0
        )
        is_disruption = np.zeros(days, dtype=bool)
        events: list[list[str]] = [[] for _ in range(days)]

        # Inject Scenario-B earthquake at day _QUAKE_DAY.
        self._inject_earthquake(
            rng=rng,
            day=_QUAKE_DAY,
            magnitude=_QUAKE_MAGNITUDE,
            proximity_km=_QUAKE_PROXIMITY_KM,
            earthquake=earthquake,
            tsunami=tsunami,
            is_disruption=is_disruption,
            events=events,
            total_days=days,
        )

        # Sprinkle minor tremors throughout the year.
        for d in range(days):
            if rng.random() < _MINOR_TREMOR_RATE and not is_disruption[d]:
                mag = rng.uniform(*_MINOR_TREMOR_MAG_RANGE)
                far_proximity_km = rng.uniform(800, 2500)
                weight = self._proximity_weight(far_proximity_km)
                quake_sev = self._magnitude_to_severity(mag) * weight
                earthquake[d] = max(earthquake[d], quake_sev)
                if quake_sev > 0.10:
                    events[d].append(
                        f"Magnitude {mag:.1f} tremor {int(far_proximity_km)}km from Strait"
                    )

        composite = self._composite(
            earthquake=earthquake, tsunami=tsunami,
            cyclone=cyclone, weather=weather,
            weights=_DEFAULT_WEIGHTS,
        )

        df = pd.DataFrame({
            "timestamp": timestamps,
            "earthquake_severity": np.round(earthquake, 4),
            "tsunami_risk": np.round(tsunami, 4),
            "cyclone_severity": np.round(cyclone, 4),
            "severe_weather_index": np.round(weather, 4),
            "composite_disaster_risk": np.round(composite, 4),
            "active_events": [json.dumps(items) for items in events],
            "is_disruption": is_disruption,
        })
        logger.info(
            "[DisasterConnector/synthetic] generated %d rows; "
            "scenario-B disruption_days=%d",
            days, int(is_disruption.sum()),
        )
        return df

    def load_csv(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load CSV at ``path`` (or ``self.csv_path``) and validate schema."""
        csv_path = Path(path) if path is not None else Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Disaster CSV not found at {csv_path}. "
                "Provide one or set data_mode='synthetic'."
            )
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if not self.validate(df):
            raise ValueError(
                f"Disaster CSV at {csv_path} failed schema validation."
            )
        return df

    def fetch_api(self) -> pd.DataFrame:
        """Fetch current disaster risk from the Ambee Disasters API.

        Queries ``/disasters/latest/by-lat-lng`` for every monitoring point
        configured under ``monitoring_points[location]`` (``location``
        defaults to ``"hormuz"``), maps Ambee's two categorical severity
        fields (``proximity_severity_level``, ``default_alert_levels``) onto
        a ``[0, 1]`` score via ``severity_mapping`` in config, and takes the
        worst-case (max) severity per feature column across all points and
        events. ``tsunami_risk`` has no Ambee equivalent — it is approximated
        from any sufficiently severe earthquake event (damped by 0.7), per
        the project's documented design decision; this is a coarse proxy,
        not a real tsunami signal.

        Earthquake magnitude/depth (and a real USGS-based ``tsunami_risk``)
        remain on the FRED-style "planned" path noted in the class docs —
        USGS is unused here because Ambee already covers the chokepoint
        bounding box end to end for this connector.

        Returns:
            Single-row DataFrame matching the synthetic-mode schema
            (``timestamp``, the four feature columns, ``composite_disaster_risk``,
            ``active_events``, ``is_disruption``).

        Raises:
            ValueError: If no Ambee API key or no monitoring points are
                configured — callers (e.g. :class:`~src.orchestrator.Orchestrator`)
                catch ``ValueError`` and fall back to synthetic.
        """
        api_key = resolve_env_value(self.config.get("api", {}).get("ambee_api_key", ""))
        if not api_key:
            raise ValueError(
                "Ambee API key not configured — set agents.natural_disaster.api."
                "ambee_api_key (or AMBEE_API_KEY in .env)."
            )

        location = str(self.config.get("location", "hormuz"))
        monitoring_points = self.config.get("monitoring_points", {}).get(location, [])
        if not monitoring_points:
            raise ValueError(f"No monitoring_points configured for location={location!r}.")

        severity_cfg = self.config.get("severity_mapping", {}) or {}
        proximity_map = severity_cfg.get("proximity", _AMBEE_PROXIMITY_MAP)
        alert_map = severity_cfg.get("alert", _AMBEE_ALERT_MAP)
        prox_weight = float(severity_cfg.get("proximity_weight", _AMBEE_PROXIMITY_WEIGHT))
        alert_weight = float(severity_cfg.get("alert_weight", _AMBEE_ALERT_WEIGHT))
        base_url = self.config.get("api", {}).get("ambee_base_url", "https://api.ambeedata.com")

        feature_maxes: dict[str, float] = {
            "earthquake_severity": 0.0,
            "tsunami_risk": 0.0,
            "cyclone_severity": 0.0,
            "severe_weather_index": 0.0,
        }
        active_events: list[str] = []
        headers = {"x-api-key": api_key, "Content-type": "application/json"}

        for point in monitoring_points:
            lat, lng, name = point["lat"], point["lng"], point.get("name", "")
            try:
                response = requests.get(
                    f"{base_url}/disasters/latest/by-lat-lng",
                    headers=headers,
                    params={"lat": lat, "lng": lng},
                    timeout=30,
                )
                response.raise_for_status()
                events = response.json().get("result", [])
            except requests.RequestException as exc:
                logger.error("[DisasterConnector/api] Ambee request failed for %s: %s", name, exc)
                continue

            for event in events:
                event_type = event.get("event_type", "Misc")
                feature_col = _AMBEE_EVENT_TYPE_TO_FEATURE.get(event_type, "severe_weather_index")
                base_sev = proximity_map.get(event.get("proximity_severity_level", "Low"), 0.2)
                alert_sev = alert_map.get(event.get("default_alert_levels", "Green"), 0.1)
                severity = round(prox_weight * base_sev + alert_weight * alert_sev, 4)

                if severity > feature_maxes[feature_col]:
                    feature_maxes[feature_col] = severity
                if event_type == "EQ" and severity > 0.5:
                    feature_maxes["tsunami_risk"] = max(
                        feature_maxes["tsunami_risk"], severity * 0.7
                    )
                if severity >= 0.35:
                    active_events.append(
                        f"{event.get('event_name', event_type)} near {name} "
                        f"(severity={severity:.2f})"
                    )

        weights = self.config.get("weights") or _DEFAULT_WEIGHTS
        composite = float(np.clip(
            weights["earthquake"] * feature_maxes["earthquake_severity"]
            + weights["tsunami"] * feature_maxes["tsunami_risk"]
            + weights["cyclone"] * feature_maxes["cyclone_severity"]
            + weights["severe_weather"] * feature_maxes["severe_weather_index"],
            0.0, 1.0,
        ))
        max_single = max(feature_maxes.values())
        is_disruption = (
            composite >= float(self.config.get("threshold", 0.30))
            or max_single >= float(self.config.get("single_event_threshold", 0.40))
        )

        df = pd.DataFrame([{
            "timestamp": pd.Timestamp.utcnow().tz_localize(None),
            "earthquake_severity": round(feature_maxes["earthquake_severity"], 4),
            "tsunami_risk": round(feature_maxes["tsunami_risk"], 4),
            "cyclone_severity": round(feature_maxes["cyclone_severity"], 4),
            "severe_weather_index": round(feature_maxes["severe_weather_index"], 4),
            "composite_disaster_risk": round(composite, 4),
            "active_events": json.dumps(active_events),
            "is_disruption": is_disruption,
        }])
        logger.info(
            "[DisasterConnector/api] Ambee live fetch | location=%s | composite=%.4f | "
            "is_disruption=%s | events=%d",
            location, composite, is_disruption, len(active_events),
        )
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Schema + range checks; returns True iff all checks pass."""
        required = {
            "timestamp",
            "earthquake_severity",
            "tsunami_risk",
            "cyclone_severity",
            "severe_weather_index",
            "composite_disaster_risk",
            "is_disruption",
        }
        missing = required - set(df.columns)
        if missing:
            logger.error("[DisasterConnector] missing columns: %s", sorted(missing))
            return False
        score_cols = [c for c in required if c not in {"timestamp", "is_disruption"}]
        for c in score_cols:
            if df[c].isna().any():
                logger.error("[DisasterConnector] NaN in %s", c)
                return False
            if not df[c].between(0.0, 1.0).all():
                logger.error("[DisasterConnector] %s out of [0, 1]", c)
                return False
        return True

    def save_raw(
        self, path: str | Path = _DEFAULT_CSV_PATH
    ) -> Path:
        df = self.fetch()
        if not self.validate(df):
            raise ValueError("Disaster data failed validation prior to save.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("[DisasterConnector] wrote %d rows to %s", len(df), out)
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
    def _inject_earthquake(
        self, *,
        rng: np.random.Generator,
        day: int,
        magnitude: float,
        proximity_km: float,
        earthquake: np.ndarray,
        tsunami: np.ndarray,
        is_disruption: np.ndarray,
        events: list[list[str]],
        total_days: int,
    ) -> None:
        """Stamp a Scenario-B earthquake + tsunami response into the arrays."""
        if day >= total_days:
            return
        weight = self._proximity_weight(proximity_km)
        quake_peak = min(self._magnitude_to_severity(magnitude) * weight, 1.0)
        tsunami_peak = min(quake_peak * 0.85, 1.0)

        for offset in range(_QUAKE_DECAY_DAYS):
            idx = day + offset
            if idx >= total_days:
                break
            decay = max(0.0, 1.0 - (offset / _QUAKE_DECAY_DAYS))
            earthquake[idx] = max(earthquake[idx], quake_peak * decay)
            tsunami[idx] = max(tsunami[idx], tsunami_peak * decay)
            is_disruption[idx] = True
            if offset == 0:
                events[idx].append(
                    f"Magnitude {magnitude:.1f} earthquake "
                    f"{int(proximity_km)}km from Strait of Hormuz"
                )
            if offset == 0 and tsunami_peak >= 0.5:
                events[idx].append("Regional tsunami advisory issued")

        logger.info(
            "[DisasterConnector/synthetic] injected M%.1f quake at day %d "
            "(quake peak=%.2f, tsunami peak=%.2f, proximity=%dkm)",
            magnitude, day, quake_peak, tsunami_peak, int(proximity_km),
        )

    def _proximity_weight(self, distance_km: float) -> float:
        """Linear-decay weight: 1.0 within full_weight, 0.0 beyond decay_km."""
        if distance_km <= self.full_weight_km:
            return 1.0
        if distance_km >= self.decay_km:
            return 0.0
        span = self.decay_km - self.full_weight_km
        return max(0.0, 1.0 - (distance_km - self.full_weight_km) / span * 0.8)

    @staticmethod
    def _magnitude_to_severity(magnitude: float) -> float:
        """Map Richter magnitude → severity in [0, 1]."""
        # Logistic-ish: M3=0.05, M5=0.30, M6=0.50, M6.5=0.70, M7=0.90.
        return float(min(1.0, max(0.0, 1.0 / (1.0 + math.exp(-1.7 * (magnitude - 6.0))))))

    @staticmethod
    def _composite(
        *,
        earthquake: np.ndarray,
        tsunami: np.ndarray,
        cyclone: np.ndarray,
        weather: np.ndarray,
        weights: dict[str, float],
    ) -> np.ndarray:
        return np.clip(
            weights["earthquake"] * earthquake
            + weights["tsunami"] * tsunami
            + weights["cyclone"] * cyclone
            + weights["severe_weather"] * weather,
            0.0, 1.0,
        )
