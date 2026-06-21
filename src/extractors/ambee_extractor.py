"""Ambee Disasters API extractor for natural disaster data.

Covers: natural_disaster agent. Replaces ReliefWebExtractor in the default
extraction pipeline (ReliefWeb now gates access behind an appname-approval
step; see reliefweb_extractor.py, kept as a fallback).

Important limitation: on the free/registered-key tier, Ambee's
``/disasters/history/by-lat-lng`` endpoint only covers roughly the last 30
days ("For data older than one month, contact us!") — it cannot backfill
the project's deep historical disruption cases (Cyclone Gonu 2007, the 2011
Japan tsunami, etc.). This extractor is therefore most useful for recent /
ongoing events; :class:`~src.extractors.reliefweb_extractor.ReliefWebExtractor`
remains the intended source for deep historical backfill once approved.

Note the two Ambee endpoints use different response envelopes:
``/latest`` returns events under ``"result"``; ``/history`` returns them
under ``"data"``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from src.extractors.base_extractor import BaseExtractor, resolve_env_value

logger = logging.getLogger(__name__)

PROXIMITY_MAP: dict[str, float] = {"Low": 0.20, "Moderate": 0.50, "High Risk": 0.85}
ALERT_MAP: dict[str, float] = {"Green": 0.10, "Yellow": 0.35, "Orange": 0.65, "Red": 0.90}
PROX_WEIGHT: float = 0.6
ALERT_WEIGHT: float = 0.4

EVENT_TYPE_LABELS: dict[str, str] = {
    "EQ": "Earthquake",
    "CY": "Cyclone",
    "FL": "Flood",
    "SW": "Severe Weather",
    "Misc": "Miscellaneous Disaster",
}

EVENT_TYPE_TO_AGENT_FEATURE: dict[str, str] = {
    "EQ": "earthquake_severity",
    "CY": "cyclone_severity",
    "FL": "severe_weather_index",
    "SW": "severe_weather_index",
    "Misc": "severe_weather_index",
}

# Ambee only allows ~30 days of lookback on /history before requiring a paid plan.
_HISTORY_LOOKBACK_DAYS = 30


class AmbeeExtractor(BaseExtractor):
    """Extract natural disaster data from the Ambee Disasters API."""

    @property
    def source_name(self) -> str:
        return "ambee"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        disaster_cfg = config.get("agents", {}).get("natural_disaster", {}) or {}
        self.api_key = resolve_env_value(disaster_cfg.get("api", {}).get("ambee_api_key", ""))
        self.base_url = disaster_cfg.get("api", {}).get("ambee_base_url", "https://api.ambeedata.com")
        self.monitoring_points = disaster_cfg.get("monitoring_points", {}) or {}

        if not self.api_key:
            logger.warning(
                "Ambee API key not configured — set agents.natural_disaster.api."
                "ambee_api_key (or AMBEE_API_KEY in .env)."
            )

    @staticmethod
    def _compute_severity(event: dict) -> float:
        """Combined: 0.6 * proximity_severity + 0.4 * alert_level."""
        base = PROXIMITY_MAP.get(event.get("proximity_severity_level", "Low"), 0.2)
        alert_val = ALERT_MAP.get(event.get("default_alert_levels", "Green"), 0.1)
        return round(PROX_WEIGHT * base + ALERT_WEIGHT * alert_val, 3)

    @staticmethod
    def _classify_severity(score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def _fetch_disasters_at_point(
        self, lat: float, lng: float, endpoint: str = "latest",
        date_from: str = "", date_to: str = "",
    ) -> list[dict]:
        """Fetch disasters from Ambee for a single coordinate.

        ``/latest`` responses key events under ``"result"``; ``/history``
        responses key them under ``"data"``.
        """
        if not self.api_key:
            return []
        self._rate_limit_wait()

        path = "disasters/history/by-lat-lng" if endpoint == "history" else "disasters/latest/by-lat-lng"
        url = f"{self.base_url}/{path}"
        headers = {"x-api-key": self.api_key, "Content-type": "application/json"}
        params: dict[str, Any] = {"lat": lat, "lng": lng}
        if endpoint == "history":
            if date_from:
                params["from"] = date_from
            if date_to:
                params["to"] = date_to

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Ambee API failed for (%s, %s) [%s]: %s", lat, lng, endpoint, exc)
            return []

        key = "data" if endpoint == "history" else "result"
        return payload.get(key, [])

    def extract_historical(self, region: str, **kwargs) -> list[dict]:
        """Extract disaster events for all monitoring points in a region.

        Tries the ``/history`` endpoint over the last ~30 days first (the
        only window this plan supports), then falls back to ``/latest``
        (currently active disasters only) if history is empty or rejected.
        """
        monitoring_points = self.monitoring_points.get(region, [])
        if not monitoring_points:
            logger.warning("No Ambee monitoring points configured for region=%s", region)
            return []

        now = datetime.now(timezone.utc)
        date_from = (now - timedelta(days=_HISTORY_LOOKBACK_DAYS)).strftime("%Y-%m-%d 00:00:00")
        date_to = now.strftime("%Y-%m-%d 00:00:00")

        documents: list[dict] = []
        seen_events: set[str] = set()

        for point in monitoring_points:
            lat, lng = point["lat"], point["lng"]
            point_name = point.get("name", f"{lat},{lng}")

            events = self._fetch_disasters_at_point(
                lat, lng, endpoint="history", date_from=date_from, date_to=date_to
            )
            if not events:
                events = self._fetch_disasters_at_point(lat, lng, endpoint="latest")

            logger.info("Ambee [%s] %s: %d events", region, point_name, len(events))

            for event in events:
                event_name = event.get("event_name", "Unknown")
                event_type = event.get("event_type", "Misc")
                event_date = event.get("date", "")

                dedup_key = f"{event_name}_{event_date}"
                if dedup_key in seen_events:
                    continue
                seen_events.add(dedup_key)

                severity = self._compute_severity(event)
                severity_label = self._classify_severity(severity)
                event_label = EVENT_TYPE_LABELS.get(event_type, "Disaster")
                feature_col = EVENT_TYPE_TO_AGENT_FEATURE.get(event_type, "severe_weather_index")
                prox_level = event.get("proximity_severity_level", "Unknown")
                alert_level = event.get("default_alert_levels", "Unknown")

                text = (
                    f"{event_label}: {event_name} near {point_name} in the "
                    f"{region.replace('_', ' ')} region. Date: {event_date}. "
                    f"Proximity severity: {prox_level}. Alert level: {alert_level}. "
                    f"Computed severity score: {severity:.2f}/1.0. "
                    f"Event type code: {event_type}. Maps to agent feature: {feature_col}."
                )

                countries = (
                    self.config.get("extraction", {}).get("chokepoints", {}).get(region, {}).get("countries", [])
                )

                documents.append(self._normalize_document(
                    doc_id=f"{event_type}_{event_name}_{event_date}".replace(" ", "_")[:100],
                    text=text,
                    event_date=event_date[:10] if event_date else "",
                    region=region,
                    countries=countries,
                    primary_agents=["natural_disaster"],
                    event_type=event_label,
                    severity=severity_label,
                    extra_metadata={
                        "event_name": event_name[:200],
                        "event_type_code": event_type,
                        "proximity_severity": prox_level,
                        "alert_level": alert_level,
                        "computed_severity": severity,
                        "agent_feature": feature_col,
                        "lat": event.get("lat", lat),
                        "lng": event.get("lng", lng),
                        "monitoring_point": point_name,
                    },
                ))

        return documents
