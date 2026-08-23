"""USGS earthquake extractor — seismic events for the RAG knowledge base.

Complements :mod:`src.extractors.gdacs_extractor` rather than duplicating it.
GDACS reports an earthquake only once it clears a humanitarian alert bar, so a
moderate quake near a chokepoint — enough to close a port or trigger a tsunami
advisory — never appears there. USGS publishes every event above a magnitude
threshold, with the magnitude, depth and tsunami flag GDACS omits.

Overlap is expected and handled: both sources emit their own document ids and
``KnowledgeBaseBuilder`` deduplicates by id, so one quake can legitimately
appear once from each source carrying different detail.

Unlike GDACS this endpoint filters by bounding box *and* date server-side, and
reports its own truncation, so a single request per region usually suffices.
No credentials required.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

#: Magnitude thresholds for the three-level severity vocabulary. Chosen for
#: infrastructure effect rather than geophysics: below 5.5 a quake rarely
#: disrupts port operations, 6.5+ routinely does.
_HIGH_MAGNITUDE = 6.5
_MEDIUM_MAGNITUDE = 5.5


class USGSExtractor(BaseExtractor):
    """Extract USGS earthquakes inside a chokepoint's bounding box."""

    @property
    def source_name(self) -> str:
        return "usgs"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        usgs_cfg = config.get("extraction", {}).get("usgs", {}) or {}
        #: Below this, seismicity is noise for supply-chain purposes.
        self.min_magnitude: float = float(usgs_cfg.get("min_magnitude", 4.0))
        #: FDSN caps a result set; the span is split when the cap is reached.
        self.max_per_request: int = int(usgs_cfg.get("max_per_request", 2000))

    def _chokepoint(self, region: str) -> dict:
        return (
            self.config.get("extraction", {})
            .get("chokepoints", {})
            .get(region, {})
        ) or {}

    @staticmethod
    def _severity(magnitude: float) -> str:
        if magnitude >= _HIGH_MAGNITUDE:
            return "high"
        if magnitude >= _MEDIUM_MAGNITUDE:
            return "medium"
        return "low"

    def _fetch(self, box: dict, start: str, end: str) -> list[dict] | None:
        """Fetch one window, or ``None`` on failure (distinct from no events)."""
        self._rate_limit_wait()
        try:
            response = requests.get(
                _QUERY_URL,
                params={
                    "format": "geojson",
                    "starttime": start,
                    "endtime": end,
                    "minmagnitude": self.min_magnitude,
                    "minlatitude": box["lat_min"],
                    "maxlatitude": box["lat_max"],
                    "minlongitude": box["lon_min"],
                    "maxlongitude": box["lon_max"],
                    "limit": self.max_per_request,
                    "orderby": "time",
                },
                timeout=90,
            )
            response.raise_for_status()
            return response.json().get("features", []) or []
        except Exception as exc:
            logger.warning("USGS window %s..%s failed (%s)", start, end, exc)
            return None

    def extract_historical(
        self, region: str, start_year: int | None = None,
        end_year: int | None = None, **kwargs,
    ) -> list[dict]:
        """Extract earthquakes inside ``region``'s bounding box.

        Args:
            region: Chokepoint key with a ``bounding_box`` in config. The box
                is read from config rather than hardcoded, so it stays in step
                with the region overlays that project it.
            start_year: Overrides ``extraction.historical_range.start_year``.
            end_year: Overrides ``extraction.historical_range.end_year``.

        Returns:
            Normalized documents, one per earthquake.
        """
        chokepoint = self._chokepoint(region)
        box = chokepoint.get("bounding_box", {}) or {}
        if not all(k in box for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
            logger.warning(
                "USGS: no bounding box for region=%s — extracting nothing.", region
            )
            return []

        countries = chokepoint.get("countries", []) or []
        hist = self.config.get("extraction", {}).get("historical_range", {}) or {}
        start_year = int(hist.get("start_year", 2010) if start_year is None else start_year)
        end_year = int(hist.get("end_year", 2025) if end_year is None else end_year)

        features = self._fetch(box, f"{start_year}-01-01", f"{end_year}-12-31")
        if features is None:
            return []
        if len(features) >= self.max_per_request:
            logger.warning(
                "USGS [%s] returned the %d-result limit — raise "
                "extraction.usgs.max_per_request or narrow the span.",
                region, self.max_per_request,
            )

        documents: list[dict] = []
        for feature in features:
            doc = self._feature_to_doc(feature, region, countries)
            if doc is not None:
                documents.append(doc)

        logger.info(
            "USGS [%s] -> %d quakes M%.1f+ (%d-%d)",
            region, len(documents), self.min_magnitude, start_year, end_year,
        )
        return documents

    def _feature_to_doc(
        self, feature: dict, region: str, countries: list[str]
    ) -> dict | None:
        """Convert one USGS feature to a document, or ``None`` to skip it."""
        props = feature.get("properties", {}) or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        magnitude = props.get("mag")
        if magnitude is None or len(coords) < 2:
            return None

        try:
            magnitude = float(magnitude)
            lon, lat = float(coords[0]), float(coords[1])
            depth_km = float(coords[2]) if len(coords) > 2 else None
        except (TypeError, ValueError):
            return None

        # The event id lives on the feature, not in properties.
        event_id = str(feature.get("id") or props.get("code") or "")
        if not event_id:
            return None

        # USGS timestamps are epoch milliseconds, UTC.
        try:
            event_date = datetime.fromtimestamp(
                float(props.get("time")) / 1000.0, tz=timezone.utc
            ).strftime("%Y-%m-%d")
        except (TypeError, ValueError, OSError):
            event_date = ""

        place = str(props.get("place") or "").strip()
        tsunami_flag = bool(props.get("tsunami"))
        depth_note = "" if depth_km is None else f" at {depth_km:.0f} km depth"

        text = (
            f"USGS recorded a magnitude {magnitude:.1f} earthquake "
            f"{place or f'near the {region} chokepoint'}{depth_note} on "
            f"{event_date or 'an unrecorded date'}."
            + (" A tsunami alert was issued for this event." if tsunami_flag else "")
        )

        agents = ["natural_disaster"]
        if tsunami_flag or magnitude >= _HIGH_MAGNITUDE:
            # A tsunami advisory or a major quake closes ports — a shipping
            # signal as much as a hazard one.
            agents.append("shipping")

        return self._normalize_document(
            doc_id=event_id,
            text=text,
            event_date=event_date,
            region=region,
            countries=countries,
            primary_agents=agents,
            event_type="earthquake",
            severity=self._severity(magnitude),
            extra_metadata={
                "magnitude": round(magnitude, 2),
                "depth_km": "" if depth_km is None else round(depth_km, 1),
                "tsunami_alert": tsunami_flag,
                "place": place[:200],
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "usgs_event_id": event_id,
            },
        )
