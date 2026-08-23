"""GDACS extractor — natural-hazard events for the RAG knowledge base.

Replaces :mod:`src.extractors.ambee_extractor` as the primary natural-disaster
source. Ambee held a valid key but returned zero documents for every region
with no error to debug against; GDACS is the UN/EC Global Disaster Alert and
Coordination System, needs no credentials, and publishes a documented GeoJSON
event list.

Covers earthquakes (EQ), tropical cyclones (TC), floods (FL), volcanoes (VO),
wildfires (WF) and droughts (DR).

Two API behaviours drive the design, both measured rather than assumed:

* **There is no working server-side geographic filter.** ``countrycode``,
  ``iso3`` and ``countrylist`` are all accepted and all silently ignored —
  ``countrycode=PA``, ``countrycode=IR`` and no country parameter at all
  return byte-identical global result sets. (``country=`` with a full name
  is worse: it answers HTTP 204.) Scoping therefore has to happen
  client-side, on the ``affectedcountries`` array each feature carries, which
  lists ``iso2``/``iso3``/``countryname`` per affected country.
* **Every response is capped at 100 features, silently.** A truncated window
  is indistinguishable from a complete one, which is exactly the silent-zero
  failure that made Ambee useless.

Hence: fetch the *global* event list with *adaptive* windowing — one request
for the whole span, halved only where the cap is actually hit, down to a floor
of one month — then filter it per region. The global list is fetched once and
shared across regions, so adding a region costs no extra requests. At
Orange/Red the whole planet yields a few hundred events per decade, so this is
cheaper than the per-country querying it replaces *and* correct, which the
per-country querying was not.
"""

from __future__ import annotations

import calendar
import logging
import time

import requests

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

_EVENT_LIST_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"

#: Hard server-side cap on features per response. Undocumented and silent, so
#: hitting it is treated as "this window is incomplete", not as a result.
_RESULT_CAP = 100

#: Attempts per window before it is declared unfetchable. Long windows time out
#: intermittently, and a dropped window is a silent hole in the record.
_FETCH_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 5

#: GDACS event-type codes -> readable labels used in document text/metadata.
EVENT_TYPES: dict[str, str] = {
    "EQ": "earthquake",
    "TC": "tropical cyclone",
    "FL": "flood",
    "VO": "volcanic activity",
    "WF": "wildfire",
    "DR": "drought",
}

#: GDACS alert level -> the knowledge base's three-level severity vocabulary.
#: Green is kept rather than dropped: quiet periods are evidence too, and the
#: benchmark's negative classes need them.
ALERT_TO_SEVERITY: dict[str, str] = {
    "Green": "low",
    "Orange": "medium",
    "Red": "high",
}

#: Hazards that plausibly disrupt shipping as well as being natural hazards.
#: A cyclone or flood closes ports; an inland wildfire generally does not.
_SHIPPING_RELEVANT = {"TC", "FL"}


class GDACSExtractor(BaseExtractor):
    """Extract GDACS natural-hazard events, scoped to a chokepoint region."""

    @property
    def source_name(self) -> str:
        return "gdacs"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        gdacs_cfg = config.get("extraction", {}).get("gdacs", {}) or {}
        self.alert_levels: str = str(gdacs_cfg.get("alert_levels", "Green;Orange;Red"))
        self.event_types: str = str(gdacs_cfg.get("event_types", ";".join(EVENT_TYPES)))
        #: Stop subdividing once a window is this short, even if still capped.
        #: A single month is the practical floor: below that the request cost
        #: outruns the events gained, and no country produces 100 Orange/Red
        #: alerts in one month.
        self.min_window_months: int = int(gdacs_cfg.get("min_window_months", 1))
        #: Global event list per (start_year, end_year). The API cannot filter
        #: by geography, so every region reads the same fetched list — caching
        #: it makes a four-region run cost one traversal instead of four.
        self._global_cache: dict[tuple[int, int], list[dict]] = {}

    # ----------------------------------------------------------- config
    def _chokepoint(self, region: str) -> dict:
        return (
            self.config.get("extraction", {})
            .get("chokepoints", {})
            .get(region, {})
        ) or {}

    def _iso2_codes(self, region: str) -> list[str]:
        """ISO-2 codes for ``region``, falling back to nothing if unset.

        ``extraction.chokepoints[region].iso2`` mirrors ``countries``
        positionally; GDACS keys on the code, not the name.
        """
        return [str(c).strip().upper() for c in self._chokepoint(region).get("iso2", [])]

    # -------------------------------------------------------------- api
    def _fetch(
        self, from_date: str, to_date: str, attempt: int = 1
    ) -> list[dict] | None:
        """Fetch one global time window.

        No country parameter is sent: the endpoint ignores every geographic
        filter it accepts, so sending one would only imply a scoping that is
        not happening.

        Returns:
            The feature list, or ``None`` if the request failed — distinct from
            ``[]`` (a genuine "no events"), so a failure is never mistaken for
            a quiet period.
        """
        self._rate_limit_wait()
        try:
            response = requests.get(
                _EVENT_LIST_URL,
                params={
                    "fromDate": from_date,
                    "toDate": to_date,
                    "alertlevel": self.alert_levels,
                    "eventlist": self.event_types,
                },
                timeout=90,
            )
            response.raise_for_status()
            if "json" not in response.headers.get("content-type", ""):
                # GDACS answers some queries with an HTML error page or 204.
                return []
            return response.json().get("features", []) or []
        except Exception as exc:
            if attempt < _FETCH_ATTEMPTS:
                # Long windows time out intermittently under load. Back off
                # and retry before treating the window as unfetchable.
                logger.info(
                    "GDACS %s..%s attempt %d/%d failed (%s) — retrying",
                    from_date, to_date, attempt, _FETCH_ATTEMPTS, exc,
                )
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                return self._fetch(from_date, to_date, attempt + 1)
            logger.warning("GDACS %s..%s failed (%s)", from_date, to_date, exc)
            return None

    @staticmethod
    def _window_bounds(start: int, end: int) -> tuple[str, str]:
        """Convert a month-ordinal window to inclusive ``YYYY-MM-DD`` bounds.

        Ordinals are ``year * 12 + (month - 1)``, which makes halving a window
        plain integer arithmetic regardless of where it falls in a year.
        """
        start_year, start_month = divmod(start, 12)
        end_year, end_month = divmod(end, 12)
        last_day = calendar.monthrange(end_year, end_month + 1)[1]
        return (
            f"{start_year:04d}-{start_month + 1:02d}-01",
            f"{end_year:04d}-{end_month + 1:02d}-{last_day:02d}",
        )

    def _collect(self, start: int, end: int, out: list[dict]) -> None:
        """Accumulate global features, splitting the window when the cap is hit.

        Recursive halving over month ordinals: one request for a quiet span,
        more only where the data is dense enough to truncate. The recursion
        goes below a year because it has to — a busy year exceeds 100
        Orange/Red alerts worldwide on its own, and stopping at a year would
        leave that year silently incomplete.

        Args:
            start: First month of the window, as a month ordinal.
            end: Last month of the window, inclusive.
            out: Accumulator for the raw features.
        """
        from_date, to_date = self._window_bounds(start, end)
        features = self._fetch(from_date, to_date)
        span_months = end - start + 1

        if features is None:
            # Retries are exhausted, but a long window that times out often
            # succeeds once halved. Dropping it instead would leave a silent
            # hole — a real run lost every event after 2024-10 this way.
            if span_months > self.min_window_months:
                midpoint = start + span_months // 2
                logger.warning(
                    "GDACS %s..%s unfetchable — splitting and retrying",
                    from_date, to_date,
                )
                self._collect(start, midpoint - 1, out)
                self._collect(midpoint, end, out)
            else:
                logger.error(
                    "GDACS %s..%s could not be fetched; events in this window "
                    "are MISSING from the knowledge base.", from_date, to_date,
                )
            return

        if len(features) >= _RESULT_CAP and span_months > self.min_window_months:
            midpoint = start + span_months // 2
            logger.info(
                "GDACS %s..%s capped at %d — splitting", from_date, to_date, _RESULT_CAP
            )
            self._collect(start, midpoint - 1, out)
            self._collect(midpoint, end, out)
            return

        if len(features) >= _RESULT_CAP:
            # Cannot subdivide further; say so rather than silently truncating.
            logger.warning(
                "GDACS %s..%s still capped at %d after splitting to the "
                "minimum window — this slice is incomplete.",
                from_date, to_date, _RESULT_CAP,
            )

        out.extend(features)

    def _global_events(self, start_year: int, end_year: int) -> list[dict]:
        """The global event list for a span, fetched at most once per span."""
        key = (start_year, end_year)
        if key not in self._global_cache:
            features: list[dict] = []
            self._collect(start_year * 12, end_year * 12 + 11, features)
            logger.info(
                "GDACS fetched %d global events for %d-%d",
                len(features), start_year, end_year,
            )
            self._global_cache[key] = features
        return self._global_cache[key]

    @staticmethod
    def _feature_iso2(feature: dict) -> set[str]:
        """ISO-2 codes of every country a feature reports as affected.

        ``affectedcountries`` is the authoritative list — the top-level
        ``iso3``/``country`` fields name only the primary country, so a
        cyclone crossing four countries would match on just one of them.
        """
        props = feature.get("properties", {}) or {}
        codes = {
            str(entry.get("iso2", "")).strip().upper()
            for entry in (props.get("affectedcountries") or [])
            if isinstance(entry, dict)
        }
        return codes - {""}

    def extract_historical(
        self, region: str, start_year: int | None = None,
        end_year: int | None = None, **kwargs,
    ) -> list[dict]:
        """Extract hazard events affecting ``region``'s countries.

        Args:
            region: Chokepoint key. Needs ``iso2`` under
                ``extraction.chokepoints``; ``bounding_box`` is optional and
                only sets the ``near_chokepoint`` flag.
            start_year: Overrides ``extraction.historical_range.start_year``.
            end_year: Overrides ``extraction.historical_range.end_year``.

        Returns:
            Normalized documents, one per distinct event.
        """
        codes = set(self._iso2_codes(region))
        if not codes:
            logger.warning(
                "GDACS: no iso2 codes for region=%s — filtering happens "
                "client-side on those codes, so there is nothing to match.",
                region,
            )
            return []

        chokepoint = self._chokepoint(region)
        countries = chokepoint.get("countries", []) or []
        box = chokepoint.get("bounding_box", {}) or {}
        hist = self.config.get("extraction", {}).get("historical_range", {}) or {}
        start_year = int(hist.get("start_year", 2010) if start_year is None else start_year)
        end_year = int(hist.get("end_year", 2025) if end_year is None else end_year)

        features = self._global_events(start_year, end_year)

        documents: list[dict] = []
        seen: set[str] = set()   # long hazards can repeat across split windows
        for feature in features:
            if not (codes & self._feature_iso2(feature)):
                continue
            doc = self._feature_to_doc(feature, region, countries, box, seen)
            if doc is not None:
                documents.append(doc)

        logger.info(
            "GDACS [%s] %d of %d global events matched %s (%d-%d)",
            region, len(documents), len(features), sorted(codes), start_year, end_year,
        )
        return documents

    # ------------------------------------------------- normalisation
    @staticmethod
    def _in_box(lon: float, lat: float, box: dict) -> bool:
        return (
            float(box["lat_min"]) <= lat <= float(box["lat_max"])
            and float(box["lon_min"]) <= lon <= float(box["lon_max"])
        )

    def _feature_to_doc(
        self, feature: dict, region: str, countries: list[str],
        box: dict, seen: set[str],
    ) -> dict | None:
        """Convert one GeoJSON feature to a document, or ``None`` to skip it.

        Skips only what cannot be used: an event with no id, or a repeat of one
        already emitted (the same event is returned per affected country, and
        long-running hazards such as droughts span several windows).

        Position is recorded, never used to drop an event. GDACS centroids are
        country-scale, so a strait-sized box would discard almost everything —
        trading one silent zero for another. Proximity is a
        ``near_chokepoint`` flag instead.
        """
        props = feature.get("properties", {}) or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []

        lon = lat = None
        if len(coords) >= 2:
            try:
                lon, lat = float(coords[0]), float(coords[1])
            except (TypeError, ValueError):
                lon = lat = None

        near = bool(
            box and lon is not None
            and all(k in box for k in ("lat_min", "lat_max", "lon_min", "lon_max"))
            and self._in_box(lon, lat, box)
        )

        event_id = str(props.get("eventid") or "")
        code = str(props.get("eventtype") or "").upper()
        key = f"{code}_{event_id}"
        if not event_id or key in seen:
            return None
        seen.add(key)

        label = EVENT_TYPES.get(code, code.lower() or "hazard")
        alert = str(props.get("alertlevel") or "Green").title()
        event_date = str(props.get("fromdate") or "")[:10]
        name = str(props.get("eventname") or props.get("name") or "").strip()
        country = str(props.get("country") or "").strip()
        description = str(
            props.get("htmldescription") or props.get("description") or ""
        ).strip()

        text = (
            f"GDACS {alert} alert: {label} "
            f"{('(' + name + ') ') if name else ''}"
            f"near {country or region} on {event_date or 'an unrecorded date'}. "
            f"{description}"
        ).strip()

        agents = ["natural_disaster"]
        if code in _SHIPPING_RELEVANT:
            agents.append("shipping")

        return self._normalize_document(
            # Region-qualified: a hazard can affect two chokepoints' countries
            # (Saudi Arabia sits in both hormuz and bab_el_mandeb), and the
            # builder deduplicates by id — an unqualified id would silently
            # drop the second region's copy along with its region metadata.
            doc_id=f"{region}_{code}_{event_id}",
            text=text,
            event_date=event_date,
            region=region,
            countries=countries,
            primary_agents=agents,
            event_type=label,
            severity=ALERT_TO_SEVERITY.get(alert, "low"),
            extra_metadata={
                "event_type_code": code,
                "alert_level": alert,
                "alert_score": props.get("alertscore", ""),
                "event_name": name[:200],
                "iso3": str(props.get("iso3") or ""),
                "latitude": "" if lat is None else round(lat, 4),
                "longitude": "" if lon is None else round(lon, 4),
                "near_chokepoint": near,
                "gdacs_event_id": event_id,
            },
        )
