"""ACLED extractor for geopolitical conflict data.

Covers: geopolitical agent. Free for research use; register at
https://acleddata.com/register/. ACLED retired the legacy email+API-key
query-param scheme in favour of OAuth (username/password token exchange,
24h access token + 14-day refresh token); this extractor uses the
``acled`` PyPI client (>=1.0.0), which auto-selects OAuth and handles the
token lifecycle internally — no raw HTTP/auth code needed here, with one
exception: :meth:`ACLEDExtractor.fetch_all_events` calls the client's private
transport to follow the API's pagination cursor (``get_data`` returns one page).
"""

from __future__ import annotations

import logging
from typing import Any

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

MILITARY_EVENT_TYPES = {"Battles", "Explosions/Remote violence"}
CIVILIAN_EVENT_TYPES = {"Violence against civilians"}

#: Curated scenarios matching the static RAG cases. Region keys must exist in
#: the registry — ``red_sea`` was retired in favour of ``bab_el_mandeb``, and
#: documents tagged with the old key are unreachable by any region filter.
HISTORICAL_SCENARIOS: list[dict] = [
    {"name": "hormuz_2019_tanker", "country": "Iran", "year": 2019, "region": "hormuz"},
    {"name": "iran_sanctions_2012", "country": "Iran", "year": 2012, "region": "hormuz"},
    {"name": "houthi_red_sea_2024", "country": "Yemen", "year": 2024, "region": "bab_el_mandeb"},
    {"name": "somali_piracy_2011", "country": "Somalia", "year": 2011, "region": "bab_el_mandeb"},
]


class ACLEDFetchError(RuntimeError):
    """An ACLED request failed or came back inconsistent.

    Raised instead of returning what was fetched so far: a partial country-year
    is indistinguishable from a complete one. Messages never contain the
    credentials (:meth:`ACLEDExtractor._mask`).
    """


class ACLEDExtractor(BaseExtractor):
    """Extract geopolitical conflict event data from ACLED."""

    @property
    def source_name(self) -> str:
        return "acled"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.username = self._api_key("acled_username")
        self.password = self._api_key("acled_password")
        self._client: Any | None = None
        if not self.username or not self.password:
            logger.warning(
                "ACLED credentials not configured — set api_keys.acled_username/acled_password."
            )

    def _get_client(self):
        """Lazy-init the OAuth-authenticated ACLED client.

        ``AcledClient(username=..., password=...)`` auto-selects OAuth
        (over legacy key/email or cookie auth) and manages the access/refresh
        token lifecycle internally.
        """
        if self._client is None:
            from acled import AcledClient

            self._client = AcledClient(username=self.username, password=self.password)
            logger.info("ACLED client initialized (OAuth auth).")
        return self._client

    def _fetch_events(self, country: str, year: int, limit: int = 100, notes_filter: str = "") -> list[dict]:
        if not self.username or not self.password:
            return []
        self._rate_limit_wait()

        try:
            client = self._get_client()
            kwargs: dict[str, Any] = {"country": country, "year": year, "limit": limit}
            if notes_filter:
                kwargs["query_params"] = {"notes": notes_filter, "notes_where": "LIKE"}
            return list(client.get_data(**kwargs))
        except Exception as exc:
            logger.error("ACLED fetch failed for %s/%s: %s", country, year, exc)
            return []

    def _mask(self, text: str) -> str:
        """``text`` with the username and password replaced by ``***``."""
        for secret in (self.username, self.password):
            if secret:
                text = text.replace(secret, "***")
        return text

    def fetch_all_events(
        self, country: str, year: int, page_size: int = 5000, max_pages: int = 200,
    ) -> tuple[list[dict], int]:
        """Every event for one country-year, following the API's cursor.

        ``AcledClient.get_data`` returns a single page and drops the response's
        ``next_cursor``, so a country-year with more than ``page_size`` events
        was silently cut at the first page (Yemen 2023: 5,000 of 7,627). The
        API paginates with that cursor, passed back as ``cursor``; ``page`` is
        rejected with HTTP 400 and ``offset`` is ignored, and ``count`` /
        ``total_count`` come back null. So the last page is the one without a
        ``next_cursor``. This calls the client's own transport
        (``_get`` / ``_parse_event`` / ``endpoint`` of its data client), as
        ``get_data`` does, and names the problem if a future ``acled`` release
        no longer provides them.

        Args:
            page_size: Events per request (the API's limit parameter).
            max_pages: Refuse to follow more pages than this.

        Returns:
            ``(events, pages)``: the parsed events, in order, and the number of
            requests made.

        Raises:
            ACLEDFetchError: On missing credentials, any failed request, an API
                error response, a repeated cursor, more than ``max_pages``
                pages, or duplicate event ids across pages. Never returns a
                partial result.
        """
        where = f"{country}/{year}"

        def fail(message: str) -> ACLEDFetchError:
            return ACLEDFetchError(self._mask(f"ACLED {where}: {message}"))

        if not self.username or not self.password:
            raise fail("credentials not configured (api_keys.acled_username / acled_password)")
        try:
            data_client = self._get_client()._acled_data_client
            transport, parse, endpoint = (
                data_client._get, data_client._parse_event, data_client.endpoint)
        except AttributeError:
            raise fail("the installed acled client does not expose the data client's "
                       "transport (_get / _parse_event / endpoint); pagination needs it") from None
        except Exception as exc:
            raise fail(f"client setup failed: {type(exc).__name__}: {exc}") from None

        events: list[dict] = []
        seen: set = set()
        seen_ids: set = set()
        cursor = None
        pages = 0
        page = 1
        try:
            while True:
                page = pages + 1
                if pages >= max_pages:
                    raise fail(f"more than {max_pages} pages; refusing to continue")
                self._rate_limit_wait()
                params: dict[str, Any] = {"country": country, "year": year, "limit": page_size}
                if cursor is not None:
                    params["cursor"] = cursor
                response = transport(endpoint, params=params)
                pages += 1
                if not response.get("success"):
                    errors = response.get("error") or [{}]
                    raise fail(f"API error on page {page}: "
                               f"{(errors[0] or {}).get('message', 'unknown')}")
                batch = [parse(row) for row in (response.get("data") or [])]
                for event in batch:
                    # Checked per page, so overlapping pages stop at the first repeat.
                    event_id = event.get("event_id_cnty")
                    if event_id:
                        if event_id in seen_ids:
                            raise fail(f"duplicate event id {event_id} on page {page} "
                                       "(pages overlap)")
                        seen_ids.add(event_id)
                events.extend(batch)
                cursor = response.get("next_cursor")
                if not cursor:
                    break
                if cursor in seen:
                    raise fail(f"cursor {cursor} repeated on page {page}")
                seen.add(cursor)
        except ACLEDFetchError:
            raise
        except Exception as exc:
            raise fail(f"request for page {page} failed: "
                       f"{type(exc).__name__}: {exc}") from None
        return events, pages

    @staticmethod
    def _compute_risk_profile(events: list[dict]) -> dict:
        if not events:
            return {"total_events": 0, "military_events": 0, "civilian_events": 0,
                     "total_fatalities": 0, "risk_level": "low"}

        military = sum(1 for e in events if e.get("event_type") in MILITARY_EVENT_TYPES)
        civilian = sum(1 for e in events if e.get("event_type") in CIVILIAN_EVENT_TYPES)
        total_fatalities = sum(int(e.get("fatalities", 0) or 0) for e in events)
        total = len(events)

        if total > 100 or total_fatalities > 50:
            risk_level = "high"
        elif total > 30 or total_fatalities > 10:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "total_events": total,
            "military_events": military,
            "civilian_events": civilian,
            "total_fatalities": total_fatalities,
            "risk_level": risk_level,
        }

    def extract_historical(
        self, region: str, start_year: int | None = None,
        end_year: int | None = None, **kwargs,
    ) -> list[dict]:
        """Summarize ACLED conflict activity per country-year for ``region``.

        One document per country-year: a risk profile (event counts, military
        vs civilian split, fatalities) rather than one document per incident.
        A single country-year can hold thousands of ACLED rows, and individual
        incidents retrieve poorly — the aggregate is the supply-chain signal.

        Args:
            region: Chokepoint key.
            start_year: Overrides ``extraction.historical_range.start_year``.
            end_year: Overrides ``extraction.historical_range.end_year``.
        """
        countries = (
            self.config.get("extraction", {}).get("chokepoints", {}).get(region, {}).get("countries", [])
        )
        hist = self.config.get("extraction", {}).get("historical_range", {}) or {}
        start_year = int(hist.get("start_year", 2010) if start_year is None else start_year)
        end_year = int(hist.get("end_year", 2025) if end_year is None else end_year)

        # Stride 1 covers every year. The previous default of 3 sampled only a
        # third of the span (2018, 2021, 2024, 2026 over 2018-2026), leaving
        # gaps that read as quiet years rather than as unsampled ones.
        stride = int(
            (self.config.get("extraction", {}).get("acled", {}) or {}).get("year_stride", 1)
        )
        sample_years = list(range(start_year, end_year + 1, max(stride, 1)))
        if end_year not in sample_years:
            sample_years.append(end_year)

        documents: list[dict] = []
        for country in countries:
            for year in sample_years:
                events = self._fetch_events(country, year, limit=50)
                if not events:
                    continue
                profile = self._compute_risk_profile(events)
                logger.info(
                    "ACLED [%s] %s/%s -> %d events, risk=%s",
                    region, country, year, profile["total_events"], profile["risk_level"],
                )

                text = (
                    f"Geopolitical conflict in {country} ({year}): {profile['total_events']} conflict "
                    f"events recorded, {profile['military_events']} military engagements, "
                    f"{profile['civilian_events']} civilian targeting incidents, "
                    f"{profile['total_fatalities']} total fatalities. Risk level: {profile['risk_level']}."
                )

                documents.append(self._normalize_document(
                    doc_id=f"{region}_{country}_{year}",
                    text=text,
                    event_date=f"{year}-01-01",
                    region=region,
                    countries=[country],
                    primary_agents=["geopolitical"],
                    event_type="conflict_summary",
                    severity=profile["risk_level"],
                    extra_metadata={
                        "total_events": profile["total_events"],
                        "military_events": profile["military_events"],
                        "total_fatalities": profile["total_fatalities"],
                        "year": year,
                    },
                ))

        return documents

    def extract_specific_scenarios(self) -> list[dict]:
        """Extract data for specific historical scenarios matching the RAG cases."""
        documents: list[dict] = []
        for scenario in HISTORICAL_SCENARIOS:
            events = self._fetch_events(scenario["country"], scenario["year"], limit=100)
            profile = self._compute_risk_profile(events)
            logger.info(
                "ACLED scenario '%s' -> %d events, risk=%s",
                scenario["name"], len(events), profile["risk_level"],
            )

            text = (
                f"Historical scenario: {scenario['name'].replace('_', ' ')}. "
                f"{scenario['country']} in {scenario['year']}. {profile['total_events']} total conflict "
                f"events, {profile['military_events']} military, {profile['total_fatalities']} fatalities."
            )

            documents.append(self._normalize_document(
                doc_id=f"scenario_{scenario['name']}",
                text=text,
                event_date=f"{scenario['year']}-06-01",
                region=scenario["region"],
                countries=[scenario["country"]],
                primary_agents=["geopolitical", "shipping"],
                event_type="historical_scenario",
                severity=profile["risk_level"],
                extra_metadata={
                    "scenario_name": scenario["name"],
                    "total_events": profile["total_events"],
                    "total_fatalities": profile["total_fatalities"],
                    "historical_case": True,
                },
            ))

        return documents
