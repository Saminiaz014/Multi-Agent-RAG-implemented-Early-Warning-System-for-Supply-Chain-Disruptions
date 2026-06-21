"""ACLED extractor for geopolitical conflict data.

Covers: geopolitical agent. Free for research use; register at
https://acleddata.com/register/. ACLED retired the legacy email+API-key
query-param scheme in favour of OAuth (username/password token exchange,
24h access token + 14-day refresh token); this extractor uses the
``acled`` PyPI client (>=1.0.0), which auto-selects OAuth and handles the
token lifecycle internally — no raw HTTP/auth code needed here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

MILITARY_EVENT_TYPES = {"Battles", "Explosions/Remote violence"}
CIVILIAN_EVENT_TYPES = {"Violence against civilians"}

HISTORICAL_SCENARIOS: list[dict] = [
    {"name": "hormuz_2019_tanker", "country": "Iran", "year": 2019, "region": "hormuz"},
    {"name": "iran_sanctions_2012", "country": "Iran", "year": 2012, "region": "hormuz"},
    {"name": "houthi_red_sea_2024", "country": "Yemen", "year": 2024, "region": "red_sea"},
    {"name": "somali_piracy_2011", "country": "Somalia", "year": 2011, "region": "red_sea"},
]


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

    def extract_historical(self, region: str, **kwargs) -> list[dict]:
        countries = (
            self.config.get("extraction", {}).get("chokepoints", {}).get(region, {}).get("countries", [])
        )
        start_year = self.config.get("extraction", {}).get("historical_range", {}).get("start_year", 2010)
        end_year = self.config.get("extraction", {}).get("historical_range", {}).get("end_year", 2025)
        sample_years = list(range(start_year, end_year + 1, 3))
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
                    doc_id=f"{country}_{year}",
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
