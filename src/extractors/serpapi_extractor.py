"""SerpAPI (Google News) extractor for HISTORICAL news, date-unbounded.

Covers: all agent domains, depending on the historical case. This fills the
gap NewsAPI's free tier cannot cover — NewsAPI's ``from``/``to`` params are
capped to roughly the last 30 days on the free Developer plan, so it can
only feed the live news_sentiment agent (see ``newsapi_extractor.py``,
unchanged). SerpAPI's Google News engine accepts ``after:``/``before:``
date operators directly in the query string with no lookback limit, making
it the only practical source for backfilling the RAG knowledge base with
real news about the 10 historical disruption cases (2007-2024).

Role separation:
  * NewsAPI  -> ingestion layer, current (~30-day) news for live scoring.
  * SerpAPI  -> extraction layer only, historical news for RAG population.

Free tier: 250 searches/month. The 10 cases x 2 queries below use 20.

Response shape (verified against the live API — differs from SerpAPI's own
docs example): each ``news_results`` item has ``title``, ``link``,
``source`` (dict with ``name``), ``date`` (human-readable), and
``iso_date`` (ISO 8601, preferred for parsing) — there is generally no
``snippet`` field on ``google_news`` results. Some items group related
coverage under a nested ``stories`` list instead of a flat title/link; both
shapes are handled.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"

# Cap per-query results kept as documents — Google News can return up to
# ~100 hits per query, which would flood the live_extracted_context
# collection with near-duplicate headlines about the same event.
_MAX_RESULTS_PER_QUERY = 10

HISTORICAL_QUERIES: list[dict] = [
    {"case_id": "hormuz_2019", "region": "hormuz",
     "primary_agents": ["geopolitical", "shipping"],
     "queries": [
         {"q": '"Strait of Hormuz" tanker attack after:2019-05-01 before:2019-08-31',
          "context": "Hormuz tanker attacks June-July 2019"},
         {"q": '"Gulf of Oman" tanker explosion Iran after:2019-06-01 before:2019-07-31',
          "context": "Gulf of Oman tanker incidents 2019"},
     ]},
    {"case_id": "ever_given_2021", "region": "suez",
     "primary_agents": ["routing", "shipping"],
     "queries": [
         {"q": '"Ever Given" "Suez Canal" blocked after:2021-03-20 before:2021-04-15',
          "context": "Ever Given Suez Canal blockage March 2021"},
         {"q": '"Suez Canal" shipping disruption reroute after:2021-03-23 before:2021-04-10',
          "context": "Suez Canal shipping disruption and rerouting 2021"},
     ]},
    {"case_id": "houthi_red_sea_2024", "region": "red_sea",
     "primary_agents": ["geopolitical", "routing", "news_sentiment"],
     "queries": [
         {"q": 'Houthi "Red Sea" shipping attack after:2023-11-01 before:2024-03-31',
          "context": "Houthi Red Sea shipping attacks 2023-2024"},
         {"q": '"Bab el-Mandeb" Houthi ship reroute after:2024-01-01 before:2024-06-30',
          "context": "Bab el-Mandeb Houthi disruption and rerouting 2024"},
     ]},
    {"case_id": "hormuz_mine_2010", "region": "hormuz",
     "primary_agents": ["geopolitical", "shipping"],
     "queries": [
         {"q": '"Strait of Hormuz" mine threat Iran naval after:2010-01-01 before:2010-12-31',
          "context": "Hormuz mine threat 2010"},
         {"q": 'Iran "Strait of Hormuz" closure threat oil after:2010-01-01 before:2011-06-30',
          "context": "Iran Hormuz closure threat 2010-2011"},
     ]},
    {"case_id": "somali_piracy_2011", "region": "red_sea",
     "primary_agents": ["routing", "shipping"],
     "queries": [
         {"q": 'Somalia piracy shipping "Gulf of Aden" after:2011-01-01 before:2011-12-31',
          "context": "Somali piracy peak 2011"},
         {"q": 'pirate attack vessel hijack Somalia after:2011-01-01 before:2011-12-31',
          "context": "Somali pirate vessel hijackings 2011"},
     ]},
    {"case_id": "japan_earthquake_2011", "region": "malacca",
     "primary_agents": ["natural_disaster", "shipping"],
     "queries": [
         {"q": 'Japan earthquake tsunami port closure shipping after:2011-03-01 before:2011-05-31',
          "context": "Japan earthquake port closures March 2011"},
         {"q": 'Fukushima earthquake supply chain disruption port after:2011-03-10 before:2011-04-30',
          "context": "Fukushima supply chain disruption 2011"},
     ]},
    {"case_id": "covid_port_congestion_2021", "region": "hormuz",
     "primary_agents": ["shipping", "market"],
     "queries": [
         {"q": 'COVID port congestion shipping delay container after:2021-01-01 before:2021-12-31',
          "context": "COVID port congestion 2021"},
         {"q": 'supply chain crisis port backlog container ship after:2021-06-01 before:2021-12-31',
          "context": "Supply chain crisis port backlogs 2021"},
     ]},
    {"case_id": "us_port_strikes_2014", "region": "hormuz",
     "primary_agents": ["shipping", "market"],
     "queries": [
         {"q": '"West Coast" port strike shipping delay after:2014-10-01 before:2015-03-31',
          "context": "US West Coast port strikes 2014-2015"},
         {"q": 'ILWU port labor dispute container after:2014-11-01 before:2015-02-28',
          "context": "ILWU labor dispute 2014-2015"},
     ]},
    {"case_id": "iran_sanctions_2012", "region": "hormuz",
     "primary_agents": ["geopolitical", "market"],
     "queries": [
         {"q": 'Iran sanctions oil embargo tanker after:2012-01-01 before:2012-12-31',
          "context": "Iran sanctions oil embargo 2012"},
         {"q": 'Iran oil export sanctions "Strait of Hormuz" after:2012-06-01 before:2013-03-31',
          "context": "Iran oil sanctions impact on Hormuz 2012"},
     ]},
    {"case_id": "cyclone_gonu_2007", "region": "hormuz",
     "primary_agents": ["natural_disaster", "routing"],
     "queries": [
         {"q": '"Cyclone Gonu" Oman port shipping after:2007-06-01 before:2007-07-31',
          "context": "Cyclone Gonu Oman June 2007"},
         {"q": '"Cyclone Gonu" oil disruption "Persian Gulf" after:2007-05-15 before:2007-07-15',
          "context": "Cyclone Gonu Persian Gulf oil disruption 2007"},
     ]},
]


class SerpAPIExtractor(BaseExtractor):
    """Extract historical Google News results via SerpAPI for RAG backfill."""

    @property
    def source_name(self) -> str:
        return "serpapi"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.api_key = self._api_key("serpapi")
        if not self.api_key:
            logger.warning("SerpAPI key not configured — set api_keys.serpapi / SERPAPI_API_KEY.")

    @staticmethod
    def _flatten_results(news_results: list[dict]) -> list[dict]:
        """Flatten grouped ``stories`` items into individual result dicts."""
        flat: list[dict] = []
        for item in news_results:
            stories = item.get("stories")
            if stories:
                flat.extend(stories)
            elif item.get("link"):
                flat.append(item)
        return flat

    def _search_google_news(self, query: str, gl: str = "us", hl: str = "en") -> list[dict]:
        """Run a single Google News search via SerpAPI.

        Returns:
            List of ``{title, link, source, date}`` dicts (flattened — see
            :meth:`_flatten_results`), capped at ``_MAX_RESULTS_PER_QUERY``.
        """
        if not self.api_key:
            return []
        self._rate_limit_wait()

        params = {"engine": "google_news", "q": query, "api_key": self.api_key, "gl": gl, "hl": hl}
        try:
            response = requests.get(SERPAPI_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("SerpAPI request failed: %s", exc)
            return []

        results = self._flatten_results(payload.get("news_results", []))
        return results[:_MAX_RESULTS_PER_QUERY]

    def _result_to_doc(self, result: dict, case: dict, query_context: str) -> dict | None:
        title = result.get("title", "") or ""
        link = result.get("link", "") or ""
        source_name = (result.get("source") or {}).get("name", "")
        event_date = result.get("iso_date", "") or result.get("date", "")
        snippet = result.get("snippet", "") or ""

        if not title:
            return None

        text = f"Historical event: {query_context}. {title}."
        if snippet:
            text += f" {snippet}."
        text += f" Source: {source_name}. Date: {event_date}."

        countries = (
            self.config.get("extraction", {}).get("chokepoints", {}).get(case["region"], {}).get("countries", [])
        )

        return self._normalize_document(
            doc_id=f"{case['case_id']}_{hash(link) % 1_000_000}",
            text=text,
            event_date=str(event_date)[:10] if event_date else "",
            region=case["region"],
            countries=countries,
            primary_agents=case["primary_agents"],
            event_type="historical_news",
            severity="high",
            extra_metadata={
                "case_id": case["case_id"],
                "context": query_context,
                "title": title[:200],
                "source_name": source_name,
                "url": link,
            },
        )

    def extract_historical(self, region: str, **kwargs) -> list[dict]:
        """Extract historical news for the cases matching ``region``."""
        documents: list[dict] = []
        for case in HISTORICAL_QUERIES:
            if case["region"] != region:
                continue
            for q in case["queries"]:
                results = self._search_google_news(q["q"])
                logger.info(
                    "SerpAPI [%s] '%s' -> %d results", case["case_id"], q["context"], len(results)
                )
                for result in results:
                    doc = self._result_to_doc(result, case, q["context"])
                    if doc:
                        documents.append(doc)
        return documents

    def extract_all_cases(self) -> list[dict]:
        """Run all 10 historical cases regardless of region (one-time backfill)."""
        documents: list[dict] = []
        for case in HISTORICAL_QUERIES:
            for q in case["queries"]:
                results = self._search_google_news(q["q"])
                logger.info(
                    "SerpAPI [%s] '%s' -> %d results", case["case_id"], q["context"], len(results)
                )
                for result in results:
                    doc = self._result_to_doc(result, case, q["context"])
                    if doc:
                        documents.append(doc)
        return documents
