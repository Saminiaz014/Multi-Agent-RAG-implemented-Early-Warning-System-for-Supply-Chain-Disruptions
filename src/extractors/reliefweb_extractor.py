"""ReliefWeb API extractor for natural disaster data.

Covers: natural_disaster agent. Free, no API key needed — just an
``appname`` query parameter. Historical data available since 1981.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

BASE_URL = "https://api.reliefweb.int/v2"
APP_NAME = "supply-chain-dss"

MARITIME_DISASTER_TYPES = [
    "Tropical Cyclone",
    "Tsunami",
    "Earthquake",
    "Flood",
    "Storm Surge",
]

HISTORICAL_DISASTERS: list[dict] = [
    {"query": "Cyclone Gonu", "country": "Oman", "region": "hormuz",
     "agents": ["natural_disaster", "routing"]},
    {"query": "earthquake tsunami", "country": "Japan", "region": "malacca",
     "agents": ["natural_disaster", "shipping"]},
    {"query": "tsunami earthquake", "country": "Indonesia", "region": "malacca",
     "agents": ["natural_disaster", "routing"]},
    {"query": "flood storm", "country": "Egypt", "region": "suez",
     "agents": ["natural_disaster"]},
]

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str, limit: int) -> str:
    return _TAG_RE.sub("", html or "")[:limit]


class ReliefWebExtractor(BaseExtractor):
    """Extract natural disaster data from UN OCHA ReliefWeb API."""

    @property
    def source_name(self) -> str:
        return "reliefweb"

    def _search_disasters(
        self, query: str = "", country: str = "", disaster_type: str = "", limit: int = 20,
    ) -> list[dict]:
        self._rate_limit_wait()
        url = f"{BASE_URL}/disasters?appname={APP_NAME}"
        payload: dict[str, Any] = {
            "limit": limit,
            "fields": {"include": ["name", "date", "type", "country", "status", "glide", "description-html"]},
            "sort": ["date.created:desc"],
        }
        if query:
            payload["query"] = {"value": query}

        filters = []
        if disaster_type:
            filters.append({"field": "type.name", "value": disaster_type})
        if country:
            filters.append({"field": "country.name", "value": country})
        if len(filters) == 1:
            payload["filter"] = filters[0]
        elif len(filters) > 1:
            payload["filter"] = {"operator": "AND", "conditions": filters}

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("ReliefWeb disasters request failed: %s", exc)
            return []

        results = []
        for item in data.get("data", []):
            fields = item.get("fields", {})
            results.append({
                "id": item.get("id"),
                "name": fields.get("name", ""),
                "date_created": fields.get("date", {}).get("created", ""),
                "types": [t.get("name", "") for t in fields.get("type", [])],
                "countries": [c.get("name", "") for c in fields.get("country", [])],
                "status": fields.get("status", ""),
                "glide": fields.get("glide", ""),
                "description": _strip_html(fields.get("description-html", ""), 1500),
            })
        return results

    def _get_reports(self, query: str = "", country: str = "", limit: int = 10) -> list[dict]:
        self._rate_limit_wait()
        url = f"{BASE_URL}/reports?appname={APP_NAME}"
        payload: dict[str, Any] = {
            "limit": limit,
            "fields": {"include": ["title", "date", "source", "country", "disaster", "body", "url"]},
            "sort": ["date.created:desc"],
        }
        if query:
            payload["query"] = {"value": query}
        if country:
            payload["filter"] = {"field": "country.name", "value": country}

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("ReliefWeb reports request failed: %s", exc)
            return []

        results = []
        for item in data.get("data", []):
            fields = item.get("fields", {})
            results.append({
                "id": item.get("id"),
                "title": fields.get("title", ""),
                "date": fields.get("date", {}).get("created", ""),
                "body_clean": _strip_html(fields.get("body", ""), 2000),
                "url": fields.get("url", ""),
                "countries": [c.get("name", "") for c in fields.get("country", [])],
                "disasters": [d.get("name", "") for d in fields.get("disaster", [])],
            })
        return results

    def extract_historical(self, region: str, **kwargs) -> list[dict]:
        countries = (
            self.config.get("extraction", {}).get("chokepoints", {}).get(region, {}).get("countries", [])
        )
        documents: list[dict] = []

        for country in countries:
            for dtype in MARITIME_DISASTER_TYPES:
                disasters = self._search_disasters(country=country, disaster_type=dtype, limit=10)
                logger.info("ReliefWeb [%s] %s/%s -> %d disasters", region, country, dtype, len(disasters))
                for d in disasters:
                    text = (
                        f"{d['name']}. Types: {', '.join(d['types'])}. "
                        f"Countries: {', '.join(d['countries'])}. Status: {d['status']}. {d['description']}"
                    )
                    severity = "high" if any(t in ("Tsunami", "Tropical Cyclone") for t in d["types"]) else "medium"
                    documents.append(self._normalize_document(
                        doc_id=f"disaster_{d['id']}",
                        text=text,
                        event_date=d["date_created"][:10] if d["date_created"] else "",
                        region=region,
                        countries=d["countries"],
                        primary_agents=["natural_disaster"],
                        event_type=", ".join(d["types"]),
                        severity=severity,
                        extra_metadata={"disaster_name": d["name"], "glide": d["glide"], "status": d["status"]},
                    ))

            reports = self._get_reports(query="port maritime shipping coastal cyclone tsunami", country=country, limit=5)
            logger.info("ReliefWeb [%s] %s reports -> %d", region, country, len(reports))
            for r in reports:
                text = f"{r['title']}. {r['body_clean']}"
                if not text.strip(". "):
                    continue
                documents.append(self._normalize_document(
                    doc_id=f"report_{r['id']}",
                    text=text,
                    event_date=r["date"][:10] if r["date"] else "",
                    region=region,
                    countries=r["countries"],
                    primary_agents=["natural_disaster", "shipping"],
                    event_type="situation_report",
                    severity="medium",
                    extra_metadata={"title": r["title"][:200], "url": r["url"], "disasters": ", ".join(r["disasters"][:3])},
                ))

        return documents

    def extract_specific_events(self) -> list[dict]:
        """Extract data for specific historical events matching RAG cases."""
        documents: list[dict] = []
        for event in HISTORICAL_DISASTERS:
            disasters = self._search_disasters(query=event["query"], country=event["country"], limit=5)
            logger.info(
                "ReliefWeb historical '%s' / %s -> %d disasters", event["query"], event["country"], len(disasters)
            )
            for d in disasters:
                text = (
                    f"{d['name']}. Types: {', '.join(d['types'])}. "
                    f"Countries: {', '.join(d['countries'])}. {d['description']}"
                )
                documents.append(self._normalize_document(
                    doc_id=f"hist_disaster_{d['id']}",
                    text=text,
                    event_date=d["date_created"][:10] if d["date_created"] else "",
                    region=event["region"],
                    countries=d["countries"],
                    primary_agents=event["agents"],
                    event_type=", ".join(d["types"]),
                    severity="high",
                    extra_metadata={"disaster_name": d["name"], "historical_case": True},
                ))
        return documents
