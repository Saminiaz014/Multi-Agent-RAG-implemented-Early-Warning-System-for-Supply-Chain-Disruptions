"""NewsAPI extractor for news sentiment data.

Covers: news_sentiment agent. Free tier: 100 requests/day, 1-month lookback.
"""

from __future__ import annotations

import logging

import requests

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

CHOKEPOINT_QUERIES: dict[str, list[str]] = {
    "hormuz": [
        '"Strait of Hormuz" AND (shipping OR tanker OR blockade OR attack)',
        "Iran AND (sanctions OR oil OR embargo OR military)",
    ],
    "red_sea": [
        '"Red Sea" AND (Houthi OR shipping OR attack OR reroute)',
        '"Bab el-Mandeb" AND (shipping OR disruption)',
    ],
    "malacca": [
        '"Strait of Malacca" AND (piracy OR shipping OR disruption)',
        '"Malacca Strait" AND (security OR vessel)',
    ],
    "suez": [
        '"Suez Canal" AND (blockage OR disruption OR delay OR grounding)',
        '"Suez Canal" AND (shipping OR container OR reroute)',
    ],
}

HISTORICAL_EVENT_QUERIES: list[dict] = [
    {"query": "Ever Given Suez Canal blocked", "date_from": "2021-03-20", "date_to": "2021-04-15",
     "region": "suez", "agents": ["routing", "shipping"]},
    {"query": "Houthi Red Sea shipping attack", "date_from": "2023-11-01", "date_to": "2024-03-31",
     "region": "red_sea", "agents": ["geopolitical", "routing", "news_sentiment"]},
    {"query": "Iran sanctions oil embargo tanker", "date_from": "2024-01-01", "date_to": "2024-12-31",
     "region": "hormuz", "agents": ["geopolitical", "market"]},
]


class NewsAPIExtractor(BaseExtractor):
    """Extract news articles from NewsAPI.org for sentiment analysis."""

    @property
    def source_name(self) -> str:
        return "newsapi"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.api_key = self._api_key("newsapi")
        self.base_url = "https://newsapi.org/v2/everything"
        if not self.api_key:
            logger.warning("NewsAPI key not configured — set api_keys.newsapi / NEWSAPI_KEY.")

    def _search_articles(
        self,
        query: str,
        date_from: str = "",
        date_to: str = "",
        page_size: int = 50,
        language: str = "en",
        sort_by: str = "relevancy",
    ) -> list[dict]:
        if not self.api_key:
            return []

        self._rate_limit_wait()
        params = {
            "q": query,
            "apiKey": self.api_key,
            "pageSize": min(page_size, 100),
            "language": language,
            "sortBy": sort_by,
        }
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "ok":
                logger.error("NewsAPI error: %s", data.get("message", "unknown"))
                return []
            return data.get("articles", [])
        except requests.RequestException as exc:
            logger.error("NewsAPI request failed: %s", exc)
            return []

    def _region_countries(self, region: str) -> list[str]:
        return (
            self.config.get("extraction", {})
            .get("chokepoints", {})
            .get(region, {})
            .get("countries", [])
        )

    def _article_to_doc(
        self, article: dict, doc_id: str, region: str, primary_agents: list[str],
        event_type: str, severity: str,
    ) -> dict | None:
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""
        published = (article.get("publishedAt", "") or "")[:10]
        url = article.get("url", "")

        text = f"{title}. {description}. {content[:500]}"
        if not text.strip(". "):
            return None

        return self._normalize_document(
            doc_id=doc_id,
            text=text,
            event_date=published,
            region=region,
            countries=self._region_countries(region),
            primary_agents=primary_agents,
            event_type=event_type,
            severity=severity,
            extra_metadata={
                "source_name": article.get("source", {}).get("name", ""),
                "url": url,
                "title": title[:200],
            },
        )

    def extract_historical(self, region: str, **kwargs) -> list[dict]:
        documents: list[dict] = []
        for query in CHOKEPOINT_QUERIES.get(region, []):
            articles = self._search_articles(query, page_size=50)
            logger.info("NewsAPI [%s] '%s...' -> %d articles", region, query[:40], len(articles))
            for article in articles:
                url = article.get("url", "")
                doc = self._article_to_doc(
                    article,
                    doc_id=f"{region}_{article.get('publishedAt', '')[:10]}_{hash(url) % 100000}",
                    region=region,
                    primary_agents=["news_sentiment"],
                    event_type="news_article",
                    severity="medium",
                )
                if doc:
                    documents.append(doc)
        return documents

    def extract_historical_events(self) -> list[dict]:
        """Extract articles for specific known historical disruption events."""
        documents: list[dict] = []
        for event in HISTORICAL_EVENT_QUERIES:
            articles = self._search_articles(
                query=event["query"],
                date_from=event.get("date_from", ""),
                date_to=event.get("date_to", ""),
                page_size=20,
            )
            logger.info(
                "NewsAPI historical '%s...' -> %d articles", event["query"][:40], len(articles)
            )
            for article in articles:
                url = article.get("url", "")
                doc = self._article_to_doc(
                    article,
                    doc_id=f"hist_{event['region']}_{article.get('publishedAt', '')[:10]}_{hash(url) % 100000}",
                    region=event["region"],
                    primary_agents=event["agents"],
                    event_type="historical_news",
                    severity="high",
                )
                if doc:
                    documents.append(doc)
        return documents
