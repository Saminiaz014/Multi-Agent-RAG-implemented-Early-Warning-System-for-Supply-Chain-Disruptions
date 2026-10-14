"""Precedent retrieval for explained alerts — an evaluation wrapper.

The production retriever (:class:`src.rag.context_retriever.ContextRetriever`)
is left untouched. It builds its query from which agents exceed 0.4, phrased
around the Strait of Hormuz whatever the region, and applies no region or date
filter, so a 2019-2025 alert can be "grounded" in a 2026 article about another
chokepoint. This wrapper adds the four things an explanation needs:

1. **Region filter** — live and backfill documents by their ``region`` key;
   the ten static cases through :data:`STATIC_REGION_MAP` (their ``region`` is
   free text).
2. **No temporal leakage** — only documents dated strictly before the alert
   date. Static ``YYYY-MM`` dates are read as the first of the month.
3. **De-duplication** — by a hash of the document text (the live collection
   stores the same article once per region).
4. **A query built from the explanation** — the alert's two leading agents by
   exact Shapley value and their leading terms, prefixed with the region's own
   name (:func:`build_query`).

ChromaDB ``where`` clauses cannot compare date strings, so date filtering is
done in Python after over-fetching candidates per collection.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.core.regions import get_region

#: The static cases' free-text regions -> registry key (None: not one of the
#: four monitored chokepoints, so never region-matched).
STATIC_REGION_MAP: dict[str, str | None] = {
    "Strait of Hormuz": "hormuz",
    "Strait of Hormuz / Persian Gulf": "hormuz",
    "Gulf of Oman / Oman Coast": "hormuz",
    "Red Sea / Bab al-Mandab Strait": "bab_el_mandeb",
    "Gulf of Aden / Indian Ocean": "bab_el_mandeb",
    "Suez Canal": None,
    "Northeast Japan / Pacific Coast": None,
    "Global (LA/Long Beach, Shanghai, Rotterdam)": None,
    "US West Coast Ports (LA, Long Beach, Seattle)": None,
}

STATIC_COLLECTION = "disruption_cases"

#: How each agent's leading terms are put into words for the query.
_TERM_WORDS: dict[str, str] = {
    "isolation_forest": "abnormal vessel transits, delays and congestion",
    "max_zscore": "an extreme daily shipping reading",
    "duration": "a sustained drop in vessel transits",
    "brent_crude_usd": "oil prices",
    "trade_volume_index": "trade volumes",
    "freight_rate_index": "freight rates",
    "freight_services_pct_change": "freight services",
    "sanctions": "sanctions",
    "military": "military activity",
    "diplomatic": "diplomatic incidents",
    "stability": "political instability",
    "earthquake": "an earthquake", "earthquake_severity": "an earthquake",
    "tsunami": "a tsunami", "tsunami_risk": "a tsunami",
    "cyclone": "a cyclone", "cyclone_severity": "a cyclone",
    "severe_weather": "severe weather", "severe_weather_index": "severe weather",
    "sentiment": "negative news tone",
    "velocity": "rapidly worsening news tone",
    "volume": "a surge in news volume",
    "consensus": "consistent negative reporting",
}
_AGENT_WORDS: dict[str, str] = {
    "shipping": "shipping disruption",
    "market": "market stress",
    "geopolitical": "geopolitical tension",
    "natural_disaster": "a natural hazard",
    "news_sentiment": "negative media coverage",
}


def build_query(region: str, top_drivers: Sequence[Mapping]) -> str:
    """Query text from an AER's ranked drivers: region name + two leading agents.

    Args:
        region: Region key.
        top_drivers: AER ``top_drivers`` entries (``agent``, ``phi``,
            ``top_terms``), ranked; agents with non-positive Shapley value are
            skipped unless none is positive.
    """
    positive = [d for d in top_drivers if float(d["phi"]) > 0] or list(top_drivers[:1])
    phrases = []
    for driver in positive[:2]:
        terms = [_TERM_WORDS.get(t["term"], t["term"].replace("_", " "))
                 for t in driver.get("top_terms", []) if float(t["value"]) > 0]
        phrase = _AGENT_WORDS[driver["agent"]]
        if terms:
            phrase += " (" + ", ".join(dict.fromkeys(terms)) + ")"
        phrases.append(phrase)
    return f"{get_region(region).display_name}: " + "; ".join(phrases) + "."


def text_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def doc_date(meta: Mapping) -> pd.Timestamp | None:
    """A document's date: ``event_date`` (live/backfill) or ``date`` (static)."""
    raw = meta.get("event_date") or meta.get("date")
    if not raw:
        return None
    raw = str(raw)
    if len(raw) == 7:  # static YYYY-MM -> first of the month
        raw += "-01"
    date = pd.to_datetime(raw, errors="coerce")
    return None if pd.isna(date) else pd.Timestamp(date)


def doc_region(meta: Mapping, collection: str) -> str | None:
    if collection == STATIC_COLLECTION:
        return STATIC_REGION_MAP.get(str(meta.get("region", "")))
    return meta.get("region")


@dataclass
class Hit:
    doc_id: str
    collection: str
    similarity: float
    text: str
    meta: dict

    @property
    def region(self) -> str | None:
        return doc_region(self.meta, self.collection)

    @property
    def date(self) -> pd.Timestamp | None:
        return doc_date(self.meta)

    def record(self) -> dict:
        """The AER ``precedents`` entry."""
        date = self.date
        return {
            "doc_id": self.doc_id, "collection": self.collection,
            "similarity": round(self.similarity, 4), "region": self.region,
            "event_type": self.meta.get("event_type", "historical_case"),
            "event_date": None if date is None else f"{date:%Y-%m-%d}",
            "title": self.meta.get("title") or self.meta.get("event", ""),
            "source_api": self.meta.get("source_api", "static"),
            "primary_agents": self.meta.get("primary_agents", ""),
        }


class PrecedentRetriever:
    """Region- and date-filtered, de-duplicated retrieval over chosen collections.

    Args:
        db_path: A ChromaDB directory. Use an eval copy, never the production DB.
        collections: Collection names to search.
        n_fetch: Candidates fetched per collection before filtering.
    """

    def __init__(self, db_path: Path, collections: Sequence[str], n_fetch: int = 200) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._client = chromadb.PersistentClient(path=str(db_path))
        ef = DefaultEmbeddingFunction()
        self._collections = {name: self._client.get_collection(name, embedding_function=ef)
                             for name in collections}
        self.n_fetch = n_fetch
        #: Candidates per (query, region). Consecutive alert days usually share
        #: their leading drivers and so their query text; the date filter is
        #: applied afterwards, so caching the unfiltered candidates is exact.
        self._cache: dict[tuple[str, str | None], list[Hit]] = {}

    def candidates(self, query: str, region: str | None = None) -> list[Hit]:
        """Unfiltered nearest neighbours (region-scoped where the store allows it)."""
        key = (query, region)
        if key not in self._cache:
            self._cache[key] = self._query(query, region)
        return self._cache[key]

    def _query(self, query: str, region: str | None) -> list[Hit]:
        hits: list[Hit] = []
        for name, coll in self._collections.items():
            count = coll.count()
            if count == 0:
                continue
            kwargs = {"query_texts": [query], "n_results": min(self.n_fetch, count),
                      "include": ["documents", "distances", "metadatas"]}
            if region is not None and name != STATIC_COLLECTION:
                kwargs["where"] = {"region": region}
            try:
                res = coll.query(**kwargs)
            except Exception:  # no document matches the where clause
                continue
            for i, doc, dist, meta in zip(res["ids"][0], res["documents"][0],
                                          res["distances"][0], res["metadatas"][0]):
                hits.append(Hit(i, name, 1.0 - float(dist), doc, dict(meta or {})))
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits

    def retrieve(self, region: str, alert_date: str | pd.Timestamp, query: str,
                 top_k: int = 3, min_similarity: float = 0.55) -> list[Hit]:
        """Top ``top_k`` region-matched, pre-alert, de-duplicated hits above the floor."""
        cutoff = pd.Timestamp(alert_date)
        seen: set[str] = set()
        out: list[Hit] = []
        for hit in self.candidates(query, region):
            if hit.region != region:
                continue
            date = hit.date
            if date is None or date >= cutoff:
                continue
            key = text_hash(hit.text)
            if key in seen:
                continue
            seen.add(key)
            if hit.similarity < min_similarity:
                break  # sorted by similarity: nothing below can pass
            out.append(hit)
            if len(out) == top_k:
                break
        return out
