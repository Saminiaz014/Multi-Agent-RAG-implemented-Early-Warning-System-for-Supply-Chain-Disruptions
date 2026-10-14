"""Alert-blind backfill of the knowledge base for the 2023-25 test window.

Implements ``eval/explainability/PREREGISTERED_RULES.md`` section A exactly:
fixed queries, a 15-documents-per-region-month cap, and uniform selection
that never looks at alert dates. Fetching is separated from selection so the
selection rules can be tested offline, and every raw API response is cached
so a rebuild never spends quota twice.

The output goes only to an *eval copy* of the knowledge base
(``data/knowledge_base/.chromadb_xai``, collection ``backfill_2023_25``); the
production DB is read once, to make that copy, and never written.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import re
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

COLLECTION = "backfill_2023_25"
MONTHS: tuple[str, ...] = tuple(
    p.strftime("%Y-%m") for p in pd.period_range("2023-08", "2025-08", freq="M")
)
CAP_TOTAL = 15
CAP_NEWS = 10
CAP_ACLED = 5

#: A.1 — one Google News query per region.
SERP_QUERIES: dict[str, str] = {
    "hormuz": '"Strait of Hormuz" (shipping OR tanker OR attack OR military OR seized)',
    "bab_el_mandeb": '("Bab el-Mandeb" OR "Red Sea") (shipping OR Houthi OR attack OR vessel)',
    "panama": '"Panama Canal" (shipping OR transit OR drought OR vessels)',
    "malacca": '("Strait of Malacca" OR "Malacca Strait") (shipping OR piracy OR vessel OR security)',
}

#: A.2 — coastal states of each chokepoint.
ACLED_COUNTRIES: dict[str, tuple[str, ...]] = {
    "hormuz": ("Iran", "Oman", "United Arab Emirates"),
    "bab_el_mandeb": ("Yemen", "Djibouti", "Eritrea"),
    "panama": ("Panama",),
    "malacca": ("Malaysia", "Indonesia", "Singapore"),
}

MARITIME = re.compile(
    r"\b(ship|ships|shipping|vessel|vessels|tanker|tankers|boat|boats|port|ports|sea|"
    r"maritime|strait|navy|naval|coast ?guard|canal|ferry|cargo|harbou?r|offshore|seiz\w*)\b",
    re.IGNORECASE,
)

#: The region's countries as stored on production documents (``countries`` field).
REGION_COUNTRIES: dict[str, str] = {
    "hormuz": "Iran,Oman,United Arab Emirates,Saudi Arabia,Qatar,Bahrain,Kuwait,Iraq",
    "bab_el_mandeb": "Yemen,Saudi Arabia,Egypt,Eritrea,Djibouti,Sudan,Somalia",
    "panama": "Panama",
    "malacca": "Malaysia,Indonesia,Singapore",
}


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    year, mon = (int(x) for x in month.split("-"))
    return (pd.Timestamp(year, mon, 1),
            pd.Timestamp(year, mon, calendar.monthrange(year, mon)[1]))


def cdr(month: str) -> str:
    start, end = month_bounds(month)
    return f"cdr:1,cd_min:{start.month}/{start.day}/{start.year},cd_max:{end.month}/{end.day}/{end.year}"


# ------------------------------------------------------------- selection
def parse_news_date(raw: object) -> pd.Timestamp | None:
    """Parse a Google News date: 'Jan 18, 2024' or '01/18/2024, 08:00 AM, +0000 UTC'.

    Relative dates ('3 days ago') do not occur for a past custom range and are
    rejected, as are blanks.
    """
    text = str(raw or "").strip()
    if not text or "ago" in text:
        return None
    for candidate in (text, text.split(",")[0]):
        date = pd.to_datetime(candidate, errors="coerce")
        if not pd.isna(date):
            return pd.Timestamp(date).tz_localize(None).normalize()
    return None


def select_news(results: Iterable[dict], month: str, cap: int) -> list[dict]:
    """A.1 keep rule: dated inside the month, de-duplicated by URL then title, in order."""
    start, end = month_bounds(month)
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    kept = []
    for r in results:
        date = parse_news_date(r.get("date"))
        if date is None or not (start <= date <= end):
            continue
        url, title = r.get("link") or "", (r.get("title") or "").strip().lower()
        if url in seen_urls or (title and title in seen_titles):
            continue
        seen_urls.add(url)
        seen_titles.add(title)
        kept.append({**r, "_date": date})
        if len(kept) >= cap:
            break
    return kept


def maritime_events(events: Iterable[dict]) -> list[dict]:
    """A.2 maritime filter on the event notes."""
    return [e for e in events if MARITIME.search(str(e.get("notes") or ""))]


def select_evenly(events: list[dict], cap: int) -> list[dict]:
    """A.2 selection: sort by (date, id), take ``cap`` evenly spaced positions."""
    ordered = sorted(events, key=lambda e: (str(e.get("event_date")), str(e.get("event_id_cnty"))))
    if len(ordered) <= cap:
        return ordered
    positions = sorted({int(round(p)) for p in np.linspace(0, len(ordered) - 1, cap)})
    return [ordered[i] for i in positions]


def allocate(n_news: int, n_acled: int) -> tuple[int, int]:
    """Slots for news and ACLED: nominal 10 + 5, with spare slots refilled, total <= 15."""
    news = min(n_news, CAP_NEWS)
    acled = min(n_acled, CAP_ACLED)
    spare = CAP_TOTAL - news - acled
    if spare > 0:
        extra_news = min(n_news - news, spare)
        news += extra_news
        acled += min(n_acled - acled, spare - extra_news)
    return news, acled


# ------------------------------------------------------------- documents
def _doc_id(source: str, region: str, date: pd.Timestamp, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"backfill_{source}_{region}_{date:%Y-%m-%d}_{digest}"


def news_doc(r: dict, region: str) -> tuple[str, str, dict]:
    source = r.get("source") or ""
    if isinstance(source, dict):
        source = source.get("name", "")
    title = (r.get("title") or "").strip()
    text = f"{title}. {(r.get('snippet') or '').strip()} Source: {source}."
    meta = {
        "region": region, "source_api": "serpapi", "event_type": "news_article",
        "event_date": f"{r['_date']:%Y-%m-%d}", "severity": "medium",
        "primary_agents": "news_sentiment", "countries": REGION_COUNTRIES[region],
        "title": title, "url": r.get("link") or "", "source_name": str(source),
        "backfill": True,
    }
    return _doc_id("serpapi", region, r["_date"], r.get("link") or title), text, meta


def acled_doc(e: dict, region: str) -> tuple[str, str, dict]:
    date = pd.Timestamp(str(e.get("event_date")))
    fatalities = int(float(e.get("fatalities") or 0))
    severity = "low" if fatalities == 0 else ("medium" if fatalities < 10 else "high")
    meta = {
        "region": region, "source_api": "acled", "event_type": "conflict_event",
        "event_date": f"{date:%Y-%m-%d}", "severity": severity,
        "primary_agents": "geopolitical", "countries": str(e.get("country") or ""),
        "title": f"{e.get('event_type')}: {e.get('location')}, {e.get('country')}",
        "source_name": "ACLED", "backfill": True,
        "event_id_cnty": str(e.get("event_id_cnty") or ""), "fatalities": fatalities,
        "acled_event_type": str(e.get("event_type") or ""), "location": str(e.get("location") or ""),
    }
    return _doc_id("acled", region, date, meta["event_id_cnty"]), str(e.get("notes") or ""), meta


# ---------------------------------------------------------------- fetching
class SerpApiError(RuntimeError):
    """A SerpApi failure, described without the request URL (which carries the key)."""


class Fetcher:
    """API access with an on-disk cache; a cached call never touches the network."""

    def __init__(self, cache_dir: Path, serpapi_key: str | None, acled_user: str | None,
                 acled_password: str | None, pause: float = 1.0) -> None:
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.serpapi_key = serpapi_key
        self.acled_user, self.acled_password = acled_user, acled_password
        self.pause = pause
        self._acled = None
        self.calls = {"serpapi": 0, "acled": 0}

    def _cached(self, name: str, fetch) -> object:
        path = self.cache_dir / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        data = fetch()
        path.write_text(json.dumps(data, default=str), encoding="utf-8")
        time.sleep(self.pause)
        return data

    def serpapi(self, region: str, month: str) -> list[dict]:
        import requests

        def fetch():
            params = {"engine": "google", "tbm": "nws", "q": SERP_QUERIES[region], "num": 20,
                      "gl": "us", "hl": "en", "tbs": cdr(month), "api_key": self.serpapi_key}
            # Errors are re-raised without the request URL: requests puts the full
            # query string, API key included, into HTTPError / ConnectionError text.
            failure = ""
            for _attempt in range(2):  # one retry on an HTTP error or timeout (A.1, amended)
                try:
                    response = requests.get("https://serpapi.com/search.json", params=params,
                                            timeout=120)
                except (requests.Timeout, requests.ConnectionError) as exc:
                    failure = type(exc).__name__
                    continue
                self.calls["serpapi"] += 1
                if response.ok:
                    body = response.json()
                    if "error" in body and "hasn't returned any results" not in body["error"]:
                        raise SerpApiError(f"SerpApi {region} {month}: {body['error']}")
                    return body.get("news_results", [])
                failure = f"HTTP {response.status_code}"
            raise SerpApiError(f"SerpApi {region} {month}: {failure} on both attempts")

        return self._cached(f"serpapi_{region}_{month}", fetch)

    def acled(self, country: str, month: str) -> list[dict]:
        def fetch():
            if self._acled is None:
                from acled import AcledClient
                self._acled = AcledClient(username=self.acled_user, password=self.acled_password)
            start, end = month_bounds(month)
            rows: list[dict] = []
            page = 1
            while True:
                batch = [dict(r) for r in self._acled.get_data(
                    country=country, event_date=f"{start:%Y-%m-%d}|{end:%Y-%m-%d}",
                    query_params={"event_date_where": "BETWEEN"}, limit=5000, page=page)]
                self.calls["acled"] += 1
                rows.extend(batch)
                if len(batch) < 5000:
                    return rows
                page += 1

        return self._cached(f"acled_{country.replace(' ', '_')}_{month}", fetch)


def build_region_month(fetcher: Fetcher, region: str,
                       month: str) -> tuple[list[tuple[str, str, dict]], str]:
    """All selected backfill documents for one region-month (A.1-A.3).

    Returns:
        ``(docs, news_status)``. ``news_status`` is ``"ok"`` or
        ``"fetch_failed"``: per the amended A.1, a region-month whose SerpApi
        request fails on both attempts gets no news documents (ACLED still
        fills its slots), is not cached, and is therefore retried by the next
        run.
    """
    try:
        raw, status = fetcher.serpapi(region, month), "ok"
    except SerpApiError:
        raw, status = [], "fetch_failed"
    news = select_news(raw, month, cap=CAP_TOTAL)
    events = maritime_events(e for c in ACLED_COUNTRIES[region] for e in fetcher.acled(c, month))
    n_news, n_acled = allocate(len(news), len(events))
    docs = [news_doc(r, region) for r in news[:n_news]]
    docs += [acled_doc(e, region) for e in select_evenly(events, n_acled)]
    return docs, status
