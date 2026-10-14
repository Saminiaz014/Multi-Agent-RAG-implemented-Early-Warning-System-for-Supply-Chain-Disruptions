"""Backfill selection, the precedent wrapper's helpers, FP evidence tags, precision@3."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.explainability.kb_backfill import (
    allocate,
    cdr,
    maritime_events,
    parse_news_date,
    select_evenly,
    select_news,
)
from src.explainability.kb_evidence import characterise
from src.explainability.precedent_retriever import (
    STATIC_REGION_MAP,
    build_query,
    doc_date,
    doc_region,
)


# ------------------------------------------------------------------ backfill
def test_cdr_covers_whole_month() -> None:
    assert cdr("2024-02") == "cdr:1,cd_min:2/1/2024,cd_max:2/29/2024"


def test_parse_news_date_formats() -> None:
    assert parse_news_date("Jan 18, 2024") == pd.Timestamp("2024-01-18")
    assert parse_news_date("01/18/2024, 08:00 AM, +0000 UTC") == pd.Timestamp("2024-01-18")
    assert parse_news_date("3 days ago") is None
    assert parse_news_date("") is None


def test_select_news_keeps_in_month_and_deduplicates() -> None:
    results = [
        {"date": "Jan 5, 2024", "link": "a", "title": "A"},
        {"date": "Feb 1, 2024", "link": "b", "title": "B"},   # out of month
        {"date": "Jan 6, 2024", "link": "a", "title": "A2"},  # duplicate URL
        {"date": "Jan 7, 2024", "link": "c", "title": "a"},   # duplicate title
        {"date": "Jan 8, 2024", "link": "d", "title": "D"},
    ]
    kept = select_news(results, "2024-01", cap=15)
    assert [r["link"] for r in kept] == ["a", "d"]


def test_allocate_refills_spare_slots() -> None:
    assert allocate(10, 5) == (10, 5)
    assert allocate(3, 40) == (3, 12)
    assert allocate(40, 0) == (15, 0)
    assert allocate(0, 0) == (0, 0)


def test_select_evenly_is_uniform_and_ordered() -> None:
    events = [{"event_date": f"2024-01-{d:02d}", "event_id_cnty": str(d)} for d in range(1, 21)]
    picked = select_evenly(list(reversed(events)), 5)
    assert [e["event_date"] for e in picked] == [
        "2024-01-01", "2024-01-06", "2024-01-11", "2024-01-15", "2024-01-20"]


def test_maritime_filter() -> None:
    events = [{"notes": "Houthis fired a missile at a commercial vessel"},
              {"notes": "Farmers protested in Isfahan"},
              {"notes": "The coast guard seized a dhow"}]
    assert len(maritime_events(events)) == 2


# ------------------------------------------------------------------ wrapper
def test_static_region_map_covers_every_static_case_region() -> None:
    import json

    cases = json.loads(Path("data/knowledge_base/disruption_cases.json").read_text(encoding="utf-8"))
    assert {c["region"] for c in cases} <= set(STATIC_REGION_MAP)


def test_doc_date_and_region() -> None:
    assert doc_date({"date": "2019-06"}) == pd.Timestamp("2019-06-01")
    assert doc_date({"event_date": "2024-03-05"}) == pd.Timestamp("2024-03-05")
    assert doc_region({"region": "Strait of Hormuz"}, "disruption_cases") == "hormuz"
    assert doc_region({"region": "Suez Canal"}, "disruption_cases") is None
    assert doc_region({"region": "panama"}, "live_extracted_context") == "panama"


def test_build_query_uses_region_name_and_positive_drivers() -> None:
    drivers = [
        {"agent": "geopolitical", "phi": 0.2,
         "top_terms": [{"term": "military", "value": 0.38}, {"term": "diplomatic", "value": 0.0}]},
        {"agent": "shipping", "phi": 0.1, "top_terms": [{"term": "duration", "value": 0.9}]},
        {"agent": "market", "phi": -0.01, "top_terms": []},
    ]
    query = build_query("panama", drivers)
    assert query.startswith("Panama Canal:")
    assert "military activity" in query and "diplomatic" not in query
    assert "sustained drop in vessel transits" in query and "market" not in query
    assert "Hormuz" not in query


# ------------------------------------------------------------------ evidence
def _docs(rows):
    frame = pd.DataFrame(rows)
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame["is_news"] = frame["source_api"].isin(["newsapi", "serpapi"])
    frame["is_acled_event"] = (frame["source_api"] == "acled") & (frame["event_type"] != "conflict_summary")
    frame["is_geo"] = frame["is_acled_event"] | frame["title"].str.contains("attack")
    return frame


_EPISODE = {"region": "hormuz", "start": "2024-03-10", "end": "2024-03-12"}


def test_tag_risk_without_traffic_drop() -> None:
    docs = _docs([{"region": "hormuz", "source_api": "acled", "event_type": "conflict_event",
                   "event_date": "2024-03-04", "title": "x", "text": "x"}])
    assert characterise(_EPISODE, docs)["tag"] == "risk_without_traffic_drop"


def test_tag_unexplained_and_no_coverage() -> None:
    news = _docs([{"region": "hormuz", "source_api": "serpapi", "event_type": "news_article",
                   "event_date": "2024-03-11", "title": "port reopens", "text": "calm"}])
    assert characterise(_EPISODE, news)["tag"] == "unexplained"
    late = news.assign(event_date=pd.Timestamp("2024-03-13"))  # after the episode end
    assert characterise(_EPISODE, late)["tag"] == "no_coverage"


def test_annual_acled_summary_is_not_coverage() -> None:
    docs = _docs([{"region": "hormuz", "source_api": "acled", "event_type": "conflict_summary",
                   "event_date": "2024-01-01", "title": "x", "text": "x", "military_events": 4}])
    out = characterise({**_EPISODE, "start": "2024-01-03", "end": "2024-01-05"}, docs)
    assert out["tag"] == "no_coverage"
    assert out["acled_annual_military_events"] == 4


# ------------------------------------------------------------------ precision@3
def test_precision_at_3_known_values() -> None:
    spec = importlib.util.spec_from_file_location(
        "p3", Path("eval/explainability/compute_precision_at_3.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    annotated = pd.DataFrame({
        "episode_id": ["hormuz_E001"] * 3 + ["panama_E002"] * 2,
        "region": ["hormuz"] * 3 + ["panama"] * 2,
        "doc_id": ["a", "b", "c", "d", "e"],
        "relevance": [2, 1, 0, 0, 0],
    })
    key = pd.DataFrame({"episode_id": ["hormuz_E001"] * 3 + ["panama_E002"] * 2 + ["malacca_E003"],
                        "doc_id": ["a", "b", "c", "d", "e", None],
                        "episode_label": ["TP"] * 3 + ["FP"] * 3})
    table = module.precision_table(annotated, key)
    overall = table[(table.region == "all") & (table.episode_label == "all")].set_index("measure")
    assert overall.loc["precision@3 (lenient)", "value"] == pytest.approx((2 / 3 + 0) / 2)
    assert overall.loc["precision@3 (strict)", "value"] == pytest.approx((1 / 3 + 0) / 2)
    assert overall.loc["precision@3 (lenient)", "n_episodes_without_hits"] == 1
    with pytest.raises(ValueError):
        module.precision_table(annotated.assign(relevance=[2, np.nan, 0, 0, 0]), key)
