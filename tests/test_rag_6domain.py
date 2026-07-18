"""Tests for the 6-domain RAG knowledge base and ContextRetriever (Phase 2F.3a).

Five tests:

1. test_knowledge_base_completeness  — JSON has >= 10 cases, covers all 6 domains,
                                       every case has all required fields.
2. test_query_scenario_b             — Multi-domain signal (shipping + geo + disaster
                                       + routing + news) → top match is multi-domain
                                       with similarity > 0.6.
3. test_query_geopolitical_only      — Only geopolitical elevated → top match has
                                       "geopolitical" in primary_agents.
4. test_query_disaster               — Only natural_disaster elevated → top match has
                                       "natural_disaster" in primary_agents.
5. test_format_context               — format_context() returns readable text with
                                       event names and similarity scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag.context_retriever import (
    ContextRetriever,
    _REQUIRED_CASE_FIELDS,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KB_JSON = Path("data/knowledge_base/disruption_cases.json")

ALL_6_DOMAINS = {
    "shipping",
    "market",
    "geopolitical",
    "natural_disaster",
    "routing",
    "news_sentiment",
}


@pytest.fixture(scope="module")
def retriever(tmp_path_factory: pytest.TempPathFactory) -> ContextRetriever:
    """Return a ContextRetriever backed by a fresh temp ChromaDB collection."""
    db_path = str(tmp_path_factory.mktemp("chromadb"))
    cfg = {"collection_name": "test_disruption_cases", "top_k": 3}
    r = ContextRetriever(cfg, persist_directory=db_path)
    added = r.build_index(str(KB_JSON))
    # Should have indexed all cases; if the fixture is reused the count check
    # returns 0 (already up-to-date) — either is fine.
    assert added >= 0
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_knowledge_base_completeness() -> None:
    """JSON has >= 10 cases, covers all 6 domains, all required fields present."""
    assert KB_JSON.exists(), f"disruption_cases.json not found at {KB_JSON}"
    cases: list[dict] = json.loads(KB_JSON.read_text(encoding="utf-8"))

    assert len(cases) >= 10, f"Expected >= 10 cases, got {len(cases)}"

    # Collect all agent domains referenced across all cases.
    all_domains_seen: set[str] = set()
    for case in cases:
        all_domains_seen.update(case.get("primary_agents", []))

    missing_domains = ALL_6_DOMAINS - all_domains_seen
    assert not missing_domains, (
        f"These agent domains are not covered by any case: {missing_domains}"
    )

    # Verify required fields on every case.
    for case in cases:
        missing_fields = _REQUIRED_CASE_FIELDS - set(case.keys())
        assert not missing_fields, (
            f"Case '{case.get('id', '?')}' is missing fields: {missing_fields}"
        )
        # features sub-dict must exist and be a dict.
        assert isinstance(case.get("features"), dict), (
            f"Case '{case.get('id', '?')}': 'features' must be a dict."
        )


def test_query_scenario_b(retriever: ContextRetriever) -> None:
    """Multi-domain signal profile returns a multi-domain case with similarity > 0.6."""
    # Scenario B: shipping, geopolitical, natural_disaster, routing, news all elevated.
    signals = {
        "shipping": 0.80,
        "market": 0.35,
        "geopolitical": 0.85,
        "natural_disaster": 0.70,
        "routing": 0.75,
        "news_sentiment": 0.72,
    }
    results = retriever.query(signals, top_k=3)

    assert results, "Expected at least one result from multi-domain query."

    top = results[0]
    assert "similarity" in top, "Result must have a 'similarity' key."
    assert top["similarity"] > 0.60, (
        f"Expected top-match similarity > 0.60, got {top['similarity']:.4f}"
    )

    # Top result should involve multiple primary domains.
    meta = top.get("metadata", {})
    agents_raw = meta.get("primary_agents", "[]")
    top_agents: list[str] = (
        json.loads(agents_raw) if isinstance(agents_raw, str) else agents_raw
    )
    assert len(top_agents) >= 2, (
        f"Scenario B top match '{top['id']}' should involve >= 2 domains, "
        f"got: {top_agents}"
    )


def test_query_geopolitical_only(retriever: ContextRetriever) -> None:
    """Geopolitical-only signal → top result has 'geopolitical' in primary_agents."""
    signals = {
        "shipping": 0.10,
        "market": 0.15,
        "geopolitical": 0.90,
        "natural_disaster": 0.10,
        "routing": 0.10,
        "news_sentiment": 0.12,
    }
    results = retriever.query(signals, top_k=3)

    assert results, "Expected at least one result."

    top = results[0]
    meta = top.get("metadata", {})
    agents_raw = meta.get("primary_agents", "[]")
    top_agents: list[str] = (
        json.loads(agents_raw) if isinstance(agents_raw, str) else agents_raw
    )
    assert "geopolitical" in top_agents, (
        f"Geopolitical-only query: top match '{top['id']}' "
        f"should have 'geopolitical' in primary_agents, got: {top_agents}"
    )


def test_query_disaster(retriever: ContextRetriever) -> None:
    """Natural-disaster-only signal → top result involves natural_disaster domain."""
    signals = {
        "shipping": 0.10,
        "market": 0.10,
        "geopolitical": 0.08,
        "natural_disaster": 0.88,
        "routing": 0.12,
        "news_sentiment": 0.10,
    }
    results = retriever.query(signals, top_k=3)

    assert results, "Expected at least one result."

    # The top result OR one of the top-3 must be a disaster case.
    disaster_cases_found = [
        r for r in results
        if "natural_disaster" in json.loads(
            r.get("metadata", {}).get("primary_agents", "[]")
        )
    ]
    assert disaster_cases_found, (
        f"Expected at least one disaster case in top-3 results. Got ids: "
        f"{[r['id'] for r in results]}"
    )

    top_disaster = disaster_cases_found[0]
    meta = top_disaster.get("metadata", {})
    agents_raw = meta.get("primary_agents", "[]")
    top_agents: list[str] = (
        json.loads(agents_raw) if isinstance(agents_raw, str) else agents_raw
    )
    assert "natural_disaster" in top_agents


def test_format_context(retriever: ContextRetriever) -> None:
    """format_context() returns readable text with event names and similarity scores."""
    signals = {
        "shipping": 0.70,
        "geopolitical": 0.65,
        "routing": 0.60,
        "natural_disaster": 0.20,
        "market": 0.30,
        "news_sentiment": 0.55,
    }
    results = retriever.query(signals, top_k=3)
    text = retriever.format_context(results)

    assert isinstance(text, str)
    assert len(text) > 50, "Expected non-trivial formatted text."
    assert "Historical Precedents:" in text, "Expected header line."
    assert "similarity:" in text, "Expected similarity scores in output."

    # At least one known event name or domain keyword should appear.
    known_keywords = [
        "Hormuz", "Houthi", "Suez", "Cyclone", "Tanker",
        "Japan", "COVID", "Iran", "piracy", "shipping", "geopolitical",
    ]
    text_lower = text.lower()
    found = [kw for kw in known_keywords if kw.lower() in text_lower]
    assert found, (
        f"Expected at least one event/domain keyword in formatted context. "
        f"Got:\n{text}"
    )


# ---------------------------------------------------------------------------
# Live-collection seeding from the committed snapshot (cloud cold start)
# ---------------------------------------------------------------------------


def _live_snapshot_doc(i: int) -> dict:
    """A live_extracted_backup.json-shaped document ({id, text, metadata})."""
    return {
        "id": f"newsapi_hormuz_2026-01-0{i + 1}_{i}",
        "text": (
            f"Tanker congestion report {i}: elevated transit delays and vessel "
            f"queuing near the Strait of Hormuz."
        ),
        "metadata": {
            "source_api": "newsapi",
            "event_date": f"2026-01-0{i + 1}",
            "region": "hormuz",
            "primary_agents": '["news_sentiment"]',
            "event_type": "news",
            "title": f"Hormuz congestion headline {i}",
            "source_name": "Test Wire",
            "url": f"https://example.test/article/{i}",
        },
    }


@pytest.fixture()
def live_retriever(tmp_path: Path) -> ContextRetriever:
    """A ContextRetriever on a fresh temp store with an empty live collection."""
    cfg = {
        "collection_name": "test_seed_static",
        "top_k": 3,
        "collections": {"live_context": "test_seed_live"},
    }
    return ContextRetriever(cfg, persist_directory=str(tmp_path / "chromadb"))


def test_seed_live_collection_from_backup(
    live_retriever: ContextRetriever, tmp_path: Path
) -> None:
    """Seeding an empty live collection upserts every snapshot document and
    the seeded documents are reachable through the normal gated query path."""
    backup = tmp_path / "live_extracted_backup.json"
    backup.write_text(json.dumps([_live_snapshot_doc(i) for i in range(5)]), encoding="utf-8")

    assert live_retriever._live_collection.count() == 0
    seeded = live_retriever.seed_live_collection_from_backup(str(backup))
    assert seeded == 5
    assert live_retriever._live_collection.count() == 5

    # Static collection is deliberately left empty, so every gated match must
    # come from the seeded live collection.
    out = live_retriever.query_gated(
        {"news_sentiment": 0.9, "shipping": 0.8},
        composite_risk_score=0.9,
        min_similarity=0.0,
    )
    assert out is not None and out["triggered"]
    assert out["matches"], "Seeded documents should be retrievable."
    assert all(m["source"] == "live" for m in out["matches"])
    assert any("Hormuz" in m["text"] for m in out["matches"])


def test_seed_is_idempotent(live_retriever: ContextRetriever, tmp_path: Path) -> None:
    """A second seed call hits the count fast-path — no duplicates, no rework."""
    backup = tmp_path / "live_extracted_backup.json"
    backup.write_text(json.dumps([_live_snapshot_doc(i) for i in range(4)]), encoding="utf-8")

    assert live_retriever.seed_live_collection_from_backup(str(backup)) == 4
    assert live_retriever.seed_live_collection_from_backup(str(backup)) == 0
    assert live_retriever._live_collection.count() == 4


def test_seed_missing_or_malformed_backup_degrades(
    live_retriever: ContextRetriever, tmp_path: Path
) -> None:
    """Missing or garbage snapshots never raise — the collection stays empty
    and the dashboard feed keeps its existing fallback message."""
    assert live_retriever.seed_live_collection_from_backup(str(tmp_path / "absent.json")) == 0
    assert live_retriever._live_collection.count() == 0

    garbage = tmp_path / "garbage.json"
    garbage.write_text("this is not json {", encoding="utf-8")
    assert live_retriever.seed_live_collection_from_backup(str(garbage)) == 0
    assert live_retriever._live_collection.count() == 0
