"""Build the 2023-25 backfill into the eval copy of the knowledge base.

Follows ``PREREGISTERED_RULES.md`` section A. The production DB
(``data/knowledge_base/.chromadb``) is only *copied* — on the first run — into
``data/knowledge_base/.chromadb_xai``; the backfill collection is written there
and nowhere else. Raw API responses are cached in
``eval/explainability/backfill_raw/``, so a rebuild spends no quota.

Writes ``eval/explainability/backfill_coverage.csv`` (region x month).

Usage::

    python eval/explainability/run_backfill.py
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.explainability.kb_backfill import (  # noqa: E402
    ACLED_COUNTRIES,
    COLLECTION,
    MONTHS,
    Fetcher,
    build_region_month,
    maritime_events,
)

_OUT = _ROOT / "eval" / "explainability"
PROD_DB = _ROOT / "data" / "knowledge_base" / ".chromadb"
XAI_DB = _ROOT / "data" / "knowledge_base" / ".chromadb_xai"
REGIONS = ("hormuz", "bab_el_mandeb", "panama", "malacca")


def ensure_eval_copy() -> None:
    """Copy the production DB once; never write to the production path."""
    if not XAI_DB.exists():
        shutil.copytree(PROD_DB, XAI_DB)
        print(f"copied production knowledge base -> {XAI_DB.relative_to(_ROOT)}")


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
    ensure_eval_copy()

    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    fetcher = Fetcher(_OUT / "backfill_raw", os.environ.get("SERPAPI_API_KEY"),
                      os.environ.get("ACLED_USERNAME"), os.environ.get("ACLED_PASSWORD"))
    client = chromadb.PersistentClient(path=str(XAI_DB))
    live = client.get_collection("live_extracted_context").get(include=["metadatas"])["metadatas"]
    live_dates = pd.DataFrame([{"region": m.get("region"), "date": m.get("event_date")} for m in live])
    live_dates["month"] = pd.to_datetime(live_dates["date"], errors="coerce").dt.strftime("%Y-%m")

    docs, coverage = [], []
    for region in REGIONS:
        for month in MONTHS:
            selected, news_status = build_region_month(fetcher, region, month)
            docs.extend(selected)
            raw_news = fetcher.serpapi(region, month) if news_status == "ok" else []
            candidates = maritime_events(e for c in ACLED_COUNTRIES[region]
                                         for e in fetcher.acled(c, month))
            coverage.append({
                "data_origin": "kb_backfill", "region": region, "month": month,
                "serpapi_status": news_status,
                "serpapi_raw_results": len(raw_news),
                "acled_maritime_candidates": len(candidates),
                "backfill_news": sum(1 for _, _, m in selected if m["source_api"] == "serpapi"),
                "backfill_acled": sum(1 for _, _, m in selected if m["source_api"] == "acled"),
                "backfill_total": len(selected),
                "production_live_docs": int(((live_dates.region == region)
                                             & (live_dates.month == month)).sum()),
            })
        print(f"{region}: {sum(c['backfill_total'] for c in coverage if c['region'] == region)} docs "
              f"(API calls so far: {fetcher.calls})")

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, embedding_function=DefaultEmbeddingFunction(),
                                          metadata={"hnsw:space": "cosine"})
    ids, texts, metas = zip(*docs)
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate backfill document ids")
    for start in range(0, len(ids), 500):
        collection.add(ids=list(ids[start:start + 500]), documents=list(texts[start:start + 500]),
                       metadatas=list(metas[start:start + 500]))
    pd.DataFrame(coverage).to_csv(_OUT / "backfill_coverage.csv", index=False)
    print(f"wrote {collection.count()} docs to {XAI_DB.relative_to(_ROOT)}::{COLLECTION}; "
          f"API calls this run: {fetcher.calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
