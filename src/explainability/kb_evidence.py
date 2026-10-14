"""Knowledge-base evidence around false-positive alert episodes.

For each FP episode, looks for region-matched documents dated in
``[start - 7 days, end]`` and tags the episode (``PREREGISTERED_RULES.md``
section B):

* ``risk_without_traffic_drop`` — at least one geopolitical/conflict document:
  an event-level ACLED record, or a news item matching :data:`GEO_PATTERN`.
  The detector saw real risk; the label (a traffic level shift) did not move.
* ``unexplained`` — documents exist in the window, none geopolitical/conflict.
* ``no_coverage`` — no document at all in the window, so absence of evidence
  says nothing about the world.

ACLED yearly summaries (``event_type == conflict_summary``, dated 1 January)
are excluded from the window test and reported separately: they are not
events on a day.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

GEO_PATTERN = re.compile(
    r"\b(attack\w*|strike\w*|missile\w*|drone\w*|houthi\w*|militar\w*|nav(y|al)|warship\w*|"
    r"seiz\w*|hijack\w*|pira\w*|sanction\w*|war|conflict\w*|tension\w*|threat\w*|"
    r"escalat\w*|clash\w*|blockade\w*|militant\w*|terror\w*|hostil\w*)\b",
    re.IGNORECASE,
)
LOOKBACK_DAYS = 7
NEWS_SOURCES = ("newsapi", "serpapi")


def load_documents(db_path: Path, collections: Sequence[str], copy: bool = True) -> pd.DataFrame:
    """All documents of ``collections`` as a frame (read from a temporary copy by default)."""
    import chromadb

    tmp = None
    try:
        if copy:
            tmp = Path(tempfile.mkdtemp(prefix="xai_kb_"))
            shutil.copytree(db_path, tmp / "kb")
            db_path = tmp / "kb"
        client = chromadb.PersistentClient(path=str(db_path))
        rows = []
        for name in collections:
            got = client.get_collection(name).get(include=["metadatas", "documents"])
            rows += [{**(m or {}), "doc_id": i, "text": t, "collection": name}
                     for i, m, t in zip(got["ids"], got["metadatas"], got["documents"])]
        del client
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    frame = pd.DataFrame(rows)
    frame["event_date"] = pd.to_datetime(frame.get("event_date"), errors="coerce")
    for col in ("title", "source_api", "event_type", "region"):
        if col not in frame:
            frame[col] = ""
    frame["title"] = frame["title"].fillna("")
    frame["is_news"] = frame["source_api"].isin(NEWS_SOURCES)
    frame["is_acled_event"] = (frame["source_api"] == "acled") & ~frame["event_type"].isin(
        ["conflict_summary", "historical_scenario"])
    frame["is_geo"] = frame["is_acled_event"] | (
        frame["is_news"] & (frame["title"] + " " + frame["text"].fillna("")).str.contains(GEO_PATTERN))
    return frame


def characterise(episode: dict, docs: pd.DataFrame) -> dict:
    """Evidence and tag for one FP episode.

    Args:
        episode: ``region``, ``start``, ``end`` (dates) at least.
        docs: Output of :func:`load_documents`.
    """
    start, end = pd.Timestamp(episode["start"]), pd.Timestamp(episode["end"])
    lo = start - pd.Timedelta(days=LOOKBACK_DAYS)
    region_docs = docs[docs["region"] == episode["region"]]
    dated = region_docs[region_docs["is_news"] | region_docs["is_acled_event"]]
    window = dated[(dated["event_date"] >= lo) & (dated["event_date"] <= end)]
    geo = window[window["is_geo"]]
    annual = region_docs[(region_docs["event_type"] == "conflict_summary")
                         & (region_docs["event_date"].dt.year == start.year)
                         & (region_docs["event_date"] <= end)]
    gaps = [(lo - d).days if d < lo else max((d - end).days, 0)
            for d in dated["event_date"].dropna()]
    if len(geo):
        tag = "risk_without_traffic_drop"
    elif len(window):
        tag = "unexplained"
    else:
        tag = "no_coverage"
    return {
        "window_start": lo.date().isoformat(),
        "window_end": end.date().isoformat(),
        "n_docs_in_window": len(window),
        "n_news_in_window": int(window["is_news"].sum()),
        "n_acled_events_in_window": int(window["is_acled_event"].sum()),
        "n_geopolitical_in_window": len(geo),
        "n_backfill_in_window": int(window.get("backfill", pd.Series(dtype=bool))
                                    .fillna(False).astype(bool).sum()),
        "acled_annual_military_events": (float(pd.to_numeric(annual.get("military_events"),
                                                             errors="coerce").sum())
                                         if len(annual) else np.nan),
        "nearest_region_doc_days": int(min(gaps)) if gaps else None,
        "matched_titles": " | ".join(geo["title"].head(3)),
        "tag": tag,
    }
