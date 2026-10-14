"""EXPLORATORY: the stricter FP-characterisation rule (post-hoc amendment, 2026-10-15).

Runs once, per ``PREREGISTERED_RULES.md`` (amendment "stricter
FP-characterisation rule"). The primary outcome remains the pre-registered tag
and its null calm-day control (``fp_characterisation_base_rate.csv``).

Reads only existing artefacts — no agent, detection or API call:
``data/eval/{region}.csv`` (dates, labels, the 70/30 split),
``aer/{region}.jsonl`` (alert days), ``alerts_episodes.csv`` (episodes) and the
``backfill_2023_25`` collection of ``data/knowledge_base/.chromadb_xai`` (read
through a temporary copy).

Writes ``fp_strict_lift.csv``.

Usage::

    python eval/explainability/run_fp_strict.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.explainability_metrics import BOOT_SEED, N_BOOT  # noqa: E402
from src.explainability.kb_backfill import COLLECTION as BACKFILL  # noqa: E402
from src.explainability.kb_evidence import load_documents  # noqa: E402

_OUT = _ROOT / "eval" / "explainability"
XAI_DB = _ROOT / "data" / "knowledge_base" / ".chromadb_xai"
REGIONS = ("hormuz", "bab_el_mandeb", "panama", "malacca")
TEST_FRACTION = 0.30
WINDOW_DAYS = 8

ACLED_TYPES = {"Battles", "Explosions/Remote violence", "Strategic developments"}
STRONG = re.compile(
    r"\b(houthi\w*|missile\w*|hijack\w*|pirate\w*|piracy|warship\w*|blockade\w*|airstrike\w*|"
    r"air strike\w*|seized|seizure|seizes|seizing|drone strike\w*|drone attack\w*)\b", re.I)
VERB = re.compile(r"\b(attack\w*|strike\w*|struck|fired on|explosion\w*|blast\w*)\b", re.I)
OBJECT = re.compile(r"\b(ship|ships|vessel\w*|tanker\w*|merchant|cargo|navy|naval)\b", re.I)
EXCLUDE = re.compile(
    r"\b(shark\w*|heart attack\w*|panic attack\w*|cyber\w*|ransomware|hacker\w*|"
    r"hunger strike\w*|strike action|dock\w* strike\w*|union|price war|trade war|tariff war|"
    r"football|soccer|cricket)\b|\bworker\w*\b.{0,20}\bstrike\w*", re.I)


def news_counts(text: str) -> bool:
    """INCLUDE (strong term, or attack verb + maritime object) and not EXCLUDE."""
    if EXCLUDE.search(text):
        return False
    return bool(STRONG.search(text) or (VERB.search(text) and OBJECT.search(text)))


def counting_dates(docs: pd.DataFrame, region: str) -> pd.DatetimeIndex:
    """Dates of the region's backfilled documents that count under the strict rule."""
    sub = docs[docs["region"] == region]
    acled = sub[(sub["source_api"] == "acled") & sub["acled_event_type"].isin(ACLED_TYPES)]
    news = sub[sub["source_api"] == "serpapi"]
    news = news[(news["title"] + " " + news["text"].fillna("")).map(news_counts)]
    return pd.DatetimeIndex(pd.concat([acled["event_date"], news["event_date"]]).dropna())


def window_hits(days: pd.DatetimeIndex, doc_dates: pd.DatetimeIndex) -> np.ndarray:
    """1 if any counting document falls in [day - 7, day], vectorised by sorted search."""
    dates = np.sort(doc_dates.values.astype("datetime64[D]"))
    end = days.values.astype("datetime64[D]")
    start = end - np.timedelta64(WINDOW_DAYS - 1, "D")
    return (np.searchsorted(dates, end, side="right")
            - np.searchsorted(dates, start, side="left") > 0).astype(float)


def region_windows(region: str) -> dict[str, pd.DatetimeIndex]:
    """FP / TP episode first days and calm days, from existing artefacts only."""
    frame = pd.read_csv(_ROOT / "data" / "eval" / f"{region}.csv", parse_dates=["timestamp"])
    test = frame.iloc[int(len(frame) * (1 - TEST_FRACTION)):]
    alert_dates = {json.loads(line)["date"]
                   for line in (_OUT / "aer" / f"{region}.jsonl").open(encoding="utf-8")}
    stamps = test["timestamp"].dt.strftime("%Y-%m-%d")
    calm = test[~stamps.isin(alert_dates) & (test["y_true"].astype(int) == 0)]["timestamp"]
    episodes = pd.read_csv(_OUT / "alerts_episodes.csv")  # with-duration episodes (Phase 1)
    episodes = episodes[episodes.region == region]
    first = {lab: pd.DatetimeIndex(pd.to_datetime(episodes[episodes.episode_label == lab]["start"]))
             for lab in ("FP", "TP")}
    return {"FP": first["FP"], "TP": first["TP"], "calm": pd.DatetimeIndex(calm)}


def bootstrap_lift(fp: list[np.ndarray], calm: list[np.ndarray],
                   rng: np.random.Generator) -> tuple[float, float, float]:
    """Lift with a percentile CI; FP and calm windows resampled independently per stratum."""
    def rate_draws(groups: list[np.ndarray]) -> np.ndarray:
        hits = np.zeros(N_BOOT)
        total = sum(len(g) for g in groups)
        for g in groups:
            idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
            hits += g[idx].sum(axis=1)
        return hits / total

    fp_rate = sum(g.sum() for g in fp) / sum(len(g) for g in fp)
    calm_rate = sum(g.sum() for g in calm) / sum(len(g) for g in calm)
    if calm_rate == 0:
        return float("nan"), float("nan"), float("nan")
    draws_fp, draws_calm = rate_draws(fp), rate_draws(calm)
    with np.errstate(divide="ignore", invalid="ignore"):
        lifts = draws_fp / draws_calm
    lifts = lifts[np.isfinite(lifts)]
    return (fp_rate / calm_rate, float(np.percentile(lifts, 2.5)),
            float(np.percentile(lifts, 97.5)))


def main() -> int:
    docs = load_documents(XAI_DB, [BACKFILL], copy=True)
    docs["acled_event_type"] = docs.get("acled_event_type", pd.Series(dtype=str)).fillna("")
    rng = np.random.default_rng(BOOT_SEED)
    rows, pooled = [], {"FP": [], "calm": []}
    for region in REGIONS:
        windows = region_windows(region)
        dates = counting_dates(docs, region)
        hits = {k: window_hits(v, dates) for k, v in windows.items()}
        lift, lo, hi = bootstrap_lift([hits["FP"]], [hits["calm"]], rng)
        rows.append({"data_origin": "real_eval", "analysis": "EXPLORATORY (post-hoc rule)",
                     "kb": "backfill_2023_25 only", "region": region,
                     "n_counting_docs": len(dates),
                     "n_fp_windows": len(hits["FP"]), "fp_rate": hits["FP"].mean(),
                     "n_tp_windows": len(hits["TP"]),
                     "tp_rate": hits["TP"].mean() if len(hits["TP"]) else np.nan,
                     "n_calm_windows": len(hits["calm"]), "calm_rate": hits["calm"].mean(),
                     "lift_fp_vs_calm": lift, "ci_lo": lo, "ci_hi": hi})
        pooled["FP"].append(hits["FP"])
        pooled["calm"].append(hits["calm"])
    lift, lo, hi = bootstrap_lift(pooled["FP"], pooled["calm"], rng)
    rows.append({"data_origin": "real_eval", "analysis": "EXPLORATORY (post-hoc rule)",
                 "kb": "backfill_2023_25 only", "region": "pooled",
                 "n_counting_docs": sum(r["n_counting_docs"] for r in rows),
                 "n_fp_windows": sum(len(g) for g in pooled["FP"]),
                 "fp_rate": np.concatenate(pooled["FP"]).mean(),
                 "n_calm_windows": sum(len(g) for g in pooled["calm"]),
                 "calm_rate": np.concatenate(pooled["calm"]).mean(),
                 "lift_fp_vs_calm": lift, "ci_lo": lo, "ci_hi": hi})
    table = pd.DataFrame(rows)
    table.to_csv(_OUT / "fp_strict_lift.csv", index=False)
    print(table.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
