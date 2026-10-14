"""Phase 3 — precedent grounding: diagnostics, AER precedents, annotation set, FP evidence.

Implements ``PREREGISTERED_RULES.md`` section B.

* **Diagnostics** (``rag_diagnostics.csv``): one query per alert episode, on
  its first day, under three configurations — the production retriever
  (temporary copy of the production DB), the eval wrapper over the production
  collections, and the wrapper over production + the 2023-25 backfill
  (``data/knowledge_base/.chromadb_xai``).
* **AER precedents**: every with-duration AER gets the wrapper's top 3
  (production + backfill) in its ``precedents`` field.
* **E8 / E5b** rows replace the "pending" rows in ``scorecard.csv``.
* **Annotation set** (``rag_annotation.csv``, blind) and its key
  (``rag_annotation_key.csv``).
* **FP characterisation** (``fp_characterisation.csv``) against the
  production KB and the backfilled KB, with the ``no_coverage`` tag, and an
  exploratory calm-day base rate for the same tags
  (``fp_characterisation_base_rate.csv``).

Usage::

    python eval/explainability/run_precedents.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import path_d  # noqa: E402
from src.evaluation.explainability_metrics import episode_bootstrap  # noqa: E402
from src.explainability.agent_breakdown import (  # noqa: E402
    breakdown,
    effective_terms,
    present_columns,
)
from src.explainability.alert_sets import VARIANTS, AlertSet, build_alert_set  # noqa: E402
from src.explainability.exact_shapley import path_d_closed_form  # noqa: E402
from src.explainability.kb_backfill import COLLECTION as BACKFILL  # noqa: E402
from src.explainability.kb_evidence import characterise, load_documents  # noqa: E402
from src.explainability.precedent_retriever import (  # noqa: E402
    STATIC_COLLECTION,
    PrecedentRetriever,
    build_query,
    doc_date,
    doc_region,
    text_hash,
)

_OUT = _ROOT / "eval" / "explainability"
PROD_DB = _ROOT / "data" / "knowledge_base" / ".chromadb"
XAI_DB = _ROOT / "data" / "knowledge_base" / ".chromadb_xai"
LIVE = "live_extracted_context"
MIN_SIM = 0.55
TOP_K = 3
ANNOTATION_SEED = 20261014
ANNOTATION_EPISODES = 20
CONFIGS = ("production", "wrapper_production", "wrapper_backfill")


# ------------------------------------------------------------------ drivers
_PRESENT: dict[tuple[str, str], set[str]] = {}


def _present(aset: AlertSet, agent: str) -> set[str]:
    key = (aset.region, agent)
    if key not in _PRESENT:
        _PRESENT[key] = present_columns(aset.data.agents[agent].test_frame)
    return _PRESENT[key]


def day_drivers(aset: AlertSet, idx: int) -> tuple[list[dict], dict[str, float]]:
    """Ranked drivers (AER ``top_drivers`` shape) and exact phi for one alert day."""
    scores = {a: float(aset.test_scores[a][idx]) for a in aset.agents}
    phi = path_d_closed_form(scores, aset.baselines, aset.weights, aset.agents)
    drivers = []
    for agent in sorted(phi, key=phi.get, reverse=True):
        run = aset.data.agents[agent]
        result = breakdown(agent, run.agent, run.test_frame.iloc[idx], _present(aset, agent))
        if agent == "shipping" and not aset.include_duration:
            result = {**result, "branch_winner": "shock"}
        terms = sorted(effective_terms(result).items(), key=lambda kv: kv[1], reverse=True)[:2]
        drivers.append({"agent": agent, "phi": phi[agent],
                        "top_terms": [{"term": t, "value": v} for t, v in terms]})
    return drivers, phi


# --------------------------------------------------------------- production
class ProductionRetriever:
    """The unchanged production ContextRetriever, on a temporary copy of the production DB."""

    def __init__(self) -> None:
        from src.core.config_manager import load_config_for_region
        from src.rag.context_retriever import ContextRetriever

        self._tmp = Path(tempfile.mkdtemp(prefix="xai_prodkb_"))
        shutil.copytree(PROD_DB, self._tmp / "kb")
        cfg = dict(load_config_for_region("hormuz")["rag"])
        self.threshold = float(cfg.get("composite_threshold", 0.65))
        self._retriever = ContextRetriever(cfg, persist_directory=str(self._tmp / "kb"))

    def top(self, agent_scores: dict[str, float], composite: float) -> list[dict]:
        """Ungated top 3 (similarity floor removed), as hit dicts."""
        out = self._retriever.query_gated(current_signals=agent_scores,
                                          composite_risk_score=max(composite, self.threshold),
                                          top_k=TOP_K, min_similarity=-1.0)
        hits = []
        for m in out["matches"]:
            collection = STATIC_COLLECTION if m["source"] == "static" else LIVE
            hits.append({"collection": collection, "similarity": float(m["similarity"]),
                         "text": m["text"], "meta": m["metadata"]})
        return hits

    def close(self) -> None:
        del self._retriever
        shutil.rmtree(self._tmp, ignore_errors=True)


def _hit_dict(hit) -> dict:
    return {"collection": hit.collection, "similarity": hit.similarity, "text": hit.text,
            "meta": hit.meta, "doc_id": hit.doc_id}


def diagnose(hits: list[dict], region: str, alert_date: pd.Timestamp) -> dict:
    """The four Phase 3 diagnostics for one query's returned top 3."""
    if not hits:
        return {"temporal_valid": np.nan, "region_match": np.nan, "duplicate_rate": np.nan,
                "any_hit_ge_0.55": 0.0, "sims": []}
    dates = [doc_date(h["meta"]) for h in hits]
    regions = [doc_region(h["meta"], h["collection"]) for h in hits]
    hashes = [text_hash(h["text"]) for h in hits]
    return {
        "temporal_valid": float(np.mean([d is not None and d < alert_date for d in dates])),
        "region_match": float(np.mean([r == region for r in regions])),
        "duplicate_rate": (len(hashes) - len(set(hashes))) / len(hashes),
        "any_hit_ge_0.55": float(max(h["similarity"] for h in hits) >= MIN_SIM),
        "sims": [h["similarity"] for h in hits],
    }


# ---------------------------------------------------------------------- main
def run_diagnostics(asets: dict[str, AlertSet], retrievers: dict) -> tuple[list[dict], dict]:
    """Per-episode diagnostics under each configuration, plus the annotation candidates."""
    per_episode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    wrapper_top: dict[str, list] = {}
    for region, aset in asets.items():
        labels, ids = aset.episode_labels(), aset.episode_ids()
        for k, members in enumerate(aset.episodes):
            idx = members[0]
            date = pd.Timestamp(aset.dates[idx])
            drivers, _ = day_drivers(aset, idx)
            query = build_query(region, drivers)
            scores = {a: float(aset.test_scores[a][idx]) for a in aset.agents}
            results = {
                "production": retrievers["production"].top(scores, float(aset.composite[idx])),
                "wrapper_production": [_hit_dict(h) for h in retrievers["wrapper_production"]
                                       .retrieve(region, date, query, TOP_K, -1.0)],
                "wrapper_backfill": [_hit_dict(h) for h in retrievers["wrapper_backfill"]
                                     .retrieve(region, date, query, TOP_K, -1.0)],
            }
            for config, hits in results.items():
                diag = diagnose(hits, region, date)
                diag["triggered"] = (float(aset.composite[idx] >= retrievers["production"].threshold)
                                     if config == "production" else 1.0)
                per_episode[(config, region)].append(diag)
            wrapper_top[ids[k]] = {
                "region": region, "label": labels[k], "date": date, "query": query,
                "drivers": drivers,
                "hits": [h for h in results["wrapper_backfill"] if h["similarity"] >= MIN_SIM],
            }

    rows = []
    for (config, region), diags in per_episode.items():
        for metric in ("temporal_valid", "region_match", "duplicate_rate", "any_hit_ge_0.55",
                       "triggered"):
            values = [d[metric] for d in diags]
            mean, lo, hi = episode_bootstrap(values)
            rows.append({"data_origin": path_d.DATA_ORIGIN, "config": config, "region": region,
                         "metric": metric, "level": "episode", "value": mean, "ci_lo": lo,
                         "ci_hi": hi, "n_episodes": len(diags)})
        sims = [s for d in diags for s in d["sims"]]
        for name, fn in (("similarity_median", np.median), ("similarity_p10",
                          lambda v: np.percentile(v, 10)), ("similarity_p90",
                          lambda v: np.percentile(v, 90))):
            rows.append({"data_origin": path_d.DATA_ORIGIN, "config": config, "region": region,
                         "metric": name, "level": "returned docs",
                         "value": float(fn(sims)) if sims else np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan, "n_episodes": len(diags)})
    return rows, wrapper_top


def fill_aers_and_e8(asets_by_variant: dict, retriever: PrecedentRetriever) -> list[dict]:
    """AER precedents (with-duration) and E8 / E5b rows for both variants."""
    rows = []
    for variant, asets in asets_by_variant.items():
        for region, aset in asets.items():
            per_day: dict[int, dict] = {}
            for idx in aset.alert_idx:
                date = pd.Timestamp(aset.dates[idx])
                drivers, phi = day_drivers(aset, int(idx))
                hits = retriever.retrieve(region, date, build_query(region, drivers), TOP_K, MIN_SIM)
                agents_cited = set()
                for h in hits:
                    raw = str(h.meta.get("primary_agents", ""))
                    try:
                        parsed = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        parsed = raw.split(",")
                    agents_cited |= {str(a).strip() for a in parsed if str(a).strip()}
                top_agent = max(phi, key=phi.get)
                per_day[int(idx)] = {
                    "precedents": [h.record() for h in hits],
                    "grounded": float(bool(hits)),
                    "coherent": float(top_agent in agents_cited) if hits else np.nan,
                }
            if variant == "with_duration":
                _write_aer_precedents(region, per_day)
            episode_of = {i: k for k, g in enumerate(aset.episodes) for i in g}
            labels = aset.episode_labels()
            for metric, key, note in (
                ("E8", "grounded", "share with >= 1 region-matched, pre-alert precedent "
                                   ">= 0.55 (wrapper, production + backfill)"),
                ("E5", "coherent", "top exact agent among the precedents' primary_agents "
                                   "(days with >= 1 precedent)"),
            ):
                by_episode = defaultdict(list)
                for idx, d in per_day.items():
                    if np.isfinite(d[key]):
                        by_episode[episode_of[idx]].append(d[key])
                mean, lo, hi = episode_bootstrap([np.mean(v) for v in by_episode.values()])
                day_values = [d[key] for d in per_day.values() if np.isfinite(d[key])]
                base = {"data_origin": path_d.DATA_ORIGIN, "path": "D", "weight_mode": "hand_tuned",
                        "tier": 5, "variant": variant, "region": region,
                        "n_episodes": len(aset.episodes),
                        "n_tp_episodes": sum(1 for lab in labels if lab == "TP"),
                        "n_alert_days": len(per_day), "metric_id": metric,
                        "metric": ("precedent grounding" if metric == "E8"
                                   else "top agent agrees with precedents' primary_agents"),
                        "method": "exact"}
                rows.append({**base, "level": "episode", "value": mean, "ci_lo": lo, "ci_hi": hi,
                             "n_days_used": len(day_values), "note": note})
                rows.append({**base, "level": "day",
                             "value": float(np.mean(day_values)) if day_values else np.nan,
                             "ci_lo": np.nan, "ci_hi": np.nan, "n_days_used": len(day_values),
                             "note": "secondary: alert days are not independent"})
    return rows


def _write_aer_precedents(region: str, per_day: dict[int, dict]) -> None:
    path = _OUT / "aer" / f"{region}.jsonl"
    aset_dates = build_alert_set(region, "with_duration").dates
    by_date = {aset_dates[i]: v["precedents"] for i, v in per_day.items()}
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            rec["precedents"] = by_date[rec["date"]]
            fh.write(json.dumps(rec) + "\n")


def annotation_set(wrapper_top: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-registered stratified sample of episodes; blind annotation file + key."""
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for episode_id, info in wrapper_top.items():
        strata[(info["region"], info["label"])].append(episode_id)
    rng = np.random.default_rng(ANNOTATION_SEED)
    chosen: dict[tuple[str, str], list[str]] = {}
    for key in sorted(strata):
        ids = sorted(strata[key])
        take = min(3, len(ids))
        chosen[key] = [ids[i] for i in sorted(rng.choice(len(ids), size=take, replace=False))]
    while sum(len(v) for v in chosen.values()) > ANNOTATION_EPISODES:
        largest_fp = max((k for k in chosen if k[1] == "FP"), key=lambda k: len(strata[k]))
        chosen[largest_fp] = chosen[largest_fp][:-1]

    blind, key_rows = [], []
    for (region, label), ids in sorted(chosen.items()):
        for episode_id in ids:
            info = wrapper_top[episode_id]
            drivers = ", ".join(f"{d['agent']} ({d['phi']:+.3f})" for d in info["drivers"][:3])
            for hit in info["hits"]:
                date = doc_date(hit["meta"])
                title = (hit["meta"].get("title") or hit["meta"].get("event")
                         or f"[{hit['meta'].get('event_type', 'document')}, "
                            f"{hit['meta'].get('source_api', 'static')}]")
                blind.append({
                    "episode_id": episode_id, "region": region,
                    "alert_date": f"{info['date']:%Y-%m-%d}", "top_drivers": drivers,
                    "doc_id": hit["doc_id"], "doc_title": title,
                    "doc_date": None if date is None else f"{date:%Y-%m-%d}",
                    "doc_snippet": " ".join(str(hit["text"]).split())[:300],
                    "relevance": "",
                })
                key_rows.append({
                    "data_origin": path_d.DATA_ORIGIN, "episode_id": episode_id,
                    "doc_id": hit["doc_id"], "episode_label": label,
                    "similarity": round(hit["similarity"], 4), "collection": hit["collection"],
                    "source_api": hit["meta"].get("source_api", "static"),
                    "backfill": bool(hit["meta"].get("backfill", False)), "query": info["query"],
                })
            if not info["hits"]:
                key_rows.append({"data_origin": path_d.DATA_ORIGIN, "episode_id": episode_id,
                                 "doc_id": None, "episode_label": label, "similarity": np.nan,
                                 "collection": None, "source_api": None, "backfill": None,
                                 "query": info["query"]})
    return pd.DataFrame(blind), pd.DataFrame(key_rows)


def fp_characterisation(asets: dict[str, AlertSet]) -> list[dict]:
    """FP episodes against the production KB and the backfilled KB."""
    kbs = {
        "production": load_documents(XAI_DB, [LIVE], copy=True),
        "production+backfill": load_documents(XAI_DB, [LIVE, BACKFILL], copy=True),
    }
    rows = []
    for region, aset in asets.items():
        labels, ids = aset.episode_labels(), aset.episode_ids()
        for k, members in enumerate(aset.episodes):
            if labels[k] != "FP":
                continue
            phi_sum: dict[str, float] = defaultdict(float)
            for idx in members:
                for a, v in day_drivers(aset, idx)[1].items():
                    phi_sum[a] += v
            episode = {"region": region, "start": aset.dates[members[0]],
                       "end": aset.dates[members[-1]]}
            for kb_variant, docs in kbs.items():
                rows.append({
                    "data_origin": path_d.DATA_ORIGIN, "path": "D", "variant": aset.variant,
                    "kb_variant": kb_variant, "region": region, "episode_id": ids[k],
                    "start": episode["start"], "end": episode["end"],
                    "n_alert_days": len(members),
                    "top_exact_driver": max(phi_sum, key=phi_sum.get),
                    **characterise(episode, docs),
                })
    return rows


def fp_tag_base_rate(asets: dict[str, AlertSet]) -> list[dict]:
    """EXPLORATORY control (not pre-registered): the same tag on calm days.

    A tag that also fires on most ordinary days says nothing about false
    positives. Every window here is 8 days ending on the given day (the
    episode's first day, or a calm day), so FP, TP and calm windows are compared
    on the same footing.
    """
    docs = load_documents(XAI_DB, [LIVE, BACKFILL], copy=True)
    rows = []
    for region, aset in asets.items():
        alert = aset.composite > aset.threshold
        labels = aset.episode_labels()
        groups = {
            "FP episode first day": [g[0] for g, lab in zip(aset.episodes, labels) if lab == "FP"],
            "TP episode first day": [g[0] for g, lab in zip(aset.episodes, labels) if lab == "TP"],
            "calm non-alert day": [i for i in range(len(alert)) if not alert[i] and aset.y[i] == 0],
        }
        for window, days in groups.items():
            if not days:
                continue
            tags = [characterise({"region": region, "start": aset.dates[d], "end": aset.dates[d]},
                                 docs)["tag"] for d in days]
            rows.append({"data_origin": path_d.DATA_ORIGIN, "analysis": "exploratory",
                         "kb_variant": "production+backfill", "region": region,
                         "window": f"8 days ending on the {window}", "n": len(days),
                         **{f"share_{t}": float(np.mean([x == t for x in tags]))
                            for t in ("risk_without_traffic_drop", "unexplained", "no_coverage")}})
    return rows


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    if not XAI_DB.exists():
        raise SystemExit("data/knowledge_base/.chromadb_xai missing: run run_backfill.py first")

    asets_by_variant = {v: {r: build_alert_set(r, v) for r in path_d.REGIONS} for v in VARIANTS}
    with_duration = asets_by_variant["with_duration"]

    production = ProductionRetriever()
    retrievers = {
        "production": production,
        "wrapper_production": PrecedentRetriever(XAI_DB, [STATIC_COLLECTION, LIVE]),
        "wrapper_backfill": PrecedentRetriever(XAI_DB, [STATIC_COLLECTION, LIVE, BACKFILL]),
    }
    try:
        diag_rows, wrapper_top = run_diagnostics(with_duration, retrievers)
    finally:
        production.close()
    pd.DataFrame(diag_rows).to_csv(_OUT / "rag_diagnostics.csv", index=False)

    e8_rows = fill_aers_and_e8(asets_by_variant, retrievers["wrapper_backfill"])
    scorecard = pd.read_csv(_OUT / "scorecard.csv")
    pending = (scorecard.metric_id == "E8") | (
        (scorecard.metric_id == "E5") & scorecard.metric.str.contains("precedents"))
    pd.concat([scorecard[~pending], pd.DataFrame(e8_rows)], ignore_index=True).to_csv(
        _OUT / "scorecard.csv", index=False)

    blind, key = annotation_set(wrapper_top)
    blind.to_csv(_OUT / "rag_annotation.csv", index=False)
    key.to_csv(_OUT / "rag_annotation_key.csv", index=False)

    pd.DataFrame(fp_characterisation(with_duration)).to_csv(_OUT / "fp_characterisation.csv",
                                                            index=False)
    pd.DataFrame(fp_tag_base_rate(with_duration)).to_csv(
        _OUT / "fp_characterisation_base_rate.csv", index=False)
    print(f"wrote rag_diagnostics.csv, rag_annotation.csv ({len(blind)} pairs, "
          f"{key.episode_id.nunique()} episodes), rag_annotation_key.csv, "
          "fp_characterisation.csv; updated scorecard.csv (E8, E5 precedents) and AER precedents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
