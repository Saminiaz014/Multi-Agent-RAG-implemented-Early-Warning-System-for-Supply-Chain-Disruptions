"""Precision@3 of the precedent wrapper, from the author's relevance annotations.

Reads ``rag_annotation.csv`` (the blind file, with ``relevance`` filled in as
0 / 1 / 2) and joins ``rag_annotation_key.csv`` for each episode's TP/FP label.
Per ``PREREGISTERED_RULES.md`` section B:

* lenient precision@3 — share of retrieved documents with relevance >= 1;
* strict precision@3 — share with relevance == 2.

Episodes are the unit: precision is computed per episode over its retrieved
documents (up to 3), then averaged with an episode-level bootstrap CI.
Episodes for which nothing passed the 0.55 similarity floor have no retrieved
documents and are reported as a count, not as precision 0.

Writes ``rag_precision_at_3.csv``.

Usage::

    python eval/explainability/compute_precision_at_3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.explainability_metrics import episode_bootstrap  # noqa: E402

_OUT = _ROOT / "eval" / "explainability"


def precision_table(annotated: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Per region x label (and overall) lenient/strict precision@3 with CIs."""
    relevance = pd.to_numeric(annotated["relevance"], errors="coerce")
    if relevance.isna().any() or not relevance.isin([0, 1, 2]).all():
        bad = annotated.loc[relevance.isna() | ~relevance.isin([0, 1, 2]), "doc_id"].tolist()
        raise ValueError(f"relevance must be 0, 1 or 2 for every row; check {bad[:5]}")
    frame = annotated.assign(relevance=relevance.astype(int)).merge(
        key[["episode_id", "doc_id", "episode_label"]], on=["episode_id", "doc_id"], how="left")
    labels = key.drop_duplicates("episode_id").set_index("episode_id")["episode_label"]
    empty = key[key.doc_id.isna()].episode_id.unique()

    per_episode = frame.groupby("episode_id").agg(
        region=("region", "first"),
        lenient=("relevance", lambda r: (r >= 1).mean()),
        strict=("relevance", lambda r: (r == 2).mean()),
    )
    per_episode["episode_label"] = labels.reindex(per_episode.index)

    rows = []
    groups = [("all", "all", per_episode)]
    groups += [(r, "all", g) for r, g in per_episode.groupby("region")]
    groups += [("all", lab, g) for lab, g in per_episode.groupby("episode_label")]
    for region, label, group in groups:
        for measure in ("lenient", "strict"):
            mean, lo, hi = episode_bootstrap(group[measure].tolist())
            rows.append({"data_origin": "real_eval", "annotator": "author (single)",
                         "region": region, "episode_label": label,
                         "measure": f"precision@3 ({measure})", "value": mean, "ci_lo": lo,
                         "ci_hi": hi, "n_episodes": len(group),
                         "n_docs": int(frame[frame.episode_id.isin(group.index)].shape[0]),
                         "n_episodes_without_hits": int(sum(
                             1 for e in empty
                             if (region == "all" or key.loc[key.episode_id == e, "episode_id"].str
                                 .startswith(region).any())
                             and (label == "all" or labels.get(e) == label)))})
    return pd.DataFrame(rows)


def main() -> int:
    annotated = pd.read_csv(_OUT / "rag_annotation.csv")
    key = pd.read_csv(_OUT / "rag_annotation_key.csv")
    table = precision_table(annotated, key)
    table.to_csv(_OUT / "rag_precision_at_3.csv", index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
