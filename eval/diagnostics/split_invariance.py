"""Split-invariance diagnosis: agent and tier scores under two split dates.

For one region, every agent and every tier is scored four ways on the same
frame (``data/eval/<region>.csv``):

    A  fitted before 2023-07-01, scored from 2023-07-01   (the main split)
    B  fitted before 2024-01-01, scored from 2024-01-01   (the late split)
    C  fitted before 2023-07-01, scored from 2024-01-01   (restart only)
    D  fitted before 2024-01-01, scored from 2023-07-01   (refit only)

On the shared days (2024-01-01 .. end_date) it reports, per series, the max
and mean absolute score difference and the number of days that differ, for
A vs B, A vs C (does a day's score depend on where scoring starts?) and A vs
D (how much does refitting on a longer window move it?). It also reports AUC
on exactly those days (masked news days excluded) under each configuration,
which separates a code restart from a change of evaluation window.

Scores go through ``scripts/run_method_comparison.py``'s own
``_agent_scores`` / ``_compose``, so the harness is what is tested. C passes
the whole frame as history, because its fitting window and test window are
not adjacent (train + test would skip July-December 2023).

Usage::

    python eval/diagnostics/split_invariance.py <region>

Writes ``data/eval/diagnostics/inv_<region>.csv`` (the table) and
``inv_scores_<region>.csv`` (the daily scores; read by ``where_diff.py``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "run_method_comparison", ROOT / "scripts" / "run_method_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)

from src.core.config_manager import load_config_for_region  # noqa: E402
from src.evaluation.window import eval_window, split_by_date  # noqa: E402

OUT = ROOT / "data" / "eval" / "diagnostics"
EARLY, LATE = "2023-07-01", "2024-01-01"
CONFIGS = {
    "A": "fit<2023-07, score from 2023-07",
    "B": "fit<2024-01, score from 2024-01",
    "C": "fit<2023-07, score from 2024-01",
    "D": "fit<2024-01, score from 2023-07",
}


def main(region: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, manifest = rmc._load(region)
    config = load_config_for_region(region)
    masked = set(pd.to_datetime((manifest.get("news_source") or {}).get("days_masked") or []))
    history = frame[pd.to_datetime(frame["timestamp"]) <= eval_window()["end_date"]]

    train_a, test_a = split_by_date(frame, split_date=EARLY)
    train_b, test_b = split_by_date(frame, split_date=LATE)
    overlap = pd.DatetimeIndex(test_b["timestamp"])
    runs = {"A": (train_a, test_a, None), "B": (train_b, test_b, None),
            "C": (train_a, test_b, history), "D": (train_b, test_a, None)}

    series: dict[str, dict[str, pd.Series | None]] = {}
    for tag, (train, test, hist) in runs.items():
        agents = {}
        for domain in rmc._TIER_ORDER:
            s = rmc._agent_scores(domain, config, train, test, history=hist)
            agents[domain] = (pd.Series(s, index=pd.DatetimeIndex(test["timestamp"]))
                              if s is not None else None)
        index = next(s.index for s in agents.values() if s is not None)
        arrays = {d: (s.to_numpy() if s is not None else None) for d, s in agents.items()}
        tiers = {f"Tier {k}": pd.Series(rmc._compose(k, arrays, config)[0], index=index)
                 for k in range(1, len(rmc._TIER_ORDER) + 1)}
        series[tag] = {**{f"agent {d}": s for d, s in agents.items()}, **tiers}
        print(f"  scored {tag}: {CONFIGS[tag]}", flush=True)

    y = frame.set_index("timestamp")["y_true"].astype(int)
    keep = overlap[~overlap.isin(masked)]
    yy = y.reindex(keep).to_numpy()
    rows = []
    for key, a in series["A"].items():
        if a is None:
            continue
        a = a.reindex(overlap)
        row = {"region": region, "series": key, "n_overlap": len(overlap)}
        for other, name in (("B", "AvsB"), ("C", "restart"), ("D", "refit")):
            d = (a - series[other][key].reindex(overlap)).abs()
            row[f"{name}_max"] = float(d.max())
            row[f"{name}_mean"] = float(d.mean())
            row[f"{name}_ndiff"] = int((d > 1e-12).sum())
        if len(np.unique(yy)) > 1:
            for tag in CONFIGS:
                s = series[tag][key].reindex(keep).to_numpy()
                row[f"auc_overlap_{tag}"] = float(roc_auc_score(yy, np.nan_to_num(s)))
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"inv_{region}.csv", index=False)
    pd.DataFrame({f"{tag}|{k}": v for tag, d in series.items()
                  for k, v in d.items() if v is not None}).to_csv(
        OUT / f"inv_scores_{region}.csv")
    return table


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python eval/diagnostics/split_invariance.py <region>")
    print(main(sys.argv[1]).round(4).to_string(index=False))
