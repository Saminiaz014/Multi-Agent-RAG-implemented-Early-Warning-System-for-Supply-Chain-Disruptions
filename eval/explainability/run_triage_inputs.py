"""Phase 4, step C0: one-time export of everything the triage analysis needs.

The only agent run in Phase 4 (``PREREGISTERED_RULES.md`` §C.0). It re-scores
Path D once per region and refuses to write anything unless the recomputed
Tier 1-5 metrics equal ``eval/method_comparison_results.csv`` within 1e-9. It
then fits the production SHAP surrogate (synthetic, as deployed) once.

Writes ``eval/explainability/triage_inputs/``:

* ``days.csv`` — region x variant x window x day: agent scores (shipping =
  shock score in ``without_duration``), composite, threshold, alert flag, label.
* ``baselines.json`` — calm baselines per variant x region x agent.
* ``surrogate.csv`` — the production surrogate's per-agent attribution on every
  test alert day of both variants.

Usage::

    python eval/explainability/run_triage_inputs.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import path_d  # noqa: E402
from src.explainability.alert_sets import VARIANTS, _composite, build_alert_set  # noqa: E402
from src.explainability.surrogate_comparator import (  # noqa: E402
    agent_attribution,
    production_surrogate,
    real_features,
)

_OUT = _ROOT / "eval" / "explainability" / "triage_inputs"
_CSV = _ROOT / "eval" / "method_comparison_results.csv"


def check_bit_identity() -> None:
    """Tier 1-5 metrics from the cached scores must equal the CSV (1e-9)."""
    mod = path_d.comparison_module()
    csv = pd.read_csv(_CSV)
    for region in path_d.REGIONS:
        data = path_d.load_region(region)
        for tier in range(1, 6):
            metrics = mod._score_metrics(data.y_test, data.tier_scores(tier, "test"),
                                         data.tier_scores(tier, "train"))
            row = csv[(csv.region == region) & (csv.method == f"Tier {tier}")].iloc[0]
            for key in ("auc", "f1", "fpr", "alert_rate"):
                a, b = float(metrics[key]), float(row[key])
                same = (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9
                if not same:
                    raise SystemExit(f"bit-identity check failed: {region} Tier {tier} {key} "
                                     f"{a!r} != {b!r}; nothing written")


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    t0 = time.time()
    check_bit_identity()
    print(f"bit-identity with method_comparison_results.csv: OK ({time.time() - t0:.0f}s)")

    rows, baselines, alert_days = [], {}, []
    for variant in VARIANTS:
        baselines[variant] = {}
        for region in path_d.REGIONS:
            aset = build_alert_set(region, variant)
            baselines[variant][region] = aset.baselines
            data = aset.data
            train_composite = _composite(aset.train_scores, aset.agents, aset.weights)
            windows = {
                "train": (pd.to_datetime(data.train["timestamp"]), data.y_train,
                          aset.train_scores, train_composite),
                "test": (pd.to_datetime(data.test["timestamp"]), aset.y,
                         aset.test_scores, aset.composite),
            }
            for window, (stamps, y, scores, composite) in windows.items():
                frame = pd.DataFrame({
                    "data_origin": path_d.DATA_ORIGIN, "region": region, "variant": variant,
                    "window": window, "date": stamps.dt.strftime("%Y-%m-%d").to_numpy(),
                    "y_true": y, "composite": composite, "threshold": aset.threshold,
                    "alert": composite > aset.threshold,
                })
                for agent in aset.agents:
                    frame[f"score_{agent}"] = scores[agent]
                rows.append(frame)
            alert_days += [(region, variant, int(i)) for i in aset.alert_idx]
    _OUT.mkdir(parents=True, exist_ok=True)
    days = pd.concat(rows, ignore_index=True)
    days.to_csv(_OUT / "days.csv", index=False)
    (_OUT / "baselines.json").write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    print(f"days.csv: {len(days)} rows ({time.time() - t0:.0f}s)")

    surrogate = production_surrogate()
    sur_rows = []
    for region in path_d.REGIONS:
        data = path_d.load_region(region)
        agents = data.tier_agents(5)
        idx = sorted({i for r, _, i in alert_days if r == region})
        shap = surrogate.shap(real_features(data.test).iloc[idx])
        dates = pd.to_datetime(data.test["timestamp"]).dt.strftime("%Y-%m-%d").to_numpy()
        for row, i in zip(shap, idx):
            phi, outside = agent_attribution(row, agents)
            sur_rows.append({"data_origin": path_d.DATA_ORIGIN, "region": region,
                             "date": dates[i], "outside_share": outside,
                             **{f"phi_sur_{a}": v for a, v in phi.items()}})
    pd.DataFrame(sur_rows).to_csv(_OUT / "surrogate.csv", index=False)
    print(f"surrogate.csv: {len(sur_rows)} alert days ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
