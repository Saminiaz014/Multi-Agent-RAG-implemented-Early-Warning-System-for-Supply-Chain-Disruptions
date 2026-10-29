"""Explain every test-window alert in each region with :func:`explain_alert`.

Alerts are the evaluation harness's own (``scripts/run_method_comparison.py``):
the all-agent tier's composite above the 90th percentile of its training-window
scores, on the last 30% of each region's frame. Nothing about the model, label,
thresholds or aggregation is changed.

SHAP values come from one of two surrogates:

* ``--surrogate real`` (default): the per-region real-data surrogate fitted by
  ``scripts/run_real_surrogate.py`` (``outputs/surrogate/real_rf.joblib``). Only
  the features the region supplies are explained; nothing is zero-filled.
* ``--surrogate synthetic``: the deployed synthetic surrogate, trained per region
  as the orchestrator trains it, kept for comparison.

Writes ``outputs/explanations/`` (real) or ``outputs/explanations_synthetic/``:
one JSON per alert, a summary per region, and ``coverage.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import run_method_comparison as harness  # noqa: E402

from src.core.config_manager import load_config_for_region  # noqa: E402
from src.explainability.alert_explainer import explain_alert, write_explanation  # noqa: E402
from src.explainability.shap_explainer import (  # noqa: E402
    ALL_FEATURE_NAMES,
    SurrogateShapExplainer,
    build_shap_training_data,
)

_OUTPUTS = _PROJECT_ROOT / "outputs"
_REAL_MODELS = _OUTPUTS / "surrogate" / "real_rf.joblib"
_REAL_METRICS = _OUTPUTS / "surrogate" / "real_surrogate_metrics.json"
_TIER = len(harness._TIER_ORDER)


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip ``domain__`` prefixes; keep the surrogate's features that exist."""
    renamed = frame.rename(columns={c: c.split("__", 1)[1] for c in frame.columns if "__" in c})
    keep = ["timestamp"] + [f for f in ALL_FEATURE_NAMES if f in renamed.columns]
    return renamed[keep].reset_index(drop=True)


def _real_shap(region: str, features: pd.DataFrame, rows: list[int]) -> tuple[list[dict], dict]:
    import joblib
    import shap

    bundle = joblib.load(_REAL_MODELS)
    columns = bundle["features"][region]
    X = features[columns].iloc[rows].to_numpy(dtype=float)
    values = shap.TreeExplainer(bundle["models"][region]).shap_values(X) if rows else np.empty((0, 0))
    metrics = json.loads(_REAL_METRICS.read_text(encoding="utf-8"))["results"][region]["metrics"]
    fit = {"r2_test": metrics["r2_test"], "spearman_test": metrics["spearman_test"],
           "r2_train": metrics["r2_train"]}
    return [dict(zip(columns, map(float, v))) for v in values], fit


def _synthetic_shap(config: dict, features: pd.DataFrame, rows: list[int]) -> tuple[list[dict], dict]:
    explainer = SurrogateShapExplainer()
    syn_features, syn_risk = build_shap_training_data(config)
    r2 = explainer.train_surrogate(syn_features, syn_risk)
    return [explainer.explain(features.iloc[[t]])["shap_values"] for t in rows], {"r2_synthetic_in_sample": r2}


def run_region(region: str, surrogate: str, out: Path) -> dict:
    frame, _ = harness._load(region)
    config = load_config_for_region(region)
    train, test = harness._split(frame)

    scores, used = harness.tier_scores(_TIER, config, train, test)
    train_scores, _ = harness.tier_scores(_TIER, config, train, train)
    threshold = float(np.quantile(np.nan_to_num(train_scores), 0.90))
    alert_rows = [len(train) + int(i) for i in np.flatnonzero(np.nan_to_num(scores) > threshold)]

    features = _features(frame)
    if surrogate == "real":
        shap_rows, fit = _real_shap(region, features, alert_rows)
    else:
        shap_rows, fit = _synthetic_shap(config, features, alert_rows)

    complete = 0
    for t, shap_values in zip(alert_rows, shap_rows):
        explanation = explain_alert(features, shap_values, t, region)
        write_explanation(explanation, out)
        complete += explanation["complete"]

    return {
        "region": region,
        "surrogate": surrogate,
        "agents": used,
        "alerts": len(alert_rows),
        "complete": complete,
        "coverage": complete / len(alert_rows) if alert_rows else None,
        "surrogate_fit": fit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--surrogate", choices=("real", "synthetic"), default="real")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    out = _OUTPUTS / ("explanations" if args.surrogate == "real" else "explanations_synthetic")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    results = [run_region(p.stem, args.surrogate, out) for p in sorted(harness._CACHE.glob("*.csv"))]
    (out / "coverage.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results:
        cov = "n/a" if r["coverage"] is None else f"{r['coverage']:.1%}"
        print(f"[{r['region']}] alerts={r['alerts']} complete={r['complete']} ({cov}) "
              f"fit={r['surrogate_fit']} agents={','.join(r['agents'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
