"""Fit the SHAP surrogate on real corridor data, per region and pooled.

Re-runnable pipeline. For each region:

1. Load the real frame (2019-01-01 to 2025-08-30) from ``data/eval/<region>.csv``,
   the api-mode cache built by ``scripts/build_eval_dataset.py``. Only if no
   cache exists is the existing connector pipeline called, and its output saved
   to ``data/cache/<region>.csv``.
2. Split temporally: first 70 % of rows train, last 30 % test.
3. Score every agent exactly as the evaluation harness does (fit on the
   training window, score each window) and pass the scores through
   ``RiskEngine.compute_risk`` — the target and its per-agent breakdown.
4. Fit ``RandomForestRegressor(n_estimators=100, random_state=42)`` on the 17
   canonical features and evaluate on the test window.

Missing features: a feature the region does not supply is **dropped** from that
region's surrogate; the pooled surrogate keeps every feature present in at least
one region and leaves it **NaN** where a region lacks it. Features absent from
every region are dropped from both. Nothing is zero-filled.

The synthetic-trained surrogate is evaluated on the same test windows, unchanged
(including its zero-fill of absent features), and written to separate files.

Outputs (``outputs/surrogate/``): ``real_surrogate_metrics.json``,
``real_surrogate_metrics.md``, ``real_rf.joblib``,
``synthetic_surrogate_metrics.json``, ``synthetic_rf.joblib``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import run_method_comparison as harness  # noqa: E402

from src.core.config_manager import load_config_for_region  # noqa: E402
from src.explainability.real_surrogate import (  # noqa: E402
    canonical_features,
    composite_from_agent_scores,
    evaluate_surrogate,
    fit_surrogate,
    temporal_split,
)
from src.explainability.shap_explainer import (  # noqa: E402
    ALL_FEATURE_NAMES,
    SurrogateShapExplainer,
    build_shap_training_data,
)

_EVAL_CACHE = _PROJECT_ROOT / "data" / "eval"
_FALLBACK_CACHE = _PROJECT_ROOT / "data" / "cache"
_OUT = _PROJECT_ROOT / "outputs" / "surrogate"
REGIONS = ("hormuz", "bab_el_mandeb", "panama", "malacca")
MISSING_POLICY = (
    "per-region: features the region does not supply are dropped; pooled: features present in "
    "at least one region are kept and left NaN where absent (RandomForest native NaN handling); "
    "features absent from every region are dropped. No zero-filling, no scaler, no imputer."
)


def load_region_frame(region: str) -> tuple[pd.DataFrame, str]:
    """Real frame from cache; the connector pipeline only when no cache exists."""
    for path in (_EVAL_CACHE / f"{region}.csv", _FALLBACK_CACHE / f"{region}.csv"):
        if path.exists():
            return pd.read_csv(path, parse_dates=["timestamp"]), str(path.relative_to(_PROJECT_ROOT))
    import build_eval_dataset

    frame, manifest = build_eval_dataset.build_region(region)
    _FALLBACK_CACHE.mkdir(parents=True, exist_ok=True)
    path = _FALLBACK_CACHE / f"{region}.csv"
    frame.to_csv(path, index=False)
    (_FALLBACK_CACHE / f"{region}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return frame, str(path.relative_to(_PROJECT_ROOT))


def agent_scores(config: dict, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-day agent scores over train then test, as the evaluation harness computes them."""
    out: dict[str, np.ndarray] = {}
    for domain in harness._TIER_ORDER:
        on_train = harness._agent_scores(domain, config, train, train)
        on_test = harness._agent_scores(domain, config, train, test)
        if on_train is None or on_test is None:
            continue
        out[domain] = np.concatenate([on_train, on_test])
    return out


def prepare_region(region: str) -> dict:
    frame, source = load_region_frame(region)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    config = load_config_for_region(region)
    train, test = temporal_split(frame)

    scores = agent_scores(config, train, test)
    target = composite_from_agent_scores(scores, config)
    features, missing = canonical_features(frame)
    flagged = (target["risk_level"] != "low").to_numpy()

    n_train = len(train)
    return {
        "region": region,
        "source": source,
        "config": config,
        "timestamps": frame["timestamp"],
        "n_train": n_train,
        "features": features,
        "missing": missing,
        "y": target["risk_score"].to_numpy(dtype=float),
        "flagged": flagged,
        "top_agent": target["top_agent"].to_numpy(dtype=object),
        "agents": list(scores),
        "train_dates": [str(frame["timestamp"].iloc[0].date()), str(frame["timestamp"].iloc[n_train - 1].date())],
        "test_dates": [str(frame["timestamp"].iloc[n_train].date()), str(frame["timestamp"].iloc[-1].date())],
    }


def _parts(d: dict, columns: list[str]):
    n = d["n_train"]
    X = d["features"][columns]
    return (X.iloc[:n], d["y"][:n], X.iloc[n:], d["y"][n:], d["flagged"][n:], d["top_agent"][n:])


def fit_and_evaluate(d: dict, columns: list[str]) -> tuple[object, dict]:
    X_tr, y_tr, X_te, y_te, flagged, top = _parts(d, columns)
    model = fit_surrogate(X_tr, y_tr)
    return model, evaluate_surrogate(model, X_tr, y_tr, X_te, y_te, flagged, top)


def pooled(data: list[dict], columns: list[str]) -> tuple[object, dict, dict]:
    parts = [_parts(d, columns) for d in data]
    X_tr = pd.concat([p[0] for p in parts], ignore_index=True)
    y_tr = np.concatenate([p[1] for p in parts])
    X_te = pd.concat([p[2] for p in parts], ignore_index=True)
    y_te = np.concatenate([p[3] for p in parts])
    flagged = np.concatenate([p[4] for p in parts])
    top = np.concatenate([p[5] for p in parts])
    model = fit_surrogate(X_tr, y_tr)
    metrics = evaluate_surrogate(model, X_tr, y_tr, X_te, y_te, flagged, top)
    split = {
        "train_max_date": max(d["train_dates"][1] for d in data),
        "test_min_date": min(d["test_dates"][0] for d in data),
    }
    return model, metrics, split


def synthetic_comparison(d: dict) -> tuple[object, dict]:
    """The deployed synthetic surrogate on this region's real windows, as deployed."""
    explainer = SurrogateShapExplainer()
    syn_X, syn_y = build_shap_training_data(d["config"])
    r2_synthetic = explainer.train_surrogate(syn_X, syn_y)
    deployed = d["features"].fillna(0.0)  # the deployed _align_features behaviour
    n = d["n_train"]
    metrics = evaluate_surrogate(
        explainer._rf, deployed.iloc[:n], d["y"][:n], deployed.iloc[n:], d["y"][n:],
        d["flagged"][n:], d["top_agent"][n:],
    )
    metrics["r2_synthetic_in_sample"] = r2_synthetic
    metrics["note"] = ("r2_train here is on the real training window; the surrogate was fitted on the "
                       "synthetic series, whose in-sample R² is r2_synthetic_in_sample.")
    return explainer._rf, metrics


def _fmt(v, spec=".3f") -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def markdown_table(results: dict) -> str:
    head = ("| Model | Features | R² train | R² test | MAE test | Spearman test | "
            "Test days ≥ medium | MAE on those | Top-1 agent agreement | Top-3 agent agreement |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for name, r in results.items():
        m = r["metrics"]
        share = m["share_test_medium_or_above"]
        rows.append(
            f"| {name} | {len(r['features_used'])} | {_fmt(m['r2_train'])} | {_fmt(m['r2_test'])} | "
            f"{_fmt(m['mae_test'], '.4f')} | {_fmt(m['spearman_test'])} | "
            f"{_fmt(share, '.1%')} ({m['n_test_medium_or_above']}) | {_fmt(m['mae_test_medium_or_above'], '.4f')} | "
            f"{_fmt(m['top1_agent_agreement'], '.1%')} | {_fmt(m['top3_agent_agreement'], '.1%')} |"
        )
    return head + "\n".join(rows) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    _OUT.mkdir(parents=True, exist_ok=True)
    import joblib

    data = [prepare_region(r) for r in REGIONS]

    results: dict[str, dict] = {}
    models: dict[str, object] = {}
    for d in data:
        columns = [f for f in ALL_FEATURE_NAMES if f not in d["missing"]]
        model, metrics = fit_and_evaluate(d, columns)
        models[d["region"]] = model
        results[d["region"]] = {
            "metrics": metrics,
            "features_used": columns,
            "features_dropped": d["missing"],
            "agents_in_target": d["agents"],
            "source": d["source"],
            "train_dates": d["train_dates"],
            "test_dates": d["test_dates"],
            "medium_threshold": float(d["config"]["thresholds"]["risk_medium"]),
        }

    everywhere_missing = set.intersection(*(set(d["missing"]) for d in data))
    pooled_columns = [f for f in ALL_FEATURE_NAMES if f not in everywhere_missing]
    model, metrics, split = pooled(data, pooled_columns)
    models["pooled"] = model
    results["pooled"] = {
        "metrics": metrics,
        "features_used": pooled_columns,
        "features_dropped": sorted(everywhere_missing),
        "features_nan_in": {d["region"]: [f for f in pooled_columns if f in d["missing"]] for d in data},
        **split,
    }

    synthetic: dict[str, dict] = {}
    synthetic_models: dict[str, object] = {}
    for d in data:
        rf, m = synthetic_comparison(d)
        synthetic_models[d["region"]] = rf
        synthetic[d["region"]] = {"metrics": m, "features_used": list(ALL_FEATURE_NAMES)}

    payload = {
        "target": "RiskEngine.compute_risk risk_score (weighted mean over active agents, agreement-amplified)",
        "model": "RandomForestRegressor(n_estimators=100, random_state=42)",
        "split": "temporal per region: first 70% of rows train, last 30% test; no shuffling",
        "missing_feature_policy": MISSING_POLICY,
        "flagged_definition": "test days with RiskEngine.classify_risk >= medium (risk_score >= risk_medium)",
        "fidelity_mapping": "FEATURE_AGENT_MAP (thesis Table 12): top-1 / any of top-3 |SHAP| features' agent "
                            "== agent with the largest pre-amplification contribution",
        "results": results,
    }
    (_OUT / "real_surrogate_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (_OUT / "synthetic_surrogate_metrics.json").write_text(json.dumps(
        {"model": "deployed synthetic surrogate (build_shap_training_data + train_surrogate), "
                  "evaluated on the same real windows with absent features zero-filled as deployed",
         "results": synthetic}, indent=2), encoding="utf-8")
    joblib.dump({"models": models, "features": {k: v["features_used"] for k, v in results.items()},
                 "missing_feature_policy": MISSING_POLICY}, _OUT / "real_rf.joblib", compress=3)
    joblib.dump({"models": synthetic_models, "features": list(ALL_FEATURE_NAMES)},
                _OUT / "synthetic_rf.joblib", compress=3)

    table = markdown_table(results)
    syn_table = markdown_table(synthetic)
    md = ("# Real-data SHAP surrogate\n\n"
          f"Target: {payload['target']}. Model: {payload['model']}. Split: {payload['split']}.\n\n"
          f"Missing features: {MISSING_POLICY}\n\n" + table +
          "\n## For comparison: the deployed synthetic-trained surrogate on the same real windows\n\n" + syn_table)
    (_OUT / "real_surrogate_metrics.md").write_text(md, encoding="utf-8")

    print("\nReal-data surrogate\n" + table)
    print("Synthetic-trained surrogate (comparison)\n" + syn_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
