"""Phase 5 (SYNTHETIC, seed 44): METRIC 8 refresh and P0-surrogate vs P0-exact.

Implements ``PREREGISTERED_RULES.md`` §D (commit ``f0f2253``). Uses the
synthetic train/validation/test realisations (seeds 42/43/44), never the real
evaluation data, and makes no API call.

* **Refresh:** ``notebooks/evaluation.py::metric_8_decision``, unchanged, for
  both weight modes, next to the stale 2026-07-30 numbers.
* **Comparison:** the stock day records with ``top_agent`` taken either from
  the production SHAP surrogate on the *seed-42* row (what the dashboard does)
  or from exact Path C Shapley values of the *seed-44* day itself.

Writes ``eval/explainability/metric8_refresh.json``.

Usage::

    python eval/explainability/run_metric8.py
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import logging
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

from src.evaluation.decision_effectiveness import ACTIONS, predict_action  # noqa: E402
from src.explainability.baselines import synthetic_baselines  # noqa: E402
from src.explainability.exact_shapley import shapley_agents, value_fn_path_c  # noqa: E402

_OUT = _ROOT / "eval" / "explainability"
_STALE = _ROOT / "data" / "processed" / "evaluation_results.json"


def _notebook():
    spec = importlib.util.spec_from_file_location("evaluation_nb", _ROOT / "notebooks" / "evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision_scores(records: list[dict]) -> dict:
    """Accuracy, macro-F1 and per-class recall of predict_action on day records."""
    from sklearn.metrics import f1_score, recall_score

    y = [r["correct_action"] for r in records]
    pred = [predict_action(r["risk_level"], r["top_drivers"], r["historical_context"],
                           sustained=r["sustained"]) for r in records]
    labels = [a for a in ACTIONS if a in set(y) | set(pred)]
    recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(np.mean([a == b for a, b in zip(y, pred)])),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "recall": {lab: float(r) for lab, r in zip(labels, recall)},
        "support": {lab: int(sum(1 for v in y if v == lab)) for lab in labels},
        "correct": [a == b for a, b in zip(y, pred)],
        "predicted": pred,
    }


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Two-sided exact McNemar test on discordant pairs."""
    from scipy.stats import binomtest

    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_right_b_wrong": b, "b_right_a_wrong": c, "p_two_sided": p}


def surrogate_top_agents(config: dict, n_days: int) -> list[str]:
    """The dashboard's driver: surrogate explained on seed-42 row i for test day i."""
    from src.explainability.shap_explainer import SurrogateShapExplainer, build_shap_training_data

    features_df, risk = build_shap_training_data(config)
    explainer = SurrogateShapExplainer()
    explainer.train_surrogate(features_df, risk)
    return [explainer.explain(features_df.iloc[[min(i, len(features_df) - 1)]])["top_drivers"][0]["agent"]
            for i in range(n_days)]


def exact_top_agents(scores_df: pd.DataFrame, params: dict, baselines: dict) -> list[str]:
    """Largest exact Path C Shapley value per day (evaluator aggregation)."""
    inter, thr = params["inter_weights"], params["thresholds"]
    agents = [a for a in scores_df.columns if inter.get(a, 0.0) > 0]
    value_fn = value_fn_path_c(inter, agents, float(thr.get("agreement_bonus_3", 1.15)),
                               float(thr.get("agreement_bonus_5", 1.25)))
    filled = scores_df[agents].fillna(0.0)
    tops = []
    for _, row in filled.iterrows():
        phi = shapley_agents({a: float(row[a]) for a in agents}, baselines, value_fn)
        tops.append(max(phi, key=phi.get))
    return tops


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    t0 = time.time()
    nb = _notebook()
    import yaml

    from src.optimization.data_split import DataSplitManager
    from src.optimization.pipeline_evaluator import PipelineEvaluator
    from src.optimization.weight_config import load_optimized_weights, resolve_active_weights

    config = yaml.safe_load(nb._CONFIG_PATH.read_text(encoding="utf-8"))
    modes = {"hand_tuned": nb._layout_to_params(
        resolve_active_weights({**config, "weight_mode": "hand_tuned"}))}
    opt = load_optimized_weights(config)
    modes["optimized"] = nb._layout_to_params(opt) if opt and opt.get("inter_agent_weights") \
        else copy.deepcopy(modes["hand_tuned"])
    manager = DataSplitManager(config)
    manager.get_splits()
    evaluator = PipelineEvaluator(manager, config.get("optimization", {}).get("objective_weights"))

    with contextlib.redirect_stdout(io.StringIO()):
        refresh = nb.metric_8_decision(evaluator, modes)
    stale = json.loads(_STALE.read_text(encoding="utf-8")).get("metric_8_decision_effectiveness", {}) \
        if _STALE.exists() else {}

    sur_cache: list[str] | None = None
    out = {"data_origin": "synthetic_seed44", "label": "SYNTHETIC",
           "preregistration": "f0f2253", "modes": {}}
    for mode, params in modes.items():
        records = nb._build_day_records(evaluator, params, "train", "test")
        scores_df = pd.DataFrame(nb._score_series(evaluator, params, "train", "test")) \
            .sort_index().reset_index(drop=True)
        baselines = synthetic_baselines(config, params, manager)
        exact = exact_top_agents(scores_df, params, baselines)
        if sur_cache is None:
            sur_cache = surrogate_top_agents(config, len(records))
        variants = {"stock (weight x score)": records}
        for name, tops in (("P0-surrogate", sur_cache), ("P0-exact", exact)):
            variants[name] = [{**r, "top_drivers": [{"agent": a}]} for r, a in zip(records, tops)]
        scored = {name: decision_scores(recs) for name, recs in variants.items()}
        high_days = [i for i, r in enumerate(records) if r["risk_level"] == "high" and not r["sustained"]]
        out["modes"][mode] = {
            "refresh_metric8": {k: v for k, v in refresh[mode].items() if k != "confusion_matrix"}
                               | {"confusion_matrix": refresh[mode]["confusion_matrix"]},
            "stale_2026_07_30": {k: stale.get(mode, {}).get(k) for k in
                                 ("overall_accuracy", "per_case_accuracy", "per_scenario_accuracy",
                                  "baseline_accuracy", "action_distribution")},
            "variants": {name: {k: v for k, v in s.items() if k not in ("correct", "predicted")}
                         for name, s in scored.items()},
            "comparison_P0_exact_vs_P0_surrogate": {
                **mcnemar_exact(scored["P0-exact"]["correct"], scored["P0-surrogate"]["correct"]),
                "days_top_agent_differs": int(sum(a != b for a, b in zip(sur_cache, exact))),
                "days_action_differs": int(sum(a != b for a, b in zip(
                    scored["P0-surrogate"]["predicted"], scored["P0-exact"]["predicted"]))),
                "high_non_sustained_days": len(high_days),
                "top_agent_counts_surrogate": pd.Series(sur_cache).value_counts().to_dict(),
                "top_agent_counts_exact": pd.Series(exact).value_counts().to_dict(),
            },
            "baselines_seed42": baselines,
        }
    (_OUT / "metric8_refresh.json").write_text(json.dumps(out, indent=2, default=str),
                                               encoding="utf-8")
    print(f"wrote metric8_refresh.json in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
