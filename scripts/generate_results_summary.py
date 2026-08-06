"""Generate an honest, thesis-ready results summary from aggregated data.

Every number in the output markdown is read from the CSVs produced by
``scripts/aggregate_all_results.py`` (run that first). Nothing here is
invented: metrics that were never computed (D1/D2 VUS, D11-D15
event/range/PA-F1 variants) are listed as such with a reason, and there
is no "full system" / pre-declared-target section, since no such run or
spec exists in this repo — see the Limitations section this script writes.

Run from project root, after ``scripts/aggregate_all_results.py``::

    python scripts/generate_results_summary.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_RESULTS_ROOT = _PROJECT_ROOT / "results"

_UNMEASURED_METRICS: list[tuple[str, str, str]] = [
    ("D1 VUS-PR", "tslearn not a project dependency", "pip install tslearn; implement in BaselineEvaluator"),
    ("D2 VUS-ROC", "tslearn not a project dependency", "pip install tslearn; implement in BaselineEvaluator"),
    ("D11 Event-based F1", "segment-level eval not implemented", "Tatbul et al. 2018-style range-overlap scoring"),
    ("D12 Range-based F1", "segment-level eval not implemented", "same as D11"),
    ("D13 Affiliation F1", "segment-level eval not implemented", "affiliation-based metric (Huet et al. 2022)"),
    ("D14 PA-F1", "point-adjust eval not implemented", "point-adjustment protocol (Kim et al. 2022 critique noted)"),
    ("D15 PA%K", "point-adjust eval not implemented", "PA%K variant of D14"),
    ("E1 Lead time / MTTD", "no onset-detection logic exists", "evaluator that finds first alarm relative to event onset_day"),
    ("A1-A6 Decision accuracy", "no decision-label ground truth beyond y_action_int", "extend R3 scenarios + annotate decision_labels.json"),
    ("C1-C4 Calibration", "not implemented", "Brier score / ECE computation in BaselineEvaluator"),
]


def load_summary_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_all = pd.read_csv(_RESULTS_ROOT / "results_all_runs.csv")
    df_by_baseline = pd.read_csv(_RESULTS_ROOT / "results_by_baseline.csv", index_col=0)
    df_by_scenario = pd.read_csv(_RESULTS_ROOT / "results_by_scenario.csv", index_col=0)
    df_ablations = pd.read_csv(_RESULTS_ROOT / "ablation_findings.csv", index_col=0)
    return df_all, df_by_baseline, df_by_scenario, df_ablations


def compute_key_findings(df_all: pd.DataFrame) -> dict:
    """Compute every value the narrative needs, straight from df_all."""
    findings: dict = {}
    df_baselines = df_all[df_all["config_id"].isna()]

    for metric in ("D3_auc_pr", "D6_best_f1", "D9_fpr_tau"):
        per_baseline = df_baselines.groupby("baseline_name")[metric].mean()
        if len(per_baseline) == 0:
            continue
        findings[f"best_{metric}_name"] = per_baseline.idxmax()
        findings[f"best_{metric}_val"] = per_baseline.max()

    # Per-baseline means, for the narrative bullets (every Tier 0-2 baseline,
    # not just the two headline ones).
    findings["per_baseline"] = df_baselines.groupby("baseline_name").agg(
        {"D3_auc_pr": "mean", "D6_best_f1": "mean", "D9_fpr_tau": "mean"}
    ).round(4)

    df_ablations = df_all[df_all["config_id"].notna()].copy()
    if len(df_ablations) > 0:
        df_ablations["scenario_type"] = df_ablations["scenario_id"].str.extract(
            r"(P_CRIT|P_HIGH|N_QUIET|N_DECOY)", expand=False
        )
        a5_decoy = df_ablations[
            (df_ablations["config_id"] == "A5") & (df_ablations["scenario_type"] == "N_DECOY")
        ]
        a6_decoy = df_ablations[
            (df_ablations["config_id"] == "A6") & (df_ablations["scenario_type"] == "N_DECOY")
        ]
        if len(a5_decoy) > 0:
            findings["a5_decoy_f1"] = a5_decoy["D6_best_f1"].mean()
            findings["a5_decoy_fpr"] = a5_decoy["D9_fpr_tau"].mean()
        if len(a6_decoy) > 0:
            findings["a6_decoy_f1"] = a6_decoy["D6_best_f1"].mean()
            findings["a6_decoy_fpr"] = a6_decoy["D9_fpr_tau"].mean()

    return findings


def _decoy_interpretation(findings: dict) -> str:
    """Interpret the A5-vs-A6 N-DECOY comparison from whatever was actually measured."""
    a5_fpr = findings.get("a5_decoy_fpr")
    a6_fpr = findings.get("a6_decoy_fpr")
    if a5_fpr is None or a6_fpr is None:
        return "No N-DECOY ablation results were found to compare."
    if np.isnan(a5_fpr) or np.isnan(a6_fpr):
        return (
            "FPR is undefined for at least one of A5/A6 on N-DECOY "
            "(no negative predictions ever made, or no evaluable days)."
        )
    if a6_fpr < a5_fpr:
        return (
            f"A6's FPR ({a6_fpr:.4f}) is lower than A5's ({a5_fpr:.4f}) on N-DECOY: "
            "the agreement bonus reduced false alarms on this benchmark, as predicted."
        )
    if a6_fpr == a5_fpr:
        return (
            f"A5 and A6 have identical FPR ({a6_fpr:.4f}) on N-DECOY. This scenario's "
            "test window (days 281-364) has zero positive days, so both the F1-optimal "
            "threshold search and the Optuna weight search degenerate to the same "
            "equal-weights fallback for A5 and A6 alike — the bonus condition (>=3 "
            "domains >= 0.65) was never met for either config in this window, so it "
            "had no opportunity to differ. This is a limitation of evaluating the bonus "
            "on a scenario whose test split contains no positive days, not evidence "
            "the bonus doesn't work."
        )
    return (
        f"A6's FPR ({a6_fpr:.4f}) is *higher* than A5's ({a5_fpr:.4f}) on N-DECOY — "
        "the opposite of the prediction. Worth investigating before citing this as a win."
    )


def _decoy_interpretation_short(findings: dict) -> str:
    """One-sentence version of the A5-vs-A6 N-DECOY comparison, for the Ch. 6 blurb."""
    a5_fpr = findings.get("a5_decoy_fpr")
    a6_fpr = findings.get("a6_decoy_fpr")
    if a5_fpr is None or a6_fpr is None or np.isnan(a5_fpr) or np.isnan(a6_fpr):
        return "No comparable A5/A6 N-DECOY result was available to test the agreement-bonus hypothesis."
    if a6_fpr < a5_fpr:
        return (
            f"The agreement bonus (A6) reduced FPR on N-DECOY from {a5_fpr:.4f} (A5) to "
            f"{a6_fpr:.4f}, as predicted."
        )
    if a6_fpr == a5_fpr:
        return (
            "A5 and A6 showed identical FPR on N-DECOY (Section 2) — that scenario's test "
            "window has no positive days, so the bonus condition was never evaluated there, "
            "not disproven."
        )
    return (
        f"A6's FPR on N-DECOY ({a6_fpr:.4f}) was higher than A5's ({a5_fpr:.4f}) — the "
        "opposite of the prediction (Section 2)."
    )


def generate_summary_md(
    findings: dict,
    df_by_baseline: pd.DataFrame,
    df_by_scenario: pd.DataFrame,
    df_ablations: pd.DataFrame,
) -> str:
    best_auc_pr = findings.get("best_D3_auc_pr_name", "N/A")
    best_auc_pr_val = findings.get("best_D3_auc_pr_val", np.nan)
    best_f1 = findings.get("best_D6_best_f1_name", "N/A")
    best_f1_val = findings.get("best_D6_best_f1_val", np.nan)

    per_baseline_bullets = "\n".join(
        f"- **{name}:** mean AUC-PR = {row['D3_auc_pr']:.4f}, "
        f"mean Best-F1 = {row['D6_best_f1']:.4f}, mean FPR@tau = {row['D9_fpr_tau']:.4f}"
        for name, row in findings["per_baseline"].iterrows()
    )

    unmeasured_rows = "\n".join(
        f"| {name} | {reason} | {requirement} |"
        for name, reason, requirement in _UNMEASURED_METRICS
    )

    decoy_note = _decoy_interpretation(findings)

    return f"""# Results Summary: EVAL01 Baseline & Ablation Benchmark

*Generated by `scripts/generate_results_summary.py` from
`results/results_all_runs.csv`. Every number below is read from that file
— nothing here is invented.*

## Executive Summary

This benchmark evaluates **baseline competitiveness** (Tiers 0-2: 3
controls + 5 statistical/SPC + 2 classical unsupervised, each over 4
Hormuz scenarios x 5 seeds) and **aggregation strategy value** (ablations
A0-A7, using lightweight domain-scorer proxies, 4 scenarios x seed=42
only — see `docs/ABLATION_RATIONALE.md`).

**Measured:** D3-D10 detection metrics, aggregation-strategy comparisons,
false-alarm robustness on the N-DECOY scenario class.
**Not measured:** production-agent performance, full-system integration
(SHAP/RAG), real-world validation. See Section 3.

---

## 1. Baseline Results (Tiers 0-2)

Best mean AUC-PR: **{best_auc_pr}** ({best_auc_pr_val:.4f})
Best mean Best-F1: **{best_f1}** ({best_f1_val:.4f})

### By baseline (mean over 4 scenarios x 5 seeds = 20 runs each)

{per_baseline_bullets}

Full table with std: `results/results_by_baseline.csv`.

### By scenario type (mean Best-F1)

{df_by_scenario.to_markdown()}

Positive scenarios (P_CRIT, P_HIGH) generally separate more cleanly than
negative ones (N_QUIET, N_DECOY) — expected, since the negative classes'
test windows (days 281-364) carry zero positive labels by construction
(see R1's scenario design), so precision/recall/F1 there are dominated by
false-positive behavior rather than true detection.

---

## 2. Ablation Results (A0-A7, seed=42 only)

{df_ablations.to_markdown()}

Full per-scenario detail: `results/ablation_findings.csv` and
`results/baselines/ablations/*.json`.

### Agreement Bonus on N-DECOY (A5 vs A6)

- A5 (no bonus): Best-F1 = {findings.get('a5_decoy_f1', float('nan')):.4f}, FPR = {findings.get('a5_decoy_fpr', float('nan')):.4f}
- A6 (+bonus): Best-F1 = {findings.get('a6_decoy_f1', float('nan')):.4f}, FPR = {findings.get('a6_decoy_fpr', float('nan')):.4f}

{decoy_note}

---

## 3. Honest Limitations

### Metrics not measured in this benchmark

| Metric | Why not | Would require |
|---|---|---|
{unmeasured_rows}

### What this benchmark does not test

1. **Production-agent performance.** Ablations use lightweight domain
   scorers (rolling z-score per domain), not the production
   `ShippingAgent`/`MarketAgent`/etc., which need a multi-column schema
   R3's synthetic generator doesn't produce. See `docs/ABLATION_RATIONALE.md`.
2. **Full-system integration.** No run here combines detection with SHAP
   explanation, RAG retrieval, or decision recommendation — those are
   post-detection in the production pipeline and untouched by this
   detection-focused benchmark.
3. **Real-world validation.** EVAL01 scenarios are synthetic
   (R1-R3), grounded in documented event shapes but not tested against
   live disruption data.
4. **Operational performance.** No measurement of alert fatigue, cost-
   weighted decision quality, or deployment latency.

There is no pre-declared-target comparison table in this summary: no
metric-specification document with numeric targets exists anywhere in
this repository to compare against, so none is fabricated here.

---

## 4. For Thesis Chapter 6

This section can be adapted directly:

> To understand baseline competitiveness and the value added by
> aggregation strategy, we evaluated {len(findings['per_baseline'])}
> baseline detectors (Tiers 0-2) and ablated domain weighting and
> consensus logic (A0-A7) on a controlled benchmark
> (`results/results_by_baseline.csv`, `results/ablation_findings.csv`).
>
> {best_auc_pr} achieved the highest mean AUC-PR ({best_auc_pr_val:.4f})
> and {best_f1} the highest mean Best-F1 ({best_f1_val:.4f}) among Tier
> 0-2 baselines. {_decoy_interpretation_short(findings)}
>
> **Limitations:** this benchmark evaluates detection and aggregation
> strategy only, using domain-scorer proxies rather than the production
> agents; full-system integration (SHAP + RAG) and real-world validation
> are out of scope here (Section 3).

---

## References

- `docs/ABLATION_RATIONALE.md` — why domain scorers, not production agents, for ablations A0-A7.
"""


def main() -> None:
    logger.info("Loading aggregated results...")
    df_all, df_by_baseline, df_by_scenario, df_ablations = load_summary_data()

    logger.info("Computing findings from measured data...")
    findings = compute_key_findings(df_all)

    logger.info("Generating markdown...")
    summary_md = generate_summary_md(findings, df_by_baseline, df_by_scenario, df_ablations)

    output_file = _RESULTS_ROOT / "benchmark_summary.md"
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(summary_md)

    logger.info("Saved summary to %s", output_file)
    print(summary_md)


if __name__ == "__main__":
    main()
