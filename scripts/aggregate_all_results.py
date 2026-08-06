"""Aggregate all baseline (Tier 0-2) and ablation (A0-A7) results into comprehensive tables.

Run from project root, after the R4/R5/R6/R7 result-generating scripts::

    python scripts/aggregate_all_results.py
"""

from __future__ import annotations

import json
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
_TIER0_DIR = _RESULTS_ROOT / "baselines" / "tier0"
_TIER1_DIR = _RESULTS_ROOT / "baselines" / "tier1"
_TIER2_DIR = _RESULTS_ROOT / "baselines" / "tier2"
_ABLATIONS_DIR = _RESULTS_ROOT / "baselines" / "ablations"

# D3-D10 are actually computed by BaselineEvaluator (see R4). D1/D2/D11-D15
# are declared NaN placeholders there (D1/D2 need tslearn, not a project
# dependency; D11-D15 need segment-level eval logic not yet implemented) --
# carried through here rather than silently dropped, so every result row
# has the full metric schema and it's visible in the CSVs which columns
# are genuinely unmeasured.
_MEASURED_METRICS: tuple[str, ...] = (
    "D3_auc_pr", "D4_auc_roc", "D5_f1_tau", "D6_best_f1",
    "D7_precision_tau", "D8_recall_tau", "D9_fpr_tau", "D10_macro_f1",
)
_UNMEASURED_METRICS: tuple[str, ...] = (
    "D1_vus_pr", "D2_vus_roc", "D11_event_f1", "D12_range_f1",
    "D13_affiliation_f1", "D14_pa_f1", "D15_pa_k_0_2",
)


def load_results(results_dir: Path, pattern: str = "*.json") -> list[dict]:
    """Load all JSON results matching ``pattern`` under ``results_dir``."""
    results = []
    for json_file in sorted(results_dir.glob(pattern)):
        try:
            with open(json_file, encoding="utf-8") as fh:
                results.append(json.load(fh))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", json_file, exc)
    return results


def extract_baseline_name(result: dict) -> str:
    """Extract a display name from either a Tier 0-2 or an ablation result.

    Tier 0-2 results carry a top-level ``baseline_name``. Ablation results
    (from ``scripts/run_ablations.py``) instead carry ``ablation_config``
    at the top level and the human-readable ``ablation_name`` nested in
    ``metadata`` — there is no top-level ``ablation_name`` key.
    """
    if "baseline_name" in result:
        return result["baseline_name"]
    if "ablation_config" in result:
        return result.get("metadata", {}).get(
            "ablation_name", result.get("ablation_config", "unknown")
        )
    return "unknown"


def aggregate_results(results_list: list[dict]) -> pd.DataFrame:
    """Flatten loaded results into one row per scenario x baseline x seed."""
    rows = []
    for result in results_list:
        row = {
            "scenario_id": result.get("scenario_id", "unknown"),
            "baseline_name": extract_baseline_name(result),
            "config_id": result.get("ablation_config"),  # None for Tier 0-2
            "seed": result.get("seed", "unknown"),
        }
        metrics = result.get("metrics", {})
        for metric_id in _MEASURED_METRICS:
            row[metric_id] = metrics.get(metric_id, np.nan)
        for metric_id in _UNMEASURED_METRICS:
            row[metric_id] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/count per Tier 0-2 baseline across seeds and scenarios."""
    df_baselines = df[df["config_id"].isna()].copy()
    summary = df_baselines.groupby("baseline_name").agg(
        {
            "D3_auc_pr": ["mean", "std", "count"],
            "D4_auc_roc": ["mean", "std"],
            "D5_f1_tau": ["mean", "std"],
            "D6_best_f1": ["mean", "std"],
            "D7_precision_tau": ["mean", "std"],
            "D8_recall_tau": ["mean", "std"],
            "D9_fpr_tau": ["mean", "std"],
            "D10_macro_f1": ["mean", "std"],
        }
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary.round(4)


def summarize_by_scenario_type(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Best-F1 per Tier 0-2 baseline per scenario type (P_CRIT, etc.)."""
    df = df.copy()
    df["scenario_type"] = df["scenario_id"].str.extract(
        r"(P_CRIT|P_HIGH|N_QUIET|N_DECOY)", expand=False
    )
    df_baselines = df[df["config_id"].isna()]
    pivot = df_baselines.pivot_table(
        index="baseline_name", columns="scenario_type", values="D6_best_f1", aggfunc="mean"
    )
    return pivot.round(4)


def summarize_ablations(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per ablation config (A0-A7, seed=42 only)."""
    df_ablations = df[df["config_id"].notna()].copy()
    summary = df_ablations.groupby("config_id").agg(
        {
            "D3_auc_pr": ["mean"],
            "D6_best_f1": ["mean"],
            "D8_recall_tau": ["mean"],
            "D9_fpr_tau": ["mean"],
        }
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary.round(4)


def main() -> None:
    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading results from all tiers...")
    tier0_results = load_results(_TIER0_DIR)
    tier1_results = load_results(_TIER1_DIR)
    tier2_results = load_results(_TIER2_DIR)
    ablation_results = load_results(_ABLATIONS_DIR)
    logger.info(
        "Loaded: %d Tier0, %d Tier1, %d Tier2, %d ablations",
        len(tier0_results), len(tier1_results), len(tier2_results), len(ablation_results),
    )

    all_results = tier0_results + tier1_results + tier2_results + ablation_results
    if not all_results:
        logger.warning(
            "No results found under %s — run the R4-R7 result-generating scripts first.",
            _RESULTS_ROOT,
        )
        return

    df_all = aggregate_results(all_results)
    summary_by_baseline = summarize_by_baseline(df_all)
    summary_by_scenario = summarize_by_scenario_type(df_all)
    summary_ablations = summarize_ablations(df_all)

    df_all.to_csv(_RESULTS_ROOT / "results_all_runs.csv", index=False)
    logger.info("Saved results_all_runs.csv (%d rows)", len(df_all))

    summary_by_baseline.to_csv(_RESULTS_ROOT / "results_by_baseline.csv")
    logger.info("Saved results_by_baseline.csv")

    summary_by_scenario.to_csv(_RESULTS_ROOT / "results_by_scenario.csv")
    logger.info("Saved results_by_scenario.csv")

    summary_ablations.to_csv(_RESULTS_ROOT / "ablation_findings.csv")
    logger.info("Saved ablation_findings.csv")

    print("\n" + "=" * 120)
    print("BASELINE SUMMARY (mean +/- std over seeds/scenarios, Tiers 0-2 only)")
    print("=" * 120)
    print(summary_by_baseline.to_string())

    print("\n" + "=" * 120)
    print("BEST-F1 BY SCENARIO TYPE (mean over seeds)")
    print("=" * 120)
    print(summary_by_scenario.to_string())

    print("\n" + "=" * 120)
    print("ABLATION SUMMARY (A0-A7, seed=42 only)")
    print("=" * 120)
    print(summary_ablations.to_string())


if __name__ == "__main__":
    main()
