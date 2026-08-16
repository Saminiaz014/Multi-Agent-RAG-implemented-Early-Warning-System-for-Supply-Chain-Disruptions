"""Aggregate all baseline (Tier 0-2) and ablation (A0-A7) results into comprehensive tables.

Run from project root, after the R4/R5/R6/R7 result-generating scripts::

    python scripts/aggregate_all_results.py [--region REGION]

``--region`` defaults to ``hormuz`` (matching every R4-R7 runner script). It
filters every results directory to only that region's result files, so a
second region's results on disk never silently blend into this region's
tables — this script and ``aggregate_ablation_results.py`` were found to be
the two remaining unfiltered aggregation entry points (2026-08-14, during
A6), after the four R4-R7 runner scripts had already been fixed; see
docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md §6 for the full writeup.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.benchmark.regions import resolve_region_key  # noqa: E402

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


def load_results(results_dir: Path, region: str, pattern: str = "*.json") -> list[dict]:
    """Load ``region``'s JSON results matching ``pattern`` under ``results_dir``.

    Args:
        results_dir: Directory to glob.
        region: Canonical region key (already resolved via
            :func:`~src.benchmark.regions.resolve_region_key`) — only files
            whose name starts with ``f"{region}_"`` are loaded, so a second
            region's results on disk never silently blend into this call's
            output.
        pattern: Glob pattern, applied before the region filter.
    """
    results = []
    for json_file in sorted(results_dir.glob(pattern)):
        if not json_file.stem.startswith(f"{region}_"):
            continue
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


def main(region: str = "hormuz") -> None:
    """Aggregate ``region``'s Tier 0-2 + ablation results into summary tables.

    Args:
        region: Canonical region key, alias, or display name (see
            :func:`src.benchmark.regions.resolve_region_key`). Defaults to
            ``"hormuz"``, matching every R4-R7 runner script.
    """
    region = resolve_region_key(region)
    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading region=%s results from all tiers...", region)
    tier0_results = load_results(_TIER0_DIR, region)
    tier1_results = load_results(_TIER1_DIR, region)
    tier2_results = load_results(_TIER2_DIR, region)
    ablation_results = load_results(_ABLATIONS_DIR, region)
    logger.info(
        "Loaded: %d Tier0, %d Tier1, %d Tier2, %d ablations",
        len(tier0_results), len(tier1_results), len(tier2_results), len(ablation_results),
    )

    all_results = tier0_results + tier1_results + tier2_results + ablation_results
    if not all_results:
        logger.warning(
            "No results found for region %r under %s — run the R4-R7 "
            "result-generating scripts with --region %s first.",
            region, _RESULTS_ROOT, region,
        )
        return

    df_all = aggregate_results(all_results)
    summary_by_baseline = summarize_by_baseline(df_all)
    summary_by_scenario = summarize_by_scenario_type(df_all)
    summary_ablations = summarize_ablations(df_all)

    df_all.to_csv(_RESULTS_ROOT / f"results_all_runs_{region}.csv", index=False)
    logger.info("Saved results_all_runs_%s.csv (%d rows)", region, len(df_all))

    summary_by_baseline.to_csv(_RESULTS_ROOT / f"results_by_baseline_{region}.csv")
    logger.info("Saved results_by_baseline_%s.csv", region)

    summary_by_scenario.to_csv(_RESULTS_ROOT / f"results_by_scenario_{region}.csv")
    logger.info("Saved results_by_scenario_%s.csv", region)

    summary_ablations.to_csv(_RESULTS_ROOT / f"ablation_findings_{region}.csv")
    logger.info("Saved ablation_findings_%s.csv", region)

    print("\n" + "=" * 120)
    print(f"BASELINE SUMMARY (region={region}, mean +/- std over seeds/scenarios, Tiers 0-2 only)")
    print("=" * 120)
    print(summary_by_baseline.to_string())

    print("\n" + "=" * 120)
    print(f"BEST-F1 BY SCENARIO TYPE (region={region}, mean over seeds)")
    print("=" * 120)
    print(summary_by_scenario.to_string())

    print("\n" + "=" * 120)
    print(f"ABLATION SUMMARY (region={region}, A0-A7, seed=42 only)")
    print("=" * 120)
    print(summary_ablations.to_string())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all baseline (Tier 0-2) and ablation (A0-A7) "
        "results into comprehensive tables for a region.",
    )
    parser.add_argument(
        "--region",
        default="hormuz",
        help="Region key, alias, or display name to aggregate (default: hormuz). "
        "Only result files for this region are read, so runs never silently "
        "mix regions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().region)
