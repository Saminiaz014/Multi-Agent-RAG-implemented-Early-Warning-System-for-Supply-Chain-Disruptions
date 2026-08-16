"""Aggregate ablation results (R7) into a summary table.

Run from project root, after ``scripts/run_ablations.py``::

    python scripts/aggregate_ablation_results.py [--region REGION]

``--region`` defaults to ``hormuz`` (matching every R4-R7 runner script). It
filters ``results/baselines/ablations/`` to only that region's result files,
so a second region's ablation results on disk never silently blend into this
region's summary — this script and ``aggregate_all_results.py`` were found
to be the two remaining unfiltered aggregation entry points (2026-08-14,
during A6), after the four R4-R7 runner scripts had already been fixed; see
docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md §6 for the full writeup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.benchmark.regions import resolve_region_key  # noqa: E402

_RESULTS_DIR = _PROJECT_ROOT / "results" / "baselines" / "ablations"
_METRIC_COLS: tuple[str, ...] = (
    "D3_auc_pr", "D4_auc_roc", "D5_f1_tau", "D6_best_f1", "D8_recall_tau", "D9_fpr_tau",
)


def aggregate_ablations(region: str = "hormuz") -> None:
    """Load ``region``'s ablation result JSONs, compute mean over scenarios, print + save.

    Args:
        region: Canonical region key, alias, or display name (see
            :func:`src.benchmark.regions.resolve_region_key`). Defaults to
            ``"hormuz"``, matching every R4-R7 runner script.
    """
    region = resolve_region_key(region)
    json_files = sorted(
        p for p in _RESULTS_DIR.glob("*.json")
        if p.stem.startswith(f"{region}_")
    )
    if not json_files:
        print(
            f"No ablation results found for region {region!r} in {_RESULTS_DIR}. "
            f"Run scripts/run_ablations.py --region {region} first."
        )
        return

    rows = []
    for json_file in json_files:
        with open(json_file, encoding="utf-8") as fh:
            result = json.load(fh)
        row = {
            "ablation_config": result["ablation_config"],
            "ablation_name": result["metadata"]["ablation_name"],
            "scenario_id": result["scenario_id"],
        }
        row.update({col: result["metrics"].get(col, np.nan) for col in _METRIC_COLS})
        rows.append(row)

    df_results = pd.DataFrame(rows)
    n_scenarios = df_results["scenario_id"].nunique()

    summary = df_results.groupby("ablation_config")[list(_METRIC_COLS)].mean().round(4)

    print("\n" + "=" * 90)
    print(f"ABLATION SUMMARY (region={region}, mean over {n_scenarios} scenarios, seed=42)")
    print("=" * 90)
    print(summary.to_string())
    print("=" * 90)

    print("\nBest-F1 by scenario and ablation:")
    pivot = df_results.pivot_table(
        index="scenario_id", columns="ablation_config", values="D6_best_f1"
    )
    print(pivot.round(4).to_string())

    print("\n" + "=" * 90)
    print("KEY FINDING: Agreement Bonus Impact (A5 vs A6) on N-DECOY")
    print("=" * 90)
    decoy = df_results["scenario_id"].str.contains("N_DECOY", na=False)
    a5_decoy = df_results.loc[decoy & (df_results["ablation_config"] == "A5"), "D9_fpr_tau"]
    a6_decoy = df_results.loc[decoy & (df_results["ablation_config"] == "A6"), "D9_fpr_tau"]

    if len(a5_decoy) > 0 and len(a6_decoy) > 0:
        print(f"A5 (no bonus) FPR on N-DECOY: {a5_decoy.mean():.4f}")
        print(f"A6 (+bonus)   FPR on N-DECOY: {a6_decoy.mean():.4f}")
        print(f"Difference: {a6_decoy.mean() - a5_decoy.mean():+.4f}")
        if a6_decoy.mean() < a5_decoy.mean():
            print("Prediction confirmed: agreement bonus reduces false alarms on decoy scenarios.")
        elif a6_decoy.mean() == a5_decoy.mean():
            print("No difference observed (bonus never triggered, or FPR already at floor).")
        else:
            print("Prediction missed: bonus did not reduce FPR on decoys (investigate).")
    else:
        print("No N-DECOY results found for A5/A6.")

    output_file = _RESULTS_DIR / f"ablation_summary_{region}.csv"
    summary.to_csv(output_file)
    print(f"\nSaved summary to {output_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ablation results (R7) into a summary table for a region.",
    )
    parser.add_argument(
        "--region",
        default="hormuz",
        help="Region key, alias, or display name to aggregate (default: hormuz). "
        "Only ablation result files for this region are read, so runs never "
        "silently mix regions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    aggregate_ablations(_parse_args().region)
