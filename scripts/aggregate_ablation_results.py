"""Aggregate ablation results (R7) into a summary table.

Run from project root, after ``scripts/run_ablations.py``::

    python scripts/aggregate_ablation_results.py [--region REGION] [--positives-only]

``--region`` defaults to ``hormuz`` (matching every R4-R7 runner script). It
filters ``results/baselines/ablations/`` to only that region's result files,
so a second region's ablation results on disk never silently blend into this
region's summary — this script and ``aggregate_all_results.py`` were found
to be the two remaining unfiltered aggregation entry points (2026-08-14,
during A6), after the four R4-R7 runner scripts had already been fixed; see
docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md §6 for the full writeup.

``--positives-only`` excludes the two negative-control scenarios (N_QUIET,
N_DECOY) from the mean. A5-A7's per-domain weights are Optuna-tuned per
scenario (``scripts/run_ablations.py``/``src/baselines/ablation_runner.py``);
N_QUIET and N_DECOY have zero positive days in the validation window by
design (negative controls), which always hits ``tune_weights_optuna``'s own
documented equal-weights fallback ("Falls back to equal weights if no trial
beat an all-zero F1"). The default (blended) mean therefore averages two
genuinely-tuned scenario results (P_CRIT, P_HIGH) with two fallback-weight
results (N_QUIET, N_DECOY) into one number for A5/A6/A7. ``--positives-only``
reports the tuned-weights-only figures instead; default behavior (blended,
all 4 scenarios) is unchanged. See docs/multiregion/CROSS_REGION_RESULTS.md
for the region-by-region comparison of blended vs. positives-only figures.
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


def aggregate_ablations(region: str = "hormuz", positives_only: bool = False) -> None:
    """Load ``region``'s ablation result JSONs, compute mean over scenarios, print + save.

    Args:
        region: Canonical region key, alias, or display name (see
            :func:`src.benchmark.regions.resolve_region_key`). Defaults to
            ``"hormuz"``, matching every R4-R7 runner script.
        positives_only: If True, exclude N_QUIET/N_DECOY from the mean (see
            module docstring) — the mean is then over P_CRIT/P_HIGH only,
            both of which have genuinely Optuna-tuned A5-A7 weights. Reflected
            in both the printed header and the output filename. Defaults to
            False (existing behavior: mean over all 4 scenarios).
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

    if positives_only:
        df_results = df_results[
            ~df_results["scenario_id"].str.endswith(("_N_QUIET", "_N_DECOY"))
        ]

    n_scenarios = df_results["scenario_id"].nunique()

    summary = df_results.groupby("ablation_config")[list(_METRIC_COLS)].mean().round(4)

    mode_label = "positives-only (P_CRIT/P_HIGH, tuned weights)" if positives_only else "all scenarios, blended"
    print("\n" + "=" * 90)
    print(f"ABLATION SUMMARY (region={region}, {mode_label}, mean over {n_scenarios} scenarios, seed=42)")
    print("=" * 90)
    print(summary.to_string())
    print("=" * 90)

    print("\nBest-F1 by scenario and ablation:")
    pivot = df_results.pivot_table(
        index="scenario_id", columns="ablation_config", values="D6_best_f1"
    )
    print(pivot.round(4).to_string())

    if positives_only:
        print(
            "\n(--positives-only: N-DECOY excluded from this run, so the "
            "Agreement Bonus / N-DECOY section is skipped — see the default, "
            "blended run for that comparison.)"
        )
    else:
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

    suffix = "_positives_only" if positives_only else ""
    output_file = _RESULTS_DIR / f"ablation_summary_{region}{suffix}.csv"
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
    parser.add_argument(
        "--positives-only",
        action="store_true",
        help="Exclude N_QUIET/N_DECOY from the mean, so A5-A7's tuned weights "
        "aren't diluted by the two scenarios that always fall back to equal "
        "weights (zero positive days in the validation window, by design). "
        "Default: off (existing behavior, mean over all 4 scenarios).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    aggregate_ablations(_args.region, _args.positives_only)
