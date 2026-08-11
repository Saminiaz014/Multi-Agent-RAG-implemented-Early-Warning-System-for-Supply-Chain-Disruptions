"""Run Tier 0 control baselines on a region's 4x5 benchmark grid.

Run from project root::

    python scripts/run_tier0_baselines.py [--region REGION]

``--region`` defaults to ``hormuz`` (backward-compatible with every prior
invocation of this script). It filters ``scenarios_generated/`` to only that
region's parquet files, so adding a second region's data never silently
mixes regions into a run — see docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md
§6 gap 15.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.baselines.baseline_evaluator import BaselineEvaluator  # noqa: E402
from src.baselines.tier0_controls import (  # noqa: E402
    AlwaysAlarmBaseline,
    NeverAlarmBaseline,
    RandomBaseline,
)
from src.benchmark.regions import resolve_region_key  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SCENARIOS_DIR = _PROJECT_ROOT / "data" / "benchmark" / "scenarios_generated"
_RESULTS_DIR = _PROJECT_ROOT / "results" / "baselines" / "tier0"

# Matches the train(0-200)/val(201-280)/test(281-364) split used across the
# benchmark (see tests/test_benchmark_regions.py::test_scenario_split).
_VAL_WINDOW = (201, 281)   # [start, end) — 80 days
_TEST_WINDOW = (281, 365)  # [start, end) — 84 days


def run_tier0_baselines(region: str = "hormuz") -> None:
    """Run all Tier 0 baselines on every ``region`` scenario parquet.

    Args:
        region: Canonical region key, alias, or display name (see
            :func:`src.benchmark.regions.resolve_region_key`). Defaults to
            ``"hormuz"`` — every prior call site (no argument) reproduces
            identical behavior, since every parquet file in
            ``scenarios_generated/`` today is Hormuz's.
    """
    region = resolve_region_key(region)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baselines = [RandomBaseline(), AlwaysAlarmBaseline(), NeverAlarmBaseline()]
    parquet_files = sorted(
        p for p in _SCENARIOS_DIR.glob("*.parquet")
        if p.stem.rpartition("_seed_")[0].startswith(f"{region}_")
    )
    if not parquet_files:
        logger.warning(
            "No scenario parquet files found for region %r in %s — run "
            "scripts/generate_hormuz_benchmark.py first (or generate that "
            "region's benchmark grid).",
            region, _SCENARIOS_DIR,
        )
        return

    for parquet_file in parquet_files:
        # Filename format: {scenario_id}_seed_{seed}.parquet
        scenario_id, _, seed_str = parquet_file.stem.rpartition("_seed_")
        seed = int(seed_str)

        df = pd.read_parquet(parquet_file)
        y_true = df["y_disruption"].to_numpy()

        for baseline in baselines:
            anomaly_scores, metadata = baseline.run(df, scenario_id, seed)

            val_scores = anomaly_scores[_VAL_WINDOW[0]:_VAL_WINDOW[1]]
            val_y = y_true[_VAL_WINDOW[0]:_VAL_WINDOW[1]]
            threshold = baseline._compute_threshold(val_scores, val_y)

            test_scores = anomaly_scores[_TEST_WINDOW[0]:_TEST_WINDOW[1]]
            test_y = y_true[_TEST_WINDOW[0]:_TEST_WINDOW[1]]
            metrics = BaselineEvaluator.evaluate(
                test_y, test_scores, scenario_id, threshold=threshold
            )

            metadata["threshold"] = threshold
            metadata["val_window"] = list(_VAL_WINDOW)
            metadata["test_window"] = list(_TEST_WINDOW)
            metadata["region"] = region

            result = {
                "scenario_id": scenario_id,
                "baseline_name": baseline.name,
                "seed": seed,
                "region": region,
                "metrics": metrics,
                "metadata": metadata,
            }

            output_file = _RESULTS_DIR / f"{scenario_id}_{baseline.name}_seed_{seed}.json"
            with open(output_file, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, default=str)

            logger.info(
                "%s / %s / seed=%d | tau=%.3f | D5_f1_tau=%.4f D4_auc_roc=%s -> %s",
                scenario_id, baseline.name, seed, threshold,
                metrics["D5_f1_tau"], metrics["D4_auc_roc"], output_file.name,
            )

    total = len(parquet_files) * len(baselines)
    logger.info("=" * 60)
    logger.info("region=%s | Wrote %d result files to %s", region, total, _RESULTS_DIR)
    logger.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tier 0 control baselines on a region's benchmark grid.",
    )
    parser.add_argument(
        "--region",
        default="hormuz",
        help="Region key, alias, or display name to evaluate (default: hormuz). "
        "Only scenario parquet files for this region are read, so runs never "
        "silently mix regions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_tier0_baselines(_parse_args().region)
