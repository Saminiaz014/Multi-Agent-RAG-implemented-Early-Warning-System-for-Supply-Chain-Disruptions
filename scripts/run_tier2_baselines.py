"""Run Tier 2 classical unsupervised baselines on the Hormuz 4x5 benchmark grid.

Uses the same pre-declared train(0-200)/val(201-280)/test(281-364)
protocol as ``scripts/run_tier0_baselines.py`` / ``run_tier1_baselines.py``
so results are comparable across tiers.

Run from project root::

    python scripts/run_tier2_baselines.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.baselines.baseline_evaluator import BaselineEvaluator  # noqa: E402
from src.baselines.tier2_classical import (  # noqa: E402
    IsolationForestBaseline,
    MatrixProfileBaseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SCENARIOS_DIR = _PROJECT_ROOT / "data" / "benchmark" / "scenarios_generated"
_RESULTS_DIR = _PROJECT_ROOT / "results" / "baselines" / "tier2"

# Matches run_tier0_baselines.py / run_tier1_baselines.py.
_VAL_WINDOW = (201, 281)   # [start, end) — 80 days
_TEST_WINDOW = (281, 365)  # [start, end) — 84 days


def run_tier2_baselines() -> None:
    """Run all Tier 2 baselines on every Hormuz 4x5 scenario parquet."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baselines = [
        IsolationForestBaseline(contamination=0.1, n_estimators=100),
        MatrixProfileBaseline(m=30),
    ]
    parquet_files = sorted(_SCENARIOS_DIR.glob("*.parquet"))
    if not parquet_files:
        logger.warning(
            "No scenario parquet files found in %s — run "
            "scripts/generate_hormuz_benchmark.py first.",
            _SCENARIOS_DIR,
        )
        return

    written = 0
    for parquet_file in parquet_files:
        scenario_id, _, seed_str = parquet_file.stem.rpartition("_seed_")
        seed = int(seed_str)

        df = pd.read_parquet(parquet_file)
        y_true = df["y_disruption"].to_numpy()

        for baseline in baselines:
            try:
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

                result = {
                    "scenario_id": scenario_id,
                    "baseline_name": baseline.name,
                    "seed": seed,
                    "metrics": metrics,
                    "metadata": metadata,
                }

                output_file = _RESULTS_DIR / f"{scenario_id}_{baseline.name}_seed_{seed}.json"
                with open(output_file, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, indent=2, default=str)
                written += 1

                logger.info(
                    "%s / %s / seed=%d | tau=%.3f | D5_f1_tau=%.4f D6_best_f1=%.4f -> %s",
                    scenario_id, baseline.name, seed, threshold,
                    metrics["D5_f1_tau"], metrics["D6_best_f1"], output_file.name,
                )
            except Exception:
                logger.exception(
                    "Error running %s on %s seed=%d", baseline.name, scenario_id, seed
                )

    logger.info("=" * 60)
    logger.info("Wrote %d result files to %s", written, _RESULTS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_tier2_baselines()
