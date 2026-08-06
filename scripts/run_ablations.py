"""Run Tier 5 ablations (A0-A7) on the Hormuz 4x5 benchmark grid, seed=42 only.

Uses the same pre-declared val(201-280)/test(281-364) protocol as
``scripts/run_tier0/1/2_baselines.py`` for comparability: A5/A6/A7 tune
their per-domain weights on the validation split (rule 5), then — like
every other baseline — the decision threshold is also fit on validation
and applied to test.

Run from project root::

    python scripts/run_ablations.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.baselines.ablation_runner import AblationRunner  # noqa: E402
from src.baselines.ablations import ABLATIONS  # noqa: E402
from src.baselines.baseline_evaluator import BaselineEvaluator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SCENARIOS_DIR = _PROJECT_ROOT / "data" / "benchmark" / "scenarios_generated"
_RESULTS_DIR = _PROJECT_ROOT / "results" / "baselines" / "ablations"

_VAL_WINDOW = (201, 281)   # [start, end) — 80 days
_TEST_WINDOW = (281, 365)  # [start, end) — 84 days

_SEED = 42  # single seed for speed, per rule 6


def run_ablations() -> None:
    """Run A0-A7 on every Hormuz scenario at seed=42."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(_SCENARIOS_DIR.glob(f"*_seed_{_SEED}.parquet"))
    if not parquet_files:
        logger.warning(
            "No seed=%d scenario parquet files found in %s — run "
            "scripts/generate_hormuz_benchmark.py first.",
            _SEED, _SCENARIOS_DIR,
        )
        return

    written = 0
    for parquet_file in parquet_files:
        scenario_id, _, seed_str = parquet_file.stem.rpartition("_seed_")
        seed = int(seed_str)

        df = pd.read_parquet(parquet_file)
        y_true = df["y_disruption"].to_numpy()

        for config_id, config in ABLATIONS.items():
            runner = AblationRunner(config)
            try:
                anomaly_scores, metadata = runner.run(df, scenario_id, seed)

                val_scores = anomaly_scores[_VAL_WINDOW[0]:_VAL_WINDOW[1]]
                val_y = y_true[_VAL_WINDOW[0]:_VAL_WINDOW[1]]
                threshold = runner._compute_threshold(val_scores, val_y)

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
                    "ablation_config": config_id,
                    "seed": seed,
                    "metrics": metrics,
                    "metadata": metadata,
                }

                output_file = _RESULTS_DIR / f"{scenario_id}_{config_id}_seed_{seed}.json"
                with open(output_file, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, indent=2, default=str)
                written += 1

                logger.info(
                    "%s / %s / seed=%d | tau=%.3f | D5_f1_tau=%.4f D6_best_f1=%.4f -> %s",
                    scenario_id, config_id, seed, threshold,
                    metrics["D5_f1_tau"], metrics["D6_best_f1"], output_file.name,
                )
            except Exception:
                logger.exception(
                    "Error running %s on %s seed=%d", config_id, scenario_id, seed
                )

    logger.info("=" * 60)
    logger.info("Wrote %d result files to %s", written, _RESULTS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ablations()
