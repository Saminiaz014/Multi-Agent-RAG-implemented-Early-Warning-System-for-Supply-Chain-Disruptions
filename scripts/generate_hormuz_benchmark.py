"""Generate the Hormuz 4x5 (4 scenarios x 5 seeds) ChokeBench-16 benchmark grid.

Run from project root::

    python scripts/generate_hormuz_benchmark.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmark.scenario_batch_generator import (  # noqa: E402
    DEFAULT_SCENARIOS,
    DEFAULT_SEEDS,
    ScenarioBatchGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_OUTPUT_DIR = _PROJECT_ROOT / "data" / "benchmark" / "scenarios_generated"


def generate_hormuz_benchmark() -> None:
    """Generate Hormuz 4 scenarios x 5 seeds = 20 parquet files."""
    generator = ScenarioBatchGenerator(_OUTPUT_DIR)
    results = generator.generate_batch(
        list(DEFAULT_SCENARIOS), list(DEFAULT_SEEDS), region_name="hormuz"
    )

    total_files = sum(len(seeds) for seeds in results.values())
    logger.info("=" * 60)
    logger.info("Generated %d parquet files", total_files)
    logger.info("Output directory: %s", _OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_hormuz_benchmark()
