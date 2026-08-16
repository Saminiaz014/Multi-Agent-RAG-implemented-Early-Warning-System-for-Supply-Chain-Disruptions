"""Run Tier 5 ablations (A0-A7) on a region's 4x5 benchmark grid, seed=42 only.

Uses the same pre-declared val(201-280)/test(281-364) protocol as
``scripts/run_tier0/1/2_baselines.py`` for comparability: A5/A6/A7 tune
their per-domain weights on the validation split (rule 5), then — like
every other baseline — the decision threshold is also fit on validation
and applied to test.

Run from project root::

    python scripts/run_ablations.py [--region REGION]

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

from src.baselines.ablation_runner import AblationRunner  # noqa: E402
from src.baselines.ablations import ABLATIONS, AblationConfig, scope_to_domains  # noqa: E402
from src.baselines.baseline_evaluator import BaselineEvaluator  # noqa: E402
from src.benchmark.regions import load_region, resolve_region_key  # noqa: E402

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


def _compute_degeneracy(scoped_configs: dict[str, AblationConfig]) -> dict[str, str | None]:
    """Flag configs whose domain-scoped agent set duplicates an earlier config's.

    A region missing a domain that some ``ABLATIONS`` entries share (e.g.
    every non-Hormuz region is missing at least one of geopolitical/disaster)
    can collapse two nominally different configs onto the same surviving
    domain set once :func:`~src.baselines.ablations.scope_to_domains` runs —
    e.g. malacca has no ``market``, so A1 (``shipping``, ``market``) scopes
    down to just ``shipping``, identical to A0.

    Args:
        scoped_configs: ``{config_id: scoped AblationConfig}``, in
            ``ABLATIONS``' declared order (A0 first).

    Returns:
        ``{config_id: earlier_config_id_with_same_domain_set_or_None}``.
        The first config to reach a given domain set is never flagged
        (``None``); later ones pointing at it are.
    """
    seen: dict[frozenset[str], str] = {}
    degeneracy: dict[str, str | None] = {}
    for config_id, config in scoped_configs.items():
        domain_set = frozenset(config.agents)
        degeneracy[config_id] = seen.get(domain_set)
        seen.setdefault(domain_set, config_id)
    return degeneracy


def run_ablations(region: str = "hormuz") -> None:
    """Run A0-A7 on every ``region`` scenario at seed=42.

    Each of the eight ``ABLATIONS`` configs is scoped to ``region``'s
    ``active_domains`` before running (see
    :func:`~src.baselines.ablations.scope_to_domains`) — A2-A7 hardcode
    domains (geopolitical, disaster, ...) that not every region has active,
    and scoring an inactive domain silently poisons the composite score
    with ``NaN`` rather than erroring. Scoping can make two configs
    degenerate into the same surviving domain set for a given region; each
    result's ``metadata["degenerate_of"]`` names the earlier config it
    collapsed onto (``None`` if it's still distinct). See
    docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md §6 gap 21.

    Args:
        region: Canonical region key, alias, or display name (see
            :func:`src.benchmark.regions.resolve_region_key`). Defaults to
            ``"hormuz"`` — every prior call site (no argument) reproduces
            identical behavior, since Hormuz has all six domains active, so
            domain-scoping is a no-op for it.
    """
    region = resolve_region_key(region)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    active_domains = load_region(region).active_domains
    scoped_configs = {
        config_id: scope_to_domains(config, active_domains)
        for config_id, config in ABLATIONS.items()
    }
    degeneracy = _compute_degeneracy(scoped_configs)
    n_distinct = sum(1 for d in degeneracy.values() if d is None)
    logger.info(
        "region=%s | active_domains=%s | %d/%d ablation configs remain distinct "
        "after domain-scoping: %s",
        region, active_domains, n_distinct, len(scoped_configs),
        {cid: (d or "distinct") for cid, d in degeneracy.items()},
    )

    parquet_files = sorted(
        p for p in _SCENARIOS_DIR.glob(f"*_seed_{_SEED}.parquet")
        if p.stem.rpartition("_seed_")[0].startswith(f"{region}_")
    )
    if not parquet_files:
        logger.warning(
            "No seed=%d scenario parquet files found for region %r in %s — "
            "run scripts/generate_hormuz_benchmark.py first (or generate "
            "that region's benchmark grid).",
            _SEED, region, _SCENARIOS_DIR,
        )
        return

    written = 0
    for parquet_file in parquet_files:
        scenario_id, _, seed_str = parquet_file.stem.rpartition("_seed_")
        seed = int(seed_str)

        df = pd.read_parquet(parquet_file)
        y_true = df["y_disruption"].to_numpy()

        for config_id, config in scoped_configs.items():
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
                metadata["region"] = region
                metadata["degenerate_of"] = degeneracy[config_id]

                result = {
                    "scenario_id": scenario_id,
                    "ablation_config": config_id,
                    "seed": seed,
                    "region": region,
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
    logger.info("region=%s | Wrote %d result files to %s", region, written, _RESULTS_DIR)
    logger.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tier 5 ablations (A0-A7) on a region's benchmark grid, seed=42 only.",
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
    run_ablations(_parse_args().region)
