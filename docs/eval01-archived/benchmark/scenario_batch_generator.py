"""Batch scenario generator — a thin wrapper around R1's materialize_scenario().

Generates the Hormuz 4x5 (scenario x seed) benchmark grid and writes one
parquet file per combination. All signal generation, ramp/decay, noise, and
the string ``y_band`` / binary ``y_action`` labels come from
:func:`src.benchmark.scenario_generator.materialize_scenario` — this module
adds nothing to that pipeline except a convenience integer encoding of the
band/action labels and the batch/IO plumbing around it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark.regions import Region, load_region
from src.benchmark.scenario_generator import (
    BAND_TO_INT as _BAND_TO_INT,
    load_scenario,
    materialize_scenario,
)

logger = logging.getLogger(__name__)

DEFAULT_SCENARIOS: tuple[str, ...] = (
    "hormuz_P_CRIT",
    "hormuz_P_HIGH",
    "hormuz_N_QUIET",
    "hormuz_N_DECOY",
)
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 999)


class ScenarioBatchGenerator:
    """Generate a batch of scenarios by calling R1's materialize_scenario().

    For each ``scenario_id`` x ``seed`` combination:
        1. Load the region spec (R1's :func:`load_region`).
        2. Load the scenario spec (R1's :func:`load_scenario`).
        3. Call R1's :func:`materialize_scenario` to generate signals plus
           the string ``y_band`` and binary ``y_action`` labels.
        4. Add ``y_band_int`` (0-3) and ``y_action_int`` (0-3), derived
           deterministically from ``y_band`` + ``region.reroutable``.
        5. Write the result to parquet.
    """

    # Re-exported from scenario_generator.py, the single source of truth —
    # load_scenario() now validates event.peak_band against this same dict
    # at load time (docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md §6 fix
    # 3), so an unrecognized band can no longer reach this mapping at all.
    # Kept as a class attribute here for backward compatibility with any
    # caller doing ScenarioBatchGenerator.BAND_TO_INT.
    BAND_TO_INT: dict[str, int] = _BAND_TO_INT

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _derive_integer_labels(self, df: pd.DataFrame, region: Region) -> pd.DataFrame:
        """Derive integer-coded ``y_band_int`` / ``y_action_int`` from R1's outputs.

        Args:
            df: DataFrame with R1's columns (``y_band`` as string).
            region: Region dataclass — only ``reroutable`` is used here.

        Returns:
            ``df`` with two new columns appended: ``y_band_int`` (0-3) and
            ``y_action_int`` (0=no_action, 1=monitor, 2=reroute,
            3=escalate).
        """
        df = df.copy()
        df["y_band_int"] = df["y_band"].map(self.BAND_TO_INT)

        y_action_int = np.zeros(len(df), dtype=int)
        y_action_int[df["y_band_int"] == 1] = 1  # monitor

        high_mask = (df["y_band_int"] == 2).to_numpy()
        y_action_int[high_mask] = 2 if region.reroutable else 3  # reroute / escalate

        critical_mask = (df["y_band_int"] == 3).to_numpy()
        y_action_int[critical_mask] = 3  # escalate

        df["y_action_int"] = y_action_int
        return df

    def generate_scenario(
        self, scenario_id: str, seed: int, region_name: str = "hormuz"
    ) -> tuple[str, pd.DataFrame]:
        """Generate a single scenario (``scenario_id`` x ``seed``).

        Args:
            scenario_id: e.g. ``"hormuz_P_CRIT"``.
            seed: Random seed override for the scenario spec.
            region_name: Region to load.

        Returns:
            ``(output_file_path, df)``.
        """
        region = load_region(region_name)
        scenario_spec = load_scenario(f"config/benchmark/scenarios/{scenario_id}.yaml")
        scenario_spec.seed = seed

        df = materialize_scenario(scenario_spec, region)
        df = self._derive_integer_labels(df, region)

        output_file = self.output_dir / f"{scenario_id}_seed_{seed}.parquet"
        df.to_parquet(output_file, index=False)

        logger.info(
            "Generated %s seed=%d | shape=%s | disruption_days=%d | %s",
            scenario_id,
            seed,
            df.shape,
            int(df["y_disruption"].sum()),
            output_file,
        )
        return str(output_file), df

    def generate_batch(
        self,
        scenario_ids: list[str],
        seeds: list[int],
        region_name: str = "hormuz",
    ) -> dict[str, dict[int, tuple[str, pd.DataFrame]]]:
        """Generate a batch of scenarios across the full scenario x seed grid.

        Args:
            scenario_ids: e.g. ``["hormuz_P_CRIT", "hormuz_P_HIGH", ...]``.
            seeds: Random seeds to materialize each scenario under.
            region_name: Region name shared by every scenario in the batch.

        Returns:
            ``{scenario_id: {seed: (output_file, df)}}``.
        """
        results: dict[str, dict[int, tuple[str, pd.DataFrame]]] = {}
        for scenario_id in scenario_ids:
            results[scenario_id] = {}
            for seed in seeds:
                results[scenario_id][seed] = self.generate_scenario(
                    scenario_id, seed, region_name
                )
        return results
