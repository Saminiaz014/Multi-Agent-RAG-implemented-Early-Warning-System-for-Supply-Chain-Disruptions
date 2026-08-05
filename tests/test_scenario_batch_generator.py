"""Tests for the Hormuz 4x5 scenario batch generator (R3).

ScenarioBatchGenerator is a thin wrapper around R1's materialize_scenario();
these tests confirm it doesn't reimplement signal generation, that the
derived integer labels are correct, and that parquet round-trips.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.benchmark.regions import load_region
from src.benchmark.scenario_batch_generator import ScenarioBatchGenerator
from src.benchmark.scenario_generator import load_scenario, materialize_scenario


def test_batch_generator_calls_r1_materialize() -> None:
    """Generator wraps R1's materialize_scenario(), doesn't reimplement it."""
    with TemporaryDirectory() as tmpdir:
        generator = ScenarioBatchGenerator(Path(tmpdir))

        region = load_region("hormuz")
        scenario_spec = load_scenario("config/benchmark/scenarios/hormuz_P_CRIT.yaml")
        scenario_spec.seed = 42

        df_r1 = materialize_scenario(scenario_spec, region)
        _, df_gen = generator.generate_scenario("hormuz_P_CRIT", 42, "hormuz")

        r1_cols = set(df_r1.columns)
        gen_cols = set(df_gen.columns)

        assert r1_cols.issubset(gen_cols), f"Missing R1 columns: {r1_cols - gen_cols}"
        assert "y_band_int" in gen_cols
        assert "y_action_int" in gen_cols

        assert (df_gen["y_disruption"] == df_r1["y_disruption"]).all()
        assert (df_gen["y_band"] == df_r1["y_band"]).all()


def test_derive_integer_labels() -> None:
    """Integer labels are derived correctly from string y_band + region.reroutable."""
    with TemporaryDirectory() as tmpdir:
        generator = ScenarioBatchGenerator(Path(tmpdir))

        df_mock = pd.DataFrame({
            "y_disruption": [0, 0, 1, 1, 1, 0, 0],
            "y_band": ["low", "medium", "high", "high", "critical", "medium", "low"],
        })

        region = load_region("hormuz")  # Hormuz: reroutable=False
        df_extended = generator._derive_integer_labels(df_mock, region)

        assert (df_extended.loc[df_extended["y_band"] == "low", "y_band_int"] == 0).all()
        assert (df_extended.loc[df_extended["y_band"] == "medium", "y_band_int"] == 1).all()
        assert (df_extended.loc[df_extended["y_band"] == "high", "y_band_int"] == 2).all()
        assert (df_extended.loc[df_extended["y_band"] == "critical", "y_band_int"] == 3).all()

        # Hormuz: reroutable=False, so high -> escalate (3), not reroute.
        assert (df_extended.loc[df_extended["y_band_int"] == 0, "y_action_int"] == 0).all()
        assert (df_extended.loc[df_extended["y_band_int"] == 1, "y_action_int"] == 1).all()
        assert (df_extended.loc[df_extended["y_band_int"] == 2, "y_action_int"] == 3).all()
        assert (df_extended.loc[df_extended["y_band_int"] == 3, "y_action_int"] == 3).all()


def test_signals_not_clipped() -> None:
    """R1's signal domains are preserved (not clipped to [0, 1])."""
    with TemporaryDirectory() as tmpdir:
        generator = ScenarioBatchGenerator(Path(tmpdir))
        _, df = generator.generate_scenario("hormuz_P_CRIT", 42, "hormuz")

        # Shipping is vessel counts (~70 +/- 5), not a [0,1] proportion.
        assert df["shipping"].mean() > 1.0
        # Market is an unbounded z-score-like signal and does go negative.
        assert df["market"].min() < 0.0


def test_parquet_output() -> None:
    """Scenarios are written to parquet and round-trip correctly."""
    with TemporaryDirectory() as tmpdir:
        generator = ScenarioBatchGenerator(Path(tmpdir))
        output_file, df = generator.generate_scenario("hormuz_P_CRIT", 42, "hormuz")

        assert Path(output_file).exists()

        df_read = pd.read_parquet(output_file)
        assert "y_band_int" in df_read.columns
        assert "y_action_int" in df_read.columns
        assert df_read.shape == df.shape
