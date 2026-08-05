"""Tests for the ChokeBench-16 region registry and scenario spec loader."""

from __future__ import annotations

from src.benchmark.regions import load_region
from src.benchmark.scenario_generator import load_scenario, materialize_scenario


def test_load_region() -> None:
    region = load_region("hormuz")
    assert region.name == "hormuz"
    assert "shipping" in region.active_domains
    assert region.reroutable == False


def test_load_scenario() -> None:
    spec = load_scenario("config/benchmark/scenarios/hormuz_P_CRIT.yaml")
    assert spec.region == "hormuz"
    assert spec.event["onset_day"] == 240


def test_materialize_hormuz_4_scenarios() -> None:
    """Hormuz 4 scenarios x 1 seed each."""
    for scenario_file in ["hormuz_P_CRIT", "hormuz_P_HIGH", "hormuz_N_QUIET", "hormuz_N_DECOY"]:
        spec = load_scenario(f"config/benchmark/scenarios/{scenario_file}.yaml")
        region = load_region(spec.region)
        df = materialize_scenario(spec, region)
        assert len(df) == 365
        assert set(region.active_domains).issubset(df.columns)
        if scenario_file == "hormuz_P_CRIT":
            assert df.loc[240:300, "y_disruption"].sum() > 0  # event window has positives


def test_silent_agent_handling() -> None:
    """Disaster agent effect is null; verify it stays at baseline."""
    spec = load_scenario("config/benchmark/scenarios/hormuz_P_CRIT.yaml")
    region = load_region(spec.region)
    df = materialize_scenario(spec, region)
    # disaster signal should be flat ~0.02 +/- noise, no ramp
    disaster_mean = df["disaster"].mean()
    assert 0.01 < disaster_mean < 0.04


def test_scenario_split() -> None:
    """Train 0-200, val 201-280, test 281-364."""
    spec = load_scenario("config/benchmark/scenarios/hormuz_P_CRIT.yaml")
    region = load_region(spec.region)
    df = materialize_scenario(spec, region)
    train, val, test = df[0:201], df[201:281], df[281:365]
    assert len(train) == 201 and len(val) == 80 and len(test) == 84
