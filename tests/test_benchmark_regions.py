"""Tests for the EVAL01 region registry and scenario spec loader."""

from __future__ import annotations

import pytest

from src.benchmark.regions import (
    CANONICAL_REGIONS,
    REGION_ALIASES,
    REGION_STATUS,
    load_region,
    resolve_region_key,
)
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


def test_resolve_region_key_canonical_hit() -> None:
    assert resolve_region_key("hormuz") == "hormuz"
    assert resolve_region_key("malacca") == "malacca"


def test_resolve_region_key_alias_hit() -> None:
    """red_sea (the pre-existing settings.yaml/extraction.chokepoints key)
    resolves to the canonical bab_el_mandeb key."""
    assert resolve_region_key("red_sea") == "bab_el_mandeb"


def test_resolve_region_key_display_name_hit() -> None:
    assert resolve_region_key("Strait of Hormuz") == "hormuz"
    assert resolve_region_key("Bab el-Mandeb") == "bab_el_mandeb"


def test_resolve_region_key_case_insensitive() -> None:
    assert resolve_region_key("HORMUZ") == "hormuz"
    assert resolve_region_key("Red_Sea") == "bab_el_mandeb"


def test_resolve_region_key_malacca_is_canonical() -> None:
    """malacca is a full canonical key, not an alias — it already agrees
    with extraction.chokepoints, so no aliasing is needed."""
    assert "malacca" in CANONICAL_REGIONS
    assert "malacca" not in REGION_ALIASES
    assert resolve_region_key("malacca") == "malacca"


def test_resolve_region_key_unknown_raises_with_options() -> None:
    with pytest.raises(ValueError, match="Unknown region"):
        resolve_region_key("nonexistent_region")


def test_resolve_region_key_taiwan_strait_not_supported() -> None:
    """taiwan_strait is explicitly out of scope for this project (per an
    earlier external plan this codebase never implemented) and must raise."""
    with pytest.raises(ValueError):
        resolve_region_key("taiwan_strait")


def test_canonical_regions_status() -> None:
    """Only hormuz is populated; the other four canonical keys are declared
    but not pretending to have data."""
    assert REGION_STATUS["hormuz"] == "populated"
    for key in ("bab_el_mandeb", "panama", "suez", "malacca"):
        assert REGION_STATUS[key] == "planned"
    assert set(REGION_STATUS) == set(CANONICAL_REGIONS)
