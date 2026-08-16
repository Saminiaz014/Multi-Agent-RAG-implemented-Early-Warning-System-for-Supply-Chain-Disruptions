"""Test for the P_CRIT/P_HIGH val+test positive-day invariant (A6-FIX,
2026-08-16, docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md §6 gap 22).

Every positive scenario must have at least one y_disruption=1 day in BOTH
the val window (201-280) and the test window (281-364) -- the same
pre-declared split every R4-R8 script scores against. Before this fix, the
"P_HIGH onset = P_CRIT onset * 0.5" convention placed every region's P_HIGH
event entirely before day 201, and malacca/suez's P_CRIT events didn't
reach day 281 either -- 7 of the 10 positive scenarios silently had a
zero-signal test window, including hormuz's own committed P_HIGH results.
"""

from __future__ import annotations

import pytest

from src.benchmark.regions import load_region
from src.benchmark.scenario_generator import load_scenario, materialize_scenario

# Matches every R4-R8 script's pre-declared split (e.g. scripts/run_ablations.py).
_VAL_WINDOW = (201, 281)
_TEST_WINDOW = (281, 365)

_POSITIVE_SCENARIOS = [
    (region, scenario_class)
    for region in ("hormuz", "bab_el_mandeb", "malacca", "panama", "suez")
    for scenario_class in ("P_CRIT", "P_HIGH")
]


@pytest.mark.parametrize("region,scenario_class", _POSITIVE_SCENARIOS)
def test_positive_scenario_has_val_and_test_positives(region: str, scenario_class: str) -> None:
    scenario_id = f"{region}_{scenario_class}"
    region_spec = load_region(region)
    spec = load_scenario(f"config/benchmark/scenarios/{scenario_id}.yaml")
    df = materialize_scenario(spec, region_spec)

    val_positives = int(df["y_disruption"].iloc[_VAL_WINDOW[0]:_VAL_WINDOW[1]].sum())
    test_positives = int(df["y_disruption"].iloc[_TEST_WINDOW[0]:_TEST_WINDOW[1]].sum())

    assert val_positives > 0, f"{scenario_id}: zero positive days in val window {_VAL_WINDOW}"
    assert test_positives > 0, f"{scenario_id}: zero positive days in test window {_TEST_WINDOW}"
