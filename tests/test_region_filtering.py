"""Tests for the region filter added to scripts/run_tier{0,1,2}_baselines.py
and scripts/run_ablations.py (docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md
§6 gap 15 / Fix 1).

Before this fix, these scripts globbed every parquet file in
data/benchmark/scenarios_generated/ unconditionally — adding a second
region's scenario data would silently mix regions into every subsequent
run, including re-runs of Hormuz. These tests prove the real, shipped
script functions filter to exactly the requested region and self-identify
it in their output, using tiny synthetic fixtures rather than the full
Hormuz 4x5 grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_DAYS = 365


def _write_fixture_parquet(path: Path, region_key: str, scenario_suffix: str, seed: int) -> None:
    """A minimal-but-valid scenario parquet: just enough columns for the
    tier0/1/2/ablation baselines to run without touching real Hormuz data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=_DAYS, freq="D"),
        "shipping": rng.normal(70, 5, size=_DAYS),
        "market": rng.normal(0, 1, size=_DAYS),
        "y_disruption": np.zeros(_DAYS, dtype=int),
    })
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / f"{region_key}_{scenario_suffix}_seed_{seed}.parquet", index=False)


@pytest.fixture
def two_region_scenarios_dir(tmp_path: Path) -> Path:
    """A scenarios_generated/-like directory holding two regions' files,
    the exact situation that used to make every evaluation script mix
    regions silently."""
    scenarios_dir = tmp_path / "scenarios_generated"
    _write_fixture_parquet(scenarios_dir, "hormuz", "N_QUIET", 42)
    _write_fixture_parquet(scenarios_dir, "panama", "N_QUIET", 42)
    return scenarios_dir


def test_tier0_region_filter_selects_correctly(two_region_scenarios_dir, tmp_path, monkeypatch) -> None:
    import run_tier0_baselines as tier0

    results_dir = tmp_path / "results" / "tier0"
    monkeypatch.setattr(tier0, "_SCENARIOS_DIR", two_region_scenarios_dir)
    monkeypatch.setattr(tier0, "_RESULTS_DIR", results_dir)

    tier0.run_tier0_baselines("panama")

    written = sorted(p.name for p in results_dir.glob("*.json"))
    assert written, "expected tier0 to write result files"
    assert all(name.startswith("panama_") for name in written)
    assert not any(name.startswith("hormuz_") for name in written)


def test_tier0_region_filter_defaults_to_hormuz(two_region_scenarios_dir, tmp_path, monkeypatch) -> None:
    """Backward compatibility: calling with no region argument still only
    ever touches hormuz, matching every pre-Fix-1 call site."""
    import run_tier0_baselines as tier0

    results_dir = tmp_path / "results" / "tier0"
    monkeypatch.setattr(tier0, "_SCENARIOS_DIR", two_region_scenarios_dir)
    monkeypatch.setattr(tier0, "_RESULTS_DIR", results_dir)

    tier0.run_tier0_baselines()  # no region argument

    written = sorted(p.name for p in results_dir.glob("*.json"))
    assert written
    assert all(name.startswith("hormuz_") for name in written)


def test_tier0_results_self_identify_region(two_region_scenarios_dir, tmp_path, monkeypatch) -> None:
    import json

    import run_tier0_baselines as tier0

    results_dir = tmp_path / "results" / "tier0"
    monkeypatch.setattr(tier0, "_SCENARIOS_DIR", two_region_scenarios_dir)
    monkeypatch.setattr(tier0, "_RESULTS_DIR", results_dir)

    tier0.run_tier0_baselines("panama")

    result_file = next(results_dir.glob("*.json"))
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["region"] == "panama"
    assert result["metadata"]["region"] == "panama"


def test_tier0_region_filter_accepts_alias(two_region_scenarios_dir, tmp_path, monkeypatch) -> None:
    """--region accepts anything resolve_region_key understands (alias,
    display name), not just a bare canonical key."""
    import run_tier0_baselines as tier0

    results_dir = tmp_path / "results" / "tier0"
    monkeypatch.setattr(tier0, "_SCENARIOS_DIR", two_region_scenarios_dir)
    monkeypatch.setattr(tier0, "_RESULTS_DIR", results_dir)

    tier0.run_tier0_baselines("Panama Canal")  # display name, not "panama"

    written = sorted(p.name for p in results_dir.glob("*.json"))
    assert written
    assert all(name.startswith("panama_") for name in written)


def test_tier0_unknown_region_raises_loudly(two_region_scenarios_dir, tmp_path, monkeypatch) -> None:
    import run_tier0_baselines as tier0

    monkeypatch.setattr(tier0, "_SCENARIOS_DIR", two_region_scenarios_dir)
    monkeypatch.setattr(tier0, "_RESULTS_DIR", tmp_path / "results" / "tier0")

    with pytest.raises(ValueError, match="Unknown region"):
        tier0.run_tier0_baselines("nonexistent_region")


def test_ablations_region_filter_selects_correctly(two_region_scenarios_dir, tmp_path, monkeypatch) -> None:
    """run_ablations.py has an extra seed=42-suffix filter layered on top
    of the region filter — confirm both compose correctly."""
    import run_ablations as ablations

    results_dir = tmp_path / "results" / "ablations"
    monkeypatch.setattr(ablations, "_SCENARIOS_DIR", two_region_scenarios_dir)
    monkeypatch.setattr(ablations, "_RESULTS_DIR", results_dir)

    ablations.run_ablations("hormuz")

    written = sorted(p.name for p in results_dir.glob("*.json"))
    assert written
    assert all(name.startswith("hormuz_") for name in written)
