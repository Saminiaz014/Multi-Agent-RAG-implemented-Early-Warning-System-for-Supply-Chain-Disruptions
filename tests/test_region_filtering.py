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


# --------------------------------------------------------------------------
# Aggregation scripts (scripts/aggregate_ablation_results.py,
# scripts/aggregate_all_results.py) — found unfiltered during A6 (2026-08-14)
# after the four R4-R7 runner scripts above had already been fixed. Both
# globbed their results directory with no region filter at all: with a
# second region's results on disk, aggregate_ablation_results.py silently
# pooled both regions into one groupby(...).mean() under a header hardcoded
# to say "mean over Hormuz 4 scenarios" regardless of what was actually
# aggregated. These tests prove the fix the same way the tests above prove
# it for the runner scripts: tiny synthetic fixtures, two regions on disk
# at once, assert the filter actually restricts to one.
# --------------------------------------------------------------------------


def _write_fixture_ablation_result(
    path: Path, region_key: str, scenario_suffix: str, config: str, seed: int = 42
) -> None:
    import json

    path.mkdir(parents=True, exist_ok=True)
    result = {
        "scenario_id": f"{region_key}_{scenario_suffix}",
        "ablation_config": config,
        "seed": seed,
        "region": region_key,
        "metadata": {"ablation_name": f"{config}_fixture"},
        "metrics": {
            "D3_auc_pr": 0.5, "D4_auc_roc": 0.5, "D5_f1_tau": 0.5,
            "D6_best_f1": 0.5, "D8_recall_tau": 0.5, "D9_fpr_tau": 0.5,
        },
    }
    out = path / f"{region_key}_{scenario_suffix}_{config}_seed_{seed}.json"
    out.write_text(json.dumps(result), encoding="utf-8")


def _write_fixture_tier_result(
    path: Path, region_key: str, scenario_suffix: str, baseline: str, seed: int = 42
) -> None:
    import json

    path.mkdir(parents=True, exist_ok=True)
    result = {
        "scenario_id": f"{region_key}_{scenario_suffix}",
        "baseline_name": baseline,
        "seed": seed,
        "region": region_key,
        "metadata": {"region": region_key},
        "metrics": {"D3_auc_pr": 0.5, "D6_best_f1": 0.5, "D8_recall_tau": 0.5, "D9_fpr_tau": 0.5},
    }
    out = path / f"{region_key}_{scenario_suffix}_{baseline}_seed_{seed}.json"
    out.write_text(json.dumps(result), encoding="utf-8")


@pytest.fixture
def two_region_ablation_results_dir(tmp_path: Path) -> Path:
    results_dir = tmp_path / "results" / "ablations"
    _write_fixture_ablation_result(results_dir, "hormuz", "N_QUIET", "A0")
    _write_fixture_ablation_result(results_dir, "panama", "N_QUIET", "A0")
    return results_dir


def test_aggregate_ablations_region_filter_selects_correctly(
    two_region_ablation_results_dir, monkeypatch, capsys
) -> None:
    import aggregate_ablation_results as agg

    monkeypatch.setattr(agg, "_RESULTS_DIR", two_region_ablation_results_dir)

    agg.aggregate_ablations("panama")

    out = capsys.readouterr().out
    assert "region=panama" in out
    assert "panama_N_QUIET" in out
    assert "hormuz_N_QUIET" not in out
    written = two_region_ablation_results_dir / "ablation_summary_panama.csv"
    assert written.exists()


def test_aggregate_ablations_region_filter_defaults_to_hormuz(
    two_region_ablation_results_dir, monkeypatch, capsys
) -> None:
    import aggregate_ablation_results as agg

    monkeypatch.setattr(agg, "_RESULTS_DIR", two_region_ablation_results_dir)

    agg.aggregate_ablations()  # no region argument

    out = capsys.readouterr().out
    assert "region=hormuz" in out
    assert "hormuz_N_QUIET" in out
    assert "panama_N_QUIET" not in out


def test_aggregate_ablations_output_filenames_do_not_collide_across_regions(
    two_region_ablation_results_dir, monkeypatch
) -> None:
    """Before the fix, both regions' summaries would have overwritten the
    same ablation_summary.csv — the region-suffixed filename prevents that."""
    import aggregate_ablation_results as agg

    monkeypatch.setattr(agg, "_RESULTS_DIR", two_region_ablation_results_dir)

    agg.aggregate_ablations("hormuz")
    agg.aggregate_ablations("panama")

    assert (two_region_ablation_results_dir / "ablation_summary_hormuz.csv").exists()
    assert (two_region_ablation_results_dir / "ablation_summary_panama.csv").exists()


@pytest.fixture
def two_region_all_tiers_dir(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    _write_fixture_tier_result(root / "baselines" / "tier0", "hormuz", "N_QUIET", "random")
    _write_fixture_tier_result(root / "baselines" / "tier0", "panama", "N_QUIET", "random")
    _write_fixture_ablation_result(root / "baselines" / "ablations", "hormuz", "N_QUIET", "A0")
    _write_fixture_ablation_result(root / "baselines" / "ablations", "panama", "N_QUIET", "A0")
    return root


def test_aggregate_all_results_load_results_filters_by_region(
    two_region_all_tiers_dir,
) -> None:
    from scripts.aggregate_all_results import load_results

    tier0_dir = two_region_all_tiers_dir / "baselines" / "tier0"
    panama_results = load_results(tier0_dir, "panama")
    hormuz_results = load_results(tier0_dir, "hormuz")

    assert len(panama_results) == 1
    assert panama_results[0]["scenario_id"] == "panama_N_QUIET"
    assert len(hormuz_results) == 1
    assert hormuz_results[0]["scenario_id"] == "hormuz_N_QUIET"


def test_aggregate_all_results_main_filters_and_names_output_by_region(
    two_region_all_tiers_dir, monkeypatch
) -> None:
    import aggregate_all_results as agg

    monkeypatch.setattr(agg, "_RESULTS_ROOT", two_region_all_tiers_dir)
    monkeypatch.setattr(agg, "_TIER0_DIR", two_region_all_tiers_dir / "baselines" / "tier0")
    monkeypatch.setattr(agg, "_TIER1_DIR", two_region_all_tiers_dir / "baselines" / "tier1")
    monkeypatch.setattr(agg, "_TIER2_DIR", two_region_all_tiers_dir / "baselines" / "tier2")
    monkeypatch.setattr(agg, "_ABLATIONS_DIR", two_region_all_tiers_dir / "baselines" / "ablations")

    agg.main("panama")

    all_runs = two_region_all_tiers_dir / "results_all_runs_panama.csv"
    assert all_runs.exists()
    content = all_runs.read_text(encoding="utf-8")
    assert "panama_N_QUIET" in content
    assert "hormuz_N_QUIET" not in content
    assert not (two_region_all_tiers_dir / "results_all_runs_hormuz.csv").exists()
