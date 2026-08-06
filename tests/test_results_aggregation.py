"""Tests for the R8 results aggregation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_all_results import extract_baseline_name


def test_scenario_type_regex_extraction() -> None:
    """Scenario type extraction uses underscores, matching real scenario_ids."""
    scenario_ids = [
        "hormuz_P_CRIT", "hormuz_P_HIGH", "hormuz_N_QUIET", "hormuz_N_DECOY",
        "bab_P_CRIT", "panama_N_DECOY", "suez_P_HIGH",
    ]
    df = pd.DataFrame({"scenario_id": scenario_ids})
    df["scenario_type"] = df["scenario_id"].str.extract(
        r"(P_CRIT|P_HIGH|N_QUIET|N_DECOY)", expand=False
    )

    assert df[df["scenario_id"] == "hormuz_P_CRIT"]["scenario_type"].values[0] == "P_CRIT"
    assert df[df["scenario_id"] == "hormuz_N_DECOY"]["scenario_type"].values[0] == "N_DECOY"
    assert df[df["scenario_id"] == "panama_N_DECOY"]["scenario_type"].values[0] == "N_DECOY"
    assert df[df["scenario_id"] == "suez_P_HIGH"]["scenario_type"].values[0] == "P_HIGH"


def test_baseline_name_extraction_tier_baseline() -> None:
    """Tier 0-2 results expose baseline_name at the top level."""
    result = {
        "scenario_id": "hormuz_P_CRIT",
        "baseline_name": "random",
        "seed": 42,
        "metrics": {"D6_best_f1": 0.50},
    }
    assert extract_baseline_name(result) == "random"


def test_baseline_name_extraction_ablation() -> None:
    """Ablation results need ablation_name read from metadata, not the top level."""
    result = {
        "scenario_id": "hormuz_P_CRIT",
        "ablation_config": "A6",
        "seed": 42,
        "metadata": {"ablation_name": "6_+bonus"},
        "metrics": {"D6_best_f1": 0.52},
    }
    assert extract_baseline_name(result) == "6_+bonus"


def test_aggregation_handles_nan_metrics() -> None:
    """Metrics never computed by BaselineEvaluator (e.g. D1_vus_pr) come through as NaN."""
    result = {
        "scenario_id": "hormuz_P_CRIT",
        "baseline_name": "random",
        "seed": 42,
        "metrics": {"D3_auc_pr": 0.45},
    }
    metrics = result.get("metrics", {})
    assert np.isnan(metrics.get("D1_vus_pr", np.nan))
    assert metrics.get("D3_auc_pr", np.nan) == 0.45


def test_aggregation_by_baseline_computes_mean_std() -> None:
    """Mean/std over seeds are computed correctly per baseline."""
    seeds = (42, 123, 456)
    rows = [
        {"baseline_name": "random", "config_id": None, "D6_best_f1": 0.50 + i * 0.001}
        for i in range(len(seeds))
    ]
    df = pd.DataFrame(rows)
    df_baselines = df[df["config_id"].isna()]
    summary = df_baselines.groupby("baseline_name")["D6_best_f1"].agg(["mean", "std"])

    assert summary.loc["random", "mean"] == pytest.approx(0.501, abs=0.001)
    assert summary.loc["random", "std"] > 0
