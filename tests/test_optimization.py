"""Tests for the Optuna weight-optimization layer.

Covers the train/val/test split manager, the Optuna parameter space and
objective, the ``weight_mode`` hand-tuned ↔ optimized switch, an end-to-end
short optimization run, and a guard proving the test split is never touched
during optimization.
"""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest
import yaml

from src.aggregation.risk_engine import RiskEngine
from src.agents.base_agent import DetectionResult
from src.optimization.data_split import DataSplitManager
from src.optimization.pipeline_evaluator import PipelineEvaluator
from src.optimization.weight_config import (
    load_optimized_weights,
    resolve_active_weights,
)
from src.optimization.weight_optimizer import WeightOptimizer
from src.optimization import weight_optimizer as _wo

import numpy as np

_INVALID = _wo._INVALID_OBJECTIVE

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CONFIG_PATH = Path("config/settings.yaml")
_EXPECTED_DISRUPTION_DAYS = 47  # 3 injected scenarios at fixed day positions


def _load_config() -> dict:
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def config() -> dict:
    return _load_config()


@pytest.fixture(scope="module")
def data_manager(config: dict) -> DataSplitManager:
    """Module-scoped manager so the splits are generated only once."""
    manager = DataSplitManager(config)
    manager.generate_splits()
    return manager


# ---------------------------------------------------------------------------
# 1. Data splits
# ---------------------------------------------------------------------------
def test_data_splits(data_manager: DataSplitManager) -> None:
    splits = data_manager.get_splits()
    assert set(splits) == {"train", "validation", "test"}

    for split_name, frames in splits.items():
        assert set(frames) == {
            "shipping", "market", "geopolitical",
            "natural_disaster", "routing", "news_sentiment",
        }
        for conn_name, frame in frames.items():
            assert len(frame) == 365, f"{split_name}/{conn_name} not 365 rows"

    # Same disruption structure across splits (~46-47 days each).
    counts = {s: int(data_manager.get_ground_truth(s).sum()) for s in splits}
    assert len(set(counts.values())) == 1, f"disruption days differ: {counts}"
    assert counts["train"] == _EXPECTED_DISRUPTION_DAYS

    # Noise differs: normal-day vessel_count is decorrelated across splits.
    report = data_manager.validate_splits()
    assert abs(report["normal_day_correlation"]) < 0.5
    assert report["rows_ok"] and report["disruption_ok"] and report["noise_ok"]


def test_ground_truth_not_a_feature(data_manager: DataSplitManager) -> None:
    """is_disruption is an evaluation label, indexed by timestamp."""
    gt = data_manager.get_ground_truth("validation")
    assert gt.dtype == bool
    assert gt.index.name == "timestamp" or gt.index.dtype.kind == "M"
    assert int(gt.sum()) == _EXPECTED_DISRUPTION_DAYS


# ---------------------------------------------------------------------------
# 2. Parameter space
# ---------------------------------------------------------------------------
def test_parameter_space(config: dict, data_manager: DataSplitManager) -> None:
    optimizer = WeightOptimizer(config, data_manager=data_manager)
    trial = optimizer.study.ask()
    params = optimizer.define_parameter_space(trial)

    # Inter-agent weights normalise to 1.0.
    assert sum(params["inter_weights"].values()) == pytest.approx(1.0, abs=1e-6)

    # Every intra-agent weight group normalises to 1.0.
    for agent, group in params["intra"].items():
        assert sum(group.values()) == pytest.approx(1.0, abs=1e-6), agent

    # Thresholds present and within suggested bounds.
    thr = params["thresholds"]
    assert 0.55 <= thr["risk_high"] <= 0.85
    assert 0.25 <= thr["risk_medium"] <= 0.55
    assert thr["agreement_bonus_5"] >= 1.1


def test_parameter_space_disabled_layers_use_hand_tuned(config: dict) -> None:
    cfg = {**config}
    cfg["optimization"] = {
        **config["optimization"],
        "parameter_space": {
            "inter_agent_weights": False,
            "intra_agent_weights": False,
            "thresholds": False,
        },
    }
    optimizer = WeightOptimizer(cfg, data_manager=DataSplitManager(cfg))
    trial = optimizer.study.ask()
    params = optimizer.define_parameter_space(trial)
    hand = resolve_active_weights({**cfg, "weight_mode": "hand_tuned"})
    assert params["inter_weights"] == hand["inter_agent_weights"]


# ---------------------------------------------------------------------------
# 3. Objective function
# ---------------------------------------------------------------------------
def test_objective_function(config: dict, data_manager: DataSplitManager) -> None:
    optimizer = WeightOptimizer(config, data_manager=data_manager)

    # Ask several trials so at least one clears the threshold constraints and
    # actually runs an evaluation (constraint-violating trials short-circuit
    # to the invalid sentinel without touching any split).
    values = []
    for _ in range(10):
        trial = optimizer.study.ask()
        values.append(optimizer.objective(trial))

    assert all(isinstance(v, float) for v in values)
    assert any(v > _INVALID for v in values), "no valid trial produced a score"
    # Objective evaluates on validation, never on test.
    assert "validation" in optimizer.evaluator.evaluated_splits
    assert "test" not in optimizer.evaluator.evaluated_splits


# ---------------------------------------------------------------------------
# 4. weight_mode switch
# ---------------------------------------------------------------------------
def _sample_results() -> list[DetectionResult]:
    rng = np.random.default_rng(0)
    names = [
        "shipping", "market", "geopolitical",
        "natural_disaster", "routing", "news_sentiment",
    ]
    out = []
    for n in names:
        scores = rng.uniform(0.4, 0.8, size=20)
        out.append(
            DetectionResult(
                agent_name=n,
                anomaly_scores=scores,
                anomaly_flags=scores > 0.5,
                feature_names=["f"],
            )
        )
    return out


def test_weight_mode_switch(config: dict, tmp_path: Path) -> None:
    # Hand-tuned engine.
    hand_cfg = {**config, "weight_mode": "hand_tuned"}
    hand_engine = RiskEngine(hand_cfg)
    hand_risk = hand_engine.compute_risk(_sample_results())["risk_score"]

    # A dummy optimized file with clearly different inter-agent weights.
    opt_file = tmp_path / "optimized_weights.yaml"
    opt_file.write_text(
        yaml.safe_dump(
            {
                "inter_agent_weights": {
                    "shipping": 0.50, "market": 0.05, "geopolitical": 0.05,
                    "natural_disaster": 0.05, "routing": 0.30, "news_sentiment": 0.05,
                },
                "thresholds": {
                    "risk_high": 0.55, "risk_medium": 0.30,
                    "agreement_bonus_3": 1.20, "agreement_bonus_5": 1.40,
                },
            }
        ),
        encoding="utf-8",
    )
    opt_cfg = {
        **config,
        "weight_mode": "optimized",
        "optimization": {
            **config["optimization"],
            "optimized_weights_path": str(opt_file),
        },
    }

    # The resolver loads the different weights.
    loaded = load_optimized_weights(opt_cfg)
    assert loaded["inter_agent_weights"]["shipping"] == 0.50

    opt_engine = RiskEngine(opt_cfg)
    assert opt_engine.weights != hand_engine.weights
    assert opt_engine.agreement_bonus_5 == 1.40

    opt_risk = opt_engine.compute_risk(_sample_results())["risk_score"]
    # Same inputs, different weights → different composite risk.
    assert opt_risk != pytest.approx(hand_risk)


def test_weight_mode_missing_file_falls_back(config: dict, tmp_path: Path) -> None:
    cfg = {
        **config,
        "weight_mode": "optimized",
        "optimization": {
            **config["optimization"],
            "optimized_weights_path": str(tmp_path / "does_not_exist.yaml"),
        },
    }
    layout = resolve_active_weights(cfg)
    assert layout["source"] == "hand_tuned"  # graceful fallback


# ---------------------------------------------------------------------------
# 5. End-to-end short optimization
# ---------------------------------------------------------------------------
def test_optimization_short(
    config: dict, data_manager: DataSplitManager, tmp_path: Path
) -> None:
    opt_file = tmp_path / "optimized_weights.yaml"
    cfg = {
        **config,
        "optimization": {
            **config["optimization"],
            "optimized_weights_path": str(opt_file),
        },
    }
    optimizer = WeightOptimizer(cfg, data_manager=data_manager)
    results = optimizer.optimize(n_trials=5, timeout=60)

    assert opt_file.exists(), "optimized_weights.yaml not written"
    assert Path("data/processed/optimization_results.json").exists()

    # The written file round-trips into the resolver.
    written = load_optimized_weights(cfg)
    assert "inter_agent_weights" in written
    assert sum(written["inter_agent_weights"].values()) == pytest.approx(1.0, abs=1e-3)

    # Results carry both validation and test metrics + a baseline comparison.
    assert "validation_metrics" in results
    assert "test_metrics" in results
    assert "hand_tuned_metrics" in results
    assert 0.0 <= results["test_metrics"]["f1"] <= 1.0


# ---------------------------------------------------------------------------
# 6. No test-set leakage
# ---------------------------------------------------------------------------
def test_no_test_leakage(config: dict, data_manager: DataSplitManager) -> None:
    optimizer = WeightOptimizer(config, data_manager=data_manager)
    optimizer.study.optimize(optimizer.objective, n_trials=3)

    # During optimization only train (fit) + validation (eval) are touched.
    assert "test" not in optimizer.evaluator.evaluated_splits
    assert "validation" in optimizer.evaluator.evaluated_splits

    # The test split is reached only via the explicit final evaluation.
    params = optimizer._hand_tuned_params()
    optimizer.evaluate_on_test(params)
    assert "test" in optimizer.evaluator.evaluated_splits
