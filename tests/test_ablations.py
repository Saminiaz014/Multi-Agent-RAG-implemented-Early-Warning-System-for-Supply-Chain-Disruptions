"""Tests for the Tier 5 ablations (R7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.ablation_runner import AblationRunner
from src.baselines.ablations import ABLATIONS
from src.baselines.agreement_bonus import AgreementBonusCalculator
from src.baselines.domain_scorers import DOMAIN_SCORERS


@pytest.fixture
def mock_scenario_df() -> pd.DataFrame:
    """A mock scenario with a multivariate event window (days 240-299)."""
    rng = np.random.default_rng(42)
    n_days = 365

    shipping = np.ones(n_days) * 70 + rng.normal(0, 3, n_days)
    market = rng.normal(0, 1, n_days)
    geopolitical = np.ones(n_days) * 0.05 + rng.normal(0, 0.02, n_days)
    routing = np.ones(n_days) * 0.08 + rng.normal(0, 0.03, n_days)
    news = np.ones(n_days) * 0.15 + rng.normal(0, 0.08, n_days)
    disaster = np.ones(n_days) * 0.02 + rng.normal(0, 0.01, n_days)

    shipping[240:300] = 40 + rng.normal(0, 2, 60)
    market[240:300] = market[240:300] + 1.5
    geopolitical[240:300] = geopolitical[240:300] + 0.5
    routing[240:300] = routing[240:300] + 0.3

    y_disruption = np.zeros(n_days, dtype=int)
    y_disruption[240:300] = 1

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_days),
        "shipping": shipping,
        "market": market,
        "geopolitical": geopolitical,
        "routing": routing,
        "news": news,
        "disaster": disaster,
        "y_disruption": y_disruption,
    })


def test_ablation_a0_single_domain() -> None:
    """A0 uses only the shipping domain."""
    config = ABLATIONS["A0"]
    assert config.agents == ["shipping"]
    assert sum(config.weights.values()) == pytest.approx(1.0)


def test_ablation_a6_agreement_bonus() -> None:
    """A6 has the agreement bonus enabled over all 6 domains."""
    config = ABLATIONS["A6"]
    assert config.use_agreement_bonus is True
    assert len(config.agents) == 6


def test_domain_scorer_shipping(mock_scenario_df: pd.DataFrame) -> None:
    """The shipping DomainScorer produces valid-range output."""
    scores = DOMAIN_SCORERS["shipping"].score(mock_scenario_df)

    assert scores.shape == (365,)
    assert scores.min() >= 0 and scores.max() <= 1


def test_agreement_bonus_calculator() -> None:
    """Agreement bonus applies the correct multiplier for >=3 agreeing domains."""
    calc = AgreementBonusCalculator(
        agreement_threshold=0.65, bonus_3_agents=0.15, bonus_5_agents=0.25
    )

    domain_scores = {
        "shipping": 0.80,
        "market": 0.70,
        "geopolitical": 0.75,
        "routing": 0.30,
        "news": 0.40,
        "disaster": 0.20,
    }
    composite = 0.50
    boosted, meta = calc.apply(composite, domain_scores, list(domain_scores.keys()))

    expected = 0.50 * 1.15  # 3 domains >= 0.65 -> 1.15x
    assert boosted == pytest.approx(expected)
    assert meta["agreeing_domains"] == 3


def test_ablation_runner_runs(mock_scenario_df: pd.DataFrame) -> None:
    """AblationRunner executes A0 (untuned) without error."""
    config = ABLATIONS["A0"]
    runner = AblationRunner(config)
    scores, metadata = runner.run(mock_scenario_df, "test_scenario", seed=42)

    assert scores.shape == (365,)
    assert scores.min() >= 0 and scores.max() <= 1
    assert metadata["ablation_config"] == "A0"
    assert metadata["weights_tuned"] is False


def test_ablation_config_weights_normalized() -> None:
    """All ablation config weights sum to 1.0."""
    for config_id, config in ABLATIONS.items():
        assert sum(config.weights.values()) == pytest.approx(1.0), (
            f"{config_id} weights don't sum to 1.0"
        )
