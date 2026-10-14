"""Exact agent Shapley values: the axioms, the closed form, and Path C's bonus/clip."""

import numpy as np
import pytest

from src.explainability.exact_shapley import (
    decompose_path_c,
    efficiency_residual,
    minimal_drop,
    path_d_closed_form,
    shapley_agents,
    value_fn_path_c,
    value_fn_path_d,
)

AGENTS = ["shipping", "market", "geopolitical", "natural_disaster", "news_sentiment"]
WEIGHTS = {"shipping": 0.25, "market": 0.15, "geopolitical": 0.25,
           "natural_disaster": 0.10, "news_sentiment": 0.10}


def _random_point(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    return (dict(zip(AGENTS, rng.uniform(0, 1, 5))),
            dict(zip(AGENTS, rng.uniform(0, 0.4, 5))))


@pytest.mark.parametrize("seed", range(20))
def test_efficiency_path_c(seed: int) -> None:
    scores, base = _random_point(seed)
    f = value_fn_path_c(WEIGHTS, AGENTS, 1.15, 1.25)
    phi = shapley_agents(scores, base, f)
    assert abs(efficiency_residual(phi, scores, base, f)) < 1e-12


def test_symmetry() -> None:
    weights = {"a": 1.0, "b": 1.0, "c": 2.0}
    f = value_fn_path_c(weights, ["a", "b", "c"], 1.15, 1.25)
    phi = shapley_agents({"a": 0.7, "b": 0.7, "c": 0.2}, {"a": 0.1, "b": 0.1, "c": 0.1}, f)
    assert phi["a"] == pytest.approx(phi["b"], abs=1e-15)


def test_dummy_agent_gets_zero() -> None:
    weights = {**WEIGHTS, "news_sentiment": 0.0}
    scores, base = _random_point(3)
    phi = shapley_agents(scores, base, value_fn_path_d(weights, AGENTS))
    assert phi["news_sentiment"] == 0.0


def test_agent_at_baseline_gets_zero_in_path_d() -> None:
    scores, base = _random_point(4)
    scores["market"] = base["market"]
    phi = shapley_agents(scores, base, value_fn_path_d(WEIGHTS, AGENTS))
    assert abs(phi["market"]) < 1e-15


@pytest.mark.parametrize("seed", range(10))
def test_path_d_closed_form_equals_generic(seed: int) -> None:
    scores, base = _random_point(seed)
    agents = AGENTS[: 1 + seed % 5]
    scores = {a: scores[a] for a in agents}
    generic = shapley_agents(scores, base, value_fn_path_d(WEIGHTS, agents))
    closed = path_d_closed_form(scores, base, WEIGHTS, agents)
    for a in agents:
        assert generic[a] == pytest.approx(closed[a], abs=1e-15)


def test_path_c_five_agents_distributes_bonus() -> None:
    scores = dict.fromkeys(AGENTS, 0.6)
    scores["shipping"] = 0.9
    base = dict.fromkeys(AGENTS, 0.1)
    parts = decompose_path_c(scores, base, WEIGHTS, AGENTS, 1.15, 1.25)

    f_lin = value_fn_path_d(WEIGHTS, AGENTS)
    expected_bonus = (1.25 - 1.0) * f_lin(scores)  # baseline has no agreement
    assert sum(parts["bonus"].values()) == pytest.approx(expected_bonus, abs=1e-12)
    assert all(v > 0 for v in parts["bonus"].values())
    assert all(abs(v) < 1e-15 for v in parts["clip"].values())
    for a in AGENTS:
        recomposed = parts["linear"][a] + parts["bonus"][a] + parts["clip"][a]
        assert recomposed == pytest.approx(parts["total"][a], abs=1e-12)


def test_clipped_case_sums_to_one() -> None:
    scores = dict.fromkeys(AGENTS, 0.95)
    base = dict.fromkeys(AGENTS, 0.0)
    parts = decompose_path_c(scores, base, WEIGHTS, AGENTS, 1.15, 1.25)
    assert sum(parts["total"].values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(parts["clip"].values()) < 0  # the clip removed score


def test_minimal_drop_matches_path_d_closed_form() -> None:
    scores, base = _random_point(7)
    f = value_fn_path_d(WEIGHTS, AGENTS)
    threshold = f(scores) - 0.02
    t = minimal_drop("shipping", scores, f, threshold)
    mass = sum(WEIGHTS.values())
    expected = scores["shipping"] - 0.02 * mass / WEIGHTS["shipping"]
    assert t == pytest.approx(expected, abs=1e-12)


def test_minimal_drop_none_when_one_agent_cannot_clear() -> None:
    scores = dict.fromkeys(AGENTS, 0.9)
    f = value_fn_path_d(WEIGHTS, AGENTS)
    assert minimal_drop("news_sentiment", scores, f, threshold=0.5) is None


def test_minimal_drop_path_c_crosses_agreement_step() -> None:
    weights = {"a": 1.0, "b": 1.0, "c": 1.0}
    f = value_fn_path_c(weights, ["a", "b", "c"], 1.5, 1.5)
    scores = {"a": 0.55, "b": 0.55, "c": 0.55}  # amplified: 0.825
    t = minimal_drop("a", scores, f, threshold=0.6)
    # Dropping a to 0.5 removes the bonus (linear 0.5333 <= 0.6).
    assert t == pytest.approx(0.5, abs=1e-9)
