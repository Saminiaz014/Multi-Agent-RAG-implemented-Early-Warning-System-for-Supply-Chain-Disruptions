"""Alert Explanation Records: episodes, exactness, counterfactual and uncertainty."""

import numpy as np
import pytest

from src.explainability.alert_explanation import build_aer, episodes, noise_scales

AGENTS = ["shipping", "market", "geopolitical"]
WEIGHTS = {"shipping": 0.25, "market": 0.15, "geopolitical": 0.25}


def _aer(scores, baselines, threshold, path="D", **kw):
    breakdowns = {a: {"agent": a, "terms": {"x": scores[a]}, "branch_winner": "linear"}
                  for a in AGENTS}
    return build_aer(
        alert_id="r_2024-01-01", episode_id="r_E001", region="r", date="2024-01-01",
        path=path, weight_mode="hand_tuned", data_origin="toy", tier=3, agents=AGENTS,
        weights=WEIGHTS, scores=scores, baselines=baselines, threshold=threshold,
        breakdowns=breakdowns, noise_sigma=dict.fromkeys(AGENTS, 0.0), y_true=1, **kw,
    )


def test_episodes_merge_gaps_of_three_calendar_days() -> None:
    dates = ["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-11", "2024-01-12"]
    alert = np.array([True, True, True, True, False])
    # 01-02 -> 01-06 has 3 quiet days between (merge); 01-06 -> 01-11 has 4 (split).
    assert episodes(dates, alert) == [[0, 1, 2], [3]]


def test_episodes_use_dates_not_row_positions() -> None:
    dates = ["2024-01-01", "2024-02-01"]  # adjacent rows, a month apart
    assert episodes(dates, np.array([True, True])) == [[0], [1]]


def test_path_d_record_is_exact() -> None:
    scores = {"shipping": 0.9, "market": 0.5, "geopolitical": 0.3}
    base = {"shipping": 0.4, "market": 0.5, "geopolitical": 0.3}
    rec = _aer(scores, base, threshold=0.4)
    mass = 0.65
    assert rec.agent_shapley["shipping"] == pytest.approx(0.25 * 0.5 / mass, abs=1e-15)
    assert rec.agent_shapley["market"] == pytest.approx(0.0, abs=1e-15)
    assert abs(rec.efficiency_residual) < 1e-12
    assert rec.top_drivers[0]["agent"] == "shipping"
    assert rec.margin == pytest.approx(rec.score - 0.4)
    assert rec.label == {"y_true": 1, "alert_correct": "TP"}
    assert rec.bonus_share is None


def test_counterfactual_feasibility() -> None:
    scores = {"shipping": 0.9, "market": 0.5, "geopolitical": 0.3}
    base = {"shipping": 0.4, "market": 0.5, "geopolitical": 0.3}
    f = (0.25 * 0.9 + 0.15 * 0.5 + 0.25 * 0.3) / 0.65
    # Just above threshold: a small drop in shipping clears it, staying above calm.
    rec = _aer(scores, base, threshold=f - 0.01)
    assert rec.counterfactual["feasible"] is True
    assert rec.counterfactual["required_score_drop"] == pytest.approx(0.01 * 0.65 / 0.25, abs=1e-9)
    # Threshold below the calm composite: shipping would have to go below calm.
    rec = _aer(scores, base, threshold=0.30)
    assert rec.counterfactual["feasible"] is False


def test_path_c_reports_bonus_share() -> None:
    scores = dict.fromkeys(AGENTS, 0.8)
    base = dict.fromkeys(AGENTS, 0.1)
    rec = _aer(scores, base, threshold=0.5, path="C", bonus3=1.15, bonus5=1.25)
    assert rec.bonus_share["bonus_total"] == pytest.approx(0.15 * 0.8, abs=1e-12)
    assert sum(rec.agent_shapley.values()) == pytest.approx(rec.score - rec.baseline_value)


def test_zero_noise_band_collapses_to_score() -> None:
    scores = {"shipping": 0.9, "market": 0.5, "geopolitical": 0.3}
    rec = _aer(scores, dict.fromkeys(AGENTS, 0.2), threshold=0.4)
    assert rec.uncertainty["bootstrap_lo"] == pytest.approx(rec.score)
    assert rec.uncertainty["bootstrap_hi"] == pytest.approx(rec.score)


def test_noise_scale_ignores_pairs_touching_disruption_days() -> None:
    scores = {"a": np.array([0.0, 0.0, 1.0, 0.0])}
    calm = np.array([True, True, False, True])
    assert noise_scales(scores, calm)["a"] == 0.0
