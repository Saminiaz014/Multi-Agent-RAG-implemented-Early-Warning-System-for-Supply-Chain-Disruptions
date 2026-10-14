"""METRIC 9 primitives on toy inputs with known answers, plus the real-data inputs."""

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation import path_d
from src.evaluation.explainability_metrics import (
    auc,
    bootstrap_auc,
    deletion_insertion,
    episode_bootstrap,
    k_cover,
    order_by,
    random_order_auc,
    spearman,
    top1_agreement,
)
from src.explainability.feature_shapley import coalition_scores, feature_shapley

_W = {"a": 0.5, "b": 0.3, "c": 0.2}


def _linear(coalition: frozenset) -> float:
    return sum(_W[p] for p in coalition)


# ------------------------------------------------------------------ E2
def test_exact_order_beats_random_on_linear_game() -> None:
    best = deletion_insertion(order_by(_W, list(_W)), _linear)
    rand = random_order_auc(list(_W), _linear)
    worst = deletion_insertion(["c", "b", "a"], _linear)
    assert best[0] < rand[0] < worst[0]
    assert best[1] > rand[1] > worst[1]


def test_random_order_is_half_for_linear_game() -> None:
    assert random_order_auc(list(_W), _linear) == pytest.approx((0.5, 0.5))


def test_known_deletion_value() -> None:
    # Curve 1, 0.5, 0.2, 0 at x = 0, 1/3, 2/3, 1 -> trapezoid area 0.4.
    d, i = deletion_insertion(["a", "b", "c"], _linear)
    assert d == pytest.approx((0.75 + 0.35 + 0.1) / 3)
    assert i == pytest.approx(1 - d)


def test_small_gap_is_undefined() -> None:
    assert np.isnan(deletion_insertion(["a"], lambda c: 0.001 * len(c))[0])


# ------------------------------------------------------------------ E3-E7
def test_k_cover() -> None:
    assert k_cover({"a": 0.7, "b": 0.2, "c": 0.1}) == 2
    assert k_cover({"a": 0.9, "b": 0.1}) == 1
    assert k_cover({"a": -0.5, "b": 0.5}) == 2
    assert np.isnan(k_cover({"a": 0.0}))


def test_stability_of_identical_inputs_is_one() -> None:
    phi = {"a": 0.3, "b": 0.1, "c": -0.2}
    assert spearman(phi, dict(phi)) == pytest.approx(1.0)
    assert top1_agreement(phi, dict(phi)) == 1.0


def test_auc_perfect_and_undefined() -> None:
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert np.isnan(auc([0.1, 0.2], [1, 1]))
    point, lo, hi = bootstrap_auc([0.1, 0.2, 0.8, 0.9, 0.85], [0, 0, 1, 1, 1])
    assert point == 1.0 and lo == 1.0 and hi == 1.0


def test_episode_bootstrap_is_deterministic_and_brackets_mean() -> None:
    values = [0.1, 0.4, 0.5, 0.9, np.nan]
    first, second = episode_bootstrap(values), episode_bootstrap(values)
    assert first == second
    mean, lo, hi = first
    assert mean == pytest.approx(0.475)
    assert lo <= mean <= hi


# ------------------------------------------------------------- feature games
def _market_agent():
    return SimpleNamespace(
        _z_threshold=2.0,
        _active_features=lambda: ["brent_crude_usd", "trade_volume_index"],
        _feature_weights=lambda: {"brent_crude_usd": 0.6, "trade_volume_index": 0.4},
    )


def test_market_feature_game_with_clip() -> None:
    agent = _market_agent()
    row = pd.Series({"oil_zscore": 4.0, "trade_volume_zscore": 2.0,
                     "anomaly_score": 1.0})  # 1.2 + 0.4 -> clipped to 1
    calm = {"brent_crude_usd": 0.2, "trade_volume_index": 0.1}
    res = feature_shapley("market", agent, row, calm)
    assert sum(res["phi"].values()) == pytest.approx(1.0 - 0.3)
    # Oil alone already saturates: v({oil}) = 1, v({trade}) = 0.6, v({}) = 0.3.
    assert res["phi"]["brent_crude_usd"] == pytest.approx(0.5 * (0.7 + 0.4))
    assert res["phi"]["trade_volume_index"] == pytest.approx(0.5 * (0.3 + 0.0))


def test_feature_game_rejects_mismatch() -> None:
    row = pd.Series({"oil_zscore": 4.0, "trade_volume_zscore": 2.0, "anomaly_score": 0.5})
    with pytest.raises(AssertionError):
        feature_shapley("market", _market_agent(), row,
                        {"brent_crude_usd": 0.2, "trade_volume_index": 0.1})


def test_coalition_table_is_complete() -> None:
    full = {"brent_crude_usd": 1.2, "trade_volume_index": 0.4}
    table = coalition_scores("market", _market_agent(), full,
                             {"brent_crude_usd": 0.2, "trade_volume_index": 0.1})
    expected = {frozenset(c) for k in range(3) for c in combinations(full, k)}
    assert set(table) == expected


# ------------------------------------------------------------- real data
@pytest.mark.skipif(not path_d.eval_data_available(), reason="data/eval cache not built")
@pytest.mark.parametrize("region", path_d.REGIONS)
def test_alert_set_matches_csv_composite(region: str) -> None:
    from src.explainability.alert_sets import build_alert_set

    aset = build_alert_set(region, "with_duration")
    np.testing.assert_array_equal(aset.composite, aset.data.tier_scores(5))
    assert aset.threshold == aset.data.tier_threshold(5)
    nd = build_alert_set(region, "without_duration")
    assert len(nd.alert_idx) > 0
