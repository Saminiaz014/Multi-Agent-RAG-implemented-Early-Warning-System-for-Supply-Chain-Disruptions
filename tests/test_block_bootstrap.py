"""Moving-block bootstrap confidence intervals for the method comparison."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.block_bootstrap import block_bootstrap_ci, moving_block_indices


def test_indices_are_contiguous_blocks_of_the_requested_length() -> None:
    idx = moving_block_indices(100, 30, np.random.default_rng(0))
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100
    for start in range(0, 90, 30):                        # first three full blocks
        block = idx[start:start + 30]
        assert (np.diff(block) == 1).all()


def test_block_not_shorter_than_series_is_refused() -> None:
    # Every resample would be the series itself: a zero-width interval that
    # reads as certainty.
    for n in (10, 30):
        with pytest.raises(ValueError, match="block"):
            moving_block_indices(n, 30, np.random.default_rng(0))
    with pytest.raises(ValueError, match="block"):
        block_bootstrap_ci(np.r_[np.zeros(15), np.ones(10)], np.arange(25.0),
                           threshold=0.5, block=30, n_resamples=10)


@pytest.mark.parametrize("bad", [{"block": 0}, {"block": -3}, {"n_resamples": 0},
                                 {"ci": 0.0}, {"ci": 1.0}, {"ci": 95}])
def test_bad_settings_are_refused(bad) -> None:
    y = (np.arange(200) // 40 % 2).astype(int)
    with pytest.raises(ValueError):
        block_bootstrap_ci(y, y.astype(float), threshold=0.5,
                           **{"block": 30, "n_resamples": 10, **bad})


def test_same_seed_same_resamples() -> None:
    a = moving_block_indices(500, 30, np.random.default_rng(7))
    b = moving_block_indices(500, 30, np.random.default_rng(7))
    assert (a == b).all()


def _runs(n: int = 900, run: int = 60) -> np.ndarray:
    return ((np.arange(n) // run) % 3 == 0).astype(int)   # 60-day episodes


def test_ci_brackets_the_point_estimate() -> None:
    y = _runs()
    rng = np.random.default_rng(1)
    scores = y * 0.6 + rng.normal(0, 0.5, len(y))
    out = block_bootstrap_ci(y, scores, threshold=0.5, block=30, n_resamples=400, seed=42)
    from sklearn.metrics import roc_auc_score
    point = roc_auc_score(y, scores)
    assert out["auc_ci_low"] <= point <= out["auc_ci_high"]
    assert out["auc_ci_low"] < out["auc_ci_high"]
    assert out["f1_ci_low"] <= out["f1_ci_high"]
    assert 0.0 <= out["alert_rate_ci_low"] <= out["alert_rate_ci_high"] <= 1.0


def test_perfect_separation_gives_a_degenerate_interval() -> None:
    y = _runs()
    out = block_bootstrap_ci(y, y.astype(float), threshold=0.5, block=30,
                             n_resamples=200, seed=42)
    assert out["auc_ci_low"] == out["auc_ci_high"] == 1.0


def test_one_class_resamples_are_skipped_and_counted() -> None:
    y = np.zeros(600, dtype=int)
    y[100:115] = 1                                         # one short episode
    scores = y + np.random.default_rng(3).normal(0, 0.3, 600)
    out = block_bootstrap_ci(y, scores, threshold=0.5, block=30, n_resamples=300, seed=42)
    assert 0 < out["bootstrap_valid"] < 300                # some resamples miss it


def test_single_class_test_window_gives_no_auc_ci_but_an_alert_rate_ci() -> None:
    y = np.zeros(300, dtype=int)                           # Malacca: no positives
    scores = np.random.default_rng(4).uniform(size=300)
    out = block_bootstrap_ci(y, scores, threshold=0.9, block=30, n_resamples=200, seed=42)
    assert np.isnan(out["auc_ci_low"]) and np.isnan(out["auc_ci_high"])
    assert out["bootstrap_valid"] == 0
    assert out["alert_rate_ci_low"] <= 0.1 <= out["alert_rate_ci_high"]


def test_blocks_widen_the_interval_for_autocorrelated_data() -> None:
    y = _runs()
    rng = np.random.default_rng(5)
    drift = np.cumsum(rng.normal(0, 0.15, len(y)))         # slow, autocorrelated error
    scores = y * 0.5 + drift
    iid = block_bootstrap_ci(y, scores, threshold=0.5, block=1, n_resamples=400, seed=42)
    blk = block_bootstrap_ci(y, scores, threshold=0.5, block=30, n_resamples=400, seed=42)
    assert (blk["auc_ci_high"] - blk["auc_ci_low"]) > (iid["auc_ci_high"] - iid["auc_ci_low"])


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        block_bootstrap_ci(np.zeros(10), np.zeros(9), threshold=0.5)


# ------------------------------------------------- masked days, paired deltas
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.evaluation import block_bootstrap as bb  # noqa: E402


def _pair(n: int = 900, seed: int = 11):
    y = _runs(n)
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.6, n)                 # shared error: correlated methods
    a = y * 0.6 + common + rng.normal(0, 0.2, n)
    b = y * 0.5 + common + rng.normal(0, 0.2, n)
    return y, a, b


def test_masked_days_are_left_out_of_point_and_interval() -> None:
    y, a, _ = _pair()
    keep = np.ones(len(y), dtype=bool)
    keep[300:318] = False                          # an 18-day outage
    out = block_bootstrap_ci(y, a, 0.5, keep=keep, block=30, n_resamples=50)
    full = block_bootstrap_ci(y, a, 0.5, block=30, n_resamples=50)
    a_masked = a.copy()
    a_masked[~keep] = 99.0                         # garbage on masked days must not matter
    again = block_bootstrap_ci(y, a_masked, 0.5, keep=keep, block=30, n_resamples=50)
    assert out == again
    assert out != full


def test_keep_all_true_is_the_same_as_no_mask() -> None:
    y, a, _ = _pair()
    keep = np.ones(len(y), dtype=bool)
    assert (block_bootstrap_ci(y, a, 0.5, keep=keep, block=30, n_resamples=50)
            == block_bootstrap_ci(y, a, 0.5, block=30, n_resamples=50))


def test_delta_of_a_method_with_itself_is_exactly_zero() -> None:
    y, a, _ = _pair()
    d = bb.paired_auc_delta_ci(y, a, a, block=30, n_resamples=100)
    assert d["delta_auc"] == d["delta_auc_ci_low"] == d["delta_auc_ci_high"] == 0.0


def test_delta_ci_is_the_percentile_of_per_resample_differences() -> None:
    y, a, b = _pair()
    d = bb.paired_auc_delta_ci(y, a, b, block=30, n_resamples=200, seed=42)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(200):
        idx = moving_block_indices(len(y), 30, rng)
        if y[idx].min() != y[idx].max():
            diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    assert d["delta_auc"] == pytest.approx(roc_auc_score(y, a) - roc_auc_score(y, b))
    assert d["delta_auc_ci_low"] == pytest.approx(lo)
    assert d["delta_auc_ci_high"] == pytest.approx(hi)
    assert d["delta_valid"] == len(diffs)


def test_paired_delta_is_narrower_than_either_marginal_interval() -> None:
    y, a, b = _pair()
    d = bb.paired_auc_delta_ci(y, a, b, block=30, n_resamples=300)
    ma = block_bootstrap_ci(y, a, 0.5, block=30, n_resamples=300)
    mb = block_bootstrap_ci(y, b, 0.5, block=30, n_resamples=300)
    width = d["delta_auc_ci_high"] - d["delta_auc_ci_low"]
    assert width < ma["auc_ci_high"] - ma["auc_ci_low"]
    assert width < mb["auc_ci_high"] - mb["auc_ci_low"]


def test_every_method_and_every_delta_sees_the_same_resampled_blocks(monkeypatch) -> None:
    calls: list[list[int]] = []
    real = bb.moving_block_indices

    def record(n, block, rng):
        idx = real(n, block, rng)
        calls.append(idx.tolist())
        return idx

    monkeypatch.setattr(bb, "moving_block_indices", record)
    y, a, b = _pair()
    bb.block_bootstrap_ci(y, a, 0.5, block=30, n_resamples=20)
    first = calls[:]
    calls.clear()
    bb.block_bootstrap_ci(y, b * 3 - 1, 0.1, block=30, n_resamples=20)   # other scores
    second = calls[:]
    calls.clear()
    bb.paired_auc_delta_ci(y, a, b, block=30, n_resamples=20)
    assert first == second == calls


def test_delta_rejects_mismatched_lengths() -> None:
    y, a, b = _pair()
    with pytest.raises(ValueError):
        bb.paired_auc_delta_ci(y, a, b[:-1], block=30, n_resamples=10)
