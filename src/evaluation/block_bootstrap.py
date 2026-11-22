"""Moving-block bootstrap confidence intervals for time-series detection metrics.

Daily detection scores and labels are strongly autocorrelated: the label marks
runs of 14+ sustained days, and features move over weeks. Resampling single
days (the ordinary bootstrap) treats each day as independent and gives
intervals that are too narrow. The moving-block bootstrap (Kuensch, 1989)
resamples contiguous blocks instead, which keeps the within-block dependence.

The decision threshold is held fixed across resamples. It was set on the
training window, so the interval reflects test-sample uncertainty only, not
re-calibration.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One moving-block resample of ``range(n)``.

    Blocks of ``block`` consecutive indices start at uniform random offsets.
    They are concatenated and the result is truncated to ``n``.

    Raises:
        ValueError: If ``block`` is below 1, or not shorter than the series.
            A block covering the whole series makes every resample the series
            itself, and the interval collapses to a false zero width.
    """
    if block < 1:
        raise ValueError(f"block must be at least 1 day, got {block}")
    if block >= n:
        raise ValueError(f"block ({block} days) must be shorter than the series ({n} days)")
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def block_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    block: int = 30,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile CIs for AUC, F1 and alert rate on one test window.

    With the same ``seed`` and series length, every method gets the same
    resamples, so intervals within a region are paired.

    Args:
        y_true: Binary labels for the test window, in time order.
        scores: The method's scores for the same days.
        threshold: Alert threshold, fixed from the training window.
        block: Block length in days.
        n_resamples: Number of bootstrap resamples.
        ci: Coverage of the interval, e.g. 0.95.
        seed: RNG seed.

    Returns:
        ``{metric}_ci_low`` / ``{metric}_ci_high`` for auc, f1 and alert_rate,
        plus ``bootstrap_valid``: the number of resamples containing both
        classes, on which AUC and F1 are computed. AUC and F1 bounds are NaN
        when there are none (a test window without positives).

    Raises:
        ValueError: If ``y_true`` and ``scores`` differ in length, if
            ``block`` is below 1 or not shorter than the series, if
            ``n_resamples`` is below 1, or if ``ci`` is not in (0, 1).
    """
    y = np.asarray(y_true).astype(int)
    s = np.nan_to_num(np.asarray(scores, dtype=float))
    if len(y) != len(s):
        raise ValueError(f"y_true has {len(y)} days, scores {len(s)}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be a fraction in (0, 1), got {ci}")
    if block < 1 or block >= len(y):
        raise ValueError(f"block ({block} days) must be in [1, {len(y)})")

    rng = np.random.default_rng(seed)
    auc, f1, alert = [], [], []
    for _ in range(n_resamples):
        idx = moving_block_indices(len(y), block, rng)
        yb, sb = y[idx], s[idx]
        predicted = (sb > threshold).astype(int)
        alert.append(predicted.mean())
        if yb.min() != yb.max():
            auc.append(roc_auc_score(yb, sb))
            f1.append(f1_score(yb, predicted, zero_division=0))

    tail = (1.0 - ci) / 2.0 * 100.0

    def bounds(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        lo, hi = np.percentile(values, [tail, 100.0 - tail])
        return float(lo), float(hi)

    out: dict[str, float] = {}
    for name, values in (("auc", auc), ("f1", f1), ("alert_rate", alert)):
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = bounds(values)
    out["bootstrap_valid"] = len(auc)
    return out
