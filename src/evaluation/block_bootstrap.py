"""Moving-block bootstrap confidence intervals for time-series detection metrics.

Daily detection scores and labels are strongly autocorrelated: the label marks
runs of 14+ sustained days, and features move over weeks. Resampling single
days (the ordinary bootstrap) treats each day as independent and gives
intervals that are too narrow. The moving-block bootstrap (Kuensch, 1989)
resamples contiguous blocks instead, which keeps the within-block dependence.

The decision threshold is held fixed across resamples. It was set on the
training window, so the interval reflects test-sample uncertainty only, not
re-calibration.

Blocks are drawn over calendar days. Days that must not be scored (``keep``
False, e.g. a news outage) are dropped from each resample after drawing, so a
mask changes which days count but not the block structure, and every method
in a region still sees the same resampled blocks.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

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


def _check(n: int, block: int, n_resamples: int, ci: float) -> None:
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be a fraction in (0, 1), got {ci}")
    if block < 1 or block >= n:
        raise ValueError(f"block ({block} days) must be in [1, {n})")


def _keep_mask(keep: np.ndarray | None, n: int) -> np.ndarray:
    if keep is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(keep, dtype=bool)
    if len(mask) != n:
        raise ValueError(f"keep has {len(mask)} days, the series {n}")
    return mask


def _resamples(keep: np.ndarray, block: int, n_resamples: int,
               seed: int) -> Iterator[np.ndarray]:
    """The resampled day indices, masked days removed.

    The RNG depends only on ``seed`` and the series length, never on the
    scores, so every caller with the same length gets the same blocks.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        idx = moving_block_indices(len(keep), block, rng)
        yield idx[keep[idx]]


def _bounds(values: list[float], ci: float) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    tail = (1.0 - ci) / 2.0 * 100.0
    lo, hi = np.percentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


def block_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    keep: np.ndarray | None = None,
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
        threshold: Alert threshold, fixed from the training window; a day
            alerts when its score is ``>=`` it.
        keep: True on days that are scored. ``None`` scores every day.
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
        ValueError: If ``y_true``, ``scores`` and ``keep`` differ in length,
            if ``block`` is below 1 or not shorter than the series, if
            ``n_resamples`` is below 1, or if ``ci`` is not in (0, 1).
    """
    y = np.asarray(y_true).astype(int)
    s = np.nan_to_num(np.asarray(scores, dtype=float))
    if len(y) != len(s):
        raise ValueError(f"y_true has {len(y)} days, scores {len(s)}")
    _check(len(y), block, n_resamples, ci)
    mask = _keep_mask(keep, len(y))

    auc, f1, alert = [], [], []
    for idx in _resamples(mask, block, n_resamples, seed):
        yb, sb = y[idx], s[idx]
        predicted = (sb >= threshold).astype(int)          # same rule as the point metrics
        alert.append(predicted.mean() if len(idx) else np.nan)
        if len(idx) and yb.min() != yb.max():
            auc.append(roc_auc_score(yb, sb))
            f1.append(f1_score(yb, predicted, zero_division=0))

    out: dict[str, float] = {}
    alert = [a for a in alert if not np.isnan(a)]
    for name, values in (("auc", auc), ("f1", f1), ("alert_rate", alert)):
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = _bounds(values, ci)
    out["bootstrap_valid"] = len(auc)
    return out


def paired_auc_delta_ci(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    keep: np.ndarray | None = None,
    block: int = 30,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired CI for ``AUC(a) - AUC(b)`` on the same resampled blocks.

    Both methods are scored on each resample, and the interval is taken over
    the per-resample differences. Because the two scores share the days, the
    difference interval can be much narrower than either marginal one, so it
    can separate methods whose marginal intervals overlap. With the same
    ``seed`` it uses exactly the resamples of :func:`block_bootstrap_ci`.

    Returns:
        ``delta_auc`` (the point difference on the kept days),
        ``delta_auc_ci_low`` / ``delta_auc_ci_high``, and ``delta_valid``,
        the number of two-class resamples. All NaN / 0 without positives.

    Raises:
        ValueError: As :func:`block_bootstrap_ci`.
    """
    y = np.asarray(y_true).astype(int)
    a = np.nan_to_num(np.asarray(scores_a, dtype=float))
    b = np.nan_to_num(np.asarray(scores_b, dtype=float))
    if not len(y) == len(a) == len(b):
        raise ValueError(f"y_true has {len(y)} days, scores {len(a)} and {len(b)}")
    _check(len(y), block, n_resamples, ci)
    mask = _keep_mask(keep, len(y))

    yk = y[mask]
    point = (float(roc_auc_score(yk, a[mask]) - roc_auc_score(yk, b[mask]))
             if len(yk) and yk.min() != yk.max() else float("nan"))
    diffs = []
    for idx in _resamples(mask, block, n_resamples, seed):
        yb = y[idx]
        if len(idx) and yb.min() != yb.max():
            diffs.append(roc_auc_score(yb, a[idx]) - roc_auc_score(yb, b[idx]))
    low, high = _bounds(diffs, ci)
    return {"delta_auc": point, "delta_auc_ci_low": low, "delta_auc_ci_high": high,
            "delta_valid": len(diffs)}
