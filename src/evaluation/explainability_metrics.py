"""METRIC 9 — the explainability scorecard's measurement primitives.

Each function measures one property of an attribution, independently of how
the attribution was produced, so the exact decomposition and the surrogate are
scored by the same code. The scorecard (``eval/explainability/run_scorecard.py``)
applies them per alert day, aggregates days to episodes, and reports the
episode mean with an episode-level bootstrap CI: alert days inside one episode
are strongly dependent, so the episode, not the day, is the unit of evidence.

Metric map (Nauta et al.'s Co-12 property in brackets):

* E1 faithfulness [correctness] — efficiency residual; surrogate R² and local
  error ``|f(x) - g(x)|``.
* E2 deletion / insertion [correctness] — :func:`deletion_insertion`,
  :func:`random_order_auc`.
* E3 compactness — :func:`k_cover`.
* E4 stability [consistency / continuity] — :func:`spearman`,
  :func:`top1_agreement`.
* E5 coherence — :func:`top1_agreement` across methods.
* E6 contrastivity — counterfactual feasibility (computed by the AER builder).
* E7 uncertainty usefulness [confidence] — :func:`auc`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import permutations

import numpy as np

#: Bootstrap resamples and seed for every CI in the scorecard.
N_BOOT: int = 1000
BOOT_SEED: int = 20261014

#: E2 is scored only where the explained score sits at least this far above
#: its baseline (score units, on [0, 1]).
MIN_GAP: float = 0.01


# --------------------------------------------------------------- E2 curves
def _auc_of_curve(values: Sequence[float]) -> float:
    """Trapezoidal area under ``values`` sampled at k/n, k = 0..n."""
    v = np.asarray(values, dtype=float)
    n = len(v) - 1
    return float(np.sum((v[1:] + v[:-1]) / 2.0) / n)


def deletion_insertion(order: Sequence[str], value: Callable[[frozenset], float],
                       min_gap: float = MIN_GAP) -> tuple[float, float]:
    """Normalised deletion and insertion AUCs for one removal order.

    Deletion starts from every player present and moves players to their
    baseline in ``order``; insertion starts from all at baseline and restores
    them in ``order``. Both curves are rescaled so the full coalition is 1 and
    the empty one 0. A good order makes the deletion curve fall fast (low
    AUC) and the insertion curve rise fast (high AUC).

    Args:
        order: Players, most important first.
        value: Coalition (frozenset of present players) -> score.

    Returns:
        ``(deletion_auc, insertion_auc)``; ``(nan, nan)`` unless the full
        coalition scores at least ``min_gap`` above the empty one. Below that
        the normalisation divides by (near) zero, and a score *below* its
        baseline inverts what "most important first" means.
    """
    order = list(order)
    full, empty = value(frozenset(order)), value(frozenset())
    gap = full - empty
    if gap < min_gap:
        return float("nan"), float("nan")
    deletion = [(value(frozenset(order[k:])) - empty) / gap for k in range(len(order) + 1)]
    insertion = [(value(frozenset(order[:k])) - empty) / gap for k in range(len(order) + 1)]
    return _auc_of_curve(deletion), _auc_of_curve(insertion)


def random_order_auc(players: Sequence[str], value: Callable[[frozenset], float],
                     min_gap: float = MIN_GAP) -> tuple[float, float]:
    """Expected deletion / insertion AUC over *all* orderings (exact, not sampled)."""
    results = [deletion_insertion(p, value, min_gap) for p in permutations(players)]
    dels, ins = zip(*results)
    return float(np.mean(dels)), float(np.mean(ins))


def order_by(attribution: Mapping[str, float], players: Sequence[str]) -> list[str]:
    """Players sorted by attribution, largest first; ties keep ``players`` order."""
    rank = {p: i for i, p in enumerate(players)}
    return sorted(players, key=lambda p: (-float(attribution.get(p, 0.0)), rank[p]))


# --------------------------------------------------------------- E3 compactness
def k_cover(attribution: Mapping[str, float], share: float = 0.8) -> float:
    """Fewest players whose |attribution| covers ``share`` of the total."""
    mags = np.sort(np.abs(np.asarray(list(attribution.values()), dtype=float)))[::-1]
    total = mags.sum()
    if total <= 0:
        return float("nan")
    return float(np.searchsorted(np.cumsum(mags), share * total - 1e-12) + 1)


# --------------------------------------------------------------- E4 / E5
def spearman(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Spearman rank correlation over the shared players; nan if either is constant."""
    from scipy.stats import spearmanr

    keys = [k for k in a if k in b]
    x = np.array([a[k] for k in keys], dtype=float)
    y = np.array([b[k] for k in keys], dtype=float)
    if len(keys) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def top1(attribution: Mapping[str, float]) -> str:
    """The player with the largest attribution (first wins ties)."""
    return max(attribution, key=lambda k: float(attribution[k]))


def top1_agreement(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """1.0 if both attributions name the same top player, else 0.0."""
    return 1.0 if top1(a) == top1(b) else 0.0


# --------------------------------------------------------------- E7
def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC AUC of ``scores`` for ``labels == 1``; nan if only one class."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    keep = np.isfinite(s)
    if len(np.unique(y[keep])) < 2:
        return float("nan")
    return float(roc_auc_score(y[keep], s[keep]))


# --------------------------------------------------------------- bootstrap
def episode_bootstrap(values: Sequence[float], n_boot: int = N_BOOT,
                      seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Mean of per-episode values with a percentile bootstrap 95% CI.

    NaN episodes (metric undefined there) are dropped first.

    Returns:
        ``(mean, lo, hi)``; all nan if no episode has a value.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_auc(scores: Sequence[float], labels: Sequence[int], n_boot: int = N_BOOT,
                  seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """AUC with an episode-resampling bootstrap CI (resamples lacking a class skipped)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    point = auc(s, y)
    if not np.isfinite(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), size=len(s))
        value = auc(s[idx], y[idx])
        if np.isfinite(value):
            stats.append(value)
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
