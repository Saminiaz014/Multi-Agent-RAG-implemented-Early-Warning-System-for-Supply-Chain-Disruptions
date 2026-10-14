"""Exact Shapley attribution over agent scores.

The composite risk score is a known function of at most five agent scores, so
its Shapley values can be computed exactly: 2^5 = 32 coalitions, no sampling
and no surrogate. Absent agents are not dropped — dropping one would change
the renormalisation and so the function itself — but set to a *baseline*
score, the agent's calm-day median, so a coalition's value is "the score if
only these agents departed from calm".

Two value functions are provided, one per explained aggregation path:

* :func:`value_fn_path_d` — the real-data evaluation composite
  (``scripts/run_method_comparison.py::tier_scores``): a renormalised weighted
  mean. Additive, so Shapley reduces to ``w_i (s_i - b_i) / mass``
  (:func:`path_d_closed_form`).
* :func:`value_fn_path_c` — the dashboard composite
  (``src/dashboard/core.py::composite_series``): the same mean, amplified when
  three or five agents exceed 0.5, then clipped at 1.0. Not additive; Shapley
  distributes the amplification and the clip exactly, and
  :func:`decompose_path_c` separates the two by linearity of the Shapley value.

Efficiency holds by construction: ``sum(phi) == f(scores) - f(baseline)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from math import factorial

ValueFn = Callable[[Mapping[str, float]], float]

#: Per-agent score above which an agent counts toward agreement (the
#: ``_AGREEMENT_THRESHOLD`` of RiskEngine and ``composite_series``).
AGREEMENT_THRESHOLD: float = 0.5


def shapley_agents(
    scores: Mapping[str, float],
    baseline: Mapping[str, float],
    value_fn: ValueFn,
) -> dict[str, float]:
    """Exact Shapley values of ``value_fn`` at ``scores`` relative to ``baseline``.

    Args:
        scores: Agent name -> observed score.
        baseline: Agent name -> reference score; must cover every agent in
            ``scores``.
        value_fn: Maps a full ``{agent: score}`` assignment to the composite.

    Returns:
        Agent name -> Shapley value, in the order of ``scores``.
    """
    players = list(scores)
    n = len(players)
    cache: dict[frozenset, float] = {}

    def value(coalition: frozenset) -> float:
        if coalition not in cache:
            point = {p: (scores[p] if p in coalition else baseline[p]) for p in players}
            cache[coalition] = float(value_fn(point))
        return cache[coalition]

    weights = [factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)]
    phi: dict[str, float] = {}
    for player in players:
        others = [p for p in players if p != player]
        total = 0.0
        for k in range(n):
            for subset in combinations(others, k):
                coalition = frozenset(subset)
                total += weights[k] * (value(coalition | {player}) - value(coalition))
        phi[player] = total
    return phi


def efficiency_residual(
    phi: Mapping[str, float],
    scores: Mapping[str, float],
    baseline: Mapping[str, float],
    value_fn: ValueFn,
) -> float:
    """``sum(phi) - (f(scores) - f(baseline))``; zero up to float error."""
    gap = float(value_fn(dict(scores))) - float(value_fn(dict(baseline)))
    return float(sum(phi.values()) - gap)


# ---------------------------------------------------------------- value fns
def value_fn_path_d(weights: Mapping[str, float], agents: Sequence[str]) -> ValueFn:
    """Path D: weighted mean over ``agents``, renormalised by their weight mass.

    Accumulates in ``agents`` order, as ``tier_scores`` does.
    """
    agents = list(agents)
    w = {a: float(weights.get(a, 0.1)) for a in agents}
    mass = sum(w.values())

    def f(x: Mapping[str, float]) -> float:
        total = 0.0
        for a in agents:
            total += w[a] * float(x[a])
        return total / mass

    return f


def path_d_closed_form(
    scores: Mapping[str, float],
    baseline: Mapping[str, float],
    weights: Mapping[str, float],
    agents: Sequence[str],
) -> dict[str, float]:
    """Shapley values of :func:`value_fn_path_d`, analytically."""
    w = {a: float(weights.get(a, 0.1)) for a in agents}
    mass = sum(w.values())
    return {a: w[a] * (float(scores[a]) - float(baseline[a])) / mass for a in agents}


def _amplification(x: Mapping[str, float], agents: Sequence[str],
                   bonus3: float, bonus5: float, thr: float) -> float:
    agreement = sum(1 for a in agents if float(x[a]) > thr)
    if agreement >= 5:
        return bonus5
    if agreement >= 3:
        return bonus3
    return 1.0


def value_fn_path_c(
    weights: Mapping[str, float],
    agents: Sequence[str],
    bonus3: float,
    bonus5: float,
    thr: float = AGREEMENT_THRESHOLD,
    clip: float = 1.0,
) -> ValueFn:
    """Path C: ``min(amp(x) * weighted_mean(x), clip)``, as ``composite_series``."""
    agents = list(agents)
    base = value_fn_path_d(weights, agents)

    def f(x: Mapping[str, float]) -> float:
        return min(base(x) * _amplification(x, agents, bonus3, bonus5, thr), clip)

    return f


def decompose_path_c(
    scores: Mapping[str, float],
    baseline: Mapping[str, float],
    weights: Mapping[str, float],
    agents: Sequence[str],
    bonus3: float,
    bonus5: float,
    thr: float = AGREEMENT_THRESHOLD,
    clip: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Split each agent's Path C Shapley value into linear, bonus and clip parts.

    The Shapley value is linear in the value function, so writing
    ``f_C = f_lin + (f_amp - f_lin) + (f_C - f_amp)`` — weighted mean,
    amplification, clip — gives three attributions that sum exactly to the
    Path C one.

    Returns:
        ``{"total", "linear", "bonus", "clip"}``, each ``{agent: phi}``.
    """
    agents = list(agents)
    f_lin = value_fn_path_d(weights, agents)

    def f_amp(x: Mapping[str, float]) -> float:
        return f_lin(x) * _amplification(x, agents, bonus3, bonus5, thr)

    f_c = value_fn_path_c(weights, agents, bonus3, bonus5, thr, clip)
    total = shapley_agents(scores, baseline, f_c)
    linear = shapley_agents(scores, baseline, f_lin)
    bonus = shapley_agents(scores, baseline, lambda x: f_amp(x) - f_lin(x))
    clip_part = shapley_agents(scores, baseline, lambda x: f_c(x) - f_amp(x))
    return {"total": total, "linear": linear, "bonus": bonus, "clip": clip_part}


# ----------------------------------------------------------- counterfactual
def minimal_drop(
    agent: str,
    scores: Mapping[str, float],
    value_fn: ValueFn,
    threshold: float,
    iterations: int = 80,
) -> float | None:
    """Highest score for ``agent`` at which the composite is no longer an alert.

    An alert is ``value_fn(x) > threshold``. Both value functions are
    non-decreasing in every agent score, so the non-alert set along this one
    coordinate is an interval ``[0, t*]``; ``t*`` is found by bisection.

    Returns:
        ``t*``, or ``None`` if the composite stays above ``threshold`` even
        with the agent at 0 (no single-agent counterfactual exists).
    """
    point = dict(scores)

    def at(t: float) -> float:
        point[agent] = t
        return float(value_fn(point))

    current = float(scores[agent])
    if at(current) <= threshold:
        return current
    if at(0.0) > threshold:
        return None
    lo, hi = 0.0, current  # f(lo) <= threshold < f(hi)
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if at(mid) <= threshold:
            lo = mid
        else:
            hi = mid
    return lo
