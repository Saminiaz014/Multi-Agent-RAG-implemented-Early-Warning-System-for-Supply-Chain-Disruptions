"""Exact Shapley values *inside* an agent, over the agent's own inputs.

:mod:`~src.explainability.agent_breakdown` reports an agent's additive terms
and which branch of its non-linearity won. This module goes one step further
and attributes the agent's final score to its inputs exactly, treating the
non-linearity (``max``, clip, sigmoid, isolation forest) as part of the game.
Each agent has at most five inputs, so every coalition is evaluated.

Players per agent, and what "absent" means (its calm-day median on the train
window, like the agent-level baselines):

* **shipping** — each standardised feature the isolation forest reads
  (``vessel_count``, ``avg_delay_hours``, ``congestion_index``,
  ``tanker_count``). The duration score is a 30-day level statistic of
  ``vessel_count`` and is bundled with that player, so ``vessel_count`` absent
  means both its day reading and its level shift are at calm. The score is
  recomputed through the fitted forest: one batched ``decision_function`` call
  per day over all coalitions.
* **market** — per series, its term ``w_i |z_i| / z_threshold``; score is the
  clipped sum.
* **geopolitical** — per component, its renormalised term; score is the sigmoid.
* **natural_disaster** — per hazard, its raw severity; score is
  ``clip(max(sum w_k x_k, max x_k))``.
* **news_sentiment** — per available component, its renormalised term; score
  is the clipped sum.

Every game re-derives the agent's emitted score from the full coalition and
raises if it differs, so the attribution is of the score actually used.
Because the composite is linear in each agent score, a feature's share of the
Path D composite is ``(w_agent / mass) * phi_feature``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.explainability.agent_breakdown import breakdown
from src.explainability.exact_shapley import shapley_from_values

_TOL = 1e-9

_DISASTER_COLUMNS = {
    "earthquake": "earthquake_severity",
    "tsunami": "tsunami_risk",
    "cyclone": "cyclone_severity",
    "severe_weather": "severe_weather_index",
}


def _shipping_z(agent: Any) -> dict[str, str]:
    from src.agents.shipping_agent import _ZSCORE_NAME_MAP

    return {f: _ZSCORE_NAME_MAP[f] for f in agent._active_features()}


def inputs(domain: str, agent: Any, row: pd.Series, present: set[str] | None = None,
           include_duration: bool = True) -> dict[str, float] | None:
    """The agent's players and their values on one day; ``None`` if unscored."""
    score = row.get("anomaly_score")
    if score is None or not math.isfinite(float(score)):
        return None
    if domain == "shipping":
        out = {f: float(row[col]) for f, col in _shipping_z(agent).items()}
        if include_duration:
            out["_duration"] = float(row["duration_score"])
        return out
    if domain == "natural_disaster":
        return {k: float(row[c]) for k, c in _DISASTER_COLUMNS.items()}
    return dict(breakdown(domain, agent, row, present)["terms"])


def players_of(domain: str, values: Mapping[str, float]) -> list[str]:
    """Players of the game; the shipping duration rides with ``vessel_count``."""
    return [k for k in values if k != "_duration"]


def _shipping_scores(agent: Any, z: np.ndarray, duration: np.ndarray | None) -> np.ndarray:
    iforest = -agent._iforest.decision_function(z)
    span = agent._iforest_high - agent._iforest_low
    iforest_norm = np.clip((iforest - agent._iforest_low) / span, 0.0, 1.0)
    max_z_norm = np.minimum(np.max(np.abs(z), axis=1) / agent._z_threshold, 1.0)
    shock = agent._if_weight * iforest_norm + agent._zscore_weight * max_z_norm
    return shock if duration is None else np.maximum(shock, duration)


def _scalar_score(domain: str, agent: Any, v: Mapping[str, float]) -> float:
    if domain == "market":
        return min(sum(v.values()), 1.0)
    if domain == "geopolitical":
        return 1.0 / (1.0 + math.exp(-6.0 * (sum(v.values()) - 0.5)))
    if domain == "natural_disaster":
        composite = sum(agent._weights[k] * v[k] for k in _DISASTER_COLUMNS)
        return float(np.clip(max(composite, max(v.values())), 0.0, 1.0))
    if domain == "news_sentiment":
        return float(np.clip(sum(v.values()), 0.0, 1.0))
    raise ValueError(domain)


def coalition_scores(domain: str, agent: Any, full: Mapping[str, float],
                     calm: Mapping[str, float]) -> dict[frozenset, float]:
    """Agent score for every coalition of present players."""
    players = players_of(domain, full)
    coalitions = [frozenset(c) for k in range(len(players) + 1)
                  for c in combinations(players, k)]

    def point(coalition: frozenset) -> dict[str, float]:
        out = {p: (full[p] if p in coalition else calm[p]) for p in players}
        if "_duration" in full:
            out["_duration"] = full["_duration"] if "vessel_count" in coalition else calm["_duration"]
        return out

    points = [point(c) for c in coalitions]
    if domain == "shipping":
        order = list(_shipping_z(agent))
        z = np.array([[p[f] for f in order] for p in points], dtype=float)
        duration = (np.array([p["_duration"] for p in points], dtype=float)
                    if "_duration" in full else None)
        values = _shipping_scores(agent, z, duration)
        return dict(zip(coalitions, (float(v) for v in values)))
    return {c: _scalar_score(domain, agent, p) for c, p in zip(coalitions, points)}


def feature_shapley(domain: str, agent: Any, row: pd.Series, calm: Mapping[str, float],
                    present: set[str] | None = None,
                    include_duration: bool = True) -> dict[str, Any] | None:
    """Exact Shapley values of one agent's score over its inputs on one day.

    Args:
        calm: Player -> calm value (:func:`calm_inputs`).
        include_duration: ``False`` scores shipping as its shock branch alone
            (the circularity control).

    Returns:
        ``{"phi", "score", "calm_score", "residual"}`` or ``None`` if the
        agent did not score the day.

    Raises:
        AssertionError: If the full coalition does not reproduce the agent's
            score (the emitted one, or its shock branch without duration).
    """
    full = inputs(domain, agent, row, present, include_duration)
    if full is None:
        return None
    table = coalition_scores(domain, agent, full, calm)
    players = players_of(domain, full)
    everyone = frozenset(players)
    expected = (float(row["anomaly_score"]) if (domain != "shipping" or include_duration)
                else float(row["shock_score"]))
    if abs(table[everyone] - expected) > _TOL:
        raise AssertionError(
            f"{domain}: feature game gives {table[everyone]!r}, agent gave {expected!r}"
        )
    phi = shapley_from_values(players, table.__getitem__)
    residual = sum(phi.values()) - (table[everyone] - table[frozenset()])
    return {"phi": phi, "score": table[everyone], "calm_score": table[frozenset()],
            "residual": float(residual)}


def calm_inputs(domain: str, agent: Any, frame: pd.DataFrame, calm_mask: np.ndarray,
                present: set[str] | None = None,
                include_duration: bool = True) -> dict[str, float]:
    """Median of each player's value over the calm days of ``frame``."""
    collected: dict[str, list[float]] = {}
    for (_, row), is_calm in zip(frame.iterrows(), calm_mask):
        if not is_calm:
            continue
        values = inputs(domain, agent, row, present, include_duration)
        if values is None:
            continue
        for k, v in values.items():
            collected.setdefault(k, []).append(v)
    return {k: float(np.median(v)) for k, v in collected.items()}
