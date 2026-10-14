"""Alert Explanation Record (AER): one self-contained explanation per alert day.

An AER is built from the agent scores that produced the alert, not from a model
of them. Its core is the exact Shapley attribution of the composite over the
agents (:mod:`src.explainability.exact_shapley`), measured from each agent's
calm-day baseline (:mod:`src.explainability.baselines`); it adds the
within-agent breakdown (:mod:`src.explainability.agent_breakdown`), an
uncertainty band, and a single-agent counterfactual.

Fields reserved for later stages are present and empty: ``precedents`` (filled
by the precedent retriever) and ``narrative`` (filled by the narrative
template). Every record names the aggregation path it explains (``"D"``: the
real-data evaluation mean; ``"C"``: the dashboard's amplified, clipped mean),
the weight mode, and the data's origin, so no number can be read out of
context.

**Uncertainty.** Calibration is not estimable on this data (three of four
regions have no positive day in the training window). What is reported
instead is how fragile the score is to plausible agent-score noise: each
agent's noise scale is the standard deviation of its day-to-day score change
on calm training days divided by sqrt(2) (the per-day noise implied by a
first difference), and the composite is re-evaluated under ``n_draws``
Gaussian perturbations with a seed derived from the alert id.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.explainability.agent_breakdown import effective_terms
from src.explainability.exact_shapley import (
    decompose_path_c,
    efficiency_residual,
    minimal_drop,
    path_d_closed_form,
    shapley_agents,
    value_fn_path_c,
    value_fn_path_d,
)

#: Draws for the score-noise band.
N_DRAWS: int = 500

#: Alert days at most this many calendar days apart belong to one episode.
EPISODE_GAP_DAYS: int = 3

#: What each agent's score is computed over, for the counterfactual's caveat:
#: a "required score drop" is a drop in a windowed statistic, not one day's
#: reading.
WINDOW_NOTES: dict[str, str] = {
    "shipping": (
        "shock = isolation forest + max |z| on the day's standardised features; "
        "duration = 30-day mean transits vs trailing 365-day median, gated by "
        "14-day persistence"
    ),
    "market": "|z| of each series against its rolling 30-day mean and std",
    "geopolitical": "daily index levels (sigmoid of a weighted composite)",
    "natural_disaster": "daily hazard severities near the chokepoint",
    "news_sentiment": "recency-weighted tone, 3-day velocity, volume vs 30-day mean",
}


@dataclass(frozen=True)
class AlertExplanationRecord:
    """One explained alert day. See the module docstring for each block."""

    alert_id: str
    episode_id: str
    region: str
    date: str
    path: str
    weight_mode: str
    data_origin: str
    tier: int
    agents_used: list[str]
    weights: dict[str, float]

    score: float
    threshold: float
    margin: float

    agent_scores: dict[str, float]
    agent_baselines: dict[str, float]
    agent_shapley: dict[str, float]
    baseline_value: float
    efficiency_residual: float
    bonus_share: dict[str, Any] | None

    agent_breakdown: dict[str, dict]
    top_drivers: list[dict]
    uncertainty: dict[str, float]
    counterfactual: dict[str, Any]
    circularity_control: dict[str, Any]

    precedents: list[dict] = field(default_factory=list)
    narrative: dict[str, Any] = field(
        default_factory=lambda: {"text": "", "claims": [], "faithfulness": None}
    )
    label: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ helpers
def noise_scales(train_scores: Mapping[str, np.ndarray], calm: np.ndarray) -> dict[str, float]:
    """Per-agent noise scale: std of day-to-day change on calm days / sqrt(2)."""
    out: dict[str, float] = {}
    for agent, scores in train_scores.items():
        diffs = np.diff(np.asarray(scores, dtype=float))
        both_calm = calm[1:] & calm[:-1]
        diffs = diffs[both_calm]
        out[agent] = float(np.std(diffs) / np.sqrt(2.0)) if len(diffs) else 0.0
    return out


def episodes(dates: Sequence, alert: np.ndarray, gap_days: int = EPISODE_GAP_DAYS) -> list[list[int]]:
    """Group alert-day indices into episodes.

    Consecutive alert days belong to one episode when at most ``gap_days``
    calendar days without an alert separate them. Calendar days, not row
    positions: the evaluation frames drop incomplete days, so adjacent rows
    are not always adjacent dates.

    Returns:
        A list of episodes, each a list of row indices in date order.
    """
    stamps = pd.to_datetime(pd.Series(list(dates)))
    groups: list[list[int]] = []
    for idx in np.flatnonzero(alert):
        if groups:
            prev = groups[-1][-1]
            if (stamps.iloc[idx] - stamps.iloc[prev]).days - 1 <= gap_days:
                groups[-1].append(int(idx))
                continue
        groups.append([int(idx)])
    return groups


def _rank_drivers(phi: Mapping[str, float], breakdowns: Mapping[str, dict]) -> list[dict]:
    total = sum(abs(v) for v in phi.values()) or 1.0
    ranked = sorted(phi.items(), key=lambda kv: kv[1], reverse=True)
    drivers = []
    for agent, value in ranked:
        terms = effective_terms(breakdowns.get(agent, {}))
        top_terms = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[:2]
        drivers.append({
            "agent": agent,
            "phi": value,
            "share_abs": abs(value) / total,
            "branch_winner": breakdowns.get(agent, {}).get("branch_winner"),
            "top_terms": [{"term": t, "value": v} for t, v in top_terms],
        })
    return drivers


def _uncertainty(scores: Mapping[str, float], value_fn, threshold: float,
                 sigma: Mapping[str, float], seed: int, n_draws: int) -> dict[str, float]:
    agents = list(scores)
    values = np.array([scores[a] for a in agents], dtype=float)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, [sigma.get(a, 0.0) for a in agents], size=(n_draws, len(agents)))
    draws = np.clip(values + noise, 0.0, 1.0)
    composite = np.array([value_fn(dict(zip(agents, row))) for row in draws])
    return {
        "agent_dispersion": float(np.std(values)),
        "bootstrap_lo": float(np.percentile(composite, 2.5)),
        "bootstrap_hi": float(np.percentile(composite, 97.5)),
        "frac_draws_above_threshold": float(np.mean(composite > threshold)),
        "n_draws": n_draws,
        "seed": seed,
    }


# ------------------------------------------------------------------- build
def build_aer(
    *,
    alert_id: str,
    episode_id: str,
    region: str,
    date: str,
    path: str,
    weight_mode: str,
    data_origin: str,
    tier: int,
    agents: Sequence[str],
    weights: Mapping[str, float],
    scores: Mapping[str, float],
    baselines: Mapping[str, float],
    threshold: float,
    breakdowns: Mapping[str, dict],
    noise_sigma: Mapping[str, float],
    y_true: int | None,
    circularity_control: dict[str, Any] | None = None,
    bonus3: float | None = None,
    bonus5: float | None = None,
    n_draws: int = N_DRAWS,
) -> AlertExplanationRecord:
    """Explain one alert day exactly.

    Args:
        path: ``"D"`` (renormalised weighted mean) or ``"C"`` (amplified and
            clipped; requires ``bonus3`` and ``bonus5``).
        agents: The agents in the composite, in accumulation order.
        scores: Each agent's score that day.
        baselines: Each agent's calm baseline.
        threshold: The operating point; the day is an alert if score > it.
        breakdowns: :func:`~src.explainability.agent_breakdown.breakdown`
            output per agent.
        noise_sigma: Per-agent noise scale (:func:`noise_scales`).
        y_true: The day's label, or ``None`` if unlabelled.
        circularity_control: The same alert judged without the shipping
            duration score (built by the caller, which knows the train window).

    Returns:
        The record. Raises if the Shapley efficiency residual exceeds 1e-9.
    """
    agents = list(agents)
    scores = {a: float(scores[a]) for a in agents}
    base = {a: float(baselines[a]) for a in agents}

    bonus_share = None
    if path == "D":
        value_fn = value_fn_path_d(weights, agents)
        phi = shapley_agents(scores, base, value_fn)
        closed = path_d_closed_form(scores, base, weights, agents)
        worst = max(abs(phi[a] - closed[a]) for a in agents)
        if worst > 1e-12:
            raise AssertionError(f"Path D Shapley differs from closed form by {worst}")
    elif path == "C":
        value_fn = value_fn_path_c(weights, agents, bonus3, bonus5)
        parts = decompose_path_c(scores, base, weights, agents, bonus3, bonus5)
        phi = parts["total"]
        bonus_share = {
            "bonus_total": sum(parts["bonus"].values()),
            "clip_total": sum(parts["clip"].values()),
            "bonus_by_agent": parts["bonus"],
            "clip_by_agent": parts["clip"],
        }
    else:
        raise ValueError(f"unknown path {path!r}")

    score = float(value_fn(scores))
    residual = efficiency_residual(phi, scores, base, value_fn)
    if abs(residual) > 1e-9:
        raise AssertionError(f"efficiency residual {residual} exceeds 1e-9")

    drivers = _rank_drivers(phi, breakdowns)
    top = drivers[0]["agent"]
    required = minimal_drop(top, scores, value_fn, threshold)
    counterfactual = {
        "agent": top,
        "required_score": required,
        "required_score_drop": None if required is None else scores[top] - required,
        "feasible": bool(required is not None and required >= base[top]),
        "window_note": WINDOW_NOTES.get(top, ""),
    }

    seed = zlib.crc32(alert_id.encode("utf-8"))
    uncertainty = _uncertainty(scores, value_fn, threshold, noise_sigma, seed, n_draws)

    label: dict[str, Any] = {"y_true": y_true}
    if y_true is not None:
        label["alert_correct"] = "TP" if int(y_true) == 1 else "FP"

    return AlertExplanationRecord(
        alert_id=alert_id,
        episode_id=episode_id,
        region=region,
        date=date,
        path=path,
        weight_mode=weight_mode,
        data_origin=data_origin,
        tier=tier,
        agents_used=agents,
        weights={a: float(weights.get(a, 0.1)) for a in agents},
        score=score,
        threshold=float(threshold),
        margin=score - float(threshold),
        agent_scores=scores,
        agent_baselines=base,
        agent_shapley=phi,
        baseline_value=float(value_fn(base)),
        efficiency_residual=residual,
        bonus_share=bonus_share,
        agent_breakdown=dict(breakdowns),
        top_drivers=drivers,
        uncertainty=uncertainty,
        counterfactual=counterfactual,
        circularity_control=dict(circularity_control or {}),
        label=label,
    )
