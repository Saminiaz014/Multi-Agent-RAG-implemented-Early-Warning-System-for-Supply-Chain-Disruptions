"""Within-agent breakdown: which terms built an agent's score, and which branch won.

Exact agent-level Shapley values (:mod:`src.explainability.exact_shapley`) say
*which agent* moved the composite. This module says *what inside the agent*
did. Every agent forms an additive score and then passes it through one
non-linearity — a ``max``, a clip or a sigmoid — so the breakdown reports the
additive terms, the pre-non-linearity score, and which branch of the
non-linearity produced the final score.

Nothing is recomputed from raw inputs that the agent did not already compute:
terms are read from the agent's own ``detect`` output columns (``shock_score``,
``duration_score``, ``isolation_score``, z-score columns, ``raw_composite``,
``*_norm`` ...) and multiplied by the weights the fitted agent holds. Each
breakdown re-derives the agent's final score from those terms and raises if
it does not match the ``anomaly_score`` the agent emitted, so a drift between
this module and the agents fails loudly instead of explaining the wrong thing.

Per agent:

* **shipping** — ``score = max(shock, duration)``, ``shock = w_if * isolation
  + w_z * max_zscore_norm``. Terms: the two shock parts and the duration score;
  winner ``"shock"`` or ``"duration"``; ``max_z_feature`` names the feature
  whose |z| set ``max_zscore_norm``.
* **market** — ``score = min(sum_i w_i |z_i| / z_threshold, 1)``. Terms per
  feature; winner ``"linear"`` or ``"clipped"``.
* **geopolitical** — ``score = sigmoid(6 (raw - 0.5))``, ``raw`` a renormalised
  weighted mean (stability inverted). Terms per component;
  ``sigmoid_slope = 6 s (1 - s)`` converts a raw-term change to a score change.
* **natural_disaster** — ``score = clip(max(composite, max_single))``. Terms
  per hazard; winner ``"composite"`` or ``"single:<feature>"``.
* **news_sentiment** — ``score = clip(sum_k w_k c_k / used)`` over the
  available components. Terms per component; winner ``"linear"`` or
  ``"clipped"``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

#: Tolerance for re-deriving an agent's score from its own terms.
_RECON_TOL: float = 1e-9


class BreakdownMismatch(AssertionError):
    """An agent's terms no longer reproduce the score it emitted."""


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check(domain: str, derived: float, emitted: float) -> None:
    if abs(derived - emitted) > _RECON_TOL:
        raise BreakdownMismatch(
            f"{domain}: terms give {derived!r} but the agent emitted {emitted!r}"
        )


def _missing(domain: str) -> dict:
    """The agent did not score this day (the script fills such days with 0.0)."""
    return {"agent": domain, "terms": {}, "pre_score": 0.0, "score": 0.0,
            "branch_winner": "missing"}


# ----------------------------------------------------------------- agents
def _shipping(agent: Any, row: pd.Series) -> dict:
    from src.agents.shipping_agent import _ZSCORE_NAME_MAP

    terms = {
        "isolation_forest": agent._if_weight * float(row["isolation_score"]),
        "max_zscore": agent._zscore_weight * float(row["max_zscore_norm"]),
    }
    shock = float(row["shock_score"])
    duration = float(row["duration_score"])
    _check("shipping/shock", terms["isolation_forest"] + terms["max_zscore"], shock)
    score = max(shock, duration)
    _check("shipping", score, float(row["anomaly_score"]))

    z = {f: abs(float(row[_ZSCORE_NAME_MAP[f]])) for f in agent._active_features()}
    return {
        "agent": "shipping",
        "terms": {**terms, "duration": duration},
        "pre_score": score,
        "score": score,
        "shock": shock,
        "duration": duration,
        "branch_winner": "duration" if duration > shock else "shock",
        "max_z_feature": max(z, key=z.get),
        "abs_z": z,
    }


def _market(agent: Any, row: pd.Series) -> dict:
    from src.agents.market_agent import _ZSCORE_NAME_MAP

    weights = agent._feature_weights()
    terms = {
        f: weights[f] * abs(float(row[_ZSCORE_NAME_MAP[f]])) / agent._z_threshold
        for f in agent._active_features()
    }
    pre = sum(terms.values())
    score = min(pre, 1.0)
    _check("market", score, float(row["anomaly_score"]))
    return {"agent": "market", "terms": terms, "pre_score": pre, "score": score,
            "branch_winner": "clipped" if pre > 1.0 else "linear"}


_GEO_COLUMNS = {
    "sanctions": "sanctions_severity",
    "military": "military_activity_index",
    "diplomatic": "diplomatic_incident_score",
    "stability": "regime_stability_index",
}


def _geopolitical(agent: Any, row: pd.Series, present: set[str]) -> dict:
    used = sum(agent._weights[k] for k, col in _GEO_COLUMNS.items() if col in present)
    terms: dict[str, float] = {}
    for key, col in _GEO_COLUMNS.items():
        if col not in present:
            continue
        value = float(row[col])
        if key == "stability":
            value = 1.0 - value
        terms[key] = agent._weights[key] * value / used
    raw = sum(terms.values())
    _check("geopolitical/raw", raw, float(row["raw_composite"]))
    score = 1.0 / (1.0 + math.exp(-6.0 * (raw - 0.5)))
    _check("geopolitical", score, float(row["anomaly_score"]))
    return {"agent": "geopolitical", "terms": terms, "pre_score": raw, "score": score,
            "branch_winner": "sigmoid", "sigmoid_slope": 6.0 * score * (1.0 - score)}


_DISASTER_COLUMNS = {
    "earthquake": "earthquake_severity",
    "tsunami": "tsunami_risk",
    "cyclone": "cyclone_severity",
    "severe_weather": "severe_weather_index",
}


def _disaster(agent: Any, row: pd.Series) -> dict:
    terms = {k: agent._weights[k] * float(row[c]) for k, c in _DISASTER_COLUMNS.items()}
    composite = float(row["raw_composite"])
    _check("natural_disaster/composite", sum(terms.values()), composite)
    singles = {c: float(row[c]) for c in agent._feature_columns}
    top_single = max(singles, key=singles.get)
    max_single = singles[top_single]
    score = float(np.clip(max(composite, max_single), 0.0, 1.0))
    _check("natural_disaster", score, float(row["anomaly_score"]))
    return {
        "agent": "natural_disaster",
        "terms": terms,
        "pre_score": max(composite, max_single),
        "score": score,
        "composite": composite,
        "max_single": max_single,
        "branch_winner": "composite" if composite >= max_single else f"single:{top_single}",
    }


_NEWS_COLUMNS = {
    "sentiment": "neg_sentiment_norm",
    "consensus": "consensus_norm",
    "velocity": "velocity_norm",
    "volume": "volume_spike_norm",
}


def _news(agent: Any, row: pd.Series) -> dict:
    available = {k: c for k, c in _NEWS_COLUMNS.items() if _finite(row.get(c))}
    used = sum(agent._weights[k] for k in available)
    terms = {k: agent._weights[k] * float(row[c]) / used for k, c in available.items()}
    pre = sum(terms.values())
    score = float(np.clip(pre, 0.0, 1.0))
    _check("news_sentiment", score, float(row["anomaly_score"]))
    return {"agent": "news_sentiment", "terms": terms, "pre_score": pre, "score": score,
            "branch_winner": "clipped" if pre > 1.0 or pre < 0.0 else "linear"}


# ------------------------------------------------------------------ public
def present_columns(frame: pd.DataFrame) -> set[str]:
    """Columns an agent's renormalisation treats as present: in the frame, not all-NaN."""
    return {c for c in frame.columns if not frame[c].isna().all()}


def breakdown(domain: str, agent: Any, row: pd.Series,
              present: set[str] | None = None) -> dict:
    """Break one agent's score on one day into its additive terms.

    Args:
        domain: Agent config key (``"shipping"``, ``"market"``, ...).
        agent: The fitted agent that produced ``row``. Only read.
        row: One row of that agent's ``detect`` output.
        present: For the geopolitical agent, the columns present in the whole
            scored frame (see :func:`present_columns`); its renormalisation is
            decided frame-wide, not per row.

    Returns:
        ``{"agent", "terms", "pre_score", "score", "branch_winner", ...}``.
        ``branch_winner`` is ``"missing"`` when the agent did not score the day.

    Raises:
        BreakdownMismatch: If the terms do not reproduce ``row["anomaly_score"]``.
    """
    if not _finite(row.get("anomaly_score")):
        return _missing(domain)
    if domain == "shipping":
        return _shipping(agent, row)
    if domain == "market":
        return _market(agent, row)
    if domain == "geopolitical":
        return _geopolitical(agent, row, present if present is not None else set(row.index))
    if domain == "natural_disaster":
        return _disaster(agent, row)
    if domain == "news_sentiment":
        return _news(agent, row)
    raise ValueError(f"unknown agent domain: {domain}")


def effective_terms(result: dict) -> dict[str, float]:
    """Terms of the branch that set the score — what a reader should be shown.

    For shipping that is the duration score alone when duration won, otherwise
    the two shock parts; for the disaster agent, the single hazard when a
    single event won. Other agents have one additive branch.
    """
    terms = result.get("terms", {})
    winner = result.get("branch_winner", "")
    if result.get("agent") == "shipping":
        if winner == "duration":
            return {"duration": terms["duration"]}
        return {k: v for k, v in terms.items() if k != "duration"}
    if result.get("agent") == "natural_disaster" and winner.startswith("single:"):
        feature = winner.split(":", 1)[1]
        return {feature: result["max_single"]}
    return dict(terms)
