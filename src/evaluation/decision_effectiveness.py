"""Decision-effectiveness evaluation — answers SRQ5 ("Decision Effectiveness").

The rest of the evaluation suite measures whether the system *detects* and
*explains* disruptions.  This module measures the step that actually matters to
a decision-maker: given the system's risk level, its SHAP explanation, and any
retrieved historical precedent, would a human be led to the **correct action**?

The mapping from evidence → action is a deliberately transparent, auditable
rule set (:func:`predict_action`) — not another ML model.  Decision support has
to be interpretable, and adding a black box to evaluate a black box would defeat
the purpose.

Ground-truth "correct" actions come from two sources:

* The 10 historical RAG cases — each case's correct action is derived from its
  ``impact`` / ``lessons`` / feature fields by :func:`derive_case_action` and
  persisted to ``data/knowledge_base/decision_labels.json`` so the labelling is
  inspectable and manually correctable.
* The synthetic Scenarios A/B/C — correct actions are assigned by day range from
  the known scenario definitions (:func:`scenario_correct_action`).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

#: The fixed, small action space a decision-maker chooses from.
ACTIONS: list[str] = ["no_action", "monitor", "reroute", "escalate"]

_DEFAULT_LABELS_PATH = Path("data/knowledge_base/decision_labels.json")

# Agent domains that imply a *physical* disruption (rerouting is the lever).
_PHYSICAL_AGENTS: frozenset[str] = frozenset({"routing", "shipping"})
# Agent domains that imply a *crisis* dimension (escalation is the lever).
_CRISIS_AGENTS: frozenset[str] = frozenset({"geopolitical", "natural_disaster"})
# Historical-similarity floor above which a retrieved escalate-precedent counts.
_ESCALATE_SIM_FLOOR: float = 0.75


# ===========================================================================
# Ground-truth labelling — historical cases
# ===========================================================================


def derive_case_action(case: dict) -> str:
    """Derive the correct action for one historical case from its own fields.

    Rubric (documented so the labels in ``decision_labels.json`` are auditable):

    * **escalate** — a catastrophic natural disaster (major port destruction),
      or a sustained / high-intensity geopolitical crisis (attacks, sanctions,
      war-risk classification) at a high geopolitical risk level.
    * **monitor** — a credible threat with little physical impact, or a gradual
      congestion build-up with no meaningful rerouting response.
    * **reroute** — a physical blockage / closure / hazard that triggered an
      active rerouting response (high rerouting share, grounding, disaster).
    * **no_action** — no meaningful disruption (none of the 10 real cases).

    Args:
        case: One disruption-case dict from ``disruption_cases.json``.

    Returns:
        One of :data:`ACTIONS`.
    """
    f = case.get("features", {}) or {}
    drop = float(f.get("vessel_count_drop_pct", 0.0))
    rerouting = float(f.get("rerouting_pct", 0.0))
    geo = str(f.get("geopolitical_risk_level", "low")).lower()
    disaster = bool(f.get("natural_disaster_involved", False))
    duration = int(case.get("duration_days", 0))
    text = " ".join([
        str(case.get("impact", "")),
        str(case.get("lessons", "")),
        str(case.get("description", "")),
    ]).lower()
    attack = any(w in text for w in
                 ("attack", "missile", "drone", "war-risk", "sanction", "embargo"))

    # escalate — catastrophic disaster or sustained high-intensity geo crisis.
    if disaster and duration >= 30:
        return "escalate"
    if attack and geo == "high" and (duration >= 60 or drop >= 20):
        return "escalate"

    # monitor — credible threat with little physical impact, or gradual congestion.
    if geo == "high" and drop < 10:
        return "monitor"
    if rerouting < 10 and not disaster:
        return "monitor"

    # reroute — physical disruption with an active rerouting response.
    if rerouting >= 15 or disaster or "block" in text or "ground" in text:
        return "reroute"

    return "monitor"


def generate_decision_labels(cases: list[dict]) -> dict[str, str]:
    """Build the ``{case_id: correct_action}`` mapping for all cases."""
    return {str(c["id"]): derive_case_action(c) for c in cases}


def load_decision_labels(path: str | Path = _DEFAULT_LABELS_PATH) -> dict[str, str]:
    """Load the persisted, human-auditable decision labels."""
    p = Path(path)
    if not p.exists():
        logger.warning("[decision_effectiveness] labels file not found: %s", p)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def case_decision_inputs(case: dict) -> dict:
    """Synthesize the (risk_level, top_drivers, sustained) a live run would emit.

    The historical cases carry summary features rather than a live risk score,
    so this maps their fields onto the same inputs :func:`predict_action`
    consumes for the daily stream, keeping the two evaluation arms consistent.

    Returns:
        ``{"risk_level", "top_drivers", "sustained"}``.
    """
    f = case.get("features", {}) or {}
    drop = float(f.get("vessel_count_drop_pct", 0.0))
    rerouting = float(f.get("rerouting_pct", 0.0))
    geo = str(f.get("geopolitical_risk_level", "low")).lower()
    disaster = bool(f.get("natural_disaster_involved", False))
    duration = int(case.get("duration_days", 0))

    if geo == "high" or drop >= 25 or rerouting >= 25 or disaster:
        risk_level = "high"
    elif drop >= 10 or geo == "medium":
        risk_level = "medium"
    else:
        risk_level = "low"

    # Crisis-aware top driver: disaster/geopolitical crises outrank the raw
    # primary agent so the action mapper sees the escalation-relevant domain.
    primary = list(case.get("primary_agents", []) or [])
    if disaster:
        top_agent = "natural_disaster"
    elif geo == "high":
        top_agent = "geopolitical"
    elif primary:
        top_agent = primary[0]
    else:
        top_agent = "shipping"

    # "Sustained crisis" for a case = a long geopolitical crisis or a major
    # disaster — deliberately *not* a long capacity/congestion episode, which
    # is chronic rather than an escalation trigger.
    sustained = (geo == "high" and duration >= 60) or (disaster and duration >= 30)

    return {
        "risk_level": risk_level,
        "top_drivers": [{"agent": top_agent}],
        "sustained": sustained,
    }


# ===========================================================================
# Ground-truth labelling — synthetic scenarios
# ===========================================================================


def scenario_correct_action(day_index: int, high_run_length: int = 0) -> tuple[str, str]:
    """Correct action + scenario tag for a synthetic day.

    Windows follow the connectors' injected scenarios:

    * Scenario A (days 60-74, moderate tension)  -> ``monitor``
    * Scenario B (days 150-170, major blockage)  -> ``reroute`` at onset,
      ``escalate`` once risk has been HIGH for >5 consecutive days
    * Scenario C (days 280-290, brief incident)  -> ``monitor``
    * Normal / baseline days                     -> ``no_action``

    Args:
        day_index: Positional day (0-based) in the 365-day frame.
        high_run_length: Consecutive HIGH-risk days up to and including this day
            (used only inside Scenario B to switch reroute -> escalate).

    Returns:
        ``(correct_action, scenario_tag)`` where tag is ``"A"`` / ``"B"`` /
        ``"C"`` / ``"normal"``.
    """
    if 60 <= day_index <= 74:
        return "monitor", "A"
    if 150 <= day_index <= 170:
        # >5 consecutive HIGH days -> the disruption is sustained -> escalate.
        return ("escalate" if high_run_length > 5 else "reroute"), "B"
    if 280 <= day_index <= 290:
        return "monitor", "C"
    return "no_action", "normal"


# ===========================================================================
# Evidence -> action mapper (transparent, rule-based)
# ===========================================================================


def predict_action(
    risk_level: str,
    top_shap_drivers: list[dict] | None,
    historical_context: dict | None,
    sustained: bool = False,
) -> str:
    """Map (risk level, SHAP drivers, historical precedent) to an action.

    Transparent rules:

    * ``low``    -> ``no_action``
    * ``medium`` -> ``monitor``
    * ``high`` + sustained (HIGH for >5 consecutive days)        -> ``escalate``
    * ``high`` + top driver agent in {routing, shipping}         -> ``reroute``
    * ``high`` + top driver agent in {geopolitical, natural_disaster}
      with a >0.75-similar historical case labelled ``escalate`` -> ``escalate``
    * otherwise                                                  -> ``monitor``

    The sustained-crisis branch is checked before rerouting so that a
    prolonged blockage escalates even when its leading signal is physical —
    matching the Scenario B ground truth.  Robust to missing / ``None``
    ``historical_context`` and empty drivers; always returns a value in
    :data:`ACTIONS`.

    Args:
        risk_level: ``"low"`` / ``"medium"`` / ``"high"`` (case-insensitive).
        top_shap_drivers: Ranked driver dicts (each with an ``"agent"`` key);
            ``None`` or empty is tolerated.
        historical_context: Optional dict with ``"similarity"`` and an action
            label (``"action"`` or ``"correct_action"``); ``None`` tolerated.
        sustained: Whether risk has been HIGH for >5 consecutive days.

    Returns:
        One of :data:`ACTIONS`.
    """
    level = str(risk_level or "low").lower()
    if level == "low":
        return "no_action"
    if level == "medium":
        return "monitor"

    # risk_level == "high"
    top_agent = ""
    if top_shap_drivers:
        top_agent = str(top_shap_drivers[0].get("agent", "") or "").lower()

    if sustained:
        return "escalate"

    if top_agent in _PHYSICAL_AGENTS:
        return "reroute"

    if top_agent in _CRISIS_AGENTS:
        hist_sim = 0.0
        hist_action = ""
        if historical_context:
            hist_sim = float(historical_context.get("similarity", 0.0) or 0.0)
            hist_action = str(
                historical_context.get("action")
                or historical_context.get("correct_action")
                or ""
            ).lower()
        if hist_sim > _ESCALATE_SIM_FLOOR and hist_action == "escalate":
            return "escalate"
        return "monitor"

    return "monitor"


# ===========================================================================
# Aggregate evaluation
# ===========================================================================


def _accuracy(records: list[dict]) -> float:
    """Fraction of records whose predicted action equals the correct action."""
    if not records:
        return 0.0
    correct = sum(1 for r in records if r["predicted"] == r["correct"])
    return correct / len(records)


def _confusion(records: list[dict]) -> dict[str, dict[str, int]]:
    """Actions x actions confusion matrix (rows = correct, cols = predicted)."""
    matrix: dict[str, dict[str, int]] = {
        a: {b: 0 for b in ACTIONS} for a in ACTIONS
    }
    for r in records:
        correct = r["correct"] if r["correct"] in ACTIONS else "no_action"
        predicted = r["predicted"] if r["predicted"] in ACTIONS else "no_action"
        matrix[correct][predicted] += 1
    return matrix


def evaluate_decision_effectiveness(
    day_records: list[dict],
    case_records: list[dict],
    baseline_day_records: list[dict] | None = None,
) -> dict:
    """Score decision effectiveness for the daily stream and historical cases.

    Args:
        day_records: One dict per synthetic day, each with ``scenario``,
            ``risk_level``, ``top_drivers``, ``sustained``,
            ``historical_context`` and ``correct_action``.
        case_records: One dict per historical case, with ``case_id``,
            ``risk_level``, ``top_drivers``, ``sustained``,
            ``historical_context`` and ``correct_action``.
        baseline_day_records: Same shape as ``day_records`` but produced from
            the naive threshold baseline (no agent attribution).  Optional.

    Returns:
        Dict with ``per_case_accuracy``, ``per_scenario_accuracy`` (A/B/C),
        ``overall_accuracy`` (daily stream), ``confusion_matrix`` and
        ``baseline_accuracy``.
    """
    def _scored(records: list[dict]) -> list[dict]:
        out = []
        for r in records:
            predicted = predict_action(
                r.get("risk_level", "low"),
                r.get("top_drivers"),
                r.get("historical_context"),
                sustained=bool(r.get("sustained", False)),
            )
            out.append({"predicted": predicted, "correct": r["correct_action"],
                        "scenario": r.get("scenario", "")})
        return out

    scored_days = _scored(day_records)
    scored_cases = _scored(case_records)

    # Per-scenario accuracy (A/B/C only; "normal" reported via overall).
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in scored_days:
        by_scenario[r["scenario"]].append(r)
    per_scenario = {
        tag: round(_accuracy(by_scenario.get(tag, [])), 4) for tag in ("A", "B", "C")
    }

    overall = round(_accuracy(scored_days), 4)
    per_case = round(_accuracy(scored_cases), 4)

    baseline_acc = 0.0
    if baseline_day_records is not None:
        baseline_acc = round(_accuracy(_scored(baseline_day_records)), 4)

    # Action distribution, handy for the thesis write-up.
    action_dist = dict(Counter(r["predicted"] for r in scored_days))

    return {
        "per_case_accuracy": per_case,
        "per_scenario_accuracy": per_scenario,
        "overall_accuracy": overall,
        "confusion_matrix": _confusion(scored_days),
        "baseline_accuracy": baseline_acc,
        "action_distribution": {a: int(action_dist.get(a, 0)) for a in ACTIONS},
        "n_days": len(scored_days),
        "n_cases": len(scored_cases),
    }
