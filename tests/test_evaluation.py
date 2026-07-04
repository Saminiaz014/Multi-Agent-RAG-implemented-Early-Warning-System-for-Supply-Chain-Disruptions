"""Tests for the Phase 9a evaluation suite (``notebooks/evaluation.py``).

Five checks anchored to the synthetic world's known structure:

  Scenario A (Moderate Tension) : shipping days 60-74, no earthquake
  Scenario B (Major Blockage)   : shipping days 150-170, earthquake at day 148
  Normal period                 : days 100-130 (no injected disruption)

1. test_scenario_b_high_risk    — Scenario B reaches HIGH risk by day 151 and
   all six agents fire.
2. test_scenario_a_medium_risk  — Scenario A reaches at least MEDIUM risk and
   exactly five of six agents fire (natural_disaster does not).
3. test_normal_period_low       — the normal window raises no HIGH alert and
   its mean risk stays below the MEDIUM line.
4. test_6agent_beats_1agent     — the full six-agent ensemble's F1 is at least
   the shipping-only F1 (agent diversity adds value).
5. test_optimized_beats_handtuned — optimization improves the blended objective
   and lead time on the test split (raw F1 is a deliberate trade-off — see
   docstring).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_NOTEBOOKS = _PROJECT_ROOT / "notebooks"
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))

import evaluation as ev  # noqa: E402  (notebooks/evaluation.py)

from src.evaluation.decision_effectiveness import (  # noqa: E402
    ACTIONS,
    generate_decision_labels,
    load_decision_labels,
    predict_action,
)
from src.optimization.data_split import DataSplitManager  # noqa: E402
from src.optimization.pipeline_evaluator import PipelineEvaluator  # noqa: E402
from src.optimization.weight_config import (  # noqa: E402
    load_optimized_weights,
    resolve_active_weights,
)

_KB_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "disruption_cases.json"
_LABELS_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "decision_labels.json"

# Scenario windows (positional day indices into the 365-day synthetic frame).
_SCEN_A = range(60, 75)      # Moderate Tension
_SCEN_B = range(150, 171)    # Major Blockage
_NORMAL = range(100, 131)    # quiet stretch between scenarios
# An agent "fires" when its peak anomaly score in the window clears this cutoff
# (baseline agent scores sit well below it; a fired agent sits well above).
_FIRE = 0.30
_ALL_AGENTS = {
    "shipping", "market", "geopolitical",
    "natural_disaster", "routing", "news_sentiment",
}


# ---------------------------------------------------------------------------
# Shared, module-scoped evaluation environment (agents fitted once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env() -> dict:
    config = yaml.safe_load((_PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))

    hand_layout = resolve_active_weights({**config, "weight_mode": "hand_tuned"})
    opt_file = load_optimized_weights(config)
    opt_layout = opt_file if (opt_file and opt_file.get("inter_agent_weights")) else hand_layout

    ht_params = ev._layout_to_params(hand_layout)
    opt_params = ev._layout_to_params(opt_layout)

    dm = DataSplitManager(config)
    dm.get_splits()
    evaluator = PipelineEvaluator(dm, config.get("optimization", {}).get("objective_weights"))

    # Hand-tuned per-agent scores on the test split → per-day risk + agent peaks.
    series = ev._score_series(evaluator, ht_params, "train", "test")
    scores_df = pd.DataFrame(series).sort_index()
    risk = evaluator._aggregate_daily(scores_df, ht_params).reset_index(drop=True)
    agent_scores = scores_df.reset_index(drop=True)

    thr = ht_params["thresholds"]

    def level(x: float) -> str:
        if x >= thr["risk_high"]:
            return "high"
        if x >= thr["risk_medium"]:
            return "medium"
        return "low"

    def fired_agents(window: range) -> set[str]:
        return {
            name for name in agent_scores.columns
            if float(agent_scores[name].iloc[list(window)].max()) >= _FIRE
        }

    # --- Decision effectiveness (hand-tuned, the default weight mode) ---
    import json

    cases = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    labels = load_decision_labels(_LABELS_PATH) or generate_decision_labels(cases)
    case_records = ev._build_case_records(cases, labels)
    baseline_records = ev._build_baseline_day_records(evaluator, ht_params)
    day_records = ev._build_day_records(evaluator, ht_params, "train", "test")
    from src.evaluation.decision_effectiveness import evaluate_decision_effectiveness
    decision = evaluate_decision_effectiveness(day_records, case_records, baseline_records)

    return {
        "evaluator": evaluator,
        "ht_params": ht_params,
        "opt_params": opt_params,
        "risk": risk,
        "level": level,
        "thr": thr,
        "fired_agents": fired_agents,
        "day_records": day_records,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# 1 — Scenario B: HIGH risk by day 151, all six agents fire
# ---------------------------------------------------------------------------


def test_scenario_b_high_risk(env):
    risk, level = env["risk"], env["level"]

    # HIGH risk reached at or before day 151.
    reached_high_by_151 = any(level(float(risk[d])) == "high" for d in range(150, 152))
    assert reached_high_by_151, (
        f"Scenario B did not reach HIGH risk by day 151; "
        f"day150={risk[150]:.3f}, day151={risk[151]:.3f}"
    )
    assert level(float(risk[151])) == "high", f"day 151 risk {risk[151]:.3f} not HIGH"

    # All six domain agents fire during the blockage window.
    fired = env["fired_agents"](_SCEN_B)
    assert fired == _ALL_AGENTS, f"Expected all 6 agents to fire in Scenario B, got {sorted(fired)}"


# ---------------------------------------------------------------------------
# 2 — Scenario A: >= MEDIUM risk, five of six fire (not natural_disaster)
# ---------------------------------------------------------------------------


def test_scenario_a_medium_risk(env):
    risk, level = env["risk"], env["level"]

    peak_level = max((level(float(risk[d])) for d in _SCEN_A), key=lambda lv: {"low": 0, "medium": 1, "high": 2}[lv])
    assert peak_level in ("medium", "high"), f"Scenario A peak risk level was {peak_level}, expected >= medium"

    fired = env["fired_agents"](_SCEN_A)
    assert "natural_disaster" not in fired, "natural_disaster should not fire in Scenario A (no earthquake)"
    assert len(fired) == 5, f"Expected exactly 5 of 6 agents to fire in Scenario A, got {sorted(fired)}"


# ---------------------------------------------------------------------------
# 3 — Normal period: no HIGH alert, mean risk below MEDIUM
# ---------------------------------------------------------------------------


def test_normal_period_low(env):
    risk, thr, level = env["risk"], env["thr"], env["level"]
    window = risk.iloc[list(_NORMAL)]

    assert not (window >= thr["risk_high"]).any(), (
        f"Normal period raised a HIGH alert (max risk {window.max():.3f} >= {thr['risk_high']})"
    )
    assert float(window.mean()) < thr["risk_medium"], (
        f"Normal-period mean risk {window.mean():.3f} not below MEDIUM ({thr['risk_medium']})"
    )
    # Majority of days should classify as LOW.
    low_fraction = sum(1 for x in window if level(float(x)) == "low") / len(window)
    assert low_fraction >= 0.7, f"Only {low_fraction:.0%} of normal-period days are LOW"


# ---------------------------------------------------------------------------
# 4 — Agent diversity: 6-agent F1 >= 1-agent F1
# ---------------------------------------------------------------------------


def test_6agent_beats_1agent(env):
    evaluator, ht = env["evaluator"], env["ht_params"]

    six = evaluator.evaluate(ev._mask_to_active(ht, _ALL_AGENTS), "train", "test")
    one = evaluator.evaluate(ev._mask_to_active(ht, {"shipping"}), "train", "test")

    assert six.f1 >= one.f1, (
        f"6-agent F1 ({six.f1:.3f}) should be >= 1-agent F1 ({one.f1:.3f}); "
        "agent diversity is expected to add detection value."
    )


# ---------------------------------------------------------------------------
# 5 — Optimization improves the blended objective (F1 is a deliberate trade-off)
# ---------------------------------------------------------------------------


def test_optimized_beats_handtuned(env):
    """Optimization is multi-objective (F1 0.50 + lead-time 0.30 - FPR 0.20).

    On this dataset it trades a small amount of raw F1 for substantially more
    early-warning lead time, so the honest success criterion is the blended
    objective (and lead time), not F1 in isolation.  This mirrors the recorded
    optimization result (test objective 0.638 -> 0.726, lead time +1.7 days,
    F1 -0.024).
    """
    evaluator, ht, opt = env["evaluator"], env["ht_params"], env["opt_params"]

    ht_m = evaluator.evaluate(ht, "train", "test")
    opt_m = evaluator.evaluate(opt, "train", "test")

    assert opt_m.objective >= ht_m.objective, (
        f"Optimized objective ({opt_m.objective:.3f}) should be >= hand-tuned "
        f"({ht_m.objective:.3f})."
    )
    assert opt_m.lead_time_days >= ht_m.lead_time_days, (
        f"Optimized lead time ({opt_m.lead_time_days:.2f}d) should be >= hand-tuned "
        f"({ht_m.lead_time_days:.2f}d)."
    )
    # Raw F1 stays within a small tolerance of hand-tuned (the intended trade-off).
    assert opt_m.f1 >= ht_m.f1 - 0.05, (
        f"Optimized F1 ({opt_m.f1:.3f}) fell more than 0.05 below hand-tuned "
        f"({ht_m.f1:.3f}) — larger than the expected trade-off."
    )


# ---------------------------------------------------------------------------
# 6 — Decision labels are complete and valid
# ---------------------------------------------------------------------------


def test_decision_labels_complete():
    import json

    cases = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    case_ids = {str(c["id"]) for c in cases}
    assert len(case_ids) == 10, f"Expected 10 historical cases, got {len(case_ids)}"

    labels = load_decision_labels(_LABELS_PATH)
    assert set(labels.keys()) == case_ids, (
        f"decision_labels.json is missing entries for: {case_ids - set(labels.keys())}"
    )
    for cid, action in labels.items():
        assert action in ACTIONS, f"Case {cid} has invalid action '{action}'"


# ---------------------------------------------------------------------------
# 7 — predict_action always returns a valid action and never crashes
# ---------------------------------------------------------------------------


def test_predict_action_valid_output():
    # Missing / None / malformed inputs must not crash and must stay in ACTIONS.
    cases = [
        ("low", None, None, False),
        ("medium", [], None, False),
        ("high", None, None, False),
        ("high", [{"agent": "routing"}], None, False),
        ("high", [{"agent": "geopolitical"}], {"similarity": 0.9, "action": "escalate"}, False),
        ("high", [{"agent": "geopolitical"}], {"similarity": 0.1}, False),
        ("high", [{"agent": "shipping"}], None, True),
        ("high", [{}], None, False),               # driver with no agent key
        ("HIGH", [{"agent": "MARKET"}], {}, False),  # odd casing, empty context
        (None, None, None, False),                 # everything missing
        ("unknown_level", [{"agent": "x"}], None, False),
    ]
    for risk_level, drivers, context, sustained in cases:
        action = predict_action(risk_level, drivers, context, sustained=sustained)
        assert action in ACTIONS, (
            f"predict_action({risk_level!r}, {drivers!r}, {context!r}, {sustained}) "
            f"returned invalid action {action!r}"
        )

    # Spot-check the documented rules.
    assert predict_action("low", None, None) == "no_action"
    assert predict_action("medium", None, None) == "monitor"
    assert predict_action("high", [{"agent": "routing"}], None) == "reroute"
    assert predict_action("high", [{"agent": "shipping"}], None, sustained=True) == "escalate"


# ---------------------------------------------------------------------------
# 8 — Scenario B peak yields an actionable decision (reroute or escalate)
# ---------------------------------------------------------------------------


def test_decision_effectiveness_scenario_b(env):
    record = env["day_records"][155]  # peak of the Major Blockage
    action = predict_action(
        record["risk_level"], record["top_drivers"],
        record["historical_context"], sustained=record["sustained"],
    )
    assert action in ("reroute", "escalate"), (
        f"Scenario B peak (day 155) predicted {action!r}; expected an actionable "
        "response (reroute/escalate), not no_action/monitor."
    )


# ---------------------------------------------------------------------------
# 9 — Full pipeline decisions beat the naive-baseline decisions
# ---------------------------------------------------------------------------


def test_decision_effectiveness_beats_baseline(env):
    d = env["decision"]
    assert d["overall_accuracy"] >= d["baseline_accuracy"], (
        f"Pipeline decision accuracy ({d['overall_accuracy']:.3f}) should be >= "
        f"naive-baseline accuracy ({d['baseline_accuracy']:.3f}); agent attribution "
        "is expected to make the risk score more actionable."
    )


# ---------------------------------------------------------------------------
# 10 — Overall decision accuracy clears the 0.75 target (SRQ5)
# ---------------------------------------------------------------------------


def test_decision_effectiveness_threshold(env):
    d = env["decision"]
    assert d["overall_accuracy"] > 0.75, (
        f"Overall decision accuracy {d['overall_accuracy']:.3f} did not exceed 0.75.\n"
        f"per_scenario={d['per_scenario_accuracy']}, per_case={d['per_case_accuracy']}, "
        f"baseline={d['baseline_accuracy']}"
    )
