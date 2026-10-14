"""Deterministic narrative, the claim checker, and the Decision-view card."""

import re

import pytest

from src.explainability.alert_explanation import build_aer
from src.explainability.narrative import check_claims, narrate, render

AGENTS = ["shipping", "market", "geopolitical"]
WEIGHTS = {"shipping": 0.25, "market": 0.15, "geopolitical": 0.25}


def _aer(scores, threshold=0.4, path="D", precedent=True, **kw):
    breakdowns = {
        "shipping": {"agent": "shipping", "terms": {"isolation_forest": 0.56, "max_zscore": 0.24,
                                                    "duration": 0.1},
                     "branch_winner": "shock", "shock": 0.8, "duration": 0.1},
        "market": {"agent": "market", "terms": {"brent_crude_usd": 0.3}, "branch_winner": "linear"},
        "geopolitical": {"agent": "geopolitical", "terms": {"military": 0.2},
                         "branch_winner": "sigmoid"},
    }
    rec = build_aer(
        alert_id="hormuz_2024-03-10", episode_id="hormuz_E001", region="hormuz",
        date="2024-03-10", path=path, weight_mode="hand_tuned", data_origin="real_eval",
        tier=3, agents=AGENTS, weights=WEIGHTS, scores=scores,
        baselines={"shipping": 0.4, "market": 0.7, "geopolitical": 0.1}, threshold=threshold,
        breakdowns=breakdowns, noise_sigma=dict.fromkeys(AGENTS, 0.05), y_true=0, **kw,
    ).to_dict()
    if precedent:
        rec["precedents"] = [{"doc_id": "x", "similarity": 0.7123, "region": "hormuz",
                              "event_type": "news_article", "event_date": "2024-02-28",
                              "title": "Tanker seized near the strait", "source_api": "serpapi",
                              "primary_agents": "news_sentiment"}]
    return rec


_ALERT = {"shipping": 0.8, "market": 0.75, "geopolitical": 0.6}


@pytest.mark.parametrize("style", ["decimal", "points"])
def test_template_is_fully_faithful(style: str) -> None:
    aer = _aer(_ALERT)
    result = narrate(aer, style)
    assert result["faithfulness"] == 1.0
    assert result["source"] == "template"
    assert len(result["claims"]) >= 6


def test_points_style_has_no_raw_decimals() -> None:
    text = render(_aer(_ALERT), "points")["text"]
    assert not re.findall(r"\b\d+\.\d{2,}\b", text)


def _shift_first_score(text: str) -> str:
    return re.sub(r"risk score (\d\.\d\d)", lambda m: f"risk score {float(m.group(1)) + 0.07:.2f}",
                  text, count=1)


def test_checker_rejects_corruptions() -> None:
    aer = _aer(_ALERT)
    text = render(aer)["text"]
    corrupted = {
        "number": _shift_first_score(text),
        "order": text.replace("shipping +", "@@ +").replace("geopolitical +", "shipping +")
                     .replace("@@ +", "geopolitical +"),
        "title": text.replace("Tanker seized near the strait", "Invented blockade"),
        "date": text.replace("2024-02-28", "2024-03-30"),
        "side": text.replace(" above the alert", " below the alert"),
        "agent": text + " News coverage contributed most.",
    }
    for name, bad in corrupted.items():
        assert bad != text, name
        assert check_claims(bad, aer)["faithfulness"] < 1.0, name


def test_paraphrase_used_only_when_faithful() -> None:
    aer = _aer(_ALERT)
    good = narrate(aer, paraphrase=lambda t: t.replace("Largest contributions", "Biggest contributions"))
    assert good["source"] == "paraphrase"
    bad = narrate(aer, paraphrase=_shift_first_score)
    assert bad["source"] == "template" and "rejected_paraphrase" in bad


def test_calm_day_wording() -> None:
    aer = _aer({"shipping": 0.3, "market": 0.6, "geopolitical": 0.05}, threshold=0.6,
               precedent=False)
    result = narrate(aer)
    assert "already under the alert line" in result["text"]
    assert "No agent is above its calm level" in result["text"]
    assert result["faithfulness"] == 1.0


def test_path_c_bonus_sentence_is_checked() -> None:
    aer = _aer({"shipping": 0.9, "market": 0.8, "geopolitical": 0.7}, path="C",
               bonus3=1.15, bonus5=1.25)
    text = render(aer)["text"]
    assert "Agreement between agents added" in text
    assert check_claims(text, aer)["faithfulness"] == 1.0


def test_decision_card_lines_are_plain() -> None:
    from src.dashboard.decision_view import _alert_card_lines

    lines = _alert_card_lines(_aer(_ALERT))
    joined = " ".join(lines)
    assert joined.startswith("Risk ")
    assert "alert line" in joined and "Main drivers" in joined
    assert "Similar past event (historical analogue)" in joined
    assert not re.findall(r"\b\d+\.\d{2,}\b", joined)
    for term in ("anomaly score", "shap value", "z-score"):
        assert term not in joined.lower()


def test_decision_card_infeasible_counterfactual_wording() -> None:
    from src.dashboard.decision_view import _alert_card_lines

    aer = _aer(_ALERT, threshold=0.5)  # shipping would have to fall below its calm 0.4
    assert aer["counterfactual"]["feasible"] is False
    joined = " ".join(_alert_card_lines(aer))
    assert "to fall below its normal level, so not realistic on its own" in joined
