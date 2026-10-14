"""Deterministic narrative for an Alert Explanation Record, and a claim checker.

:func:`render` turns an AER into a short paragraph by template. Every sentence
is one *claim* tied to named AER fields, so the narrative is faithful by
construction: it can only restate numbers and names the record contains.

:func:`check_claims` verifies a text against an AER *without* knowing the
template, so it can vet any paraphrase (e.g. an optional LLM rewrite) before
it is shown. Per sentence it checks:

* every number matches a value in the AER (±1%, or half a unit of the
  precision shown, whichever is larger);
* agent names exist in the composite, and a sentence about contributions names
  agents in the AER's exact-Shapley rank order;
* "above"/"below" the threshold agrees with the sign of the margin;
* quoted titles and ISO dates are the alert's date or a precedent's.

A sentence passing every check is a supported claim. Faithfulness = supported
claims / claims.

Two styles: ``"decimal"`` (scores on [0, 1]; AERs, Analysis view) and
``"points"`` (scores x 100 as whole numbers; the Decision view, which must not
show raw decimals).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from src.explainability.agent_breakdown import effective_terms

AGENT_WORDS: dict[str, str] = {
    "shipping": "shipping",
    "market": "market",
    "geopolitical": "geopolitical",
    "natural_disaster": "natural disaster",
    "news_sentiment": "news",
}
_TERM_WORDS: dict[str, str] = {
    "isolation_forest": "unusual traffic pattern",
    "max_zscore": "extreme daily reading",
    "duration": "sustained drop in transits",
    "brent_crude_usd": "oil price",
    "trade_volume_index": "trade volume",
    "freight_rate_index": "freight rates",
    "freight_services_pct_change": "freight services",
    "sanctions": "sanctions",
    "military": "military activity",
    "diplomatic": "diplomatic incidents",
    "stability": "political instability",
    "earthquake": "earthquake", "tsunami": "tsunami", "cyclone": "cyclone",
    "severe_weather": "severe weather",
    "earthquake_severity": "earthquake", "tsunami_risk": "tsunami",
    "cyclone_severity": "cyclone", "severe_weather_index": "severe weather",
    "sentiment": "negative tone", "consensus": "consistent reporting",
    "velocity": "worsening tone", "volume": "news volume",
}
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_QUOTED = re.compile(r"[“\"]([^”\"]+)[”\"]")
_NUMBER = re.compile(r"(?<![\w.])[+\-−]?\d+(?:\.\d+)?")
_CONTRIBUTION = re.compile(r"\bcontribut", re.I)


# ------------------------------------------------------------------ format
def _fmt(value: float, style: str, signed: bool = False, digits: int = 2) -> str:
    if style == "points":
        number = round(value * 100)
        return f"{number:+d}" if signed else f"{number:d}"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _unit(style: str) -> str:
    return " points" if style == "points" else ""


# ------------------------------------------------------------------ render
def render(aer: Mapping[str, Any], style: str = "decimal",
           region_label: str | None = None) -> dict:
    """Template narrative for one AER.

    Returns:
        ``{"text", "claims": [{"text", "fields"}], "style"}``.
    """
    claims: list[tuple[str, list[str]]] = []
    u = _unit(style)
    region = region_label or str(aer.get("region", "")).replace("_", " ")
    margin = float(aer["margin"])
    side = "above" if margin > 0 else "below"
    if style == "points":
        status = (f"{region}, {aer['date']}: risk is {_fmt(aer['score'], style)} of 100, "
                  f"{_fmt(abs(margin), style)}{u} {side} the alert line at "
                  f"{_fmt(aer['threshold'], style)}.")
    else:
        status = (f"{region}, {aer['date']}: the risk score {_fmt(aer['score'], style)} is "
                  f"{_fmt(abs(margin), style)} {side} the alert threshold "
                  f"{_fmt(aer['threshold'], style)}.")
    claims.append((status, ["score", "margin", "threshold", "date"]))

    drivers = [d for d in aer["top_drivers"]
               if round(float(d["phi"]) * (100 if style == "points" else 1000)) > 0][:2]
    if drivers:
        parts = [f"{AGENT_WORDS[d['agent']]} {_fmt(d['phi'], style, signed=True, digits=3)}{u}"
                 for d in drivers]
        claims.append(("Largest contributions relative to a calm day: "
                       + ", then ".join(parts) + ".", ["top_drivers", "agent_shapley"]))
    else:
        claims.append(("No agent is above its calm level; every contribution is zero or "
                       "negative.", ["agent_shapley"]))

    lead = (drivers or aer["top_drivers"])[0]["agent"]
    breakdown = aer.get("agent_breakdown", {}).get(lead, {})
    terms = sorted(effective_terms(breakdown).items(), key=lambda kv: kv[1], reverse=True)[:2]
    terms = [(t, v) for t, v in terms if v > 0] if drivers else []
    if terms:
        listed = " and ".join(f"{_TERM_WORDS.get(t, t.replace('_', ' '))} "
                              f"{_fmt(v, style)}{u}" for t, v in terms)
        claims.append((f"Within {AGENT_WORDS[lead]}, the score was set by {listed}.",
                       [f"agent_breakdown.{lead}"]))

    bonus = aer.get("bonus_share")
    if bonus and abs(float(bonus.get("bonus_total", 0.0))) >= 0.0005:
        text = (f"Agreement between agents added "
                f"{_fmt(bonus['bonus_total'], style, signed=True, digits=3)}{u}")
        if abs(float(bonus.get("clip_total", 0.0))) >= 0.0005:
            text += f", and the cap at the maximum removed {_fmt(abs(bonus['clip_total']), style, digits=3)}{u}"
        claims.append((text + ".", ["bonus_share"]))

    unc = aer["uncertainty"]
    claims.append((
        f"Under calm-day input noise the score ranges from {_fmt(unc['bootstrap_lo'], style)} "
        f"to {_fmt(unc['bootstrap_hi'], style)}{u}; "
        f"{round(100 * unc['frac_draws_above_threshold'])}% of draws stay above the threshold.",
        ["uncertainty"]))

    cf = aer["counterfactual"]
    agent = AGENT_WORDS[cf["agent"]]
    if margin <= 0:
        claims.append(("The score is already under the alert line, so no single agent "
                       "needs to change.", ["counterfactual", "margin"]))
    elif cf["required_score"] is None:
        claims.append((f"No single agent can clear this alert on its own, not even {agent}.",
                       ["counterfactual"]))
    else:
        calm = aer["agent_baselines"][cf["agent"]]
        where = "still above" if cf["feasible"] else "below"
        claims.append((
            f"The alert would clear if {agent} fell by {_fmt(cf['required_score_drop'], style)}{u} "
            f"to {_fmt(cf['required_score'], style)}, {where} its calm level of "
            f"{_fmt(calm, style)}.", ["counterfactual", "agent_baselines"]))

    precedents = aer.get("precedents") or []
    if precedents:
        p = precedents[0]
        sim = (f"{round(100 * p['similarity'])}% similar" if style == "points"
               else f"similarity {p['similarity']:.2f}")
        claims.append((f"Closest earlier precedent in this region: “{p['title']}” "
                       f"({p['event_date']}, {sim}).", ["precedents"]))
    else:
        claims.append(("No earlier precedent from this region cleared the similarity floor.",
                       ["precedents"]))

    origin = str(aer.get("data_origin", ""))
    if origin.startswith("synthetic"):
        claims.append(("These are SYNTHETIC evaluation data, not a real event.", ["data_origin"]))
    return {"text": " ".join(c for c, _ in claims),
            "claims": [{"text": c, "fields": f} for c, f in claims], "style": style}


# ------------------------------------------------------------------ check
def _allowed_numbers(aer: Mapping[str, Any]) -> dict[str, list[float]]:
    """Numbers the AER supports (on the [0, 1] scale), grouped by claim kind."""
    unc = aer["uncertainty"]
    cf = aer["counterfactual"]
    bonus = aer.get("bonus_share") or {}
    terms: list[float] = []
    for b in aer.get("agent_breakdown", {}).values():
        terms += list(b.get("terms", {}).values())
        terms += [b[k] for k in ("max_single", "shock", "duration") if k in b]
    groups = {
        "status": [aer["score"], aer["threshold"], aer["margin"], abs(aer["margin"])],
        "contribution": list(aer["agent_shapley"].values()),
        "terms": terms,
        "bonus": [bonus.get("bonus_total", 0.0), abs(bonus.get("clip_total", 0.0))],
        "noise": [unc["bootstrap_lo"], unc["bootstrap_hi"]],
        "counterfactual": [v for v in (cf.get("required_score"), cf.get("required_score_drop"))
                           if v is not None] + list(aer["agent_baselines"].values()),
        "precedent": [p["similarity"] for p in aer.get("precedents") or []],
    }
    groups["any"] = [v for vals in groups.values() for v in vals] + list(aer["agent_scores"].values())
    return {k: [float(v) for v in vals if v is not None] for k, vals in groups.items()}


#: Which numbers a sentence may cite, recognised from its wording (first match
#: wins), so a paraphrase is held to the same binding as the template.
_SENTENCE_KINDS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"precedent", re.I), "precedent"),
    (re.compile(r"\bwould clear\b|\bcalm level\b|\bsingle agent\b", re.I), "counterfactual"),
    (re.compile(r"\bnoise\b|\bdraws\b", re.I), "noise"),
    (re.compile(r"\bagreement\b|\bcap\b", re.I), "bonus"),
    (re.compile(r"\bcontribut", re.I), "contribution"),
    (re.compile(r"\bwithin\b", re.I), "terms"),
    (re.compile(r"threshold|alert line", re.I), "status"),
)


def _kind(sentence: str) -> str:
    for pattern, kind in _SENTENCE_KINDS:
        if pattern.search(sentence):
            return kind
    return "any"


def _number_ok(token: str, allowed: list[float], style: str) -> bool:
    text = token.replace("−", "-")
    value = float(text)
    decimals = len(text.split(".")[1]) if "." in text else 0
    if style == "points":
        if value == 100:
            return True
        value /= 100.0
        half_unit = 0.005
    else:
        half_unit = 0.5 * 10 ** (-decimals) if decimals else 0.5
    for v in allowed:
        for candidate in (v, abs(v)):
            if abs(value - candidate) <= max(0.01 * abs(candidate), half_unit + 1e-12):
                return True
    return False


def check_claims(text: str, aer: Mapping[str, Any], style: str = "decimal") -> dict:
    """Rule-based faithfulness of ``text`` to ``aer`` (see module docstring)."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"])", text.strip()) if s]
    allowed = _allowed_numbers(aer)
    ranking = [d["agent"] for d in aer["top_drivers"]]
    used = set(aer["agents_used"])
    titles = {str(p["title"]) for p in aer.get("precedents") or []}
    dates = {str(aer["date"])} | {str(p["event_date"]) for p in aer.get("precedents") or []}
    percents = {round(100 * float(aer["uncertainty"]["frac_draws_above_threshold"]))}
    percents |= {round(100 * float(p["similarity"])) for p in aer.get("precedents") or []}

    failures = []
    for sentence in sentences:
        problems = []
        quoted = _QUOTED.findall(sentence)
        problems += [f"unknown title {q!r}" for q in quoted if q not in titles]
        body = _QUOTED.sub(" ", sentence)
        problems += [f"unknown date {d}" for d in _DATE.findall(body) if d not in dates]
        body = _DATE.sub(" ", body)
        group = allowed[_kind(sentence)]
        for match in _NUMBER.finditer(body):
            token = match.group(0)
            if body[match.end():].startswith("%"):
                if round(float(token.replace("−", "-"))) not in percents:
                    problems.append(f"unsupported percentage {token}%")
                continue
            if not _number_ok(token, group, style):
                problems.append(f"unsupported number {token}")
        lowered = body.lower()
        mentioned = sorted(((lowered.find(w), a) for a, w in AGENT_WORDS.items()
                            if re.search(rf"\b{w}\b", lowered)), key=lambda x: x[0])
        names = [a for _, a in mentioned]
        problems += [f"agent {a} not in the composite" for a in names if a not in used]
        if _CONTRIBUTION.search(sentence) and names and names != ranking[:len(names)]:
            problems.append(f"agents out of rank order: {names}")
        margin = float(aer["margin"])
        if re.search(r"\babove the alert", lowered) and margin <= 0:
            problems.append("says above, margin is not positive")
        if re.search(r"\bbelow the alert", lowered) and margin > 0:
            problems.append("says below, margin is positive")
        if problems:
            failures.append({"sentence": sentence, "problems": problems})
    total = len(sentences)
    supported = total - len(failures)
    return {"supported": supported, "total": total,
            "faithfulness": supported / total if total else float("nan"), "failures": failures}


def narrate(aer: Mapping[str, Any], style: str = "decimal", region_label: str | None = None,
            paraphrase: Callable[[str], str | None] | None = None) -> dict:
    """Template narrative, optionally paraphrased — but only if the paraphrase checks out.

    ``paraphrase`` (e.g. an LLM call) receives the template text. Its output is
    used only when :func:`check_claims` scores it 1.0; otherwise the template is
    kept and the rejection is recorded.

    Returns:
        ``{"text", "claims", "style", "faithfulness", "source"}``.
    """
    base = render(aer, style, region_label)
    template_check = check_claims(base["text"], aer, style)
    result = {**base, "faithfulness": template_check["faithfulness"], "source": "template"}
    if paraphrase is not None:
        candidate = paraphrase(base["text"])
        if candidate:
            check = check_claims(candidate, aer, style)
            if check["faithfulness"] == 1.0:
                return {**result, "text": candidate, "source": "paraphrase",
                        "faithfulness": 1.0}
            result["rejected_paraphrase"] = check
    return result
