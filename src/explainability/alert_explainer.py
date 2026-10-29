"""Per-alert explanation: SHAP drivers placed against their own recent history.

The surrogate SHAP explainer (:mod:`src.explainability.shap_explainer`) says
which features moved the score on a day. This module adds, for one alert day
``t``, whether each leading feature is unusual *for itself* and since when:

* status — the feature's value on ``t`` as a z-score against the 30 rows
  strictly before ``t``;
* history — for every reported feature, the earliest day in the 15 rows ending
  at ``t`` on which the feature was already outside ±2 standard deviations;
* one templated sentence per feature, opening with a short plain-language
  definition of what the feature measures (:data:`FEATURE_DEFINITIONS`).

Only rows at or before ``t`` are read, so an explanation for ``t`` cannot change
when later data arrives. Rows are positional and assumed to be consecutive
days, as in the evaluation frames under ``data/eval/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Rows strictly before ``t`` used for the trailing mean and standard deviation.
TRAILING_WINDOW: int = 30
#: Rows ending at ``t`` (inclusive) searched for the first threshold breach.
HISTORY_WINDOW: int = 15
#: |z| above which a feature counts as outside its normal range.
Z_THRESHOLD: float = 2.0
TOP_N: int = 5
TOP_N_HISTORY: int = TOP_N

#: Three-to-six-word definition of each feature, written for a reader who has
#: never seen the column name. Each states what the value measures *as this
#: system computes it* (e.g. congestion is traffic against its own recent
#: weeks, not anchorage waiting time), grounded in the source's own
#: documentation: IMF PortWatch transit calls, FRED / BLS producer price
#: indices, ACLED event types, GDACS alert levels and GDELT article tone.
FEATURE_DEFINITIONS: dict[str, str] = {
    # Shipping — IMF PortWatch daily chokepoint transits
    "vessel_count": "ships transiting the chokepoint daily",
    "tanker_count": "oil and gas tankers transiting daily",
    "vessel_count_trend": "ship traffic versus the past week",
    "avg_delay_hours": "estimated hours ships are delayed",
    "congestion_index": "ship traffic backlog versus recent weeks",
    # Market — FRED
    "brent_crude_usd": "Brent crude oil price in dollars",
    "trade_volume_index": "trade activity estimated from oil prices",
    "freight_rate_index": "price of deep-sea freight shipping",
    "freight_services_pct_change": "monthly change in freight service prices",
    # Geopolitical — ACLED
    "sanctions_severity": "strength of sanctions on the region",
    "military_activity_index": "battles and explosions near the corridor",
    "diplomatic_incident_score": "strategic political incidents in the region",
    "regime_stability_index": "political calm, fewer protests and riots",
    # Natural disaster — GDACS and USGS
    "earthquake_severity": "strength of nearby earthquakes",
    "tsunami_risk": "chance of a tsunami nearby",
    "cyclone_severity": "strength of nearby tropical cyclones",
    "severe_weather_index": "floods, wildfires and droughts nearby",
    # News — GDELT
    "sentiment_score": "tone of news about the region",
    "source_consensus": "how much news sources agree",
    "article_volume": "number of news articles published",
    "sentiment_magnitude": "strength of emotion in the news",
    "recency_weighted_score": "recent news tone, newest weighted most",
}


def trailing_z(values: np.ndarray, i: int, window: int = TRAILING_WINDOW) -> float:
    """z-score of ``values[i]`` against the ``window`` rows strictly before ``i``.

    Uses the sample standard deviation (``ddof=1``). Returns 0.0 when the
    standard deviation is zero (to within floating-point noise) or fewer than
    two prior rows exist.
    """
    prior = np.asarray(values[max(0, i - window):i], dtype=float)
    if len(prior) < 2:
        return 0.0
    mean = float(np.mean(prior))
    std = float(np.std(prior, ddof=1))
    # A flat (e.g. interpolated) series has std at floating-point noise, not 0.
    if not np.isfinite(std) or std <= 1e-9 * max(1.0, abs(mean)):
        return 0.0
    return float((float(values[i]) - mean) / std)


def history_window(n_rows: int, t: int, length: int = HISTORY_WINDOW) -> range:
    """Positional rows ``t-length+1 .. t``, clipped at the first row."""
    return range(max(0, t - length + 1), min(t, n_rows - 1) + 1)


def _status(z: float, prior_z: list[float]) -> str:
    """``unusual`` marks a first breach: outside ±2 now, inside ±2 for the
    preceding rows. Otherwise ``high`` / ``low`` / ``normal`` by sign."""
    if abs(z) > Z_THRESHOLD and prior_z and all(abs(p) <= Z_THRESHOLD for p in prior_z):
        return "unusual"
    if z > Z_THRESHOLD:
        return "high"
    if z < -Z_THRESHOLD:
        return "low"
    return "normal"


def _date_of(features_df: pd.DataFrame, i: int) -> str:
    if "timestamp" in features_df.columns:
        return str(pd.Timestamp(features_df["timestamp"].iloc[i]).date())
    label = features_df.index[i]
    return str(pd.Timestamp(label).date()) if isinstance(label, (pd.Timestamp, np.datetime64)) else str(label)


def _level(item: dict) -> str:
    """``high`` / ``low`` by the sign of z when outside ±2, else ``normal``.

    ``unusual`` (a first breach) is still high or low to the reader; the
    "first time" is carried by the duration clause instead.
    """
    if item["status"] == "normal":
        return "normal"
    return "high" if item["z"] > 0 else "low"


def _sentence(date: str, item: dict) -> str:
    feature = item["feature"].replace("_", " ").capitalize()
    named = f"{feature} ({item['definition']})"
    if item["status"] == "unavailable":
        return f"Alert on {date}: {named} has no data in this region; it {item['direction']}."
    level = _level(item)
    state = "within its normal range" if level == "normal" else level
    text = f"Alert on {date}: {named} is {state} (z = {item['z']:.2f}) and {item['direction']}."
    if level == "normal" or "days_before_alert" not in item:
        return text
    days = item["days_before_alert"]
    if not days:
        return f"{text} It turned {level} today, for the first time in the last 15 days."
    unit = "day" if days == 1 else "days"
    if item.get("since_window_start"):
        return f"{text} It has been {level} for at least {days} {unit}, since {item['since']} or earlier."
    return f"{text} It has been {level} for {days} {unit}, since {item['since']}."


def explain_alert(
    features_df: pd.DataFrame,
    shap_values: Mapping[str, float] | pd.DataFrame,
    t: int,
    region: str,
) -> dict:
    """Explain the alert on positional row ``t``.

    Args:
        features_df: One row per day, in time order. Features absent from it
            are reported as ``unavailable``.
        shap_values: SHAP value per feature for day ``t``, or a frame aligned
            with ``features_df`` from which row ``t`` is taken.
        t: Positional index of the alert day.
        region: Region key, recorded in the output.

    Returns:
        ``{region, date, t, complete, features, sentences}``. ``complete`` is
        True when every top feature is observed and ``t`` has the full
        trailing and history windows behind it.
    """
    t = int(t)
    if isinstance(shap_values, pd.DataFrame):
        shap_values = shap_values.iloc[t].to_dict()
    ranked = sorted(shap_values.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:TOP_N]
    date = _date_of(features_df, t)

    features: list[dict] = []
    for rank, (name, shap_value) in enumerate(ranked):
        item: dict[str, Any] = {
            "feature": name,
            "definition": FEATURE_DEFINITIONS.get(name, name.replace("_", " ")),
            "shap_value": round(float(shap_value), 6),
            "direction": "raises risk" if float(shap_value) > 0 else "lowers risk",
        }
        if name not in features_df.columns:
            item.update(value_t=None, mean=None, std=None, z=None, status="unavailable")
            features.append(item)
            continue

        values = pd.to_numeric(features_df[name], errors="coerce").to_numpy(dtype=float)
        prior = values[max(0, t - TRAILING_WINDOW):t]
        z = trailing_z(values, t)
        prior_z = [trailing_z(values, s) for s in range(max(0, t - TRAILING_WINDOW), t)]
        item.update(
            value_t=float(values[t]),
            mean=float(np.mean(prior)) if len(prior) else None,
            std=float(np.std(prior, ddof=1)) if len(prior) > 1 else None,
            z=round(z, 4),
            status=_status(z, prior_z),
        )

        if rank < TOP_N_HISTORY:
            window = history_window(len(values), t)
            breaches = [s for s in window if abs(trailing_z(values, s)) > Z_THRESHOLD]
            item["history_rows"] = [window.start, window.stop - 1]
            item["days_before_alert"] = (t - breaches[0]) if breaches else None
            item["since"] = _date_of(features_df, breaches[0]) if breaches else None
            # A breach on the window's first row may have started earlier.
            item["since_window_start"] = bool(breaches) and breaches[0] == window.start > 0
        features.append(item)

    complete = (
        len(features) == TOP_N
        and all(f["status"] != "unavailable" for f in features)
        and t >= 2 * TRAILING_WINDOW
    )
    return {
        "region": region,
        "date": date,
        "t": int(t),
        "complete": bool(complete),
        "features": features,
        "sentences": [_sentence(date, f) for f in features],
    }


def write_explanation(explanation: dict, out_dir: str | Path = "outputs/explanations") -> Path:
    """Write ``<region>_<date>.json`` and append a row to ``summary_<region>.md``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    region, date = explanation["region"], explanation["date"]
    path = out / f"{region}_{date}.json"
    path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")

    summary = out / f"summary_{region}.md"
    if not summary.exists():
        summary.write_text(
            f"# Alert explanations — {region}\n\n"
            "| Date | Complete | Top feature | Status | z | Sentence |\n"
            "|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    top = explanation["features"][0] if explanation["features"] else {}
    z = top.get("z")
    with summary.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {date} | {'yes' if explanation['complete'] else 'no'} | {top.get('feature', '')} "
            f"| {top.get('status', '')} | {'' if z is None else f'{z:.2f}'} "
            f"| {explanation['sentences'][0] if explanation['sentences'] else ''} |\n"
        )
    return path
