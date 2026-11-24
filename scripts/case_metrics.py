"""Numbers for the Chapter 3 dashboard demonstration (four cases).

Every figure the demonstration text cites is computed here from the files the
dashboard reads (``data/live/<region>_risk.csv`` and ``data/live/<region>.csv``)
and written to ``thesis/figures/ch3_demo/case_metrics.json`` and, byte for byte, to the
committed ``eval/case_metrics.json`` (its ``provenance`` block holds the sha256 of every
live file it read; ``capture_demo_screenshots.py`` records the sha256 of the file itself in
the screenshot manifest). Nothing in the
scoring is changed: the engine score is the series' ``risk_score``; the
Decision view's status word and recommended action are recomputed with the
dashboard's own functions, exactly as the page evaluates them.

Two tier conventions are reported side by side:

* **engine** — ``risk_level`` as written by :class:`RiskEngine.compute_risk`
  (low / medium / high; it has no critical branch);
* **four-tier** — the same score cut at risk_medium / risk_high /
  risk_critical (Low / Medium / High / Critical); Critical is the cut-off the
  dashboard colours.

Usage::

    python scripts/case_metrics.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
logging.getLogger("streamlit").setLevel(logging.ERROR)

from src.core.regions import get_region  # noqa: E402
from src.dashboard import core  # noqa: E402

OUT_DIR = _PROJECT_ROOT / "thesis" / "figures" / "ch3_demo"
LIVE_DIR = _PROJECT_ROOT / "data" / "live"
#: The committed copy, so the demonstration's numbers can be cited and checked.
COMMITTED = _PROJECT_ROOT / "eval" / "case_metrics.json"
#: Window the Decision-view snapshots are captured with, so the assessed day
#: (the window's peak) stays within a week of the snapshot date.
DECISION_WINDOW = "1 week"

#: The four cases. ``event`` splits ships/day into before/during.
CASES: list[dict] = [
    {"id": 1, "slug": "hormuz_warning", "region": "hormuz",
     "name": "Early warning before the 2026 break",
     "window": ("2024-07-01", "2026-03-31"), "event": "2026-03-07",
     "snapshots": [("2024-10-01", "High"), ("2025-06-21", "High")]},
    {"id": 2, "slug": "hormuz_crisis", "region": "hormuz",
     "name": "Active crisis with degraded inputs",
     "window": ("2025-10-01", "2026-10-25"), "event": "2026-03-07",
     "snapshots": [("2026-04-15", "High"), ("2026-10-25", "Medium")]},
    {"id": 3, "slug": "bab_houthi", "region": "bab_el_mandeb",
     "name": "Houthi shipping crisis",
     "window": ("2023-06-01", "2025-12-31"), "event": "2023-11-19",
     "snapshots": [("2023-12-20", "Critical"), ("2024-01-15", "Critical")]},
    {"id": 4, "slug": "panama_drought", "region": "panama",
     "name": "Drought, then recovery",
     "window": ("2023-09-01", "2026-10-25"), "event": "2023-11-23",
     "snapshots": [("2024-01-15", "Medium"), ("2026-10-25", "Low")]},
]


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(regions) -> dict:
    """sha256 of the live frame, the risk series and the manifest each region's numbers came from.

    A file that is absent is ``None``, so a gap says so instead of vanishing.
    """
    names = {"frame": "{r}.csv", "risk": "{r}_risk.csv", "manifest": "{r}.manifest.json"}
    out = {}
    for r in regions:
        out[r] = {k: (sha256_file(LIVE_DIR / n.format(r=r)) if (LIVE_DIR / n.format(r=r)).exists()
                      else None) for k, n in names.items()}
    return out


def write_metrics(out: dict, paths) -> str:
    """Write ``out`` as the same LF-terminated JSON to every path; returns its sha256."""
    text = json.dumps(out, indent=2, default=str)
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _r(x, n: int = 3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def runs(flags: pd.Series, dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Start/end dates of consecutive True runs."""
    out, start, prev = [], None, None
    for d, f in zip(dates, flags):
        if f and start is None:
            start = d
        if not f and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, prev))
    return out


def load(region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk = pd.read_csv(LIVE_DIR / f"{region}_risk.csv", parse_dates=["timestamp"])
    risk = risk.dropna(subset=["risk_score"]).reset_index(drop=True)
    raw = pd.read_csv(LIVE_DIR / f"{region}.csv", parse_dates=["timestamp"])
    raw = raw.set_index("timestamp").reindex(risk["timestamp"]).reset_index()
    return risk, raw


def demo_route(region: str) -> dict:
    """The corridor shown for ``region``: the one scoped to the most agents
    that have data in the series (ties go to the first listed)."""
    ts, _ = core.load_live_timeseries(region)
    active = set(get_region(region).active_agents())
    with_data = {a for a in active if a in ts.columns and ts[a].notna().any()}
    routes = core.get_routes(region)
    return max(routes, key=lambda r: (len(set(r["agents"]) & with_data), -routes.index(r)))


def decision_view_state(region: str, date: str, params: dict) -> dict:
    """What the Decision view shows for ``?region=&route=&date=&window=1 week``.

    Mirrors ``decision_view.render``: the window's peak day on the chosen
    corridor is the assessment day; its status word, drivers and the
    rule-based action follow from it.
    """
    ts, features = core.load_live_timeseries(region)
    thr = params["thresholds"]
    route = demo_route(region)
    series = core.route_risk_series(ts, params, route, region)
    dates = pd.DatetimeIndex(ts["timestamp"])
    hi = int(dates.searchsorted(pd.Timestamp(date), side="right"))
    lo = max(0, hi - {"1 week": 7}[DECISION_WINDOW])
    vals = series.iloc[lo:hi].to_numpy()
    day = lo + int(np.argmax(vals)) + 1
    score = float(series.iloc[day - 1])
    agents = set(route.get("agents", core.ALL_AGENTS)) & set(get_region(region).active_agents())
    drivers = core.live_top_drivers(ts, features, params, agents, day)
    top = drivers[0] if drivers else {"agent": "", "feature": ""}
    level = core.classify_level(score, thr["risk_high"], thr["risk_medium"])
    sustained = bool(core.sustained_high_flags(series, thr["risk_high"])[day - 1])
    hist = None
    try:
        from src.evaluation.decision_effectiveness import load_decision_labels

        signals = {a: float(np.nan_to_num(ts[a].iloc[day - 1])) for a in core.ALL_AGENTS}
        matches = core.get_retriever().query(signals, top_k=1)
        if matches:
            labels = load_decision_labels(core._LABELS_PATH)
            hist = {"similarity": float(matches[0].get("similarity", 0.0)),
                    "action": labels.get(matches[0]["id"], "")}
    except Exception as exc:   # same tolerance as the page
        logging.warning("retriever unavailable: %s", exc)
    action, rule = core.decide_action(level, top.get("agent", ""), sustained, hist)
    on_date = int(dates.searchsorted(pd.Timestamp(date)))
    return {
        "route": route["name"],
        "route_key": route["key"],
        "route_composite_on_date": _r(series.iloc[on_date]),
        "route_agents": sorted(agents),
        "assessed_day": f"{dates[day - 1]:%Y-%m-%d}",
        "route_composite": _r(score),
        "status_word": core.status_word(score, thr),
        "level_for_action": level,
        "sustained": sustained,
        "top_driver": top,
        "drivers": drivers,
        "historical_match": hist,
        "action": action,
        "rule": rule,
    }


def acled_end(region: str) -> str:
    """Last day with ACLED conflict data in the region's live frame, read from its manifest.

    Returns ``"unknown"`` if the manifest is missing or records no coverage.
    """
    try:
        manifest = json.loads((LIVE_DIR / f"{region}.manifest.json").read_text(encoding="utf-8"))
        return str(manifest["coverage"]["geopolitical"][1])
    except (OSError, KeyError, IndexError, ValueError):
        return "unknown"


def gap_notes(region: str, gaps: list[str], risk: pd.DataFrame) -> list[str]:
    """Why an agent has no score on a snapshot day, as the demonstration text states it."""
    notes = []
    if "geopolitical" in gaps:
        notes.append(f"geopolitical: ACLED conflict data end {acled_end(region)}")
    if "news_sentiment" not in risk.columns:
        notes.append("news_sentiment: no GDELT series (HTTP 429 when the series was built)")
    return notes


def snapshot(case: dict, risk: pd.DataFrame, raw: pd.DataFrame, date: str,
             params: dict, active: list[str]) -> dict:
    thr = params["thresholds"]
    i = int(risk["timestamp"].searchsorted(pd.Timestamp(date)))
    row = risk.iloc[i]
    gaps = [a for a in active if pd.isna(row.get(a))]
    notes = gap_notes(case["region"], gaps, risk)
    return {
        "date": f"{row['timestamp']:%Y-%m-%d}",
        "risk_score": _r(row["risk_score"]),
        "engine_tier": str(row["risk_level"]),
        "four_tier": core.risk_tier(float(row["risk_score"]), thr),
        "agent_scores": {a: _r(row.get(a)) for a in active},
        "dominant_agent": str(row["top_agent"]),
        "ships_per_day": _r(raw["shipping__vessel_count"].iloc[i], 1),
        "in_traffic_drop_label": bool(row["y_true"]),
        "data_gaps": notes,
        "decision_view": decision_view_state(case["region"], date, params),
    }


def pick_snapshot(win: pd.DataFrame, date: str, intended: str, thr: dict) -> tuple[str, str | None]:
    """``date`` if its four-tier matches ``intended``, else the nearest matching day."""
    tiers = win["risk_score"].apply(lambda v: core.risk_tier(float(v), thr))
    target = pd.Timestamp(date)
    at = win.index[win["timestamp"] == target]
    if len(at) and tiers.loc[at[0]] == intended:
        return date, None
    match = win.loc[tiers == intended]
    if match.empty:
        return date, f"no {intended} day in the window; kept {date}"
    nearest = match.iloc[(match["timestamp"] - target).abs().argmin()]["timestamp"]
    actual = tiers.loc[at[0]] if len(at) else "n/a"
    return (f"{nearest:%Y-%m-%d}",
            f"{date} is {actual}, not {intended}; nearest {intended} day {nearest:%Y-%m-%d} used")


def case_metrics(case: dict, params: dict) -> dict:
    thr = params["thresholds"]
    risk, raw = load(case["region"])
    active = [a for a in core.ALL_AGENTS if a in risk.columns
              and a in get_region(case["region"]).active_agents()]
    s, e = (pd.Timestamp(d) for d in case["window"])
    m = (risk["timestamp"] >= s) & (risk["timestamp"] <= e)
    win, rwin = risk.loc[m], raw.loc[m]
    four = win["risk_score"].apply(lambda v: core.risk_tier(float(v), thr))
    high = win["risk_level"].eq("high")
    label = win["y_true"].astype(bool)

    snaps, changes = [], []
    for date, intended in case["snapshots"]:
        used, note = pick_snapshot(win, date, intended, thr)
        if note:
            changes.append(note)
        snaps.append({"requested": date, "intended_tier": intended,
                      **snapshot(case, risk, raw, used, params, active)})

    label_runs = []
    for a, b in runs(label, win["timestamp"]):
        start_i = int(risk.index[risk["timestamp"] == a][0])
        prior = risk.iloc[max(0, start_i - 60):start_i]
        first_high = prior.loc[prior["risk_level"] == "high", "timestamp"]
        lead = int((a - first_high.iloc[0]).days) if len(first_high) else None
        in_run = (win["timestamp"] >= a) & (win["timestamp"] <= b)
        label_runs.append({
            "start": f"{a:%Y-%m-%d}", "end": f"{b:%Y-%m-%d}",
            "days": int(in_run.sum()),
            "starts_before_window": bool(a == win["timestamp"].iloc[0] and start_i > 0
                                         and bool(risk["y_true"].iloc[start_i - 1])),
            "high_days_inside": int((high & in_run).sum()),
            "mean_risk_inside": _r(win.loc[in_run, "risk_score"].mean()),
            "lead_days_first_high_prior_60d": lead,
            "lead_is_cap": lead == 60 or (lead is not None and bool(
                prior["risk_level"].iloc[0] == "high")),
        })

    ev = pd.Timestamp(case["event"])
    ships = rwin["shipping__vessel_count"]
    years = rwin.assign(y=rwin["timestamp"].dt.year).groupby("y")["shipping__vessel_count"].mean()
    high_runs = [(a, b) for a, b in runs(high, win["timestamp"]) if (b - a).days >= 2]
    return {
        "case": case["id"], "slug": case["slug"], "region": case["region"],
        "name": case["name"], "window": list(case["window"]), "event_date": case["event"],
        "active_agents": active,
        "passive_agents": get_region(case["region"]).passive_agents(),
        "snapshots": snaps,
        "snapshot_changes": changes,
        "window_stats": {
            "days": int(len(win)),
            "mean_risk": _r(win["risk_score"].mean()),
            "max_risk": _r(win["risk_score"].max()),
            "max_risk_date": f"{win.loc[win['risk_score'].idxmax(), 'timestamp']:%Y-%m-%d}",
            "four_tier_days": {t: int((four == t).sum()) for t in ("Low", "Medium", "High", "Critical")},
            "engine_tier_days": {t: int((win["risk_level"] == t).sum()) for t in ("low", "medium", "high")},
            "dominant_share": {k: _r(v) for k, v in
                               win["top_agent"].value_counts(normalize=True).items()},
            "agent_means": {a: _r(win[a].mean()) for a in active},
            "agent_corr": {f"{a}~{b}": _r(win[a].corr(win[b]), 2)
                           for k, a in enumerate(active) for b in active[k + 1:]},
            "ships_per_day_before_event": _r(ships[rwin["timestamp"] < ev].mean(), 1),
            "ships_per_day_from_event": _r(ships[rwin["timestamp"] >= ev].mean(), 1),
            "ships_per_day_by_year": {int(k): _r(v, 1) for k, v in years.items()},
            "high_runs_3d_plus": [
                {"start": f"{a:%Y-%m-%d}", "end": f"{b:%Y-%m-%d}",
                 "max": _r(win.loc[(win["timestamp"] >= a) & (win["timestamp"] <= b), "risk_score"].max()),
                 "dominant": win.loc[(win["timestamp"] >= a) & (win["timestamp"] <= b),
                                     "top_agent"].mode().iloc[0]}
                for a, b in high_runs],
        },
        "label": {
            "runs": label_runs,
            "label_days": int(label.sum()),
            "high_days_inside_label": int((high & label).sum()),
            "high_days_outside_label": int((high & ~label).sum()),
        },
    }


def region_summary(region: str, params: dict) -> dict:
    """Whole-record figures used outside the four cases (Malacca, cross-case)."""
    thr = params["thresholds"]
    risk, raw = load(region)
    four = risk["risk_score"].apply(lambda v: core.risk_tier(float(v), thr))
    ts, _ = core.load_live_timeseries(region)
    route_series = core.route_risk_series(ts, params, demo_route(region), region)
    high = risk["risk_level"].eq("high")
    y = risk["y_true"].astype(bool)
    by_year = risk.assign(y=risk["timestamp"].dt.year).groupby("y")
    agents = [a for a in core.ALL_AGENTS if a in risk.columns]
    return {
        "first_day": f"{risk['timestamp'].iloc[0]:%Y-%m-%d}",
        "last_day": f"{risk['timestamp'].iloc[-1]:%Y-%m-%d}",
        "engine_critical_days": int((risk["risk_level"] == "critical").sum()),
        "dashboard_critical_days_engine_score": int((four == "Critical").sum()),
        "demo_route": demo_route(region)["key"],
        "dashboard_critical_days_demo_route": int(
            (route_series >= float(thr["risk_critical"])).sum()),
        "four_tier_days": {t: int((four == t).sum()) for t in ("Low", "Medium", "High", "Critical")},
        "engine_tier_days": {t: int((risk["risk_level"] == t).sum()) for t in ("low", "medium", "high")},
        "high_days": int(high.sum()),
        "label_days": int(y.sum()),
        "high_inside_label": int((high & y).sum()),
        "high_outside_label": int((high & ~y).sum()),
        "high_runs_3d_plus": sum(1 for a, b in runs(high, risk["timestamp"]) if (b - a).days >= 2),
        "dominant_share": {k: _r(v, 2) for k, v in risk["top_agent"].value_counts(normalize=True).items()},
        "agent_means": {a: _r(risk[a].mean(), 2) for a in agents},
        "mean_risk_by_year": {int(k): _r(v, 2) for k, v in by_year["risk_score"].mean().items()},
        "high_days_by_year": {int(k): int(v) for k, v in by_year["risk_level"].apply(
            lambda s: (s == "high").sum()).items()},
        "ships_per_day_by_year": {int(k): _r(v, 1) for k, v in raw.assign(
            y=raw["timestamp"].dt.year).groupby("y")["shipping__vessel_count"].mean().items()},
        "max_risk": _r(risk["risk_score"].max()),
        "max_risk_date": f"{risk.loc[risk['risk_score'].idxmax(), 'timestamp']:%Y-%m-%d}",
        "latest": {"risk_score": _r(risk["risk_score"].iloc[-1]),
                   "engine_tier": str(risk["risk_level"].iloc[-1]),
                   "four_tier": core.risk_tier(float(risk["risk_score"].iloc[-1]), thr)},
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = core.resolve_mode_params("hand_tuned")
    hormuz, _ = load("hormuz")
    panama, _ = load("panama")
    out = {
        "thresholds": {k: params["thresholds"][k]
                       for k in ("risk_medium", "risk_high", "risk_critical")},
        "inter_agent_weights": params["inter_weights"],
        "decision_window": DECISION_WINDOW,
        "cases": [case_metrics(c, params) for c in CASES],
        "regions": {r: region_summary(r, params)
                    for r in ("hormuz", "bab_el_mandeb", "panama", "malacca")},
        "market_corr_hormuz_panama": _r(hormuz["market"].corr(panama["market"]), 4),
    }
    out["provenance"] = provenance(out["regions"])
    path = OUT_DIR / "case_metrics.json"
    digest = write_metrics(out, (path, COMMITTED))
    print(f"wrote {path} and {COMMITTED} (sha256 {digest})")
    for c in out["cases"]:
        print(f"case {c['case']}: " + "; ".join(
            f"{s['date']} {s['risk_score']} {s['four_tier']} / DV {s['decision_view']['status_word']}"
            f" {s['decision_view']['action']}" for s in c["snapshots"]))
        for note in c["snapshot_changes"]:
            print("   change:", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
