"""Geopolitical score vs its 14-day baseline in Bab el-Mandeb (issue 13; diagnostic).

For the evaluation's test window (``evaluation.split_date`` .. ``end_date``) it
plots, on one time axis:

1. the geopolitical agent's score as the harness evaluates it (fitted on the
   training days, scored on train + test, test rows kept), with its 14-day
   rolling mean (``GeopoliticalAgent._ROLLING_WINDOW``);
2. the raw ACLED event count per day over the region's ACLED countries;
3. the level-shift label, shaded behind both.

and prints the mean score on positive and negative days, with and without the
days whose ACLED events are known to be missing (below).

What the agent does with a 14-day baseline: ``preprocess`` adds per-feature
``<feature>_baseline`` / ``_deviation`` columns, but ``detect`` scores the
feature *levels* (a weighted mean of military, diplomatic and inverted
stability indices, then a sigmoid) and ``validate`` checks levels against a
fixed threshold. The baselines are computed and never read. A 14-day rolling
mean necessarily catches up with a sustained level within about two weeks;
the score does not adapt because it never compares against it. The plotted
rolling mean of the score shows what such a baseline would do.

ACLED truncation. The connector caps each country-year at
``api_events_per_year`` events (``GeopoliticalConnector``), and the agent's
frame was fetched with the same cap. A capped country-year whose last returned
event falls well before 31 December is missing its tail: those days are
hatched and left out of the "complete days" means. A capped year with events
through December is incomplete somewhere unknown; it is marked lighter and
not excluded.

Raw events are fetched once with the agent's own extractor, country list and
cap, and cached with a fetch log (events returned per country-year, recorded
at fetch time) in ``data/eval/diagnostics/`` (git-ignored); ``--refresh``
fetches again. No pipeline code is changed.

Usage::

    python eval/diagnostics/geopolitical_baseline.py [--refresh]

Writes ``data/eval/diagnostics/geopolitical_<region>.png`` and ``.csv``.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "run_method_comparison", ROOT / "scripts" / "run_method_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)

from src.agents.geopolitical_agent import _ROLLING_WINDOW  # noqa: E402
from src.core.config_manager import load_config_for_region  # noqa: E402
from src.evaluation.window import eval_window  # noqa: E402
from src.ingestion.geopolitical_connector import GeopoliticalConnector  # noqa: E402

OUT = ROOT / "data" / "eval" / "diagnostics"
REGION = "bab_el_mandeb"
#: A capped year whose last event is within this many days of 31 December is
#: treated as incomplete somewhere unknown, not as missing its tail.
_TAIL_SLACK_DAYS = 3
#: Reference palette (dataviz skill), validated light: blue, orange; ink and chrome.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#eeede8"


def fetch_events(connector: GeopoliticalConnector, years: range,
                 refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw ACLED events and the fetch log (events returned per country-year)."""
    events_path = OUT / f"acled_{REGION}_events.csv"
    log_path = OUT / f"acled_{REGION}_fetch_log.csv"
    if events_path.exists() and log_path.exists() and not refresh:
        return (pd.read_csv(events_path, parse_dates=["event_date"]),
                pd.read_csv(log_path))
    from src.extractors.acled_extractor import ACLEDExtractor

    extractor = ACLEDExtractor(connector._acled_config())
    if not extractor.username or not extractor.password:
        raise RuntimeError("ACLED credentials are not configured (ACLED_USERNAME / "
                           "ACLED_PASSWORD); nothing fetched or cached.")
    rows, log = [], []
    for country in connector.acled_countries:
        for year in years:
            batch = extractor._fetch_events(country, year, limit=connector.api_events_per_year)
            log.append({"country": country, "year": year, "returned": len(batch),
                        "cap": connector.api_events_per_year})
            rows += [{"event_date": e.get("event_date"), "country": country,
                      "event_type": e.get("event_type"),
                      "fatalities": pd.to_numeric(e.get("fatalities"), errors="coerce")}
                     for e in batch]
            print(f"  {country} {year}: {len(batch)} events")
    log = pd.DataFrame(log)
    if not log["returned"].any():
        raise RuntimeError("ACLED returned no events at all; nothing cached.")
    events = pd.DataFrame(rows)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    OUT.mkdir(parents=True, exist_ok=True)
    events.dropna(subset=["event_date"]).to_csv(events_path, index=False)
    log.to_csv(log_path, index=False)
    return pd.read_csv(events_path, parse_dates=["event_date"]), log


def capped_years(events: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Country-years whose ACLED response hit the cap, with where it stops.

    ``tail_missing`` is True when the last event returned is more than
    ``_TAIL_SLACK_DAYS`` before 31 December: the days after it are missing.
    """
    capped = log[log["returned"] >= log["cap"]].copy()
    last = events.groupby([events["country"], events["event_date"].dt.year.rename("year")])[
        "event_date"].max().rename("last").reset_index()
    capped = capped.merge(last, on=["country", "year"], how="left")
    year_end = pd.to_datetime(capped["year"].astype(str) + "-12-31")
    capped["tail_missing"] = capped["last"] < year_end - pd.Timedelta(days=_TAIL_SLACK_DAYS)
    return capped


def series(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, _ = rmc._load(REGION)
    config = load_config_for_region(REGION)
    connector = GeopoliticalConnector(config["agents"]["geopolitical"])
    train, test = rmc._split(frame)
    w = eval_window()
    score = rmc._agent_scores("geopolitical", config, train, test)
    history = rmc._agent_scores("geopolitical", config, train, train)   # rolling warm-up
    ts = pd.to_datetime(pd.concat([train["timestamp"], test["timestamp"]]))
    full = pd.Series(np.concatenate([history, score]), index=ts.to_numpy())
    rolling = full.rolling(_ROLLING_WINDOW, min_periods=2).mean()
    events, log = fetch_events(connector, range(w["split_date"].year - 1, w["end_date"].year + 1),
                               refresh)
    capped = capped_years(events, log)
    days = pd.to_datetime(test["timestamp"]).reset_index(drop=True)
    daily = events.groupby(events["event_date"].dt.normalize()).size()
    missing = pd.Series("", index=days.index)
    for _, c in capped[capped["tail_missing"]].iterrows():
        tail = (days > c["last"]) & (days.dt.year == c["year"])
        missing[tail] = missing[tail] + c["country"] + ";"
    out = pd.DataFrame({
        "timestamp": days.to_numpy(),
        "y_true": test["y_true"].astype(int).to_numpy(),
        "geo_score": score,
        "geo_rolling_14d": rolling.reindex(days).to_numpy(),
        "acled_events": daily.reindex(days, fill_value=0).to_numpy(),
        "acled_missing": missing.to_numpy(),
        "vessel_count": test["shipping__vessel_count"].to_numpy(),
    })
    for col in ("military_activity_index", "diplomatic_incident_score", "regime_stability_index"):
        out[col] = test[f"geopolitical__{col}"].to_numpy()
    return out, capped


def report(s: pd.DataFrame, capped: pd.DataFrame) -> None:
    runs = (s.y_true.diff() != 0).cumsum()
    positive_runs = s[s.y_true == 1].groupby(runs).timestamp.agg(["min", "max"])
    if len(positive_runs) != 1:
        raise ValueError(f"expected one positive run in the window, found {len(positive_runs)}")
    start, end = positive_runs.iloc[0]
    complete = s[s.acled_missing == ""]
    print(f"\n[{REGION}] {s.timestamp.min().date()} .. {s.timestamp.max().date()}: "
          f"{int(s.y_true.sum())} positive, {int((s.y_true == 0).sum())} negative days; "
          f"{len(s) - len(complete)} days with ACLED events known missing")
    for label, part in (("all days", s), ("complete days", complete)):
        pos, neg = part[part.y_true == 1], part[part.y_true == 0]
        print(f"mean geopolitical score ({label:<13}): positive {pos.geo_score.mean():.3f} "
              f"(n={len(pos)}), negative {neg.geo_score.mean():.3f} (n={len(neg)}); "
              f"ACLED/day {pos.acled_events.mean():.1f} vs {neg.acled_events.mean():.1f}")
    for label, part in (("all days", s), ("complete days", complete)):
        for name, seg in (("before the run", part[part.timestamp < start]),
                          ("in the run", part[(part.timestamp >= start) & (part.timestamp <= end)]),
                          ("after the run", part[part.timestamp > end])):
            print(f"  {label:<13} {name:<15} {len(seg):>4} days: score {seg.geo_score.mean():.3f}, "
                  f"ACLED {seg.acled_events.mean():.1f}/day, ships {seg.vessel_count.mean():.1f}/day")
    print("ACLED country-years at the cap:")
    for _, c in capped.iterrows():
        kind = "tail missing" if c["tail_missing"] else "incomplete, gaps unknown"
        print(f"  {c['country']} {c['year']}: {c['returned']} returned, last {c['last'].date()} ({kind})")
    run = s[(s.timestamp >= start) & (s.timestamp <= end)].iloc[_ROLLING_WINDOW:]
    gap = run.geo_score - run.geo_rolling_14d
    print(f"inside the run after day {_ROLLING_WINDOW}: score {run.geo_score.mean():.3f}, "
          f"14-day rolling mean {run.geo_rolling_14d.mean():.3f} (mean score - rolling "
          f"{gap.mean():+.3f}); a rolling mean tracks any sustained level by construction")


def plot(s: pd.DataFrame, capped: pd.DataFrame, cap: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor("#fcfcfb")
    runs = (s.y_true.diff() != 0).cumsum()
    spans = s[s.y_true == 1].groupby(runs).timestamp.agg(["min", "max"])
    for ax in (top, bottom):
        ax.set_facecolor("#fcfcfb")
        for _, span in spans.iterrows():
            ax.axvspan(span["min"], span["max"], color=BAND, zorder=0, lw=0)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    top.plot(s.timestamp, s.geo_score, color=BLUE, lw=2, label="geopolitical score")
    top.plot(s.timestamp, s.geo_rolling_14d, color=ORANGE, lw=2, ls=(0, (4, 2)),
             label=f"{_ROLLING_WINDOW}-day rolling mean (the agent never uses one)")
    top.set_ylim(0, 1)
    top.set_ylabel("score")
    fig.suptitle("Bab el-Mandeb: geopolitical score, a 14-day baseline, ACLED events and the label",
                 x=0.01, ha="left", color=INK, fontsize=11)
    bottom.plot(s.timestamp, s.acled_events, color=BLUE, lw=1, label="ACLED events per day")
    bottom.set_ylabel("ACLED events / day")
    lo, hi = s.timestamp.min(), s.timestamp.max()
    ymax = bottom.get_ylim()[1]
    shown = 0
    for _, c in capped.iterrows():
        y0, y1 = pd.Timestamp(f"{c['year']}-01-01"), pd.Timestamp(f"{c['year']}-12-31")
        if c["tail_missing"]:
            start, end = max(c["last"], lo), min(y1, hi)
            style = dict(facecolor="none", hatch="///", edgecolor=MUTED, linewidth=0.5)
            text = f" {c['country']} missing "
        else:
            start, end = max(y0, lo), min(y1, hi)
            style = dict(facecolor="none", hatch="..", edgecolor=GRID, linewidth=0)
            text = f" {c['country']} capped "
        if start >= end:
            continue
        bottom.axvspan(start, end, **style)
        near_end = start > hi - pd.Timedelta(days=60)
        bottom.text(end if near_end else start, ymax * (0.97 - 0.1 * shown), text, color=INK_2,
                    fontsize=7, va="top", ha="right" if near_end else "left")
        shown += 1
    band = matplotlib.patches.Patch(color=BAND, label="label positive (sustained traffic drop)")
    handles, labels = top.get_legend_handles_labels()
    top.legend(handles + [band], labels + [band.get_label()], loc="lower left",
               bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8, labelcolor=INK_2)
    tail = matplotlib.patches.Patch(facecolor="none", hatch="///", edgecolor=MUTED,
                                    label=f"events missing ({cap:,}-event cap reached)")
    unknown = matplotlib.patches.Patch(facecolor="none", hatch="..", edgecolor=MUTED,
                                       label="capped, gaps unknown")
    bottom.legend(handles=[*bottom.get_legend_handles_labels()[0], tail, unknown],
                  loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8,
                  labelcolor=INK_2)
    bottom.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / f"geopolitical_{REGION}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="fetch ACLED events again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    OUT.mkdir(parents=True, exist_ok=True)
    s, capped = series(args.refresh)
    s.to_csv(OUT / f"geopolitical_{REGION}.csv", index=False)
    report(s, capped)
    cap = int(GeopoliticalConnector(load_config_for_region(REGION)["agents"]["geopolitical"])
              .api_events_per_year)
    print("wrote", plot(s, capped, cap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
