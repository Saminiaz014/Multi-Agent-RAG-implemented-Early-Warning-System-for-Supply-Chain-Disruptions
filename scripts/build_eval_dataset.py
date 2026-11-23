"""Assemble the real per-region evaluation dataset.

Every method comparison and ablation downstream reads this. It exists so those
run against *measured* features rather than generated ones: an evaluation on
``np.random.normal`` cannot distinguish a good detector from a bad one, since
there is no signal in the data to find, and every AUC lands at 0.5 by
construction.

Each region's frame merges the live connectors on a daily index:

* shipping — IMF PortWatch chokepoint transits (2019-2026, all four regions)
* market — FRED Brent, freight PPI, freight services
* natural_disaster — GDACS alerts + USGS seismicity
* geopolitical — ACLED conflict events

* news_sentiment — GDELT 1.0 daily event exports, corridor box filter
  (``NewsConnector`` ``gdelt_events`` mode, reading the series that
  ``scripts/build_gdelt_news_csv.py`` builds under ``data/eval/news/``)

A domain that cannot be loaded is omitted, and the column set for each region
is recorded in the manifest, so a later run cannot quietly compare a
five-feature region against a four-feature one. The news filter's version and
hashes go in the manifest as ``news_source``.

No gap is filled with future data. Inside a source's coverage, a day it has
no value for is forward-filled from the last earlier value (linear
interpolation would read the next observation). FRED series are already
stamped on the day each value was first published (``MarketConnector``), so
carrying them forward is what a forecaster on that day would have known.

News days GDELT never published are not filled at all. They stay NaN in the
frame ("masked") and the method comparison leaves them out of every score;
the frame keeps the row so the other domains' days are unchanged. Any other
gap in the news series is an error.

``--refresh-news`` swaps only the news columns of the cached frames for the
current GDELT series. The other domains are not re-pulled, so their values and
the frame's rows stay exactly as they were.

Ground truth is the shipping connector's ``is_disruption`` — a 2-sigma
persistent drop in transits, plus the pinned April-May 2026 Hormuz shutdown.
Its known weakness travels with the data in the manifest: a 30-day rolling
baseline drifts down with a slow decline, so the label catches shocks and
misses slow-onset disruption. Panama's 2023-24 drought is the case in point —
transits fell 38% and the label flags almost none of it.

Output is cached under ``data/eval/`` (gitignored, like every other derived
artefact) so the comparison and ablation scripts do not re-pull the APIs.

Usage::

    python scripts/build_eval_dataset.py
    python scripts/build_eval_dataset.py --region panama --force
    python scripts/build_eval_dataset.py --refresh-news
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config_manager import load_config_for_region  # noqa: E402
from src.core.regions import list_regions  # noqa: E402

logger = logging.getLogger(__name__)

_CACHE_DIR = _PROJECT_ROOT / "data" / "eval"

#: Feature columns each domain contributes, prefixed on merge so two domains
#: cannot collide on a shared name.
_DOMAIN_FEATURES: dict[str, tuple[str, ...]] = {
    "shipping": ("vessel_count", "tanker_count", "avg_delay_hours",
                 "congestion_index"),
    "market": ("brent_crude_usd", "trade_volume_index", "freight_rate_index"),
    "natural_disaster": ("earthquake_severity", "tsunami_risk",
                         "cyclone_severity", "severe_weather_index"),
    "geopolitical": ("military_activity_index", "diplomatic_incident_score",
                     "regime_stability_index"),
    "news_sentiment": ("sentiment_score", "sentiment_magnitude",
                       "article_volume", "recency_weighted_score"),
}


#: Level-shift label parameters. A 30-day mean this far below the *trailing*
#: annual median, sustained this long, counts as a disruption.
_LS_DROP = 0.20
_LS_WINDOW = 30
_LS_BASELINE = 365
_LS_MIN_RUN = 14


def level_shift_label(vessel_count: pd.Series) -> pd.Series:
    """Label sustained drops in chokepoint traffic.

    The connector's own ``is_disruption`` is a 2-sigma drop against a 30-day
    rolling baseline. On real data that is unusably sparse for evaluation:
    across 2019-2026 it yields 0 positives for Hormuz outside the pinned 2026
    shutdown, 0 for Malacca ever, 3 for Bab el-Mandeb and 5 for Panama. A
    rolling baseline also drifts down with a slow decline, so the rule cannot
    see a drought at all.

    This compares a 30-day mean against the *trailing* annual median. Trailing
    matters: measured against a whole-series median instead, Malacca's traffic
    growth reads as three years of "disruption" and Bab el-Mandeb's post-2024
    level shift never ends, because a permanently lower normal stays below a
    fixed baseline forever.

    The parameters were not tuned against known events, and the label
    independently recovers two documented disruptions -- the Houthi Red Sea
    crisis (bab_el_mandeb, from 2024-01) and the Gatun Lake drought (panama,
    2023-11 to 2024-04) -- which is the evidence that it tracks reality rather
    than fitting it.

    Args:
        vessel_count: Daily transit counts, ordered by date.

    Returns:
        Boolean series, index-aligned with ``vessel_count``.
    """
    rolling = vessel_count.rolling(_LS_WINDOW, min_periods=_LS_WINDOW // 2).mean()
    # shift() so the baseline period cannot include the window it judges.
    baseline = (
        vessel_count.shift(_LS_WINDOW)
        .rolling(_LS_BASELINE, min_periods=_LS_BASELINE // 2)
        .median()
    )
    below = (rolling < baseline * (1 - _LS_DROP)).fillna(False).to_numpy()

    flags = [False] * len(below)
    run = 0
    for i, is_below in enumerate(below):
        run = run + 1 if is_below else 0
        if run >= _LS_MIN_RUN:
            for j in range(i - run + 1, i + 1):
                flags[j] = True
    return pd.Series(flags, index=vessel_count.index)


def _fetch_domain(domain: str, config: dict) -> pd.DataFrame | None:
    """Fetch one domain's live series, or ``None`` if it is unavailable.

    A domain that cannot be fetched is dropped and recorded, never replaced
    with synthetic values — mixing generated and measured features inside one
    evaluation frame would make every downstream number uninterpretable.
    """
    from src.ingestion import (DisasterConnector, GeopoliticalConnector,
                               MarketConnector, NewsConnector,
                               ShippingConnector)

    try:
        if domain == "shipping":
            block = dict(config["ingestion"]["shipping"])
            return ShippingConnector(source_mode="api", config=block).fetch()
        if domain == "market":
            block = dict(config["ingestion"]["market"])
            return MarketConnector(source_mode="api", config=block).fetch()

        if domain == "news_sentiment":
            return _load_news(config)
        block = dict(config["agents"][domain])
        block["data_mode"] = "api"
        connector = {
            "natural_disaster": DisasterConnector,
            "geopolitical": GeopoliticalConnector,
        }[domain]
        return connector(config=block).fetch()
    except Exception as exc:
        logger.warning("  %s unavailable: %s", domain, str(exc)[:160])
        return None


def _load_news(config: dict, news_dir: str | Path | None = None) -> pd.DataFrame:
    """The region's GDELT 1.0 news series. Errors propagate.

    The DOC 2.0 API (``api`` mode) is IP-blocked here (HTTP 429 on every
    request); the GDELT 1.0 export series built by
    ``scripts/build_gdelt_news_csv.py`` replaces it.
    """
    from src.ingestion import NewsConnector

    block = dict(config["agents"]["news_sentiment"])
    block["data_mode"] = "gdelt_events"
    block.setdefault("region", config["region"]["name"])
    if news_dir is not None:
        block["gdelt_events_dir"] = str(news_dir)
    return NewsConnector(config=block).fetch()


#: How a news day GDELT never published is treated in the evaluation frame.
_NEWS_GAP_POLICY = ("masked: days GDELT never published stay NaN and are left "
                    "out of every score; not forward-filled or interpolated")

#: How a gap inside any other domain's coverage is filled.
_GAP_FILL_POLICY = ("forward-fill from the last earlier value, inside each "
                    "column's coverage only (no interpolation, no future data)")


def attach_news(frame: pd.DataFrame, news: pd.DataFrame, *,
                outside: str = "raise") -> tuple[pd.DataFrame, dict]:
    """Merge the news series onto ``frame`` with the news gap policy.

    Days inside the news series' range that it has no row for must be days
    GDELT never published (the manifest's ``missing_days``). They are left
    NaN ("masked") and listed as ``days_masked``. Filling them would invent
    news: linear interpolation reads the next published day (up to 18 days
    ahead across the June 2025 outage), and forward-fill repeats one day's
    coverage for weeks.

    Args:
        frame: Time-ordered frame without news columns.
        news: The news connector's frame (``gdelt_events`` mode).
        outside: ``"raise"`` if a frame day lies outside the news series'
            range, or ``"nan"`` to leave such days NaN (the live series).

    Returns:
        ``(frame, news_source)``: the merged frame and the provenance block
        for the manifest, including the masked days.

    Raises:
        ValueError: If a gap is not a GDELT-missing day, if the frame is not
            time-ordered, or (``outside="raise"``) if a frame day lies
            outside the news coverage.
    """
    domain = "news_sentiment"
    columns = list(_DOMAIN_FEATURES[domain])
    prefixed = [f"{domain}__{c}" for c in columns]
    if not pd.to_datetime(frame["timestamp"]).is_monotonic_increasing:
        raise ValueError("attach_news needs a time-ordered frame.")
    block = news[["timestamp", *columns]].rename(columns=dict(zip(columns, prefixed)))
    out = frame.merge(block, on="timestamp", how="left")

    ts = pd.to_datetime(out["timestamp"])
    inside = (ts >= news["timestamp"].min()) & (ts <= news["timestamp"].max())
    if outside == "raise" and (~inside).any():
        uncovered = ts[~inside]
        raise ValueError(
            f"{len(uncovered)} frame days fall outside the news series' "
            f"coverage ({uncovered.min().date()} .. {uncovered.max().date()}); "
            "extend the GDELT backfill rather than dropping rows."
        )
    source = dict(news.attrs.get("news_source") or {})
    holes = [str(t.date()) for t in ts[inside & out[prefixed[0]].isna()]]
    unexplained = sorted(set(holes) - set(source.get("missing_days") or []))
    if unexplained:
        raise ValueError(
            f"news has no row for {len(unexplained)} days not listed as "
            f"GDELT-missing: {unexplained[:5]}; rebuild the news series."
        )
    source.update(gap_policy=_NEWS_GAP_POLICY, days_masked=holes)
    return out, source


def build_region(region: str, drop_incomplete: bool = True) -> tuple[pd.DataFrame, dict]:
    """Build one region's merged evaluation frame and its manifest.

    Args:
        drop_incomplete: Drop days on which any domain is missing, so every
            method sees the same complete rows (the evaluation's setting).
            ``False`` keeps every shipping day and leaves a domain NaN outside
            its source's coverage, for a live series that runs up to the
            latest shipping day (``scripts/build_live_risk_series.py``).
    """
    config = load_config_for_region(region)
    frames: dict[str, pd.DataFrame] = {}
    timings: dict[str, float] = {}

    for domain in _DOMAIN_FEATURES:
        started = time.monotonic()
        print(f"    {domain:<18}", end="", flush=True)
        df = _fetch_domain(domain, config)
        timings[domain] = round(time.monotonic() - started, 1)
        if df is None or df.empty:
            print(f"unavailable ({timings[domain]}s)")
            continue
        frames[domain] = df
        print(f"{len(df):>6} rows ({timings[domain]}s)")

    if "shipping" not in frames:
        raise RuntimeError(
            f"{region}: shipping is unavailable, and it carries the ground-"
            "truth label — there is nothing to evaluate against."
        )

    shipping = frames["shipping"].sort_values("timestamp").reset_index(drop=True)
    merged = shipping[["timestamp", "is_disruption"]].copy()
    # y_true is the level-shift label; the connector's 2-sigma flag is kept
    # alongside it as y_true_shock so the two are comparable rather than one
    # silently replacing the other.
    merged = merged.rename(columns={"is_disruption": "y_true_shock"})
    merged["y_true"] = level_shift_label(shipping["vessel_count"]).to_numpy()

    present: dict[str, list[str]] = {}
    for domain, df in frames.items():
        if domain == "news_sentiment":
            continue                      # attached below, with its own gap rule
        columns = [c for c in _DOMAIN_FEATURES[domain] if c in df.columns]
        if not columns:
            continue
        block = df[["timestamp", *columns]].copy()
        # Prefix so two domains cannot collide, and so an ablation can select
        # a tier's features by prefix alone.
        block = block.rename(columns={c: f"{domain}__{c}" for c in columns})
        merged = merged.merge(block, on="timestamp", how="left")
        present[domain] = columns

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    # Forward-fill only inside the observed range: extrapolating past an
    # API's coverage would invent values at the edges, and interpolating
    # would read the next observation, which a forecaster does not have yet.
    feature_columns = [c for c in merged.columns if "__" in c]
    holes = merged[feature_columns].isna()
    merged[feature_columns] = merged[feature_columns].ffill(limit_area="inside")
    filled = (holes & merged[feature_columns].notna()).sum()
    gap_fill = {"policy": _GAP_FILL_POLICY,
                "cells_filled": {c: int(n) for c, n in filled.items() if n}}
    news_source = None
    if "news_sentiment" in frames:
        # Same function as --refresh-news, so the two paths cannot diverge.
        merged, news_source = attach_news(merged, frames["news_sentiment"],
                                          outside="nan")
        present["news_sentiment"] = list(_DOMAIN_FEATURES["news_sentiment"])
        feature_columns = [c for c in merged.columns if "__" in c]
    before = len(merged)
    news_columns = [c for c in feature_columns if c.startswith("news_sentiment__")]
    masked = (pd.to_datetime(merged["timestamp"]).dt.strftime("%Y-%m-%d")
              .isin((news_source or {}).get("days_masked") or []))
    if drop_incomplete:
        # A masked news day is kept (its other domains are real data); any
        # other missing value drops the day.
        other = [c for c in feature_columns if c not in news_columns]
        incomplete = merged[other].isna().any(axis=1)
        if news_columns:
            incomplete |= merged[news_columns].isna().any(axis=1) & ~masked
        merged = merged[~incomplete]
        masked = masked[~incomplete]

    manifest = {
        "region": region,
        "rows": int(len(merged)),
        "rows_dropped_incomplete": int(before - len(merged)),
        "rows_news_masked": int(masked.sum()),
        "gap_fill": gap_fill,
        "date_range": [
            str(merged["timestamp"].min().date()),
            str(merged["timestamp"].max().date()),
        ] if not merged.empty else [],
        "domains_present": present,
        "domains_missing": sorted(set(_DOMAIN_FEATURES) - set(present)),
        "feature_columns": feature_columns,
        "positive_days": int(merged["y_true"].sum()) if not merged.empty else 0,
        "positive_rate": round(float(merged["y_true"].mean()), 4)
        if not merged.empty else 0.0,
        "fetch_seconds": timings,
        "label_definition": (
            f"y_true = level shift: 30-day mean of daily transits at least "
            f"{_LS_DROP:.0%} below the trailing {_LS_BASELINE}-day median, "
            f"sustained {_LS_MIN_RUN}+ days."
        ),
        "label_validation": (
            "Independently recovers the Houthi Red Sea crisis (bab_el_mandeb, "
            "from 2024-01) and the Gatun Lake drought (panama, 2023-11 to "
            "2024-04) without being given either."
        ),
        "label_secondary": (
            "y_true_shock = the connector's 2-sigma rule, retained for "
            "comparison. Unusable alone: 0 positives for hormuz outside the "
            "pinned 2026 shutdown, 0 for malacca ever, 3 for bab_el_mandeb, "
            "5 for panama."
        ),
        "positive_days_shock": int(merged["y_true_shock"].sum())
        if not merged.empty else 0,
    }
    if news_source is not None:
        manifest["news_source"] = news_source
    if "market" in frames and frames["market"].attrs.get("market_source"):
        manifest["market_source"] = frames["market"].attrs["market_source"]
    # ACLED events fetched in full (cursor pagination): row counts per
    # country-year and a content hash, so a one-page cap is visible.
    if "geopolitical" in frames and frames["geopolitical"].attrs.get("geopolitical_source"):
        manifest["geopolitical_source"] = frames["geopolitical"].attrs["geopolitical_source"]
    return merged, manifest


def replace_news(frame: pd.DataFrame, manifest: dict,
                 news: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Swap a cached frame's news columns for a freshly loaded news series.

    Every other column and every row stays as cached, so the other domains'
    values do not move. News is attached by :func:`attach_news`, the same
    function :func:`build_region` uses, so the two paths give the same values.

    Args:
        frame: A cached evaluation frame (``timestamp`` plus prefixed features).
        manifest: Its manifest; a copy is updated and returned.
        news: The news connector's frame (``gdelt_events`` mode).

    Returns:
        ``(frame, manifest)`` with the news columns replaced.

    Raises:
        ValueError: If any frame day falls outside the news series' coverage
            (dropping such rows would change the evaluation's row set for
            every other method), or as :func:`attach_news`.
    """
    domain = "news_sentiment"
    out = frame.drop(columns=[c for c in frame.columns
                              if c.startswith(f"{domain}__")])
    out, source = attach_news(out, news, outside="raise")

    m = dict(manifest)
    m["domains_present"] = {**m.get("domains_present", {}),
                            domain: list(_DOMAIN_FEATURES[domain])}
    m["domains_missing"] = [d for d in m.get("domains_missing", []) if d != domain]
    m["feature_columns"] = [c for c in out.columns if "__" in c]
    m["news_source"] = source
    m["rows"] = int(len(out))
    m["rows_news_masked"] = len(source["days_masked"])
    return out, m


def _replace_file(tmp: Path, target: Path, tries: int = 10) -> None:
    """``os.replace`` with retries: OneDrive briefly locks files it syncs."""
    for attempt in range(tries):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def refresh_news(region: str, news_dir: str | Path | None = None) -> dict:
    """Replace the news columns of ``region``'s cached frame on disk.

    Errors propagate (a sha mismatch, failed backfill days, a gap that is not
    GDELT-missing) rather than being reduced to "unavailable". Each file is
    written to a temporary sibling and swapped in.
    """
    target = _CACHE_DIR / f"{region}.csv"
    manifest_path = _CACHE_DIR / f"{region}.manifest.json"
    # round_trip so rewriting the file cannot nudge other domains' floats.
    frame = pd.read_csv(target, parse_dates=["timestamp"],
                        float_precision="round_trip")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    news = _load_news(load_config_for_region(region), news_dir)
    frame, manifest = replace_news(frame, manifest, news)

    tmp_csv = target.with_name(target.name + ".tmp")
    frame.to_csv(tmp_csv, index=False)
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _replace_file(tmp_csv, target)
    _replace_file(tmp_manifest, manifest_path)
    return manifest


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", action="append", dest="regions")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--relabel", action="store_true",
                      help="Recompute y_true on cached frames; no API calls.")
    mode.add_argument("--force", action="store_true",
                      help="Rebuild even if a cached frame exists.")
    mode.add_argument("--refresh-news", action="store_true",
                      help="Replace only the news columns of cached frames "
                           "with the GDELT 1.0 series; no other API calls.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    regions = args.regions or list_regions()

    if args.refresh_news:
        failed = 0
        for region in regions:
            try:
                m = refresh_news(region)
            except Exception as exc:
                failed += 1
                print(f"  {region}: FAILED: {exc}")
                continue
            src = m["news_source"]
            print(f"  {region}: news replaced -> {m['rows']} rows, "
                  f"{len(m['feature_columns'])} features, filter "
                  f"{src.get('filter_version')} ({str(src.get('csv_sha256'))[:12]}), "
                  f"{len(src['days_masked'])} days masked, "
                  f"missing={m['domains_missing']}")
        return 1 if failed else 0

    manifests = []
    for region in regions:
        target = _CACHE_DIR / f"{region}.csv"
        if target.exists() and args.relabel:
            # Recompute the label from the cached frame: it already carries
            # shipping__vessel_count, so this costs no API calls.
            cached = pd.read_csv(target, parse_dates=["timestamp"])
            cached["y_true"] = level_shift_label(
                cached["shipping__vessel_count"]
            ).to_numpy()
            cached.to_csv(target, index=False)
            manifest_path = _CACHE_DIR / f"{region}.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["positive_days"] = int(cached["y_true"].sum())
            manifest["positive_rate"] = round(float(cached["y_true"].mean()), 4)
            manifest["label_definition"] = (
                f"y_true = level shift: 30-day mean at least {_LS_DROP:.0%} "
                f"below the trailing {_LS_BASELINE}-day median, sustained "
                f"{_LS_MIN_RUN}+ days."
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            manifests.append(manifest)
            print(f"  {region}: relabelled -> {manifest['positive_days']} positives "
                  f"({manifest['positive_rate']:.1%})")
            continue
        if target.exists() and not args.force:
            print(f"  {region}: cached ({target.name}) — use --force to rebuild")
            manifests.append(json.loads(
                (_CACHE_DIR / f"{region}.manifest.json").read_text(encoding="utf-8")
            ))
            continue

        print(f"\n[{region}]")
        try:
            merged, manifest = build_region(region)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        merged.to_csv(target, index=False)
        (_CACHE_DIR / f"{region}.manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        manifests.append(manifest)
        print(f"  -> {manifest['rows']} rows, {len(manifest['feature_columns'])} "
              f"features, {manifest['positive_days']} positive days "
              f"({manifest['positive_rate']:.1%})")
        if manifest["domains_missing"]:
            print(f"     missing domains: {manifest['domains_missing']}")

    print("\n" + "=" * 70)
    for m in manifests:
        print(f"  {m['region']:<14} {m['rows']:>5} rows  "
              f"{len(m['feature_columns']):>2} feats  "
              f"{m['positive_days']:>4} pos ({m['positive_rate']:.1%})  "
              f"missing={m['domains_missing']}")
    return 0 if manifests else 1


if __name__ == "__main__":
    sys.exit(run())
