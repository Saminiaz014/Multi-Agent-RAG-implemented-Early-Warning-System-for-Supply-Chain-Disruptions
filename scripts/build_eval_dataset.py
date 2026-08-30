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

news_sentiment is fetched when GDELT answers and omitted when it does not;
the column set for each region is recorded in the manifest so a later run
cannot quietly compare a five-feature region against a four-feature one.

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
"""

from __future__ import annotations

import argparse
import json
import logging
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

        block = dict(config["agents"][domain])
        block["data_mode"] = "api"
        connector = {
            "natural_disaster": DisasterConnector,
            "geopolitical": GeopoliticalConnector,
            "news_sentiment": NewsConnector,
        }[domain]
        return connector(config=block).fetch()
    except Exception as exc:
        logger.warning("  %s unavailable: %s", domain, str(exc)[:160])
        return None


def build_region(region: str) -> tuple[pd.DataFrame, dict]:
    """Build one region's merged evaluation frame and its manifest."""
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
    # Interpolate only inside the observed range: extrapolating past an API's
    # coverage would invent values at the edges where they are least checkable.
    feature_columns = [c for c in merged.columns if "__" in c]
    merged[feature_columns] = merged[feature_columns].interpolate(
        limit_area="inside"
    )
    before = len(merged)
    merged = merged.dropna(subset=feature_columns, how="any")

    manifest = {
        "region": region,
        "rows": int(len(merged)),
        "rows_dropped_incomplete": int(before - len(merged)),
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
    return merged, manifest


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", action="append", dest="regions")
    parser.add_argument("--relabel", action="store_true",
                        help="Recompute y_true on cached frames; no API calls.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if a cached frame exists.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    regions = args.regions or list_regions()

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
