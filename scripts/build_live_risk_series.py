"""Score every day from 2019 to the latest available day for the dashboard.

The Decision View plots real composite risk on a calendar axis. This builds
that series per region:

1. Fetch the live connectors exactly as ``build_eval_dataset.py`` does, but
   keep every shipping day (``drop_incomplete=False``): a domain whose source
   ends earlier is NaN after its last day rather than cutting the whole frame.
2. Fit each active agent on the days before the evaluation's split date
   (settings.yaml ``evaluation.split_date``) and score every day with it.
3. Aggregate each day with :class:`RiskEngine` over the agents that have data
   that day, so weights renormalise exactly as when an agent is unavailable
   in a live run.

Kept apart from ``data/eval/`` on purpose: the thesis results are computed on
that frame, and rebuilding it would change them. Output goes to
``data/live/`` (gitignored, like every derived artefact):

* ``<region>.csv`` / ``<region>.manifest.json`` — the fetched feature frame;
* ``<region>_risk.csv`` — per-day agent scores, ``risk_score``,
  ``risk_level`` and ``top_agent``.

Usage::

    python scripts/build_live_risk_series.py                 # fetch + score all regions
    python scripts/build_live_risk_series.py --region panama
    python scripts/build_live_risk_series.py --score-only    # rescore the cached frames
    python scripts/build_live_risk_series.py --shipping-from-cache
        # refetch every domain except shipping, which is taken from the
        # previous data/live frame (IMF PortWatch rate-limits: HTTP 429)

``--shipping-from-cache`` records in each manifest (``shipping_source``) which
cached file the shipping columns came from, its hash and last day, and the
same record of the cache's own origin if it was itself cached. The shipping
series then ends where the cache ends.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import build_eval_dataset  # noqa: E402
import run_method_comparison as harness  # noqa: E402

from src.agents.base_agent import DetectionResult  # noqa: E402
from src.aggregation.risk_engine import RiskEngine  # noqa: E402
from src.core.config_manager import load_config_for_region  # noqa: E402
from src.core.regions import get_region, list_regions  # noqa: E402
from src.evaluation.window import eval_window  # noqa: E402

LIVE_DIR = _PROJECT_ROOT / "data" / "live"


def _write_atomically(path: Path, write) -> None:
    """Write beside ``path`` and move it into place, so a failed write leaves ``path`` as it was.

    The live frame is also the cache the next ``--shipping-from-cache`` run reads.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        build_eval_dataset._replace_file(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def cached_shipping(region: str) -> tuple[pd.DataFrame, dict]:
    """The shipping frame of the previous live frame, and a record of where it came from.

    Raises:
        FileNotFoundError: If there is no cached frame to take it from.
    """
    path = LIVE_DIR / f"{region}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: no cached live frame to take shipping from. Run once without "
            "--shipping-from-cache (it fetches PortWatch).")
    # round_trip: the default parser can differ from the written float by one ulp
    cached = pd.read_csv(path, parse_dates=["timestamp"], float_precision="round_trip")
    shipping = build_eval_dataset.shipping_from_frame(cached)
    source = {
        "source": "shipping columns of an earlier data/live frame; IMF PortWatch was not re-fetched",
        "cache": f"data/live/{region}.csv",
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cache_rows": int(len(shipping)),
        "cache_last_day": str(shipping["timestamp"].max())[:10],
        "cache_written": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        .isoformat(timespec="seconds"),
    }
    earlier = LIVE_DIR / f"{region}.manifest.json"
    if earlier.exists():
        inherited = json.loads(earlier.read_text(encoding="utf-8")).get("shipping_source")
        if inherited:
            source["inherited"] = inherited
    return shipping, source


def fetch(region: str, shipping_from_cache: bool = False) -> pd.DataFrame:
    """Fetch and cache one region's frame, up to the latest shipping day.

    Args:
        shipping_from_cache: Take the shipping columns from the previous
            ``data/live`` frame instead of calling PortWatch; every other
            domain is fetched.
    """
    print(f"\n[{region}] fetching")
    shipping = source = None
    if shipping_from_cache:
        shipping, source = cached_shipping(region)
        print(f"    shipping           cached to {source['cache_last_day']} "
              "(PortWatch not called)")
    frame, manifest = build_eval_dataset.build_region(
        region, drop_incomplete=False, shipping=shipping, shipping_source=source)
    manifest["coverage"] = {
        domain: _coverage(frame, domain) for domain in manifest["domains_present"]
    }
    _write_atomically(LIVE_DIR / f"{region}.csv", lambda p: frame.to_csv(p, index=False))
    _write_atomically(LIVE_DIR / f"{region}.manifest.json",
                      lambda p: p.write_text(json.dumps(manifest, indent=2), encoding="utf-8"))
    return frame


def _coverage(frame: pd.DataFrame, domain: str) -> list[str]:
    """First and last day on which ``domain`` has complete features."""
    cols = [c for c in frame.columns if c.startswith(f"{domain}__")]
    days = frame.loc[frame[cols].notna().all(axis=1), "timestamp"]
    return [str(pd.Timestamp(days.min()).date()), str(pd.Timestamp(days.max()).date())] \
        if len(days) else []


def score(region: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-day agent scores and composite risk for one region."""
    config = load_config_for_region(region)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    cut = eval_window()["split_date"]

    out = pd.DataFrame({"timestamp": frame["timestamp"], "y_true": frame["y_true"]})
    for domain in get_region(region).active_agents():
        cols = [c for c in frame.columns if c.startswith(f"{domain}__")]
        if not cols:
            continue
        rows = frame.dropna(subset=cols)[["timestamp", *cols]]
        train = rows[rows["timestamp"] < cut]
        if len(train) < 60:
            print(f"  {domain:<18} skipped: {len(train)} training days")
            continue
        scores = harness._agent_scores(domain, config, train, rows)
        if scores is None:
            continue
        series = pd.Series(scores, index=rows["timestamp"].to_numpy())
        out[domain] = series.reindex(out["timestamp"]).to_numpy()
        print(f"  {domain:<18} scored {len(rows)} days "
              f"({rows['timestamp'].min():%Y-%m-%d} .. {rows['timestamp'].max():%Y-%m-%d})")

    agents = [c for c in out.columns if c not in ("timestamp", "y_true")]
    engine = RiskEngine(config)
    risk, level, top = [], [], []
    for _, row in out[agents].iterrows():
        present = [a for a in agents if not np.isnan(row[a])]
        if not present:
            risk.append(np.nan), level.append(""), top.append("")
            continue
        result = engine.compute_risk([
            DetectionResult(agent_name=a, anomaly_scores=np.array([float(row[a])]),
                            anomaly_flags=np.array([False]), feature_names=[])
            for a in present
        ])
        contrib = {a: v["contribution"] for a, v in result["contributing_agents"].items()}
        risk.append(float(result["risk_score"]))
        level.append(str(getattr(result["risk_level"], "value", result["risk_level"])))
        top.append(max(contrib, key=contrib.get) if contrib else "")
    out["risk_score"], out["risk_level"], out["top_agent"] = risk, level, top
    out.to_csv(LIVE_DIR / f"{region}_risk.csv", index=False)
    last = out.dropna(subset=["risk_score"]).iloc[-1]
    print(f"  -> {len(out)} days to {last['timestamp']:%Y-%m-%d}; "
          f"latest risk {last['risk_score']:.2f} ({last['risk_level']})")
    return out


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", action="append", dest="regions")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score-only", action="store_true",
                      help="Rescore the cached data/live frames; no API calls.")
    mode.add_argument("--shipping-from-cache", action="store_true",
                      help="Fetch every domain except shipping, which is taken from the "
                           "previous data/live frame (PortWatch is not called).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    for region in args.regions or list_regions():
        cached = LIVE_DIR / f"{region}.csv"
        try:
            if args.score_only and cached.exists():
                frame = pd.read_csv(cached, parse_dates=["timestamp"])
                print(f"\n[{region}] cached frame")
            else:
                frame = fetch(region, shipping_from_cache=args.shipping_from_cache)
            score(region, frame)
        except Exception as exc:
            print(f"  FAILED: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
