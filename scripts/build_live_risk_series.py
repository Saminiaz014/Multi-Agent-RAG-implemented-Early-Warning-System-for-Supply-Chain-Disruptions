"""Score every day from 2019 to the latest available day for the dashboard.

The Decision View plots real composite risk on a calendar axis. This builds
that series per region:

1. Fetch the live connectors exactly as ``build_eval_dataset.py`` does, but
   keep every shipping day (``drop_incomplete=False``): a domain whose source
   ends earlier is NaN after its last day rather than cutting the whole frame.
2. Fit each active agent on the first 70 % of days (the evaluation's temporal
   split) and score every day with it.
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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

LIVE_DIR = _PROJECT_ROOT / "data" / "live"
TRAIN_FRACTION = 0.70


def fetch(region: str) -> pd.DataFrame:
    """Fetch and cache one region's frame, up to the latest shipping day."""
    print(f"\n[{region}] fetching")
    frame, manifest = build_eval_dataset.build_region(region, drop_incomplete=False)
    manifest["coverage"] = {
        domain: _coverage(frame, domain) for domain in manifest["domains_present"]
    }
    frame.to_csv(LIVE_DIR / f"{region}.csv", index=False)
    (LIVE_DIR / f"{region}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
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
    cut = frame["timestamp"].iloc[int(len(frame) * TRAIN_FRACTION)]

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
    parser.add_argument("--score-only", action="store_true",
                        help="Rescore the cached data/live frames; no API calls.")
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
                frame = fetch(region)
            score(region, frame)
        except Exception as exc:
            print(f"  FAILED: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
