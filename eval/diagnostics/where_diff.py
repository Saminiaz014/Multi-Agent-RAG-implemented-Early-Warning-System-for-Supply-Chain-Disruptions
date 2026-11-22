"""Where split_invariance.py's restart-only differences fall, and why.

Reads ``data/eval/diagnostics/inv_scores_<region>.csv`` (run
``split_invariance.py`` first) and lists, per agent, the days on which the
restart-only configuration C differs from A, with their date range.

It also shows the mechanism found on 2026-11-22: the shipping agent's
duration score (``ShippingAgent.level_shift_score``, a 30-day mean against a
365-day median lagged 30 days, ``min_periods`` 91) computed on the series
*starting at the split*, which is what the harness fed it before issue 11 was
fixed. Started at 2024-01-01 it stays at 0 for months, while the label (built
on the full series) marks the Red Sea crisis from January 2024.

Usage::

    python eval/diagnostics/where_diff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.shipping_agent import ShippingAgent  # noqa: E402
from src.evaluation.window import eval_window  # noqa: E402

OUT = ROOT / "data" / "eval" / "diagnostics"
REGIONS = ("bab_el_mandeb", "hormuz", "panama", "malacca")


def main() -> None:
    end = str(eval_window()["end_date"].date())
    for region in REGIONS:
        path = OUT / f"inv_scores_{region}.csv"
        if not path.exists():
            print(f"[{region}] no {path.name}; run split_invariance.py {region}")
            continue
        d = pd.read_csv(path, index_col=0, parse_dates=True).loc["2024-01-01":end]
        print(f"\n[{region}] days where C (scored from 2024-01) != A (scored from 2023-07):")
        for agent in ("shipping", "market", "geopolitical", "natural_disaster",
                      "news_sentiment"):
            col_a, col_c = f"A|agent {agent}", f"C|agent {agent}"
            if col_a not in d or col_c not in d:
                continue
            days = d.index[(d[col_a] - d[col_c]).abs() > 1e-12]
            span = f", {days.min().date()} .. {days.max().date()}" if len(days) else ""
            print(f"  {agent:<17} {len(days):>3} days{span}")

        frame = pd.read_csv(ROOT / "data" / "eval" / f"{region}.csv",
                            parse_dates=["timestamp"]).set_index("timestamp")
        vc = frame["shipping__vessel_count"]
        print("  duration score computed on the series from the split alone "
              "(the pre-fix harness input):")
        for start in ("2023-07-01", "2024-01-01"):
            part = vc.loc[start:end]
            dur = ShippingAgent.level_shift_score(part.reset_index(drop=True))
            dur.index = part.index
            on = dur[dur > 0]
            h1 = dur.loc["2024-01-01":"2024-06-30"]
            first = on.index.min().date() if len(on) else None
            print(f"    from {start}: first > 0 on {first}; mean Jan-Jun 2024 "
                  f"{h1.mean():.3f}; days > 0 {int((h1 > 0).sum())} of {len(h1)}")
        y = frame["y_true"].loc["2024-01-01":"2024-06-30"]
        print(f"  label positives Jan-Jun 2024: {int(y.sum())} of {len(y)}")


if __name__ == "__main__":
    main()
