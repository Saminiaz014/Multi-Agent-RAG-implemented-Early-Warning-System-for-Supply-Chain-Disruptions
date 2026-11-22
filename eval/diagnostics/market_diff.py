"""Why the market agent's scores moved with the split before issue 11's fix.

Reproduces the pre-fix path directly: ``MarketAgent`` fitted on the days
before 2023-07-01, then ``preprocess``/``detect`` run on the test window
*alone*, once starting 2023-07-01 and once starting 2024-01-01. Its score on
a day depends on the trailing 30 rows (rolling mean and standard deviation,
``min_periods=2``), so the two runs differ until both have 30 rows of shared
history: a warm-up over the first 30 days after the later start. Any later
difference is floating-point noise from pandas' rolling sums (seen: at most
~2.5e-12).

Usage::

    python eval/diagnostics/market_diff.py [region]      # default bab_el_mandeb
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "run_method_comparison", ROOT / "scripts" / "run_method_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)

from src.agents.market_agent import MarketAgent  # noqa: E402
from src.core.config_manager import load_config_for_region  # noqa: E402
from src.evaluation.window import eval_window, split_by_date  # noqa: E402


def main(region: str = "bab_el_mandeb") -> None:
    frame, _ = rmc._load(region)
    cfg = dict(load_config_for_region(region)["agents"]["market"])
    train, _ = split_by_date(frame, split_date="2023-07-01")
    runs = {}
    end = eval_window()["end_date"]
    for start in ("2023-07-01", "2024-01-01"):
        test = frame[(frame.timestamp >= start) & (frame.timestamp <= end)]
        agent = MarketAgent(config=cfg)
        agent.fit(rmc._agent_frame(train, "market"))
        runs[start] = agent.detect(
            agent.preprocess(rmc._agent_frame(test, "market"))).set_index("timestamp")
    a, c = runs["2023-07-01"], runs["2024-01-01"]
    idx = c.index.intersection(a.index)
    diff = (a.loc[idx, "anomaly_score"] - c.loc[idx, "anomaly_score"]).abs()
    early = diff[diff.index < "2024-01-31"]
    later = diff[diff.index >= "2024-01-31"]
    print(f"[{region}] market agent, test window scored alone (pre-fix path)")
    print(f"  first 30 days: {int((early > 1e-12).sum())} differ, max {early.max():.4f}, "
          f"mean {early.mean():.4f}")
    print(f"  later: {int((later > 1e-12).sum())} differ, max {later.max():.2e}; "
          f"> 1e-6: {int((later > 1e-6).sum())}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
