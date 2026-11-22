"""Risk bands from the composite on calm training days (issue 5).

The bands in settings.yaml ``thresholds`` cut the engine's composite risk
score into words (low / medium / high, and the dashboard's Critical). They are
quantiles of that composite on *calm* (label-negative) days, pooled over the
four regions::

    risk_medium   = p60     risk_high = p85     risk_critical = p97

The values set in f7b87ff (0.90 / 0.69 / 0.51) were taken over every
label-negative day 2019-2026, test window included. Its commit message
describes the method; the composite path, fitting window and scoring history
were never recorded, and no script was kept. This is a reconstruction of that
method, on training days only:

1. each region's evaluation frame (``data/eval/<region>.csv``);
2. each enabled agent fitted on the training days (before
   ``evaluation.split_date``) and scored on them with a training-only
   history, through ``scripts/run_method_comparison.py``'s ``_agent_scores``,
   as the harness's training thresholds are;
3. per day, :meth:`RiskEngine.compute_risk` over those agents, the composite
   the bands cut (the one ``build_live_risk_series.py`` writes for the
   dashboard);
4. label-negative training days, masked (GDELT-missing) days left out,
   pooled over the regions;
5. p60 / p85 / p97, rounded to two decimals.

``window="all"`` repeats it over every label-negative day up to
``evaluation.end_date``, the old, test-inclusive window, for comparison only.
Test days are scored as the harness scores them (fitted on training days,
history train + test). Scoring every day on one full-window history instead
would cut MarketAgent's input to the five years before the last day and zero
the market score on the earliest training days.

The output also records where the configured bands sit on the training
calm-day distribution, and, as a held-out check, which band the calm test
days fall in under each band set (test days read for evaluation only).

Usage::

    python scripts/calibrate_risk_bands.py

Writes ``data/eval/diagnostics/risk_bands.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import run_method_comparison as harness  # noqa: E402

from src.agents.base_agent import DetectionResult  # noqa: E402
from src.aggregation.risk_engine import RiskEngine  # noqa: E402
from src.core.config_manager import load_base_config, load_config_for_region  # noqa: E402
from src.evaluation.window import eval_window  # noqa: E402

_OUT = _PROJECT_ROOT / "data" / "eval" / "diagnostics" / "risk_bands.json"

#: Band -> quantile of the calm-day composite.
QUANTILES: tuple[tuple[str, float], ...] = (
    ("risk_medium", 0.60), ("risk_high", 0.85), ("risk_critical", 0.97),
)
WINDOWS: tuple[str, ...] = ("train", "all")


def calm_mask(frame: pd.DataFrame, masked: np.ndarray, window: str) -> np.ndarray:
    """True on the label-negative, unmasked days of ``window``.

    ``"train"``: before ``evaluation.split_date``. ``"all"``: up to and
    including ``evaluation.end_date`` (the old, test-inclusive window).
    """
    w = eval_window()
    ts = pd.to_datetime(frame["timestamp"])
    within = ts < w["split_date"] if window == "train" else ts <= w["end_date"]
    calm = within & (frame["y_true"].astype(int) == 0)
    return calm.to_numpy() & ~np.asarray(masked, dtype=bool)


def composite(scores: dict[str, np.ndarray | None], config: dict) -> np.ndarray:
    """Per-day :meth:`RiskEngine.compute_risk` over the agents with scores."""
    engine = RiskEngine(config)
    present = {a: np.asarray(s, dtype=float) for a, s in scores.items() if s is not None}
    n = len(next(iter(present.values())))
    out = np.empty(n)
    for i in range(n):
        outputs = [DetectionResult(agent_name=a, anomaly_scores=np.array([s[i]]),
                                   anomaly_flags=np.array([False]), feature_names=[])
                   for a, s in present.items()]
        out[i] = float(engine.compute_risk(outputs)["risk_score"])
    return out


def bands(values: np.ndarray) -> dict[str, float]:
    """The band cuts: calm-day quantiles, rounded to two decimals."""
    return {name: round(float(np.quantile(values, q)), 2) for name, q in QUANTILES}


def tier_shares(values: np.ndarray, cuts: dict[str, float]) -> dict[str, float]:
    """Share of days per band: Low / Medium / High / Critical."""
    v = np.asarray(values, dtype=float)
    edges = [cuts["risk_medium"], cuts["risk_high"], cuts["risk_critical"]]
    counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=4)
    return {t: float(c / len(v)) for t, c in zip(("Low", "Medium", "High", "Critical"), counts)}


def region_composite(region: str) -> pd.DataFrame:
    """One region's per-day composite, scored as the harness scores it.

    Training days: agents fitted on them and scored on a training-only
    history. Test days: fitted on the training days, history train + test.
    """
    frame, manifest = harness._load(region)
    config = load_config_for_region(region)
    masked = pd.Series(harness._masked_days(frame, manifest), index=frame.index)
    train, test = harness._split(frame)
    parts = []
    for days in (train, test):
        scores = {d: harness._agent_scores(d, config, train, days) for d in harness._TIER_ORDER}
        parts.append(pd.DataFrame({
            "timestamp": pd.to_datetime(days["timestamp"]).to_numpy(),
            "y_true": days["y_true"].to_numpy(),
            "masked": masked.loc[days.index].to_numpy(),
            "composite": composite(scores, config),
        }))
    return pd.concat(parts, ignore_index=True)


def calm_values(days: pd.DataFrame, window: str) -> np.ndarray:
    """The composite on ``window``'s calm days."""
    return days["composite"].to_numpy()[calm_mask(days, days["masked"].to_numpy(), window)]


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    regions = [p.stem for p in sorted(harness._CACHE.glob("*.csv"))]
    configured = {k: float(load_base_config()["thresholds"][k]) for k, _ in QUANTILES}
    per_region = {r: region_composite(r) for r in regions}
    result: dict = {"quantiles": dict(QUANTILES), "configured": configured,
                    "eval_window": {k: str(v.date()) for k, v in eval_window().items()
                                    if hasattr(v, "date")}}
    for window in WINDOWS:
        calm = {r: calm_values(d, window) for r, d in per_region.items()}
        pooled = np.concatenate(list(calm.values()))
        result[window] = {
            "bands": bands(pooled),
            "unrounded": {n: float(np.quantile(pooled, q)) for n, q in QUANTILES},
            "calm_days": {r: int(len(v)) for r, v in calm.items()},
            "calm_days_total": int(len(pooled)),
            "median": float(np.median(pooled)),
            # Where the configured bands sit on this window's calm days.
            "configured_percentile": {n: float((pooled < c).mean())
                                      for n, c in configured.items()},
        }
        print(f"{window:>5}: {result[window]['bands']} over {len(pooled)} calm days; "
              f"configured bands sit at {result[window]['configured_percentile']}")
    # Held-out check: calm *test* days, read for evaluation only.
    w = eval_window()
    held_out = np.concatenate([
        d["composite"].to_numpy()[(d["timestamp"] >= w["split_date"]).to_numpy()
                                  & calm_mask(d, d["masked"].to_numpy(), "all")]
        for d in per_region.values()])
    result["held_out_calm_test_days"] = {
        "n": int(len(held_out)),
        "shares_under_train_bands": tier_shares(held_out, result["train"]["bands"]),
        "shares_under_configured": tier_shares(held_out, configured),
    }
    print("held-out calm test days:", result["held_out_calm_test_days"])
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
