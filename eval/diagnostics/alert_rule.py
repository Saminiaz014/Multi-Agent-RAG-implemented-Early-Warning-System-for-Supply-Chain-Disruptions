"""Alert-rule diagnosis: ``score > threshold`` vs ``score >= threshold``.

For every method and tier in every region, on the main split, this reports
the alert threshold (90th percentile of the method's training scores, masked
days left out, as in the harness), the share of training and of scored test
scores *exactly* equal to it, and alert rate and F1 under ``>`` and ``>=``.
Ties matter for constant or zero-floored scores: a control that is constant
1 never alerts under ``>``, and a score floored at 0 on most training days
gets a threshold of 0.

It also flags methods whose alert rate rises by more than 20 points under
``>=``, or whose threshold equals their minimum training score, and prints
B6 CUSUM's training vs test score distribution.

Scores and thresholds come from ``scripts/run_method_comparison.py``'s own
functions, so this is the harness's view, not a re-implementation. The three
controls (M1, M2, ORACLE) no longer use the threshold rule in the harness
(``_DIRECT_ALERTS``); their rows here show what the rule would do to them.

Usage::

    python eval/diagnostics/alert_rule.py            # all regions

Writes ``data/eval/diagnostics/alert_rule.csv``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "run_method_comparison", ROOT / "scripts" / "run_method_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)

from src.core.config_manager import load_config_for_region  # noqa: E402

OUT = ROOT / "data" / "eval" / "diagnostics"
#: The harness's regions: one per cached frame.
REGIONS = tuple(p.stem for p in sorted((ROOT / "data" / "eval").glob("*.csv")))


def _scores(region: str):
    """(name, train scores used for the threshold, test scores, test keep, y) per method."""
    frame, manifest = rmc._load(region)
    config = load_config_for_region(region)
    masked = rmc._masked_days(frame, manifest)
    features = [c for c in frame.columns if "__" in c]
    train, test = rmc._split(frame)
    keep_of = pd.Series(~masked, index=frame.index)
    train_keep = keep_of.loc[train.index].to_numpy()
    keep = keep_of.loc[test.index].to_numpy()
    y = test["y_true"].astype(int).to_numpy()
    out = []
    for name, (fn, _kind) in rmc.BASELINES.items():
        row_wise = name in rmc._ROW_WISE
        fit_train = train[train_keep] if row_wise else train
        try:
            s = np.asarray(fn(fit_train, test, features), dtype=float)
        except rmc._NotTrainable:              # no positives: reported n/a by the harness
            continue
        except Exception as exc:               # the harness prints FAILED and skips it
            print(f"  [{region}] {name} FAILED: {str(exc)[:60]}")
            continue
        ref = s[: len(fit_train)] if row_wise else s[: len(train)][train_keep]
        out.append((name, np.nan_to_num(ref), np.nan_to_num(s[-len(test):]), keep, y))
    on_test = rmc._all_agent_scores(config, train, test)
    on_train = rmc._all_agent_scores(config, train, train)
    for tier in range(1, len(rmc._TIER_ORDER) + 1):
        s, _ = rmc._compose(tier, on_test, config)
        t, _ = rmc._compose(tier, on_train, config)
        if s is None:
            continue
        # The harness falls back to the test scores (and keep) without train scores.
        ref = t[train_keep] if t is not None else s[keep]
        out.append((f"Tier {tier}", np.nan_to_num(ref), np.nan_to_num(s), keep, y))
    return out, train, test


def _rule(y, s, thr, op):
    predicted = (s >= thr) if op == ">=" else (s > thr)
    f1 = (float(f1_score(y, predicted.astype(int), zero_division=0))
          if len(np.unique(y)) > 1 else np.nan)
    return float(predicted.mean()), f1


def main() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for region in REGIONS:
        methods, train, test = _scores(region)
        for name, ref, s, keep, y in methods:
            thr = rmc._threshold(s, ref)        # the harness's own threshold
            sk, yk = s[keep], y[keep]
            rate_gt, f1_gt = _rule(yk, sk, thr, ">")
            rate_ge, f1_ge = _rule(yk, sk, thr, ">=")
            rows.append({
                "region": region, "method": name, "threshold": thr,
                "train_min": float(ref.min()),
                "share_train_eq": float(np.mean(ref == thr)),
                "share_test_eq": float(np.mean(sk == thr)),
                "alert_gt": rate_gt, "alert_ge": rate_ge,
                "f1_gt": f1_gt, "f1_ge": f1_ge,
                "flag_rise_20pt": (rate_ge - rate_gt) > 0.20,
                "flag_thr_is_train_min": thr == float(ref.min()),
            })
            if name == "B6 CUSUM":
                q = [0, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
                print(f"\n[{region}] B6 CUSUM  threshold {thr:.3f}")
                print("  train quantiles:", {f"p{int(p*100)}": round(float(np.quantile(ref, p)), 3) for p in q})
                print("  test  quantiles:", {f"p{int(p*100)}": round(float(np.quantile(sk, p)), 3) for p in q})
                print(f"  max test {sk.max():.3f} vs threshold {thr:.3f}; "
                      f"train days > 0: {np.mean(ref > 0):.1%}; test days > 0: {np.mean(sk > 0):.1%}")
                tr_ts = pd.to_datetime(train["timestamp"]).to_numpy()
                peak = int(np.argmax(ref))
                print(f"  train peak {ref.max():.1f} on {pd.Timestamp(tr_ts[peak]).date()} "
                      f"(the run is a running sum over train + test, reset only at 0)")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "alert_rule.csv", index=False)
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t = main()
    print()
    print(t.round(3).to_string(index=False))
