"""One complete results table per region: ``eval/results_main.csv`` and ``.md``.

Built from ``eval/method_comparison_results.csv`` alone (nothing is rerun).
Every method of that file appears, per region, in one fixed order: the controls
M0-M2, M3, M8, B1-B8, Tiers 1-5, then the ORACLE on its own as a ceiling row
(it reads yesterday's label, so it bounds AUC and is never ranked).

Marks:

* ``= Tier k``: a tier that copies the tier before it (the agent it would add
  is disabled in that region);
* ``same numbers as Tier k``: a different agent set with identical metrics (an
  enabled agent that adds nothing, e.g. a constant one);
* ``†``: a model that fits on every feature column, geopolitical included
  (B3, B7, B8), so it moves when those columns do;
* ``n/a``: B7 and B8 need positives in the training window.

Malacca has no positives: AUC and F1 are undefined there, every alert is a
false alarm, so its false-alarm rate (FAR) equals its alert rate.

"Best non-ceiling method" is the highest AUC point estimate among the methods
and tiers, never the ORACLE; the three trivial controls are listed but cannot
win. The line names every method whose interval overlaps the winner's, because
overlap means the win is not clear. Other documents (the comparison report,
README section 4.5) point here instead of restating a "best baseline".

Usage::

    python scripts/results_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EVAL = _PROJECT_ROOT / "eval"

#: Models fitted on every feature column (``run_method_comparison._ROW_WISE``;
#: a test keeps the two equal).
READS_ALL_FEATURES: frozenset[str] = frozenset(
    {"B3 isolation forest", "B7 logistic regression", "B8 random forest"})

#: The fixed row order. A method not listed follows these, in file order.
METHOD_ORDER: tuple[str, ...] = (
    "M0 random", "M1 always alert", "M2 never alert", "M3 persistence", "M8 matrix profile",
    "B1 rolling z", "B2 MA crossover", "B3 isolation forest", "B4 EWMA deviation",
    "B5 AR residual", "B6 CUSUM", "B7 logistic regression", "B8 random forest",
    "Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5",
)
#: Columns the output carries; any the input lacks are added as NaN, so a
#: results file without intervals or deltas still builds.
_OPTIONAL = ("agents", "n_agents", "auc_ci_low", "auc_ci_high", "delta_auc", "delta_auc_ci_low",
             "delta_auc_ci_high", "f1_ci_low", "f1_ci_high", "alert_rate_ci_low",
             "alert_rate_ci_high", "alert_rule", "not_applicable", "circularity", "n_test",
             "test_start", "test_end", "bootstrap_block", "bootstrap_n", "bootstrap_ci")
_DASH = "—"


def _row_type(row: pd.Series) -> str:
    """The ORACLE is a ceiling whichever way the file labels it (family or kind)."""
    if row["family"] == "ceiling" or row["kind"] == "oracle":
        return "ceiling"
    if row["family"] == "tier":
        return "tier"
    return "control" if row["kind"] == "control" else "baseline"


def _tier_number(method: str) -> int:
    return int(str(method).split()[-1])


def _same(a, b) -> bool:
    return bool(a == b or (pd.isna(a) and pd.isna(b)))


def build_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per region and method, in the fixed order, ceiling rows last."""
    out = results.copy()
    for column in _OPTIONAL:
        if column not in out:
            out[column] = np.nan
    out["row_type"] = out.apply(_row_type, axis=1)
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    out["_ceiling"] = (out["row_type"] == "ceiling").astype(int)
    out["_rank"] = [rank.get(m, len(rank) + i) for i, m in enumerate(out["method"])]
    out = out.sort_values(["region", "_ceiling", "_rank"], kind="stable").drop(
        columns=["_ceiling", "_rank"])
    positives = out.groupby("region")["test_positives"].transform("max")
    # Without positives every alert is a false alarm: FAR is the alert rate.
    out["far"] = np.where(positives == 0, out["alert_rate"], out["fpr"])
    out["reads_geopolitical"] = out["method"].isin(READS_ALL_FEATURES)
    out["copy_of"] = ""
    out["same_numbers_as"] = ""
    metrics = ("auc", "f1", "fpr", "alert_rate")
    for _, rows in out[out["row_type"] == "tier"].groupby("region"):
        rows = rows.iloc[np.argsort([_tier_number(m) for m in rows["method"]], kind="stable")]
        previous = None
        for idx, row in rows.iterrows():
            if previous is not None:
                before = f"Tier {_tier_number(row['method']) - 1}"
                if isinstance(row["agents"], str) and row["agents"] == previous["agents"]:
                    out.loc[idx, "copy_of"] = before
                elif all(_same(row[m], previous[m]) for m in metrics):
                    out.loc[idx, "same_numbers_as"] = before
            previous = row
    keep = ["region", "method", "row_type", "n_agents", "agents", "copy_of", "same_numbers_as",
            "reads_geopolitical", "circularity", "auc", "auc_ci_low", "auc_ci_high", "delta_auc",
            "delta_auc_ci_low", "delta_auc_ci_high", "f1", "f1_ci_low", "f1_ci_high", "far",
            "alert_rate", "alert_rate_ci_low", "alert_rate_ci_high", "alert_rule",
            "not_applicable", "test_positives", "n_test", "test_start", "test_end"]
    return out[keep].reset_index(drop=True)


# ---------------------------------------------------------------- formatting
def _signed(x: float) -> str:
    """Signed 3 decimals; a nonzero value that would print as 0.000 gets 4."""
    return f"{x:+.4f}" if x != 0 and abs(x) < 0.0005 else f"{x:+.3f}"


def _auc(row: pd.Series) -> str:
    if pd.isna(row["auc"]):
        return "n/a" if isinstance(row.get("not_applicable"), str) else _DASH
    return f"{row['auc']:.3f} [{row['auc_ci_low']:.3f}, {row['auc_ci_high']:.3f}]"


def _delta(row: pd.Series) -> str:
    if row["row_type"] != "tier" or pd.isna(row.get("delta_auc")):
        return _DASH
    cell = (f"{_signed(row['delta_auc'])} "
            f"[{_signed(row['delta_auc_ci_low'])}, {_signed(row['delta_auc_ci_high'])}]")
    clear = row["delta_auc_ci_high"] < 0 or row["delta_auc_ci_low"] > 0
    return f"**{cell}**" if clear else cell


def _share(value) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1%}"


def _f1(row: pd.Series) -> str:
    if pd.isna(row["f1"]):
        return "n/a" if isinstance(row.get("not_applicable"), str) else _DASH
    return f"{row['f1']:.3f}"


def _name(row: pd.Series) -> str:
    return row["method"] + (" †" if row["reads_geopolitical"] else "")


def _method_cell(row: pd.Series) -> str:
    cell = _name(row)
    if row["copy_of"]:
        return f"{cell} (= {row['copy_of']})"
    if row["same_numbers_as"]:
        return f"{cell} (same numbers as {row['same_numbers_as']})"
    return cell


def _n_agents(row: pd.Series) -> str:
    return f"{int(row['n_agents'])}" if row["row_type"] == "tier" and pd.notna(row["n_agents"]) else _DASH


def _line(row: pd.Series, ceiling: bool = False) -> str:
    name = f"*{row['method']} (ceiling, not a method)*" if ceiling else _method_cell(row)
    return (f"| {name} | {_n_agents(row)} | {_auc(row)} | {_delta(row)} | {_f1(row)} | "
            f"{_share(row['far'])} | {_share(row['alert_rate'])} |")


def _overlap(a: pd.Series, b: pd.Series) -> bool:
    """Whether two methods' AUC intervals overlap (False if either has none)."""
    keys = ("auc_ci_low", "auc_ci_high")
    if any(pd.isna(r[k]) for r in (a, b) for k in keys):
        return False
    return not (a["auc_ci_low"] > b["auc_ci_high"] or b["auc_ci_low"] > a["auc_ci_high"])


def best_line(rows: pd.DataFrame) -> str:
    """The "best non-ceiling method" line for one region's rows.

    The highest AUC point estimate among baselines and tiers: never the ORACLE
    (a ceiling row), never a control. It names every method whose interval
    overlaps the winner's (the win is then not clear). A region without
    positives has no AUC to rank by, so it says so.
    """
    if rows["test_positives"].max() == 0:
        return ("**Best non-ceiling method:** none. This region has no positives, so AUC is "
                "undefined and only false-alarm rates exist; a low alert rate alone does not "
                "rank methods (a method that never alerts scores 0).")
    pool = rows[rows["row_type"].isin(["baseline", "tier"]) & rows["auc"].notna()]
    best = pool.loc[pool["auc"].idxmax()]
    text = (f"**Best non-ceiling method:** {_name(best)}, the highest AUC point estimate, "
            f"{_auc(best)} (circularity {best['circularity']}).")
    tied = [_name(r) for _, r in pool.drop(best.name).iterrows() if _overlap(best, r)]
    text += (f" Its interval overlaps {', '.join(tied)}: not a clear win." if tied
             else " Its interval overlaps no other method's.")
    tiers = pool[pool["row_type"] == "tier"]
    if best["row_type"] == "tier":
        return text + " It is a multi-agent tier."
    if not tiers.empty:
        top = tiers.loc[tiers["auc"].idxmax()]
        text += (f" Best multi-agent tier: {top['method']}, AUC {_auc(top)}; its interval "
                 f"{'overlaps' if _overlap(best, top) else 'does not overlap'} the best's.")
    return text


def _header(rows: pd.DataFrame) -> str:
    first = rows.iloc[0]
    return (f"{int(first['test_positives'])} positives among {int(first['n_test'])} scored test "
            f"days ({first['test_start']} .. {first['test_end']})")


def _legend(results: pd.DataFrame) -> list[str]:
    block, n, ci = (results[c].dropna().iloc[0] if c in results and results[c].notna().any() else None
                    for c in ("bootstrap_block", "bootstrap_n", "bootstrap_ci"))
    boot = (f"{ci:.0%} moving-block bootstrap intervals ({int(block)}-day blocks, {int(n):,} resamples)"
            if block is not None else "bootstrap intervals")
    return [
        "# Results: every method, per region",
        "",
        "Source: `eval/method_comparison_results.csv`; regenerate with "
        "`python scripts/results_table.py`. This is the one results table: other documents point here.",
        "",
        f"- AUC: {boot}. ΔAUC: the paired difference against Tier 1 on the same resampled "
        "blocks (tiers only); bold when its interval excludes 0. F1, FAR and alert rate are "
        "point estimates (intervals are in `results_main.csv`).",
        "- FAR: false-alarm rate, FP/(FP+TN). Alert rate: share of scored days that alert. A day "
        "alerts when its score is at or above the method's training 90th percentile; M1, M2 and "
        "the ORACLE set alerts directly (M1 every day, M2 never, the ORACLE when yesterday's "
        "label is 1).",
        "- Without positives (Malacca) AUC and F1 are undefined and every alert is false, so FAR "
        "equals the alert rate.",
        "- `= Tier k`: copies the tier before it (the agent it would add is disabled in that "
        "region). `same numbers as Tier k`: a different agent set with identical metrics (an "
        "enabled agent that adds nothing, e.g. a constant one). n/a: B7 and B8 need positives "
        "in the training window.",
        "- † fits on every feature column, geopolitical included (B3, B7, B8), so it moves when "
        "those columns do. Tiers 3-5 also include the geopolitical agent wherever it is enabled "
        "(every region except Panama); only B3, B7 and B8 are marked because only they fit on "
        "the raw columns.",
        "- Best non-ceiling method: the highest AUC point estimate among methods and tiers, with "
        "the methods whose interval overlaps it named. Controls M0-M2 are listed but cannot be "
        "the best. The ORACLE reads yesterday's label: a ceiling row, never ranked.",
        "",
    ]


def render_markdown(table: pd.DataFrame, results: pd.DataFrame) -> str:
    out = _legend(results)
    head = ("| Method | n agents | AUC [CI] | ΔAUC vs Tier 1 [CI] | F1 | FAR | Alert rate |\n"
            "|---|---|---|---|---|---|---|")
    for region in sorted(table["region"].unique()):
        rows = table[table["region"] == region]
        out += [f"### {region} — {_header(rows)}", "", head]
        out += [_line(r) for _, r in rows[rows["row_type"] != "ceiling"].iterrows()]
        out += [_line(r, ceiling=True) for _, r in rows[rows["row_type"] == "ceiling"].iterrows()]
        out += ["", best_line(rows), ""]
    return "\n".join(out)


def main() -> int:
    results = pd.read_csv(_EVAL / "method_comparison_results.csv")
    table = build_table(results)
    table.to_csv(_EVAL / "results_main.csv", index=False)
    (_EVAL / "results_main.md").write_text(render_markdown(table, results), encoding="utf-8")
    print(f"wrote {_EVAL / 'results_main.csv'} ({len(table)} rows) and results_main.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
