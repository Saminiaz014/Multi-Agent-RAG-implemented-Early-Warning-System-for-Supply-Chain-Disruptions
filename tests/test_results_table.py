"""eval/results_main: one complete table per region, from the results file alone."""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _ROOT / "eval" / "method_comparison_results.csv"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = _load("results_table", "scripts/results_table.py")

_TIERS = ["shipping", "shipping,market", "shipping,market,geopolitical",
          "shipping,market,geopolitical,natural_disaster",
          "shipping,market,geopolitical,natural_disaster,news_sentiment"]


def _row(region, method, family="baseline", kind="unsupervised", auc=0.6, **extra):
    base = {"region": region, "method": method, "kind": kind, "family": family,
            "circularity": "low", "alert_rule": "score >= training p90", "auc": auc,
            "auc_ci_low": auc - 0.05, "auc_ci_high": auc + 0.05, "f1": 0.3, "fpr": 0.1,
            "alert_rate": 0.2, "alert_rate_ci_low": 0.1, "alert_rate_ci_high": 0.3,
            "test_start": "2023-07-01", "test_end": "2025-11-21", "n_test": 800,
            "test_positives": 100, "bootstrap_block": 30.0, "bootstrap_n": 1000.0,
            "bootstrap_ci": 0.95}
    return {**base, **extra}


def _synthetic(tier_agents=None, oracle_auc=0.99, m3_auc=0.97, extra=()) -> pd.DataFrame:
    tier_agents = tier_agents or _TIERS
    rows = [_row("a", "M0 random", kind="control", auc=0.5),
            _row("a", "M1 always alert", kind="control", auc=0.5, alert_rate=1.0, fpr=1.0),
            _row("a", "M2 never alert", kind="control", auc=0.5, alert_rate=0.0, fpr=0.0),
            _row("a", "M3 persistence", auc=m3_auc, circularity="high"),
            _row("a", "M8 matrix profile", auc=0.5),
            _row("a", "ORACLE label t-1", family="ceiling", kind="oracle", auc=oracle_auc),
            _row("a", "B1 rolling z", auc=0.55),
            _row("a", "B3 isolation forest", auc=0.56),
            _row("a", "B7 logistic regression", auc=np.nan, f1=np.nan, fpr=np.nan,
                 alert_rate=np.nan, auc_ci_low=np.nan, auc_ci_high=np.nan,
                 not_applicable="no positive examples in the training window")]
    rows += list(extra)
    for k, agents in enumerate(tier_agents, start=1):
        d = {} if k == 1 else {"delta_auc": -0.01 * k, "delta_auc_ci_low": -0.01 * k - 0.02,
                               "delta_auc_ci_high": -0.01 * k + 0.02}
        rows.append(_row("a", f"Tier {k}", family="tier", kind="multi-agent", auc=0.9 - 0.05 * k,
                         agents=agents, n_agents=len(agents.split(",")), circularity="high", **d))
    # Malacca-like: no positives, so AUC, F1 and FPR are undefined.
    none = dict(auc=np.nan, f1=np.nan, fpr=np.nan, auc_ci_low=np.nan, auc_ci_high=np.nan,
                test_positives=0)
    rows += [_row("m", "M3 persistence", **none, alert_rate=0.04),
             _row("m", "ORACLE label t-1", family="ceiling", kind="oracle", **none, alert_rate=0.0),
             _row("m", "Tier 1", family="tier", kind="multi-agent", **none, alert_rate=0.27,
                  agents="shipping", n_agents=1)]
    return pd.DataFrame(rows)


def _section(text: str, region: str) -> str:
    return text.split(f"### {region}", 1)[1].split("\n### ", 1)[0]


_MARKS = re.compile(r" †| \(= Tier \d\)| \(same numbers as Tier \d\)")


def _methods_listed(section: str) -> list[str]:
    """The first cell of each table row, marks stripped, in order."""
    cells = []
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| Method") or line.startswith("|---"):
            continue
        cell = line.split("|")[1].strip()
        cell = re.sub(r"^\*(.*) \(ceiling, not a method\)\*$", r"\1", cell)
        cells.append(_MARKS.sub("", cell))
    return cells


# ------------------------------------------------------------ the real file
def test_every_method_of_the_results_file_is_in_the_table() -> None:
    results = pd.read_csv(_RESULTS)
    table = rt.build_table(results)
    assert set(zip(results["region"], results["method"])) == set(zip(table["region"], table["method"]))
    assert len(table) == len(results)
    text = rt.render_markdown(table, results)
    for region in sorted(results["region"].unique()):
        listed = _methods_listed(_section(text, region))
        wanted = [m for m in results[results["region"] == region]["method"]]
        assert sorted(listed) == sorted(wanted), region          # each exactly once, no extras
        # The fixed order: the methods and tiers as in METHOD_ORDER, the ORACLE last.
        assert listed[:-1] == [m for m in rt.METHOD_ORDER if m in wanted], region
        assert listed[-1].startswith("ORACLE"), region


def test_no_oracle_in_any_best_line_of_the_real_results() -> None:
    results = pd.read_csv(_RESULTS)
    text = rt.render_markdown(rt.build_table(results), results)
    best = [line for line in text.splitlines() if line.startswith("**Best non-ceiling method")]
    assert len(best) == results["region"].nunique()
    assert all("ORACLE" not in line for line in best)


# -------------------------------------------------------------- the rules
def test_the_oracle_is_a_separate_ceiling_row_and_never_the_best() -> None:
    results = _synthetic(oracle_auc=0.999)                 # higher than every method
    table = rt.build_table(results)
    assert table.loc[table.method.str.startswith("ORACLE") & (table.region == "a"), "row_type"
                     ].tolist() == ["ceiling"]
    assert table[table.region == "a"].iloc[-1]["method"] == "ORACLE label t-1"   # listed last
    line = rt.best_line(table[table.region == "a"])
    assert "ORACLE" not in line and "M3 persistence" in line
    section = _section(rt.render_markdown(table, results), "a")
    assert section.count("ORACLE") == 1                       # one ceiling row, nowhere else


def test_an_oracle_row_is_a_ceiling_whatever_family_the_file_gives_it() -> None:
    """The report treats kind == "oracle" as a ceiling too; so must this table."""
    odd = _row("a", "ORACLE label t-1", family="baseline", kind="oracle", auc=0.999)
    results = _synthetic(extra=[odd]).query("not (method == 'ORACLE label t-1' and family == 'ceiling' "
                                            "and region == 'a')")
    table = rt.build_table(results)
    assert table[(table.region == "a") & table.method.str.startswith("ORACLE")]["row_type"].tolist() \
        == ["ceiling"]
    assert "ORACLE" not in rt.best_line(table[table.region == "a"])


def test_the_best_line_states_its_criterion_and_names_the_best_tier() -> None:
    table = rt.build_table(_synthetic())
    line = rt.best_line(table[table.region == "a"])
    assert "M3 persistence" in line and "0.970" in line and "highest AUC point estimate" in line
    assert "Tier 1" in line                                    # best multi-agent tier
    assert "B7" not in line                                    # n/a rows never win
    assert "overlaps no other method's" in line
    assert "does not overlap" in line                          # the best tier is clearly below


def test_the_best_line_names_the_methods_whose_interval_overlaps_the_winner() -> None:
    close = _row("a", "B1 rolling z", auc=0.95)                # [0.90, 1.00] vs M3 [0.92, 1.02]
    results = _synthetic(extra=[close]).query("not (method == 'B1 rolling z' and auc == 0.55)")
    line = rt.best_line(rt.build_table(results).query("region == 'a'"))
    assert "overlaps B1 rolling z" in line and "not a clear win" in line
    assert "overlaps no other" not in line


def test_a_region_without_positives_has_no_best_method_and_alert_rates_only() -> None:
    results = _synthetic()
    table = rt.build_table(results)
    m = table[table.region == "m"]
    assert m["auc"].isna().all() and m["f1"].isna().all()
    assert (m["far"] == m["alert_rate"]).all()                  # no positives: every alert is false
    assert "no positives" in rt.best_line(m).lower()
    section = _section(rt.render_markdown(table, results), "m")
    assert "27.0%" in section and "AUC" in section


def test_copied_tiers_are_marked() -> None:
    copied = ["shipping", "shipping,market", "shipping,market",
              "shipping,market,natural_disaster", "shipping,market,natural_disaster,news_sentiment"]
    results = _synthetic(tier_agents=copied)
    a = rt.build_table(results).query("region == 'a'").set_index("method")
    assert a.loc["Tier 3", "copy_of"] == "Tier 2" and a.loc["Tier 2", "copy_of"] == ""
    assert a.loc["Tier 4", "copy_of"] == ""
    assert "Tier 3 (= Tier 2)" in rt.render_markdown(rt.build_table(results), results)


def test_without_agent_lists_no_tier_is_called_a_copy() -> None:
    results = _synthetic().drop(columns=["agents"])
    assert (rt.build_table(results)["copy_of"] == "").all()


def test_identical_numbers_from_a_different_agent_set_are_marked_but_not_a_copy() -> None:
    """Panama's Tier 4 adds a constant agent: the same metrics, not the same agents."""
    results = _synthetic()
    same = results["method"].isin(["Tier 3", "Tier 4"]) & (results["region"] == "a")
    for col in ("auc", "auc_ci_low", "auc_ci_high", "f1", "fpr", "alert_rate"):
        results.loc[same & (results["method"] == "Tier 4"), col] = \
            results.loc[same & (results["method"] == "Tier 3"), col].iloc[0]
    a = rt.build_table(results).query("region == 'a'").set_index("method")
    assert a.loc["Tier 4", "copy_of"] == "" and a.loc["Tier 4", "same_numbers_as"] == "Tier 3"
    assert a.loc["Tier 3", "same_numbers_as"] == ""
    assert "Tier 4 (same numbers as Tier 3)" in rt.render_markdown(rt.build_table(results), results)


def test_rows_that_read_the_geopolitical_columns_are_marked_and_match_the_harness() -> None:
    harness = _load("run_method_comparison", "scripts/run_method_comparison.py")
    assert rt.READS_ALL_FEATURES == set(harness._ROW_WISE)     # cannot drift from the harness
    results = _synthetic()
    table = rt.build_table(results)
    assert set(table[table.reads_geopolitical].method) == {"B3 isolation forest",
                                                           "B7 logistic regression"}
    text = rt.render_markdown(table, results)
    assert "B3 isolation forest †" in text and "†" in text.split("### a", 1)[0]       # legend
    assert "Tiers 3-5 also include the geopolitical agent" in text                    # honest


def test_the_tier_delta_is_shown_for_tiers_only_and_not_for_tier_1() -> None:
    results = _synthetic()
    cells = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[1:-1]]
             for line in _section(rt.render_markdown(rt.build_table(results), results), "a").splitlines()
             if line.startswith("| ") and not line.startswith(("| Method", "|---"))}
    delta = 3                                                  # method | n agents | AUC | ΔAUC
    assert cells["Tier 2"][delta] == "-0.020 [-0.040, +0.000]"
    assert cells["Tier 1"][delta] == "—" and cells["M3 persistence"][delta] == "—"


def test_a_delta_bound_that_rounds_to_zero_shows_more_digits() -> None:
    assert rt._signed(-0.0004) == "-0.0004" and rt._signed(0.0) == "+0.000"
    assert rt._signed(-0.053) == "-0.053"


def test_not_applicable_baselines_are_listed_with_the_reason() -> None:
    results = _synthetic()
    text = rt.render_markdown(rt.build_table(results), results)
    assert "B7 logistic regression" in text and "n/a" in text.split("B7 logistic regression", 1)[1][:80]


def test_a_results_file_without_intervals_or_deltas_still_builds() -> None:
    results = _synthetic().drop(columns=["auc_ci_low", "auc_ci_high", "delta_auc",
                                         "delta_auc_ci_low", "delta_auc_ci_high"], errors="ignore")
    text = rt.render_markdown(rt.build_table(results), results)
    assert "### a" in text and "Best non-ceiling method" in text


def test_main_writes_the_csv_and_the_markdown_from_one_table(tmp_path, monkeypatch) -> None:
    shutil.copy(_RESULTS, tmp_path / "method_comparison_results.csv")
    monkeypatch.setattr(rt, "_EVAL", tmp_path)
    assert rt.main() == 0
    csv = pd.read_csv(tmp_path / "results_main.csv")
    md = (tmp_path / "results_main.md").read_text(encoding="utf-8")
    assert len(csv) == len(pd.read_csv(_RESULTS))
    assert {"auc", "far", "alert_rate", "copy_of", "same_numbers_as", "reads_geopolitical",
            "row_type"} <= set(csv.columns)
    assert md.count("\n### ") == csv["region"].nunique()
    assert "method_comparison_results.csv" in md                 # names its source
