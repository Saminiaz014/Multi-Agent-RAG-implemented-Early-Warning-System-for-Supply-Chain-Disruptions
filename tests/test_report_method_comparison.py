"""The generated COMPARISON_REPORT.md must state what the results CSV shows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "report_method_comparison", _ROOT / "scripts" / "report_method_comparison.py")
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)


def _row(region, method, auc, family="tier", **extra):
    base = {"region": region, "method": method, "kind": "multi-agent" if family == "tier"
            else "unsupervised", "family": family, "circularity": "high", "auc": auc,
            "f1": 0.3, "fpr": 0.1, "alert_rate": 0.2,
            "test_start": "2023-07-01", "test_end": "2025-11-21"}
    return {**base, **extra}


def _results(tier_aucs: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for region, aucs in tier_aucs.items():
        rows.append(_row(region, "B1 rolling z", 0.55, family="baseline"))
        rows.append(_row(region, "B7 logistic regression", np.nan, family="baseline",
                         not_applicable="no positive examples in the training window"))
        rows += [_row(region, f"Tier {i + 1}", a) for i, a in enumerate(aucs)]
    return pd.DataFrame(rows)


def _report(tmp_path, monkeypatch, results, sensitivity=None) -> str:
    monkeypatch.setattr(rep, "_EVAL", tmp_path)
    rep.write_report(results, sensitivity=sensitivity)
    return (tmp_path / "COMPARISON_REPORT.md").read_text(encoding="utf-8")


def test_report_states_the_fixed_window_not_a_fraction(tmp_path, monkeypatch) -> None:
    text = _report(tmp_path, monkeypatch, _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]}))
    assert "2023-07-01" in text and "2025-11-21" in text
    assert "30%" not in text


def test_peak_claim_follows_the_numbers(tmp_path, monkeypatch) -> None:
    early = _report(tmp_path, monkeypatch, _results({"a": [0.9, 0.8, 0.7, 0.7, 0.7]}))
    assert "every region the peak is Tier 1 or Tier 2" in early
    late = _report(tmp_path, monkeypatch, _results({"a": [0.5, 0.6, 0.7, 0.8, 0.9]}))
    assert "every region the peak is Tier 1 or Tier 2" not in late
    assert "Tier 3 lowers AUC" not in late


def test_report_shows_bootstrap_cis_when_present(tmp_path, monkeypatch) -> None:
    results = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    results["auc_ci_low"], results["auc_ci_high"] = results["auc"] - 0.05, results["auc"] + 0.04
    text = _report(tmp_path, monkeypatch, results)
    assert "0.900 [0.850, 0.940]" in text
    assert "block bootstrap" in text
    assert "no confidence intervals" not in text


def test_report_without_ci_columns_says_so(tmp_path, monkeypatch) -> None:
    text = _report(tmp_path, monkeypatch, _results({"panama": [0.9] * 5}))
    assert "no confidence intervals" in text


def _with_ci(results: pd.DataFrame, cis: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Give every row a CI; ``cis`` overrides by method name."""
    out = results.copy()
    out["auc_ci_low"], out["auc_ci_high"] = out["auc"] - 0.02, out["auc"] + 0.02
    out["bootstrap_valid"] = 1000
    for method, (lo, hi) in cis.items():
        out.loc[out["method"] == method, ["auc_ci_low", "auc_ci_high"]] = [lo, hi]
    return out


def test_the_report_points_to_the_results_table_and_states_no_best_baseline(
        tmp_path, monkeypatch) -> None:
    """One table (eval/results_main.md) owns "best"; the report only points to it."""
    results = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    extra = [_row("panama", "ORACLE label t-1", 0.99, family="baseline", kind="oracle"),
             _row("panama", "M0 random", 0.95, family="baseline", kind="control")]
    text = _report(tmp_path, monkeypatch, pd.concat([results, pd.DataFrame(extra)]))
    assert "results_main.md" in text
    assert "best baseline" not in text.lower()
    assert "baselines score above" not in text.lower()
    assert "ORACLE label t-1 at" not in text            # no "beats the tiers" entry


def test_overlap_is_called_inconclusive_not_inseparable() -> None:
    """The wording ``_separation`` gives (used for the tier steps) for two methods."""
    def row(low, high):
        return pd.Series({"auc": (low + high) / 2, "auc_ci_low": low, "auc_ci_high": high})

    apart = rep._separation(row(0.53, 0.57), row(0.88, 0.92))
    assert "do not overlap" in apart                         # a clear difference
    close = rep._separation(row(0.40, 0.89), row(0.80, 0.95))
    assert "inconclusive" in close
    assert "do not separate" not in close and "cannot separate" not in close
    assert rep._separation(pd.Series({"auc": 0.5}), row(0.4, 0.6)) == ""    # no intervals


def test_only_a_whole_interval_below_half_is_called_anti_correlated(tmp_path, monkeypatch) -> None:
    base = _results({"hormuz": [0.6, 0.55, 0.45, 0.40, 0.42]})
    text = _report(tmp_path, monkeypatch, _with_ci(base, {
        "Tier 3": (0.30, 0.60),            # spans 0.5: indistinguishable from chance
        "Tier 4": (0.30, 0.45),            # wholly below: anti-correlated
        "Tier 5": (0.35, 0.52)}))
    section = text.split("anti-correlated with the label", 1)[1].split("###", 1)[0]
    assert "Tier 4" in section and "Tier 3" not in section and "Tier 5" not in section
    chance = next(line for line in text.splitlines()
                  if "indistinguishable from chance" in line)
    assert "hormuz Tier 3" in chance and "hormuz Tier 5" in chance
    assert "Tier 4" not in chance


def test_tier_steps_are_reported_as_point_estimates_when_cis_exist(tmp_path, monkeypatch) -> None:
    base = _with_ci(_results({"a": [0.9, 0.8, 0.7, 0.7, 0.7]}), {})
    text = _report(tmp_path, monkeypatch, base)
    assert "point estimate" in text
    assert "bootstrap_valid" not in text and "1,000 of 1,000" in text


def test_untrainable_count_and_news_coverage_are_not_hard_coded(tmp_path, monkeypatch) -> None:
    text = _report(tmp_path, monkeypatch,
                   _results({"a": [0.9] * 5, "b": [0.9] * 5}))
    assert "Panama alone has news" not in text
    assert "2 of 2 regions" in text            # B7 n/a in both synthetic regions


# ----------------------------------------------------- ceiling, deltas, splits
def _region_table(text: str, region: str) -> list[str]:
    block = text.split(f"### {region}\n", 1)[1]
    return [line for line in block.splitlines() if line.startswith("| ")][2:]


def test_oracle_is_a_ceiling_row_below_the_methods(tmp_path, monkeypatch) -> None:
    results = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    oracle = _row("panama", "ORACLE label t-1", 0.99, family="ceiling", kind="oracle")
    text = _report(tmp_path, monkeypatch, pd.concat([results, pd.DataFrame([oracle])]))
    table = _region_table(text, "panama")
    assert "ORACLE" in table[-1] and "ceiling" in table[-1]
    assert all("ORACLE" not in line for line in table[:-1])   # not ranked among methods
    assert "Ceiling: ORACLE label t-1" in text
    assert "best baseline" not in text.lower()


def _with_deltas(results: pd.DataFrame, deltas: dict[str, tuple[float, float, float]]):
    out = _with_ci(results, {})
    for method, (d, lo, hi) in deltas.items():
        sel = out["method"] == method
        out.loc[sel, ["delta_auc", "delta_auc_ci_low", "delta_auc_ci_high"]] = [d, lo, hi]
    return out


def test_paired_delta_table_and_verdicts(tmp_path, monkeypatch) -> None:
    base = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    text = _report(tmp_path, monkeypatch, _with_deltas(base, {
        "Tier 2": (-0.10, -0.15, -0.05),           # clearly below Tier 1
        "Tier 3": (-0.20, -0.30, 0.01),            # includes 0: inconclusive
        "Tier 4": (-0.20, -0.30, -0.10),
        "Tier 5": (0.05, 0.01, 0.09)}))            # clearly above
    assert "Paired" in text and "Tier k - Tier 1" in text
    assert "-0.100 [-0.150, -0.050]" in text
    assert "2 of 4" in text and "1 of 4" in text      # below / above / inconclusive counts
    # Overlap wording is not used when the paired test is available.
    assert "clearly apart in" not in text


def test_split_sensitivity_table(tmp_path, monkeypatch) -> None:
    base = _with_ci(_results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]}), {})
    rows = []
    for split, t1, t4 in (("2023-01-01", 0.88, 0.85), ("2023-07-01", 0.90, 0.70),
                          ("2024-01-01", 0.93, 0.72)):
        common = {"region": "panama", "split_date": split, "test_start": split,
                  "test_end": "2025-11-21", "n_test": 700, "test_positives": 140}
        rows.append({**common, "method": "Tier 1", "auc": t1,
                     "auc_ci_low": t1 - 0.05, "auc_ci_high": t1 + 0.03})
        rows.append({**common, "method": "Tier 4", "auc": t4,
                     "auc_ci_low": t4 - 0.05, "auc_ci_high": t4 + 0.03,
                     "delta_auc": t4 - t1, "delta_auc_ci_low": t4 - t1 - 0.04,
                     "delta_auc_ci_high": t4 - t1 + 0.04})
    text = _report(tmp_path, monkeypatch, base, sensitivity=pd.DataFrame(rows))
    section = text.split("Split-date sensitivity", 1)[1]
    assert "| panama | 2024-01-01 |" in section
    assert "0.930 [0.880, 0.960]" in section
    assert "-0.210 [-0.250, -0.170]" in section
    assert "Tier 1 AUC ranges 0.880-0.930" in section


def test_masked_days_are_stated(tmp_path, monkeypatch) -> None:
    results = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    results["n_masked"] = 18
    text = _report(tmp_path, monkeypatch, results)
    assert "18 test days" in text and "masked" in text


def test_dating_claim_only_for_frames_that_record_it(tmp_path, monkeypatch) -> None:
    results = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    assert "first-release date" not in _report(tmp_path, monkeypatch, results)
    results["frame_policy"] = "news masked; gaps forward-filled; FRED first-release"
    assert "first-release date" in _report(tmp_path, monkeypatch, results)


def test_ranking_graph_keeps_the_ceiling_out_of_the_bars(tmp_path, monkeypatch) -> None:
    results = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    oracle = _row("panama", "ORACLE label t-1", 0.99, family="ceiling", kind="oracle")
    frame = pd.concat([results, pd.DataFrame([oracle])], ignore_index=True)
    drawn: list[list[str]] = []
    real = rep.plt.Axes.set_yticklabels

    def spy(self, labels, *a, **kw):
        drawn.append(list(labels))
        return real(self, labels, *a, **kw)

    monkeypatch.setattr(rep, "_G_METHOD", tmp_path)
    monkeypatch.setattr(rep.plt.Axes, "set_yticklabels", spy)
    rep.graph_region_rankings(frame)
    assert drawn and all("ORACLE label t-1" not in labels for labels in drawn)


_RULE = "score >= training p90"


def test_alert_rule_and_always_alerting_rows_are_explained(tmp_path, monkeypatch) -> None:
    results = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]}).assign(alert_rule=_RULE)
    extra = [_row("panama", "M1 always alert", 0.5, family="baseline", kind="control",
                  alert_rate=1.0, alert_rule="direct: every day"),
             _row("panama", "M2 never alert", 0.5, family="baseline", kind="control",
                  alert_rate=0.0, alert_rule="direct: no day"),
             _row("panama", "ORACLE label t-1", 0.99, family="ceiling", kind="oracle",
                  alert_rate=0.17, alert_rule="direct: yesterday's label is 1"),
             _row("panama", "B9 constant", 0.5, family="baseline", alert_rate=1.0,
                  alert_rule=_RULE)]
    text = _report(tmp_path, monkeypatch, pd.concat([results, pd.DataFrame(extra)]))
    note = next(line for line in text.splitlines() if "Alert rule" in line)
    assert "score >= " in note and "90th percentile" in note
    # The controls' own rules, read from the CSV.
    for shown in ("M1 always alert (every day)", "M2 never alert (no day)",
                  "ORACLE label t-1 (yesterday's label is 1)"):
        assert shown in note
    # Only threshold-rule rows are listed as alerting every day through a tie.
    always = note.split("every scored test day")[1]
    assert "B9 constant" in always and "M1 always alert" not in always
    assert "Tier 1" not in note
    assert "bounds AUC only" not in note                # the ceiling's F1 is its own now


def test_alert_rule_note_without_always_alerting_rows(tmp_path, monkeypatch) -> None:
    text = _report(tmp_path, monkeypatch, _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]}))
    note = next(line for line in text.splitlines() if "Alert rule" in line)
    assert "every scored test day" not in note
    assert "directly" not in note                       # no alert_rule column: no controls


def _tiers_with(region, agents_by_tier, disabled_by_tier):
    return [_row(region, f"Tier {k}", 0.8 - 0.01 * k, agents=a, agents_disabled=d)
            for k, (a, d) in enumerate(zip(agents_by_tier, disabled_by_tier), start=1)]


def test_disabled_agents_and_the_tiers_they_repeat_are_named(tmp_path, monkeypatch) -> None:
    panama = _tiers_with(
        "panama",
        ["shipping", "shipping,market", "shipping,market",
         "shipping,market,natural_disaster", "shipping,market,natural_disaster,news_sentiment"],
        ["", "", "geopolitical", "geopolitical", "geopolitical"])
    hormuz = _tiers_with(
        "hormuz",
        ["shipping", "shipping,market", "shipping,market,geopolitical",
         "shipping,market,geopolitical,natural_disaster",
         "shipping,market,geopolitical,natural_disaster,news_sentiment"],
        [""] * 5)
    text = _report(tmp_path, monkeypatch, pd.DataFrame(panama + hormuz))
    note = next(line for line in text.splitlines() if "disabled by the region config" in line)
    assert "panama: geopolitical" in note
    assert "panama Tier 3" in note and "Tier 4" not in note    # only exact repeats
    assert "hormuz" not in note                                # nothing disabled there
    assert "method_comparison_all_agents.csv" in note and "appendix" in note


def test_no_disabled_note_without_disabled_agents(tmp_path, monkeypatch) -> None:
    text = _report(tmp_path, monkeypatch, _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]}))
    assert "disabled by the region config" not in text


# ---------------------------------------- tiers that repeat the tier before
_FULL = ["shipping", "shipping,market", "shipping,market,geopolitical",
         "shipping,market,geopolitical,natural_disaster",
         "shipping,market,geopolitical,natural_disaster,news_sentiment"]
_NO_GEO = ["shipping", "shipping,market", "shipping,market",
           "shipping,market,natural_disaster",
           "shipping,market,natural_disaster,news_sentiment"]          # Panama


def _repeat_results() -> pd.DataFrame:
    """Region a adds geopolitical at Tier 3; region p repeats Tier 2 there."""
    rows = []
    for region, aucs, agents in (("a", [0.90, 0.80, 0.70, 0.71, 0.72], _FULL),
                                 ("p", [0.90, 0.85, 0.85, 0.84, 0.86], _NO_GEO)):
        for k, (auc, ag) in enumerate(zip(aucs, agents), start=1):
            extra = {"agents": ag, "auc_ci_low": auc - 0.02, "auc_ci_high": auc + 0.02}
            if k > 1:
                d = auc - aucs[0]
                extra.update(delta_auc=d, delta_auc_ci_low=d - 0.01, delta_auc_ci_high=d + 0.01)
            rows.append(_row(region, f"Tier {k}", auc, **extra))
        rows.append(_row(region, "B1 rolling z", 0.55, family="baseline",
                         auc_ci_low=0.5, auc_ci_high=0.6))
    # Malacca: market disabled, so Tier 2 repeats Tier 1.
    mal = ["shipping", "shipping", "shipping,geopolitical",
           "shipping,geopolitical,natural_disaster",
           "shipping,geopolitical,natural_disaster,news_sentiment"]
    for k, (rate, ag) in enumerate(zip([0.3, 0.3, 0.1, 0.1, 0.1], mal), start=1):
        rows.append(_row("malacca", f"Tier {k}", np.nan, agents=ag, alert_rate=rate))
    rows.append(_row("malacca", "B1 rolling z", np.nan, family="baseline", alert_rate=0.1))
    return pd.DataFrame(rows)


def test_a_repeated_tier_is_marked_and_not_counted(tmp_path, monkeypatch) -> None:
    text = _report(tmp_path, monkeypatch, _repeat_results())
    assert "| Tier 3 (= Tier 2) |" in text                    # p's table
    assert "| Tier 3 |" in text                               # a's Tier 3 is its own
    assert "0.850 -> 0.850 (=)" in text                       # progression line
    # 8 deltas, one of them a copy: counted once, and said so.
    assert "clearly below Tier 1 in 7 of 7 comparisons" in text
    assert "1 tier that repeats the tier before it is not counted" in text
    delta_table = text.split("Paired ΔAUC against Tier 1", 1)[1]
    p_row = next(line for line in delta_table.splitlines() if line.startswith("| p |"))
    a_row = next(line for line in delta_table.splitlines() if line.startswith("| a |"))
    assert "(= Tier 2)" in p_row and "(=" not in a_row
    # Malacca: each distinct tier once (0.3, 0.1, 0.1, 0.1), not Tier 1 twice.
    assert "tiers 15%" in text


def test_the_tier3_geopolitical_claim_skips_regions_that_disable_it(tmp_path, monkeypatch) -> None:
    text = _report(tmp_path, monkeypatch, _repeat_results())
    assert "adding the geopolitical agent at Tier 3 lowers AUC each time" in text
    assert "it is disabled in p" in text


def test_sensitivity_table_shows_a_tier_with_fewer_agents(tmp_path, monkeypatch) -> None:
    base = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    common = {"region": "panama", "split_date": "2023-07-01", "test_start": "2023-07-01",
              "test_end": "2025-11-21", "n_test": 700, "test_positives": 140}
    rows = [{**common, "method": "Tier 1", "auc": 0.9, "n_agents": 1},
            {**common, "method": "Tier 4", "auc": 0.8, "n_agents": 3,
             "delta_auc": -0.1, "delta_auc_ci_low": -0.14, "delta_auc_ci_high": -0.06}]
    section = _report(tmp_path, monkeypatch, base, sensitivity=pd.DataFrame(rows)) \
        .split("Split-date sensitivity", 1)[1]
    assert "0.800 (3 agents)" in section
    assert "0.900 (1 agents)" not in section                  # only when fewer than k
