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


def _report(tmp_path, monkeypatch, results) -> str:
    monkeypatch.setattr(rep, "_EVAL", tmp_path)
    rep.write_report(results)
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


def test_oracle_and_controls_are_never_the_best_baseline(tmp_path, monkeypatch) -> None:
    results = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    extra = [_row("panama", "ORACLE label t-1", 0.99, family="baseline", kind="oracle"),
             _row("panama", "M0 random", 0.95, family="baseline", kind="control")]
    text = _report(tmp_path, monkeypatch, pd.concat([results, pd.DataFrame(extra)]))
    assert "Best baseline B1 rolling z" in text
    assert "Best baseline ORACLE" not in text and "Best baseline M0" not in text
    assert "ORACLE label t-1 at" not in text            # not a "baseline beats tiers" entry


def test_overlap_is_called_inconclusive_not_inseparable(tmp_path, monkeypatch) -> None:
    base = _results({"panama": [0.9, 0.8, 0.7, 0.7, 0.71]})
    apart = _report(tmp_path, monkeypatch, _with_ci(base, {}))
    assert "The intervals do not overlap" in apart           # 0.55 +-0.02 vs 0.90 +-0.02
    close = _with_ci(base, {"B1 rolling z": (0.40, 0.89), "Tier 1": (0.80, 0.95)})
    text = _report(tmp_path, monkeypatch, close)
    assert "inconclusive" in text
    assert "do not separate" not in text and "cannot separate" not in text


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
