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


def test_untrainable_count_and_news_coverage_are_not_hard_coded(tmp_path, monkeypatch) -> None:
    text = _report(tmp_path, monkeypatch,
                   _results({"a": [0.9] * 5, "b": [0.9] * 5}))
    assert "Panama alone has news" not in text
    assert "2 of 2 regions" in text            # B7 n/a in both synthetic regions
