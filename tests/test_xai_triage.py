"""Phase 4 statistics on toy inputs with known answers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "run_triage", Path("eval/explainability/run_triage.py"))
triage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(triage)


def test_holm_known_values() -> None:
    # Sorted p: 0.01, 0.02, 0.04 -> 0.03, 0.04, 0.04 (monotone step-down).
    assert triage.holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    adjusted = triage.holm([0.5, np.nan, 0.01])
    assert np.isnan(adjusted[1]) and adjusted[2] == pytest.approx(0.02)
    assert adjusted[0] == pytest.approx(0.5)


def test_auc_boot_separation_ties_and_degenerate() -> None:
    rng = np.random.default_rng(0)
    fp = np.array([1, 1, 1, 0, 0])
    assert triage.auc_boot(np.array([3.0, 4, 5, 1, 2]), fp, rng)[0] == 1.0
    assert triage.auc_boot(np.array([1.0, 1, 1, 1, 1]), fp, rng)[0] == 0.5   # all tied
    assert np.isnan(triage.auc_boot(np.array([1.0, 2, 3]), np.array([1, 1, 1]), rng)[0])


def test_mwu_is_one_sided_in_registered_direction() -> None:
    fp = np.array([1] * 8 + [0] * 8)
    higher_in_fp = np.r_[np.arange(10, 18), np.arange(0, 8)].astype(float)
    assert triage.mwu_p(higher_in_fp, fp) < 0.01
    assert triage.mwu_p(-higher_in_fp, fp) > 0.99


def test_rate_ci_brackets_rate() -> None:
    rate, lo, hi = triage.rate_ci(np.array([1, 0, 1, 1]), np.random.default_rng(1))
    assert rate == 0.75 and lo <= rate <= hi
