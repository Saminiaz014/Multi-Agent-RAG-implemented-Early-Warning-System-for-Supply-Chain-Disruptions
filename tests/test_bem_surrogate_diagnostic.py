"""Bab el-Mandeb surrogate diagnostic: error decomposition and out-of-range shares."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "bem_surrogate", _ROOT / "eval" / "diagnostics" / "bem_surrogate.py")
bem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bem)


def _pair(seed: int = 0, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, 1.0, 400)
    return y, y + shift + rng.normal(0, 0.1, 400)


def test_mse_is_bias_squared_plus_variance() -> None:
    y, p = _pair(shift=0.3)
    d = bem.error_decomposition(y, p)
    assert d["mse"] == pytest.approx(np.mean((p - y) ** 2), abs=1e-12)
    assert d["mse"] == pytest.approx(d["bias"] ** 2 + d["variance"], abs=1e-12)
    assert d["bias"] == pytest.approx(0.3, abs=0.02) and d["bias_sq"] == pytest.approx(d["bias"] ** 2)
    assert d["bias_share"] == pytest.approx(d["bias_sq"] / d["mse"])


def test_r2_matches_sklearn_and_the_debiased_r2_adds_the_bias_term() -> None:
    y, p = _pair(seed=3, shift=-0.25)
    d = bem.error_decomposition(y, p)
    assert d["r2"] == pytest.approx(r2_score(y, p), abs=1e-12)
    assert d["r2_debiased"] == pytest.approx(r2_score(y, p - d["bias"]), abs=1e-12)
    assert d["r2_debiased"] == pytest.approx(d["r2"] + d["bias_sq"] / np.var(y), abs=1e-12)
    assert d["r2_debiased"] > d["r2"]


def test_without_bias_the_two_r2_agree() -> None:
    y = np.array([0.1, 0.4, 0.5, 0.9])
    p = y + np.array([0.05, -0.05, 0.05, -0.05])               # mean error exactly 0
    d = bem.error_decomposition(y, p)
    assert d["bias"] == pytest.approx(0.0, abs=1e-15)
    assert d["r2"] == pytest.approx(d["r2_debiased"], abs=1e-12)


def test_predicting_the_training_mean_is_the_constant_baseline() -> None:
    y_te = np.array([0.5, 0.6, 0.7])
    d = bem.error_decomposition(y_te, y_te * 0 + 0.4, train_mean=0.4)
    assert d["r2_train_mean"] == pytest.approx(r2_score(y_te, np.full(3, 0.4)), abs=1e-12)
    assert d["target_shift"] == pytest.approx(0.6 - 0.4)


def test_outside_range_counts_values_below_and_above_the_training_range() -> None:
    r = bem.outside_range(pd.Series([0.0, 5.0, 10.0]), pd.Series([-1.0, 5.0, 11.0, 12.0, np.nan]))
    assert (r["train_min"], r["train_max"]) == (0.0, 10.0)
    assert (r["test_min"], r["test_max"]) == (-1.0, 12.0)
    assert r["below"] == pytest.approx(1 / 4) and r["above"] == pytest.approx(2 / 4)
    assert r["outside"] == pytest.approx(3 / 4) and r["n_test"] == 4        # NaN not counted


def test_a_value_on_the_training_boundary_is_inside() -> None:
    r = bem.outside_range(pd.Series([2.0, 4.0]), pd.Series([2.0, 4.0, 3.0]))
    assert r["outside"] == 0.0


def test_any_outside_is_the_share_of_days_with_at_least_one_feature_out_of_range() -> None:
    train = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    test = pd.DataFrame({"a": [0.5, 2.0, 0.5, 0.5], "b": [0.5, 0.5, -1.0, 0.5]})
    assert bem.any_outside(train, test, ["a", "b"]) == pytest.approx(2 / 4)


def test_the_table_names_every_quantity_and_each_top_feature() -> None:
    y, p = _pair(shift=0.2)
    train = pd.DataFrame({"f1": [0.0, 1.0], "f2": [0.0, 1.0]})
    test = pd.DataFrame({"f1": [0.5, 2.0], "f2": [0.5, 0.5]})
    text = bem.render_table(bem.error_decomposition(y, p, train_mean=0.5), train, test, ["f1", "f2"])
    for needle in ("MSE", "bias", "variance", "R² (held-out)", "R² after subtracting the mean bias",
                   "f1", "f2", "outside the training"):
        assert needle in text
    assert text.count("\n") >= 10 and text.lstrip().startswith("|")


# ------------------------------------------------------------ review follow-ups
def test_a_constant_target_is_refused_rather_than_divided_by() -> None:
    with pytest.raises(ValueError, match="constant"):
        bem.error_decomposition(np.full(5, 0.4), np.linspace(0, 1, 5))


def test_a_training_column_with_no_values_is_refused() -> None:
    with pytest.raises(ValueError, match="training"):
        bem.outside_range(pd.Series([np.nan, np.nan]), pd.Series([0.1, 0.2]))
    train = pd.DataFrame({"a": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="training"):
        bem.any_outside(train, pd.DataFrame({"a": [0.1, 0.2]}), ["a"])


def test_the_table_says_what_the_debiased_r2_and_the_table_cannot_show() -> None:
    y, p = _pair(shift=0.2)
    train = pd.DataFrame({"f1": [0.0, 1.0]})
    test = pd.DataFrame({"f1": [0.5, 0.6]})
    text = bem.render_table(bem.error_decomposition(y, p, train_mean=0.5), train, test, ["f1"])
    assert "upper bound" in text and "covariate shift" in text and "unexplained" in text
