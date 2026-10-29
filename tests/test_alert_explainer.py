"""Tests for the per-alert explainer: trailing z-score, history window, no leakage."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.explainability.alert_explainer import (
    FEATURE_DEFINITIONS,
    _sentence,
    explain_alert,
    history_window,
    trailing_z,
)


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "a": rng.normal(10.0, 1.0, n),
        "b": rng.normal(0.0, 2.0, n),
        "c": np.arange(n, dtype=float),
    })


def test_z_score_matches_hand_computation() -> None:
    values = [float(i % 7) for i in range(39)] + [20.0]  # 40 rows, spike on the last
    t = 39
    prior = values[t - 30:t]
    mean = sum(prior) / 30
    std = math.sqrt(sum((v - mean) ** 2 for v in prior) / 29)
    expected = (values[t] - mean) / std

    frame = pd.DataFrame({"a": values})
    result = explain_alert(frame, {"a": 0.3}, t, "test")

    assert math.isclose(trailing_z(np.array(values), t), expected, rel_tol=1e-12)
    assert math.isclose(result["features"][0]["z"], round(expected, 4))
    assert result["features"][0]["status"] in {"high", "unusual"}
    assert result["features"][0]["direction"] == "raises risk"


def test_history_window_is_15_rows_ending_at_t() -> None:
    t = 30
    window = history_window(40, t)
    assert len(window) == 15
    assert (window.start, window.stop - 1) == (t - 14, t)

    result = explain_alert(_frame(), {"a": 0.5, "b": -0.4, "c": 0.3}, t, "test")
    for item in result["features"][:3]:
        assert item["history_rows"] == [t - 14, t]


def test_values_after_t_do_not_change_the_explanation() -> None:
    frame = _frame()
    shap = {"a": 0.5, "b": -0.4, "c": 0.3}
    t = 32
    before = explain_alert(frame, shap, t, "test")

    altered = frame.copy()
    altered.loc[t + 1:, ["a", "b", "c"]] = 1e6
    after = explain_alert(altered, shap, t, "test")

    assert before == after


def test_sentence_states_definition_level_z_duration_and_start() -> None:
    n = 60
    values = [10.0 + (i % 3) * 0.1 for i in range(n)]
    for i in range(54, n):  # high for the last six days
        values[i] = 30.0
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "congestion_index": values,
    })
    t = n - 1
    result = explain_alert(frame, {"congestion_index": 0.4}, t, "test")
    item, sentence = result["features"][0], result["sentences"][0]

    assert item["definition"] == FEATURE_DEFINITIONS["congestion_index"]
    assert 3 <= len(item["definition"].split()) <= 6
    assert f"Congestion index ({item['definition']})" in sentence
    assert "is high (z = " in sentence and "raises risk" in sentence
    assert f"for {item['days_before_alert']} days, since {item['since']}" in sentence
    assert item["since"] == "2024-02-24"


def test_every_reported_feature_has_history_and_short_definition() -> None:
    result = explain_alert(_frame(), {"a": 0.5, "b": -0.4, "c": 0.3}, 32, "test")
    assert all("history_rows" in f for f in result["features"])
    for definition in FEATURE_DEFINITIONS.values():
        assert 3 <= len(definition.split()) <= 6, definition


def test_breach_reaching_window_start_reads_at_least() -> None:
    item = {"feature": "vessel_count", "definition": FEATURE_DEFINITIONS["vessel_count"],
            "direction": "raises risk", "status": "low", "z": -3.1,
            "days_before_alert": 14, "since": "2024-02-01", "since_window_start": True}
    sentence = _sentence("2024-02-15", item)
    assert sentence == (
        "Alert on 2024-02-15: Vessel count (ships transiting the chokepoint daily) is low "
        "(z = -3.10) and raises risk. It has been low for at least 14 days, "
        "since 2024-02-01 or earlier."
    )
