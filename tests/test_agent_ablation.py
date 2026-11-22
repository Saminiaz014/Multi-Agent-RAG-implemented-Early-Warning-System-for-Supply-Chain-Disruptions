"""Issue 1: each enabled agent alone, and the full set with one agent left out."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_method_comparison", _ROOT / "scripts" / "run_method_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)

_NEW_POLICY = {"gap_fill": {"policy": "forward-fill from the last earlier value"},
               "market_source": {"dating": "first-release date (ALFRED)"},
               "domains_present": {"market": ["brent_crude_usd"]}}
_WEIGHTS = {"shipping": 0.294, "market": 0.176, "geopolitical": 0.294,
            "natural_disaster": 0.118, "news_sentiment": 0.118}
#: Panama-like: geopolitical disabled.
_CONFIG = {"weights": dict(_WEIGHTS),
           "agents": {"shipping": {"enabled": True}, "geopolitical": {"enabled": False}}}
_ENABLED = ["shipping", "market", "natural_disaster", "news_sentiment"]
_MASKED = ["2024-07-04"]


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    y = (((ts >= "2023-03-01") & (ts < "2023-05-01"))
         | ((ts >= "2024-06-01") & (ts < "2024-09-01"))).astype(int)
    return pd.DataFrame({"timestamp": ts, "y_true": y,
                         "shipping__vessel_count": 100.0 - 30 * y})


def _signal(domain: str, eq1: bool) -> float:
    return {"shipping": 1.0 if eq1 else 0.3, "market": 0.6, "geopolitical": 0.9,
            "natural_disaster": 0.0, "news_sentiment": 0.2}[domain]


@pytest.fixture
def calls(monkeypatch) -> list[tuple[str, bool]]:
    """Fake agents with their own signal strength; shipping weaker with Eq. 1 off."""
    seen: list[tuple[str, bool]] = []

    def fake(domain, config, train, test, history=None):
        eq1 = (config["agents"].get("shipping") or {}).get("level_shift", True)
        seen.append((domain, eq1))
        rng = np.random.default_rng(len(test) * 7 + rmc._TIER_ORDER.index(domain))
        return test["y_true"].to_numpy() * _signal(domain, eq1) + rng.normal(0, 0.5, len(test))

    monkeypatch.setattr(rmc, "_agent_scores", fake)
    return seen


def _manifest() -> dict:
    return {"region": "x", "news_source": {"days_masked": _MASKED}, **_NEW_POLICY}


def _run(calls) -> tuple[dict, dict, dict]:
    on, off = rmc.agent_ablation("x", _frame(), _manifest(), _CONFIG, n_resamples=20)
    main = {r["method"]: r for r in rmc.run_region("x", _frame(), _manifest(), _CONFIG,
                                                   n_resamples=20) if r["family"] == "tier"}
    return ({r["method"]: r for r in on}, {r["method"]: r for r in off}, main)


def test_rows_cover_the_full_set_each_agent_alone_and_each_left_out(calls) -> None:
    on, off, _ = _run(calls)
    expected = ({"full enabled set"} | {f"{a} alone" for a in _ENABLED}
                | {f"without {a}" for a in _ENABLED})
    assert set(on) == set(off) == expected                      # no disabled agent
    assert {r["eq1"] for r in on.values()} == {"on"}
    assert {r["eq1"] for r in off.values()} == {"off"}
    assert on["without market"]["agents"] == "shipping,natural_disaster,news_sentiment"
    assert on["market alone"]["agents"] == "market" and on["market alone"]["n_agents"] == 1


_SAME_AS_TIER = ("auc", "auc_ci_low", "auc_ci_high", "f1", "fpr", "alert_rate",
                 "f1_ci_low", "f1_ci_high", "alert_rate_ci_low", "alert_rate_ci_high")


def test_the_full_set_and_shipping_alone_reproduce_the_main_tiers(calls) -> None:
    """Threshold metrics too: each row is thresholded on its own training composite."""
    on, _, main = _run(calls)
    for key in _SAME_AS_TIER:
        assert on["full enabled set"][key] == main["Tier 5"][key], key
        assert on["shipping alone"][key] == main["Tier 1"][key], key
    assert on["full enabled set"]["fusion_weights"] == main["Tier 5"]["fusion_weights"]


def _threshold_rate(test_scores, train_scores, keep, train_keep) -> float:
    threshold = np.quantile(train_scores[train_keep], 0.90)
    return float(np.mean(test_scores[keep] >= threshold))


def test_an_agent_alone_is_scored_on_its_own_scores(calls) -> None:
    on, _, _ = _run(calls)
    train, test = rmc._split(_frame())
    keep = ~test["timestamp"].isin(pd.to_datetime(_MASKED)).to_numpy()
    market = rmc._agent_scores("market", _CONFIG, train, test)
    y = test["y_true"].to_numpy()
    assert on["market alone"]["auc"] == pytest.approx(roc_auc_score(y[keep], market[keep]),
                                                      abs=1e-12)
    # Its alert threshold is the p90 of its own training scores.
    train_keep = ~train["timestamp"].isin(pd.to_datetime(_MASKED)).to_numpy()
    market_train = rmc._agent_scores("market", _CONFIG, train, train)
    assert on["market alone"]["alert_rate"] == pytest.approx(
        _threshold_rate(market, market_train, keep, train_keep), abs=1e-12)
    assert on["market alone"]["fusion_weights"] == "market=1.0000"


def test_leave_one_out_drops_the_agent_and_keeps_the_others_weights(calls) -> None:
    on, _, _ = _run(calls)
    train, test = rmc._split(_frame())
    keep = ~test["timestamp"].isin(pd.to_datetime(_MASKED)).to_numpy()
    y = test["y_true"].to_numpy()
    scores = {d: rmc._agent_scores(d, _CONFIG, train, test) for d in _ENABLED}
    rest = [d for d in _ENABLED if d != "shipping"]
    mass = sum(_WEIGHTS[d] for d in rest)
    expected = sum(_WEIGHTS[d] * scores[d] for d in rest) / mass
    row = on["without shipping"]
    assert row["auc"] == pytest.approx(roc_auc_score(y[keep], expected[keep]), abs=1e-12)
    # Thresholded on its own training composite, not the full set's.
    train_keep = ~train["timestamp"].isin(pd.to_datetime(_MASKED)).to_numpy()
    on_train = {d: rmc._agent_scores(d, _CONFIG, train, train) for d in rest}
    expected_train = sum(_WEIGHTS[d] * on_train[d] for d in rest) / mass
    assert row["alert_rate"] == pytest.approx(
        _threshold_rate(expected, expected_train, keep, train_keep), abs=1e-12)
    assert row["fusion_weights"].startswith("market=")
    # Paired: LOO - full, on the same blocks.
    assert row["loo_delta_auc"] == pytest.approx(row["auc"] - on["full enabled set"]["auc"],
                                                 abs=1e-12)
    assert row["loo_delta_auc_ci_low"] <= row["loo_delta_auc_ci_high"]
    assert row["loo_delta_auc"] < 0                              # shipping carries the signal
    assert "loo_delta_auc" not in on["market alone"]


def test_eq1_off_refits_only_shipping(calls) -> None:
    on, off, _ = _run(calls)
    assert [(d, e) for d, e in calls if not e] == [("shipping", False)] * 2   # only shipping
    for agent in ("market", "natural_disaster", "news_sentiment"):
        assert off[f"{agent} alone"]["auc"] == on[f"{agent} alone"]["auc"]
    assert off["shipping alone"]["auc"] < on["shipping alone"]["auc"]
    assert off["without shipping"]["auc"] == on["without shipping"]["auc"]
    assert on["shipping alone"]["circularity"] == "high"
    assert off["shipping alone"]["circularity"] == "medium"
    assert on["without shipping"]["circularity"] == "low"


def test_the_ablation_refuses_a_config_with_eq1_already_off() -> None:
    with pytest.raises(ValueError, match="level_shift"):
        rmc.agent_ablation("x", _frame(), _manifest(), rmc._without_eq1(_CONFIG), n_resamples=5)


def test_a_window_without_positives_reports_alert_rates_only(calls) -> None:
    frame = _frame()
    frame.loc[frame["timestamp"] >= "2023-07-01", "y_true"] = 0     # Malacca-like test
    on, _ = rmc.agent_ablation("x", frame, _manifest(), _CONFIG, n_resamples=10)
    for row in on:
        assert np.isnan(row["auc"]) and 0.0 <= row["alert_rate"] <= 1.0
        assert "loo_delta_auc" not in row


def test_one_enabled_agent_has_no_leave_one_out_row(calls) -> None:
    only = {"weights": dict(_WEIGHTS),
            "agents": {d: {"enabled": d == "shipping"} for d in rmc._TIER_ORDER}}
    on, _ = rmc.agent_ablation("x", _frame(), _manifest(), only, n_resamples=10)
    assert [r["method"] for r in on] == ["full enabled set", "shipping alone"]
    assert on[0]["agents_unavailable"] == ""


def test_an_enabled_agent_without_scores_is_named(monkeypatch) -> None:
    def fake(domain, config, train, test, history=None):
        if domain == "market":
            return None                                             # e.g. its source failed
        return test["y_true"].to_numpy() + np.random.default_rng(1).normal(0, 0.5, len(test))

    monkeypatch.setattr(rmc, "_agent_scores", fake)
    on, _ = rmc.agent_ablation("x", _frame(), _manifest(), _CONFIG, n_resamples=10)
    assert {r["agents_unavailable"] for r in on} == {"market"}
    assert "market alone" not in {r["method"] for r in on}
