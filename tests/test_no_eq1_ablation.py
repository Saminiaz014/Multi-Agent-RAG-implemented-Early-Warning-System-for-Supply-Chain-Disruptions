"""Issue 5: every tier with Eq. 1's duration term off, paired against it on."""

from __future__ import annotations

import copy
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
_CONFIG = {"weights": {"shipping": 0.294, "market": 0.176, "geopolitical": 0.294,
                       "natural_disaster": 0.118, "news_sentiment": 0.118},
           "agents": {"shipping": {"enabled": True, "threshold": 0.65},
                      "geopolitical": {"enabled": False}}}


def test_without_eq1_turns_off_only_the_shipping_duration_term() -> None:
    before = copy.deepcopy(_CONFIG)
    off = rmc._without_eq1(_CONFIG)
    assert _CONFIG == before                                   # not mutated
    assert off["agents"]["shipping"] == {"enabled": True, "threshold": 0.65,
                                         "level_shift": False}
    assert off["agents"]["geopolitical"] == _CONFIG["agents"]["geopolitical"]
    assert off["weights"] == _CONFIG["weights"]
    assert rmc._without_eq1({"agents": None})["agents"]["shipping"] == {"level_shift": False}


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    y = (((ts >= "2023-03-01") & (ts < "2023-05-01"))
         | ((ts >= "2024-06-01") & (ts < "2024-09-01"))).astype(int)
    return pd.DataFrame({"timestamp": ts, "y_true": y,
                         "shipping__vessel_count": 100.0 - 30 * y})


@pytest.fixture
def calls(monkeypatch) -> list[tuple[str, bool]]:
    """Fake agents: shipping reads the label only with Eq. 1 on."""
    seen: list[tuple[str, bool]] = []

    def fake(domain, config, train, test, history=None):
        eq1 = (config["agents"].get("shipping") or {}).get("level_shift", True)
        seen.append((domain, eq1))
        rng = np.random.default_rng(len(test) + rmc._TIER_ORDER.index(domain))
        lab = test["y_true"].to_numpy()
        signal = lab * (1.0 if domain == "shipping" and eq1 else 0.2)
        return signal + rng.normal(0, 0.5, len(test))

    monkeypatch.setattr(rmc, "_agent_scores", fake)
    return seen


def test_each_tier_is_paired_against_itself_with_eq1_on(calls) -> None:
    manifest = {"region": "x", "news_source": {"days_masked": ["2024-07-04"]}, **_NEW_POLICY}
    frame = _frame()
    rows = rmc.no_eq1_ablation("x", frame, manifest, _CONFIG, n_resamples=20)
    main = {r["method"]: r for r in rmc.run_region("x", frame, manifest, _CONFIG,
                                                   n_resamples=20) if r["family"] == "tier"}
    by = {r["method"]: r for r in rows}
    assert set(by) == {f"Tier {k}" for k in range(1, 6)}
    for name, row in by.items():
        assert row["eq1"] == rmc._NO_EQ1
        assert row["auc_with_eq1"] == main[name]["auc"]          # the main table's tier
        assert row["eq1_delta_auc"] == pytest.approx(row["auc"] - row["auc_with_eq1"],
                                                     abs=1e-12)  # without - with
        assert row["eq1_delta_auc_ci_low"] <= row["eq1_delta_auc_ci_high"]
        assert row["circularity"] == "medium"                    # no label statistic
        assert row["agents"] == main[name]["agents"]             # flags still honoured
    assert by["Tier 1"]["eq1_delta_auc"] < 0                     # shipping lost its signal


def test_the_ablation_refuses_a_config_with_eq1_already_off() -> None:
    manifest = {"region": "x", "news_source": {"days_masked": []}, **_NEW_POLICY}
    with pytest.raises(ValueError, match="level_shift"):
        rmc.no_eq1_ablation("x", _frame(), manifest, rmc._without_eq1(_CONFIG), n_resamples=5)


def test_only_the_shipping_agent_is_rescored(calls) -> None:
    manifest = {"region": "x", "news_source": {"days_masked": []}, **_NEW_POLICY}
    rmc.no_eq1_ablation("x", _frame(), manifest, _CONFIG, n_resamples=10)
    shipping = [eq1 for d, eq1 in calls if d == "shipping"]
    assert sorted(shipping) == [False, False, True, True]        # on/off x test/train
    others = [d for d, _ in calls if d != "shipping"]
    assert len(others) == 2 * (len(rmc._TIER_ORDER) - 1)         # once each, reused


def test_the_ablation_matches_the_pre_eq1_aggregation_end_to_end() -> None:
    """The real shipping agent: with the switch off, its scores are the shock score."""
    frame = _frame()
    rng = np.random.default_rng(2)
    frame["shipping__vessel_count"] += rng.normal(0, 3, len(frame))
    frame["shipping__avg_delay_hours"] = rng.gamma(2, 2, len(frame))
    frame["shipping__congestion_index"] = rng.uniform(0, 1, len(frame))
    train, test = rmc._split(frame)
    on = rmc._agent_scores("shipping", _CONFIG, train, test)
    off = rmc._agent_scores("shipping", rmc._without_eq1(_CONFIG), train, test)
    # max(shock, duration) -> shock: never higher, and lower where duration led.
    assert (off <= on + 1e-12).all() and (off < on - 1e-9).any()
    assert roc_auc_score(test["y_true"], off) > 0.5                # still a detector
