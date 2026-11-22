"""Issue 4: an agent a region disables is not run and never fused into a tier."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_method_comparison", _ROOT / "scripts" / "run_method_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)

#: The weights every region config carries (settings.yaml).
_WEIGHTS = {"shipping": 0.294, "market": 0.176, "geopolitical": 0.294,
            "natural_disaster": 0.118, "news_sentiment": 0.118}
_TIERS = range(1, len(rmc._TIER_ORDER) + 1)


def _config(*disabled: str) -> dict:
    return {"weights": dict(_WEIGHTS),
            "agents": {d: {"enabled": d not in disabled} for d in rmc._TIER_ORDER}}


def _scores(n: int = 6) -> dict[str, np.ndarray]:
    return {d: np.full(n, float(i + 1)) for i, d in enumerate(rmc._TIER_ORDER)}


def _expected(tier: int, scores: dict, disabled: tuple[str, ...]):
    enabled = [d for d in rmc._TIER_ORDER[:tier] if d not in disabled]
    if not enabled:
        return None
    mass = sum(_WEIGHTS[d] for d in enabled)
    return sum(_WEIGHTS[d] * scores[d] for d in enabled) / mass


# ------------------------------------------------------------------ fusion
@pytest.mark.parametrize("disabled", rmc._TIER_ORDER)
def test_a_disabled_agent_never_contributes_to_a_tier_score(disabled) -> None:
    scores = _scores()
    scores[disabled] = np.full(6, 1e6)              # a leak would dominate
    for tier in _TIERS:
        composite, used = rmc._compose(tier, scores, _config(disabled))
        assert disabled not in used
        expected = _expected(tier, scores, (disabled,))
        if expected is None:
            assert composite is None and used == []
        else:
            np.testing.assert_allclose(composite, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("disabled", [(), ("natural_disaster",), ("market",),
                                      ("geopolitical",), ("market", "news_sentiment")])
def test_fusion_weights_sum_to_one_over_the_enabled_agents(disabled) -> None:
    for tier in _TIERS:
        weights = rmc._fusion_weights(tier, _scores(), _config(*disabled))
        enabled = [d for d in rmc._TIER_ORDER[:tier] if d not in disabled]
        assert list(weights) == enabled
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
        mass = sum(_WEIGHTS[d] for d in enabled)
        for d in enabled:                           # renormalised, ratios kept
            assert weights[d] == pytest.approx(_WEIGHTS[d] / mass, abs=1e-12)


def test_a_tier_whose_added_agent_is_disabled_equals_the_previous_tier() -> None:
    config = _config("geopolitical")                # Panama
    tier2, used2 = rmc._compose(2, _scores(), config)
    tier3, used3 = rmc._compose(3, _scores(), config)
    assert used3 == used2 == ["shipping", "market"]
    assert np.array_equal(tier3, tier2)


def test_agents_without_a_flag_are_enabled() -> None:
    """The orchestrator's convention: ``enabled`` defaults to true."""
    composite, used = rmc._compose(5, _scores(), {"weights": dict(_WEIGHTS)})
    assert used == list(rmc._TIER_ORDER)
    np.testing.assert_allclose(composite, _expected(5, _scores(), ()), atol=1e-12)


# ---------------------------------------------------- the paths that use it
def _frames():
    ts = pd.date_range("2022-01-01", periods=400, freq="D")
    frame = pd.DataFrame({"timestamp": ts,
                          "geopolitical__conflict_intensity": np.linspace(0, 1, 400)})
    return frame.iloc[:300], frame.iloc[300:]


def test_tier_scores_skips_a_disabled_agent(monkeypatch) -> None:
    # A fake that ignores the config: only the fusion can keep it out.
    monkeypatch.setattr(rmc, "_agent_scores", lambda d, c, tr, te, history=None:
                        np.full(len(te), 1e6 if d == "geopolitical" else 1.0))
    train, test = _frames()
    scores, used = rmc.tier_scores(5, _config("geopolitical"), train, test)
    assert "geopolitical" not in used
    np.testing.assert_allclose(scores, 1.0, atol=1e-12)


def test_a_disabled_agent_is_not_built_or_run(monkeypatch) -> None:
    """Single-agent runs and diagnostics call _agent_scores directly."""
    from src.agents import geopolitical_agent

    built: list[int] = []
    real_init = geopolitical_agent.GeopoliticalAgent.__init__

    def spy(self, *args, **kwargs):
        built.append(1)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(geopolitical_agent.GeopoliticalAgent, "__init__", spy)
    train, test = _frames()
    assert rmc._agent_scores("geopolitical", _config("geopolitical"), train, test) is None
    assert built == []
    rmc._agent_scores("geopolitical", _config(), train, test)    # enabled: built
    assert built == [1]


# ------------------------------------------------- rows and the sensitivity
_NEW_POLICY = {"gap_fill": {"policy": "forward-fill from the last earlier value"},
               "market_source": {"dating": "first-release date (ALFRED)"},
               "domains_present": {"market": ["brent_crude_usd"]}}


def _region_frame() -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    y = (((ts >= "2023-03-01") & (ts < "2023-05-01"))
         | ((ts >= "2024-06-01") & (ts < "2024-09-01"))).astype(int)
    rng = np.random.default_rng(3)
    return pd.DataFrame({"timestamp": ts, "y_true": y,
                         "shipping__vessel_count": 100 - 30 * y + rng.normal(0, 5, len(ts))})


def _fake_agents(monkeypatch) -> None:
    rng = np.random.default_rng(8)

    def fake(domain, config, train, test, history=None):
        lab = test["y_true"].to_numpy()
        return lab * (0.5 + 0.1 * rmc._TIER_ORDER.index(domain)) + rng.normal(0, 0.5, len(test))

    monkeypatch.setattr(rmc, "_agent_scores", fake)


def test_tier_rows_name_their_enabled_and_disabled_agents(monkeypatch) -> None:
    _fake_agents(monkeypatch)
    manifest = {"region": "x", "news_source": {"days_masked": []}, **_NEW_POLICY}
    rows = rmc.run_region("x", _region_frame(), manifest, _config("geopolitical"),
                          n_resamples=10)
    tiers = {r["method"]: r for r in rows if r["family"] == "tier"}
    assert tiers["Tier 1"]["agents_disabled"] == ""
    assert tiers["Tier 3"]["agents"] == "shipping,market"
    assert tiers["Tier 3"]["agents_disabled"] == "geopolitical"
    assert tiers["Tier 3"]["n_agents"] == 2
    assert tiers["Tier 3"]["auc"] == tiers["Tier 2"]["auc"]
    assert tiers["Tier 5"]["agents"] == "shipping,market,natural_disaster,news_sentiment"


def test_the_all_agents_sensitivity_enables_every_agent(monkeypatch) -> None:
    _fake_agents(monkeypatch)
    config = _config("geopolitical")
    before = copy.deepcopy(config)
    manifest = {"region": "x", "news_source": {"days_masked": []}, **_NEW_POLICY}
    rows = rmc.all_agents_sensitivity("x", _region_frame(), manifest, config, n_resamples=10)
    assert config == before                          # the region config is not changed
    by = {r["method"]: r for r in rows}
    assert set(by) == {f"Tier {k}" for k in _TIERS}
    assert by["Tier 3"]["agents"] == "shipping,market,geopolitical"
    assert by["Tier 3"]["agents_disabled"] == ""
    assert {r["agent_flags"] for r in rows} == {"all agents enabled (sensitivity)"}
    assert "delta_auc" in by["Tier 3"]


# ----------------------------------------------------------- review fixes
def test_an_unchanged_tier_keeps_its_exact_earlier_value() -> None:
    """Weights added one at a time, as before issue 4.

    Python 3.12+ ``sum()`` is compensated: for these four weights it gives
    0.8819999999999999, not 0.882, which would move an unchanged Tier 4.
    """
    rng = np.random.default_rng(0)
    scores = {d: rng.random(50) for d in rmc._TIER_ORDER}
    for tier in _TIERS:
        total, mass = None, 0.0
        for d in rmc._TIER_ORDER[:tier]:
            w = _WEIGHTS[d]
            total = w * scores[d] if total is None else total + w * scores[d]
            mass += w
        assert np.array_equal(rmc._compose(tier, scores, _config())[0], total / mass)


def test_tier_rows_record_their_fusion_weights(monkeypatch) -> None:
    _fake_agents(monkeypatch)
    manifest = {"region": "x", "news_source": {"days_masked": []}, **_NEW_POLICY}
    rows = rmc.run_region("x", _region_frame(), manifest, _config("geopolitical"),
                          n_resamples=10)
    tier3 = next(r for r in rows if r["method"] == "Tier 3")
    weights = dict(item.split("=") for item in tier3["fusion_weights"].split(","))
    assert list(weights) == ["shipping", "market"]
    assert sum(map(float, weights.values())) == pytest.approx(1.0, abs=1e-3)
    assert float(weights["shipping"]) == pytest.approx(0.294 / 0.470, abs=1e-4)


def test_the_all_agents_rows_name_the_agents_they_force_on(monkeypatch) -> None:
    _fake_agents(monkeypatch)
    manifest = {"region": "x", "news_source": {"days_masked": []}, **_NEW_POLICY}
    rows = rmc.all_agents_sensitivity("x", _region_frame(), manifest,
                                      _config("geopolitical"), n_resamples=10)
    forced = {r["method"]: r["agents_forced_on"] for r in rows}
    assert forced["Tier 2"] == "" and forced["Tier 3"] == forced["Tier 5"] == "geopolitical"


def test_all_enabled_accepts_an_empty_agents_block() -> None:
    out = rmc._all_enabled({"agents": None, "weights": dict(_WEIGHTS)})
    assert all(rmc._enabled(out, d) for d in rmc._TIER_ORDER)
