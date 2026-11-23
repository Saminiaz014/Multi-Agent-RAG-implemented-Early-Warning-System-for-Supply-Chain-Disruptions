"""The comparison harness: masked days, paired deltas, split sensitivity, ceiling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_method_comparison", _ROOT / "scripts" / "run_method_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)


def _scored(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = ((np.arange(n) // 60) % 3 == 0).astype(int)
    return y, y * 0.8 + rng.normal(0, 0.5, n), rng.normal(0, 0.5, 400)


# ------------------------------------------------------------ masked days
def test_masked_days_are_not_scored_and_not_in_the_threshold() -> None:
    y, s, train = _scored()
    keep = np.ones(len(y), dtype=bool)
    keep[200:218] = False
    train_keep = np.ones(len(train), dtype=bool)
    train_keep[:5] = False
    poisoned, train_poisoned = s.copy(), train.copy()
    poisoned[~keep] = 1e6                      # values on masked days must not matter
    train_poisoned[~train_keep] = 1e6
    a = rmc._evaluate(y, s, train, keep=keep, train_keep=train_keep, n_resamples=50)
    b = rmc._evaluate(y, poisoned, train_poisoned, keep=keep, train_keep=train_keep,
                      n_resamples=50)
    assert a == b
    assert a["auc"] == pytest.approx(roc_auc_score(y[keep], s[keep]))


def test_masked_days_come_from_the_manifest() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-06-12", periods=5, freq="D")})
    manifest = {"region": "x", "news_source": {"days_masked": ["2025-06-14", "2025-06-15"]},
                **_NEW_POLICY}
    assert rmc._masked_days(frame, manifest).tolist() == [False, False, True, True, False]
    assert not rmc._masked_days(frame, {"region": "x", **_NEW_POLICY}).any()   # no news


def test_a_frame_built_with_forward_filled_news_is_refused() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-06-12", periods=3, freq="D")})
    manifest = {"region": "x", "news_source": {"days_forward_filled": ["2025-06-14"]}}
    with pytest.raises(ValueError, match="forward-filled"):
        rmc._masked_days(frame, manifest)


# ------------------------------------------------------ paired delta, tiers
def test_tiers_reuse_one_fit_per_agent(monkeypatch) -> None:
    calls: list[str] = []

    def fake(domain, config, train, test):
        calls.append(domain)
        return np.full(len(test), float(len(domain)))

    monkeypatch.setattr(rmc, "_agent_scores", fake)
    train = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=3)})
    test = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=4)})
    scores = rmc._all_agent_scores({}, train, test)
    assert calls == list(rmc._TIER_ORDER)                  # each agent once
    weights = {"weights": {d: 1.0 for d in rmc._TIER_ORDER}}
    tier2, used = rmc._compose(2, scores, weights)
    assert used == ["shipping", "market"]
    assert tier2.tolist() == [(8 + 6) / 2] * 4


def test_delta_vs_tier1_uses_the_same_resamples_as_the_cis() -> None:
    y, s, _ = _scored()
    other = s + np.random.default_rng(9).normal(0, 0.3, len(s))
    keep = np.ones(len(y), dtype=bool)
    d = rmc._delta(y, other, s, keep=keep, n_resamples=100)
    assert d["delta_auc"] == pytest.approx(roc_auc_score(y, other) - roc_auc_score(y, s))
    assert d["delta_auc_ci_low"] <= d["delta_auc"] <= d["delta_auc_ci_high"]
    assert d["delta_valid"] == rmc._evaluate(y, other, n_resamples=100)["bootstrap_valid"]


# ------------------------------------------------------------- ceiling row
def test_oracle_rows_are_a_ceiling_not_a_baseline() -> None:
    assert rmc._family("ORACLE label t-1", "oracle") == "ceiling"
    assert rmc._family("B1 rolling z", "unsupervised") == "baseline"
    assert rmc._family("M0 random", "control") == "baseline"


# ---------------------------------------------------------- sensitivity
def test_sensitivity_rows_cover_every_split_and_tier(monkeypatch) -> None:
    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    y = ((ts >= "2023-03-01") & (ts < "2023-05-01")) | ((ts >= "2024-06-01") & (ts < "2024-09-01"))
    frame = pd.DataFrame({"timestamp": ts, "y_true": y.astype(int)})
    rng = np.random.default_rng(1)

    def fake(domain, config, train, test):
        lab = test["y_true"].to_numpy()
        return lab * (1.0 if domain == "shipping" else 0.2) + rng.normal(0, 0.5, len(test))

    monkeypatch.setattr(rmc, "_agent_scores", fake)
    rows = rmc.split_sensitivity("x", frame, {"region": "x", **_NEW_POLICY}, {},
                                 n_resamples=30)
    got = {(r["split_date"], r["method"]) for r in rows}
    assert got == {(d, t) for d in ("2023-01-01", "2023-07-01", "2024-01-01")
                   for t in ("Tier 1", "Tier 4")}
    for r in rows:
        assert r["test_start"] == r["split_date"] and r["test_end"] == "2025-11-21"
        if r["method"] == "Tier 4":
            assert {"delta_auc", "delta_auc_ci_low", "delta_auc_ci_high"} <= set(r)


# ------------------------------------------------------ frame policy guard
_NEW_POLICY = {"gap_fill": {"policy": "forward-fill from the last earlier value"},
               "market_source": {"dating": "first-release date (ALFRED)"},
               "domains_present": {"market": ["brent_crude_usd"]}}


def test_frames_without_the_new_gap_and_dating_policy_are_refused() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-06-12", periods=3, freq="D")})
    news = {"news_source": {"days_masked": []}}
    # A --refresh-news frame: new news block, but the old interpolated build.
    with pytest.raises(ValueError, match="gap_fill"):
        rmc._masked_days(frame, {"region": "x", **news,
                                 "domains_present": {"market": ["brent_crude_usd"]}})
    with pytest.raises(ValueError, match="first-release"):
        rmc._masked_days(frame, {"region": "x", **news, **_NEW_POLICY,
                                 "market_source": {"dating": "reference date"}})
    assert not rmc._masked_days(frame, {"region": "x", **news, **_NEW_POLICY}).any()
    no_market = {**_NEW_POLICY, "domains_present": {}, "market_source": None}
    assert not rmc._masked_days(frame, {"region": "x", **news, **no_market}).any()


# ------------------------------------------------------- end to end, fakes
def test_run_region_scores_every_row_on_the_same_days(monkeypatch) -> None:
    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    y = (((ts >= "2023-03-01") & (ts < "2023-05-01"))
         | ((ts >= "2024-06-01") & (ts < "2024-09-01"))).astype(int)
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"timestamp": ts, "y_true": y,
                          "shipping__vessel_count": 100 - 30 * y + rng.normal(0, 5, len(ts)),
                          "news_sentiment__article_volume": rng.uniform(0, 5, len(ts))})
    masked = [str(d.date()) for d in pd.date_range("2025-06-14", "2025-07-01")]
    frame.loc[frame["timestamp"].isin(pd.to_datetime(masked)),
              "news_sentiment__article_volume"] = np.nan
    manifest = {"region": "x", "news_source": {"days_masked": masked}, **_NEW_POLICY}

    def fake(domain, config, train, test):
        lab = test["y_true"].to_numpy()
        return lab * (0.9 if domain == "shipping" else 0.1) + rng.normal(0, 0.5, len(test))

    monkeypatch.setattr(rmc, "_agent_scores", fake)
    rows = rmc.run_region("x", frame, manifest, {}, n_resamples=20)
    scored = [r for r in rows if not r.get("not_applicable")]
    assert {r["n_masked"] for r in scored} == {18}
    assert len({r["n_test"] for r in scored}) == 1
    by = {r["method"]: r for r in rows}
    for k in (2, 3, 4, 5):
        assert by[f"Tier {k}"]["delta_auc"] == pytest.approx(
            by[f"Tier {k}"]["auc"] - by["Tier 1"]["auc"])
    assert by["ORACLE label t-1"]["family"] == "ceiling"
    assert by["B1 rolling z"]["frame_policy"].startswith("news masked")


def test_row_wise_baselines_are_not_fitted_on_masked_days(monkeypatch) -> None:
    seen: list[int] = []
    real = rmc.BASELINES["B3 isolation forest"][0]

    def spy(train, test, features):
        seen.append(len(train))
        return real(train, test, features)

    monkeypatch.setitem(rmc.BASELINES, "B3 isolation forest", (spy, "unsupervised"))
    monkeypatch.setattr(rmc, "_agent_scores", lambda d, c, tr, te: np.zeros(len(te)))
    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    frame = pd.DataFrame({"timestamp": ts, "y_true": (ts.month == 8).astype(int),
                          "shipping__vessel_count": 100.0,
                          "news_sentiment__article_volume": 1.0})
    masked = ["2022-11-10", "2023-03-23"]
    manifest = {"region": "x", "news_source": {"days_masked": masked}, **_NEW_POLICY}
    rmc.run_region("x", frame, manifest, {}, n_resamples=10)
    n_train = int((ts < "2023-07-01").sum())
    assert seen == [n_train - 2]


# ------------------------------------------------------------- alert rule
def test_a_score_equal_to_the_threshold_alerts() -> None:
    """Issue 9: the rule is score >= threshold (a tie counts as an alert)."""
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.5, 0.5, 0.9, 0.2, 0.5])
    m = rmc._score_metrics(y, s, threshold=0.5)
    assert m["alert_rate"] == pytest.approx(4 / 6)              # 0.5, 0.5, 0.9, 0.5
    assert m["f1"] == pytest.approx(2 * 3 / (2 * 3 + 1))         # tp 3, fp 1, fn 0


def test_a_constant_score_alerts_every_day() -> None:
    """A constant equals its own 90th percentile, so >= alerts on every day."""
    y = np.array([0, 1, 0, 1])
    m = rmc._score_metrics(y, np.ones(4), train_scores=np.ones(10))
    assert m["alert_rate"] == 1.0
    # The single-class branch (Malacca: no positives) uses the same rule.
    quiet = rmc._score_metrics(np.zeros(4), np.ones(4), train_scores=np.ones(10))
    assert np.isnan(quiet["auc"]) and quiet["alert_rate"] == 1.0
    assert rmc._evaluate(y.repeat(50), np.ones(200), np.ones(100),
                         n_resamples=20)["alert_rate_ci_low"] == 1.0


# ------------------------------------------------ controls, defined directly
def _control_frame() -> pd.DataFrame:
    """Under 10% training positives, and a positive run across the split.

    With so few training positives the ORACLE's training 90th percentile is
    0, so the threshold rule would alert on every day; only the direct rule
    alerts on exactly the days after a positive one.
    """
    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    y = (((ts >= "2023-03-01") & (ts < "2023-03-21"))
         | ((ts >= "2023-06-20") & (ts < "2023-07-10"))
         | ((ts >= "2024-06-01") & (ts < "2024-09-01"))).astype(int)
    rng = np.random.default_rng(5)
    return pd.DataFrame({"timestamp": ts, "y_true": y,
                         "shipping__vessel_count": 100 - 30 * y + rng.normal(0, 5, len(ts))})


#: A masked test day right after a positive one: it must stay unscored.
_CONTROL_MASKED = ["2024-06-02", "2025-06-14"]


@pytest.fixture(scope="module")
def control_rows() -> dict[str, dict]:
    """One run_region over the control frame, shared by the control tests."""
    manifest = {"region": "x", "news_source": {"days_masked": _CONTROL_MASKED},
                **_NEW_POLICY}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rmc, "_agent_scores", lambda d, c, tr, te: np.zeros(len(te)))
        rows = rmc.run_region("x", _control_frame(), manifest, {}, n_resamples=20)
    return {r["method"]: r for r in rows}


def _test_days(frame: pd.DataFrame):
    train, test = rmc._split(frame)
    yesterday = frame["y_true"].shift(1).fillna(0).astype(int).to_numpy()[len(train):]
    keep = ~test["timestamp"].isin(pd.to_datetime(_CONTROL_MASKED)).to_numpy()
    return train, test, test["y_true"].to_numpy(), yesterday, keep


def test_m2_never_alerts(control_rows) -> None:
    m2 = control_rows["M2 never alert"]
    assert m2["alert_rate"] == 0.0 and m2["fpr"] == 0.0 and m2["f1"] == 0.0
    assert m2["alert_rate_ci_low"] == m2["alert_rate_ci_high"] == 0.0
    assert m2["auc"] == 0.5                       # AUC still comes from the score
    assert m2["alert_rule"].startswith("direct")


def test_m1_alerts_every_day(control_rows) -> None:
    m1 = control_rows["M1 always alert"]
    _, _, y, _, keep = _test_days(_control_frame())
    assert m1["alert_rate"] == 1.0 and m1["fpr"] == 1.0
    assert m1["alert_rate_ci_low"] == m1["alert_rate_ci_high"] == 1.0
    assert m1["f1"] == pytest.approx(f1_score(y[keep], np.ones(keep.sum())))
    assert m1["alert_rule"].startswith("direct")


def test_oracle_alerts_match_yesterdays_label(control_rows) -> None:
    frame = _control_frame()
    train, test, y, yesterday, keep = _test_days(frame)
    assert yesterday[0] == 1                      # its yesterday is a training day
    alerts = rmc._DIRECT_ALERTS["ORACLE label t-1"][1](train, test)
    assert np.array_equal(alerts, yesterday == 1)
    oracle = control_rows["ORACLE label t-1"]
    yk, ak = y[keep], yesterday[keep]
    assert oracle["alert_rate"] == pytest.approx(ak.mean())
    assert oracle["f1"] == pytest.approx(f1_score(yk, ak))
    assert oracle["fpr"] == pytest.approx(ak[yk == 0].mean())
    assert oracle["alert_rule"].startswith("direct")


def test_only_the_controls_skip_the_threshold_rule(control_rows) -> None:
    direct = {m for m, r in control_rows.items() if r["alert_rule"] != rmc._THRESHOLD_RULE}
    assert direct == {"M1 always alert", "M2 never alert", "ORACLE label t-1"}
    assert set(rmc._DIRECT_ALERTS) == direct


def test_direct_alerts_leave_auc_and_its_ci_unchanged() -> None:
    y, s, train = _scored()
    alerts = np.random.default_rng(2).random(len(y)) < 0.3
    by_rule = rmc._evaluate(y, s, train, n_resamples=50)
    direct = rmc._evaluate(y, s, train, alerts=alerts, n_resamples=50)
    for key in ("auc", "auc_ci_low", "auc_ci_high", "bootstrap_valid"):
        assert direct[key] == by_rule[key]
    assert direct["alert_rate"] == pytest.approx(alerts.mean())
    assert direct["f1"] == pytest.approx(f1_score(y, alerts))


def test_direct_alerts_on_a_window_without_positives() -> None:
    """Malacca: no positives, so only the alert rate is defined, from the alerts."""
    quiet = rmc._score_metrics(np.zeros(4), np.ones(4), alerts=np.zeros(4, dtype=bool))
    assert np.isnan(quiet["auc"]) and quiet["alert_rate"] == 0.0


def test_alerts_and_a_threshold_are_not_both_accepted() -> None:
    y, s = np.array([0, 1, 0, 1]), np.arange(4.0)
    with pytest.raises(ValueError, match="not both"):
        rmc._score_metrics(y, s, threshold=1.0, alerts=s > 1)
    with pytest.raises(ValueError, match="not both"):
        rmc._score_metrics(y, s, train_scores=s, alerts=s > 1)


# ------------------------------------------------ per-day alerts (shared by the figures)
def test_alert_flags_are_the_rule_the_reported_alert_rate_uses() -> None:
    y, s, train = _scored()
    keep = np.ones(len(y), dtype=bool)
    keep[200:218] = False                                   # a masked outage
    train_keep = np.ones(len(train), dtype=bool)
    train_keep[:5] = False
    flags = rmc.alert_flags(s, train, train_keep=train_keep)
    reported = rmc._evaluate(y, s, train, keep=keep, train_keep=train_keep, n_resamples=20)
    assert flags.dtype == bool and len(flags) == len(s)
    assert flags[keep].mean() == pytest.approx(reported["alert_rate"], abs=1e-12)


def test_alert_flags_count_a_tie_with_the_threshold_as_an_alert() -> None:
    assert rmc.alert_flags(np.ones(5), np.ones(20)).all()    # >= : a constant alerts every day


def test_alert_flags_take_a_controls_alerts_as_given_and_nan_scores_as_zero() -> None:
    given = np.array([True, False, True])
    assert rmc.alert_flags(np.zeros(3), np.ones(10), direct=given).tolist() == [True, False, True]
    assert not rmc.alert_flags(np.array([np.nan, 0.9]), np.linspace(0, 1, 50))[0]   # nan -> 0
