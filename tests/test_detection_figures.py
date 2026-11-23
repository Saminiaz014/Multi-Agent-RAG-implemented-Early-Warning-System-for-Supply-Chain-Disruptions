"""The lead-time analysis uses the harness's alerts, controls included, and says what it shows."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                 # @dataclass resolves its module by name
    spec.loader.exec_module(module)
    return module


rmc = _load("run_method_comparison", "scripts/run_method_comparison.py")
det = _load("detection_figures", "eval/ch4/detection_figures.py")

_WEIGHTS = {"shipping": 0.294, "market": 0.176, "geopolitical": 0.294,
            "natural_disaster": 0.118, "news_sentiment": 0.118}


def _context(seed: int = 0, train_positives: bool = True) -> "det.RegionContext":
    ts = pd.date_range("2022-01-01", "2025-11-21", freq="D")
    y = ((ts >= "2023-03-01") & (ts < "2023-05-01")) if train_positives else np.zeros(len(ts), bool)
    y = y | ((ts >= "2023-09-01") & (ts < "2023-10-15")) | ((ts >= "2024-06-01") & (ts < "2024-08-01"))
    frame = pd.DataFrame({"timestamp": ts, "y_true": y.astype(int),
                          "shipping__vessel_count": 100.0 - 20 * y})
    if not train_positives:
        frame.loc[frame["timestamp"] < "2023-07-01", "y_true"] = 0
    train, test = rmc._split(frame)
    keep = np.ones(len(test), dtype=bool)
    keep[400:418] = False                                   # a masked outage in the test window
    train_keep = np.ones(len(train), dtype=bool)
    train_keep[:200] = False                                # masked training days ...
    rng = np.random.default_rng(seed)
    on_train = {d: rng.uniform(0, 1, len(train)) for d in rmc._TIER_ORDER}
    for scores in on_train.values():
        scores[:200] = 5.0                                  # ... with extreme scores: they must not
    return det.RegionContext(                               # set the threshold
        region="x", train=train, test=test, keep=keep, train_keep=train_keep,
        y=test["y_true"].to_numpy(), config={"weights": dict(_WEIGHTS), "agents": {}},
        on_test={d: rng.uniform(0, 1, len(test)) for d in rmc._TIER_ORDER},
        on_train=on_train, features=["shipping__vessel_count"])


# ------------------------------------------------------------- lead time
def test_onsets_are_run_starts_and_skip_a_run_open_on_day_zero() -> None:
    assert det.onsets(np.array([1, 1, 0, 0, 1, 1, 0, 1])) == [4, 7]
    assert det.onsets(np.array([0, 0, 1, 1, 0])) == [2]
    assert det.onsets(np.zeros(5)) == []


@pytest.mark.parametrize("alert_at, expected", [(35, (5.0, False, False)),      # before onset
                                                (45, (-5.0, False, False)),     # after onset
                                                (10, (30.0, True, False)),      # active at window start
                                                (None, (np.nan, False, True))])
def test_lead_for_onset(alert_at, expected) -> None:
    alert = np.zeros(100, dtype=bool)
    if alert_at is not None:
        alert[alert_at] = True
    lead, censored, missed = det.lead_for_onset(alert, 40)
    assert (censored, missed) == expected[1:]
    assert (np.isnan(lead) and np.isnan(expected[0])) or lead == expected[0]


def test_the_window_is_clipped_at_the_series_start() -> None:
    alert = np.zeros(100, dtype=bool)
    alert[0] = True                                          # first test day
    assert det.lead_for_onset(alert, 10)[0] == 10.0
    assert det.lead_for_onset(alert, 10)[1] is False         # not censored: window starts at 0


# ------------------------------------------------- alerts, from the harness
def test_a_tiers_alerts_are_the_rule_behind_its_reported_alert_rate() -> None:
    ctx = _context()
    flags = det.method_alerts("Tier 3", ctx, rmc)
    scores, _ = rmc._compose(3, ctx.on_test, ctx.config)
    ref, _ = rmc._compose(3, ctx.on_train, ctx.config)
    reported = rmc._evaluate(ctx.y, scores, ref, keep=ctx.keep, train_keep=ctx.train_keep,
                             n_resamples=20)
    assert flags[ctx.keep].mean() == pytest.approx(reported["alert_rate"], abs=1e-12)


def test_the_random_control_follows_the_harness_baseline_path() -> None:
    ctx = _context()
    flags = det.method_alerts("M0 random", ctx, rmc)
    scores = rmc.m0_random(ctx.train, ctx.test, ctx.features)
    reported = rmc._evaluate(ctx.y, scores[-len(ctx.test):], scores[:len(ctx.train)],
                             keep=ctx.keep, train_keep=ctx.train_keep, n_resamples=20)
    assert flags[ctx.keep].mean() == pytest.approx(reported["alert_rate"], abs=1e-12)
    assert 0.05 < flags[ctx.keep].mean() < 0.15              # about 10 %


def test_masked_days_never_alert() -> None:
    ctx = _context()
    for name in ("Tier 1", "M0 random", "M1 always alert"):
        assert not det.method_alerts(name, ctx, rmc)[~ctx.keep].any(), name


def test_controls_are_taken_as_defined_not_through_the_threshold_rule() -> None:
    ctx = _context()
    assert not det.method_alerts("M2 never alert", ctx, rmc).any()
    assert det.method_alerts("M1 always alert", ctx, rmc)[ctx.keep].all()
    full = pd.concat([ctx.train, ctx.test])["y_true"].shift(1).fillna(0).to_numpy()[-len(ctx.test):]
    oracle = det.method_alerts("ORACLE label t-1", ctx, rmc)
    assert np.array_equal(oracle, (full == 1) & ctx.keep)


def test_a_method_that_cannot_be_trained_has_no_alerts() -> None:
    ctx = _context(train_positives=False)
    assert det.method_alerts("B7 logistic regression", ctx, rmc) is None


def test_lead_times_cover_every_onset_for_every_method() -> None:
    ctx = _context()
    lt = det.lead_times(regions=("x",), rmc=rmc, contexts={"x": ctx})
    onsets_found = len(det.onsets(ctx.y))
    assert onsets_found == 2
    assert set(lt["method"]) == set(det.METHODS)
    assert (lt.groupby("method").size() == onsets_found).all()
    assert lt.loc[lt["method"].str.startswith("Tier"), "agents"].str.len().min() > 0
    assert set(lt["alert_rule"]) == {rmc._THRESHOLD_RULE}


# ----------------------------------------------------------------- the figure
def _table(lead: float = 12.0, region_onsets: dict | None = None,
           rates: dict | None = None) -> pd.DataFrame:
    tiers = {"Tier 1": "shipping", "Tier 2": "shipping,market", "Tier 3": "shipping,market",
             "Tier 4": "shipping,market,natural_disaster", "Tier 5": "shipping,market,natural_disaster"}
    rows = []
    for region, dates in (region_onsets or {"hormuz": ["2023-12-12", "2024-01-27"],
                                            "panama": ["2024-01-15"]}).items():
        for method in det.METHODS:
            for k, onset in enumerate(dates):
                rows.append({"region": region, "method": method, "agents": tiers.get(method, ""),
                             "onset": onset, "lead_days": lead + k, "censored_at_window": False,
                             "missed": False, "alert_rate": (rates or {}).get((region, method), 0.10),
                             "alert_rule": "score >= training p90"})
    return pd.DataFrame(rows)


def test_the_headline_is_true_of_the_data() -> None:
    assert det.headline(_table(12.0)) == "Every method 'leads' the label, including the random control"
    worse = _table(12.0)
    worse.loc[0, "lead_days"] = -3.0
    assert det.headline(worse).startswith("Most methods")


def test_the_caption_states_why_every_method_leads() -> None:
    text = det.caption(_table())
    assert "Every method, including the random control, 'leads' the label" in text
    assert "lagged 30-day mean (its onset comes after the physical drop)" in text
    assert "96%" in text                                   # 1 - 0.9^30 for a 10 % alerter
    assert "not for the paper" in text.lower() and "not evidence of early warning" in text
    assert "Tier 3 (= Tier 2)" in text and "Tier 5 (= Tier 4)" in text      # copies named
    assert "Hormuz, Tier 1: 2023-12-12 +12 d" in text                        # every bar's value


def test_the_caption_gives_every_methods_alert_rate_and_the_chance_reference_honestly() -> None:
    rates = {("hormuz", "Tier 1"): 0.17, ("hormuz", "M0 random"): 0.104,
             ("panama", "Tier 5"): 0.79}
    text = det.caption(_table(rates=rates))
    assert "Alert rate (share of test days)" in text
    assert "Hormuz: Tier 1 17%" in text and "M0 random 10%" in text and "Panama: " in text
    # the highest rate is named, with where it is
    assert "highest alert rate is 79% (Panama, Tier 5)" in text
    # the chance arithmetic is conditional, about the 30 days BEFORE an onset, and a ceiling
    assert "if its alerts fell independently across days" in text
    assert "in the 30 days before an onset" in text
    assert "autocorrelated" in text and "upper reference" in text
    assert "96%" in text                                   # 1 - 0.896^30 at the control's rate


def test_the_caption_does_not_claim_every_method_leads_when_one_does_not() -> None:
    table = _table()
    table.loc[(table["method"] == "Tier 2") & (table["region"] == "hormuz"), "lead_days"] = -4.0
    table.loc[table["method"] == "M0 random", "missed"] = True
    table.loc[table["method"] == "M0 random", "lead_days"] = float("nan")
    text = det.caption(table)
    assert "Every method, including the random control" not in text
    n = int((table["lead_days"] > 0).sum())
    assert f"{n} of {len(table)} method-onsets 'lead' the label" in text
    assert "the random control leads 0 of 3" in text


def test_the_harness_is_registered_by_name_so_its_dataclasses_resolve() -> None:
    module = det.harness()
    assert sys.modules["run_method_comparison"] is module


def test_the_figure_writes_pdf_png_and_its_caption(tmp_path) -> None:
    lt = _table()
    paths = det.fig_lead(lt, tmp_path)
    assert {p.suffix for p in paths} == {".pdf", ".png", ".txt"} and all(p.exists() for p in paths)
    assert (tmp_path / f"{det.FIGURE_NAME}.caption.txt").read_text(encoding="utf-8") == det.caption(lt)


def test_the_figure_has_one_bar_per_method_and_onset_and_a_hatched_random_control(
        tmp_path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    seen = []
    real_close = plt.close
    monkeypatch.setattr(det.plt, "close", lambda fig=None: (seen.append(fig), real_close(fig)))
    det.fig_lead(_table(), tmp_path)
    fig = seen[0]
    hormuz, panama = fig.axes[0], fig.axes[1]
    assert len(hormuz.patches) == 6 * 2 and len(panama.patches) == 6 * 1
    hatched = [p for p in hormuz.patches if p.get_hatch()]
    assert len(hatched) == 2                                # M0 random, one per onset
    assert len(fig.legends[0].get_texts()) == 6
    assert fig._suptitle.get_text().startswith("Every method")


def test_the_committed_lead_time_file_covers_every_region_and_method() -> None:
    path = _ROOT / "eval" / "lead_time.csv"
    if not path.exists():
        pytest.skip("eval/lead_time.csv not generated yet")
    lt = pd.read_csv(path)
    for region in det.REGIONS:
        sub = lt[lt["region"] == region]
        assert set(sub["method"]) == set(det.METHODS), region
        assert sub.groupby("method")["onset"].nunique().nunique() == 1       # same onsets per method
    assert (lt["alert_rule"] == "score >= training p90").all()
