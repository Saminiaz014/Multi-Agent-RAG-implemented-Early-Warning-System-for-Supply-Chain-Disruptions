"""Agents are scored with the same history the baselines get (issue 11).

Each agent is fitted on the training window only, then scored on the training
days plus the test window, and only test-window rows are kept. A rolling
window or trailing baseline therefore never restarts at the split: with the
fitted parameters held fixed, a day's score does not depend on where the test
window starts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.config_manager import load_config_for_region

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_method_comparison", _ROOT / "scripts" / "run_method_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)


def _frame() -> pd.DataFrame:
    """Every agent's columns, 2021-01 .. 2025-11, with a sustained drop in 2024."""
    rng = np.random.default_rng(7)
    ts = pd.date_range("2021-01-01", "2025-11-21", freq="D")
    n = len(ts)
    drop = ((ts >= "2024-01-01") & (ts < "2024-09-01")).astype(float)
    vessels = 100 - 40 * drop + rng.normal(0, 4, n)
    volume = np.where(rng.uniform(size=n) < 0.5, 0.0, rng.uniform(1, 20, n))
    tone = np.where(volume > 0, rng.uniform(-0.6, 0.2, n), 0.0)
    return pd.DataFrame({
        "timestamp": ts, "y_true": drop.astype(int),
        "shipping__vessel_count": vessels,
        "shipping__tanker_count": vessels * 0.3 + rng.normal(0, 1, n),
        "shipping__avg_delay_hours": rng.gamma(2, 2, n),
        "shipping__congestion_index": rng.uniform(0, 1, n),
        "market__brent_crude_usd": 80 + np.cumsum(rng.normal(0, 1, n)) * 0.2,
        "market__trade_volume_index": rng.uniform(0.3, 0.9, n),
        "market__freight_rate_index": np.repeat(rng.normal(100, 5, n // 30 + 1), 30)[:n],
        "natural_disaster__earthquake_severity": rng.exponential(0.1, n),
        "natural_disaster__tsunami_risk": 0.0,
        "natural_disaster__cyclone_severity": (rng.uniform(size=n) < 0.02) * 0.5,
        "natural_disaster__severe_weather_index": 0.0,
        "geopolitical__military_activity_index": rng.uniform(0, 1, n),
        "geopolitical__diplomatic_incident_score": rng.integers(0, 3, n).astype(float),
        "geopolitical__regime_stability_index": rng.uniform(0.4, 0.8, n),
        "news_sentiment__sentiment_score": tone,
        "news_sentiment__sentiment_magnitude": np.abs(tone),
        "news_sentiment__article_volume": volume,
        "news_sentiment__recency_weighted_score": pd.Series(tone).rolling(7, min_periods=1).mean(),
    })


@pytest.mark.parametrize("domain", rmc._TIER_ORDER)
def test_a_days_score_does_not_depend_on_where_the_test_window_starts(domain) -> None:
    frame = _frame()
    config = load_config_for_region("hormuz")
    train = frame[frame.timestamp < "2023-07-01"]           # fitted parameters fixed
    history = frame
    early = frame[frame.timestamp >= "2023-07-01"]
    late = frame[frame.timestamp >= "2024-01-01"]
    a = rmc._agent_scores(domain, config, train, early, history=history)
    b = rmc._agent_scores(domain, config, train, late, history=history)
    assert a is not None and b is not None, domain
    shared = pd.Series(a, index=early.timestamp.to_numpy()).loc["2024-01-01":]
    np.testing.assert_allclose(shared.to_numpy(), b, atol=1e-9, rtol=0, err_msg=domain)


# ------------------------------------------- the harness's default path
_STATELESS_FIT = ("market", "geopolitical", "natural_disaster", "news_sentiment")


@pytest.mark.parametrize("domain", _STATELESS_FIT)
def test_default_path_scores_do_not_depend_on_the_split(domain) -> None:
    """Agents whose fit learns nothing from the window: whole-split invariance.

    Each side is the harness's own call (no history passed). Before issue 11
    was fixed, the test window was scored alone and the market and news
    warm-ups restarted at the split (differences up to 0.59 and 0.19).
    """
    frame = _frame()
    config = load_config_for_region("hormuz")
    a_train, a_test = (frame[frame.timestamp < "2023-07-01"],
                       frame[frame.timestamp >= "2023-07-01"])
    b_train, b_test = (frame[frame.timestamp < "2024-01-01"],
                       frame[frame.timestamp >= "2024-01-01"])
    a = rmc._agent_scores(domain, config, a_train, a_test)
    b = rmc._agent_scores(domain, config, b_train, b_test)
    shared = pd.Series(a, index=a_test.timestamp.to_numpy()).loc["2024-01-01":]
    np.testing.assert_allclose(shared.to_numpy(), b, atol=1e-9, rtol=0, err_msg=domain)


def _moderate_drop_frame() -> pd.DataFrame:
    """A 25% drop: enough for the duration score, too small to saturate shock."""
    frame = _frame()
    rng = np.random.default_rng(11)
    drop = ((frame.timestamp >= "2024-01-01") & (frame.timestamp < "2024-09-01"))
    frame["shipping__vessel_count"] = 100 - 25 * drop + rng.normal(0, 4, len(frame))
    return frame


def test_shipping_duration_score_is_live_from_the_first_test_day(monkeypatch) -> None:
    """The case behind issue 11: a test window that starts at a sustained drop.

    The duration score (30-day mean vs a lagged 365-day median) needs the
    year before the split. Scored on the test window alone it was 0 for
    months; on train + test it is live at once.
    """
    from src.agents.shipping_agent import ShippingAgent
    captured: list[pd.DataFrame] = []
    real_detect = ShippingAgent.detect

    def spy(self, data):
        out = real_detect(self, data)
        captured.append(out)
        return out

    monkeypatch.setattr(ShippingAgent, "detect", spy)
    frame = _moderate_drop_frame()
    train = frame[frame.timestamp < "2024-01-01"]
    test = frame[frame.timestamp >= "2024-01-01"]
    rmc._agent_scores("shipping", load_config_for_region("hormuz"), train, test)
    scored = captured[-1].set_index("timestamp")
    # A 25% drop scores 0.625 at full persistence (40% = 1.0), ramping in
    # over the 30-day mean; scored on the test window alone it was 0.0.
    assert scored.loc["2024-01-15":"2024-04-30", "duration_score"].mean() > 0.4
    assert scored.loc["2024-03-01":"2024-04-30", "duration_score"].min() > 0.5


def test_default_history_is_train_followed_by_test() -> None:
    frame = _frame()
    config = load_config_for_region("hormuz")
    train = frame[frame.timestamp < "2023-07-01"]
    test = frame[frame.timestamp >= "2023-07-01"]
    default = rmc._agent_scores("shipping", config, train, test)
    explicit = rmc._agent_scores("shipping", config, train, test,
                                 history=pd.concat([train, test]))
    np.testing.assert_array_equal(default, explicit)
    assert len(default) == len(test)


@pytest.mark.parametrize("domain", rmc._TIER_ORDER)
def test_scoring_train_against_itself_is_unchanged(domain) -> None:
    """The training-threshold reference: same as scoring train alone, no doubled days."""
    frame = _frame()
    config = load_config_for_region("hormuz")
    train = frame[frame.timestamp < "2023-07-01"]
    scores = rmc._agent_scores(domain, config, train, train)
    alone = rmc._agent_scores(domain, config, train, train, history=train)
    assert scores is not None and len(scores) == len(train)
    np.testing.assert_array_equal(scores, alone)


def test_refit_happens_on_train_only(monkeypatch) -> None:
    from src.agents.shipping_agent import ShippingAgent
    seen: list[pd.Timestamp] = []
    real_fit = ShippingAgent.fit

    def spy(self, df):
        seen.append(pd.to_datetime(df["timestamp"]).max())
        return real_fit(self, df)

    monkeypatch.setattr(ShippingAgent, "fit", spy)
    frame = _frame()
    train = frame[frame.timestamp < "2023-07-01"]
    rmc._agent_scores("shipping", load_config_for_region("hormuz"), train,
                      frame[frame.timestamp >= "2023-07-01"])
    assert seen == [pd.Timestamp("2023-06-30")]


# ------------------------------------------------- history override guard
def test_history_missing_test_days_is_refused() -> None:
    frame = _frame()
    train = frame[frame.timestamp < "2023-07-01"]
    test = frame[frame.timestamp >= "2023-07-01"]
    holed = frame[~frame.timestamp.between("2024-03-01", "2024-03-31")]
    with pytest.raises(ValueError, match="test days"):
        rmc._agent_scores("market", load_config_for_region("hormuz"), train, test,
                          history=holed)


def test_history_running_past_the_test_window_is_refused() -> None:
    frame = _frame()
    train = frame[frame.timestamp < "2023-07-01"]
    test = frame[frame.timestamp.between("2023-07-01", "2024-12-31")]
    with pytest.raises(ValueError, match="after"):
        rmc._agent_scores("market", load_config_for_region("hormuz"), train, test,
                          history=frame)
