"""Issue 5: ``agents.shipping.level_shift`` turns Eq. 1's duration term off.

Off restores the pre-f7b87ff aggregation, ``combined = w_if * iforest +
w_z * max_z`` (the ``shock`` score), and changes nothing else; on (the
default) is ``max(shock, duration)``, as now.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.shipping_agent import ShippingAgent


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    ts = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    drop = ((ts >= "2024-03-01") & (ts < "2024-09-01")).astype(float)   # sustained
    vessels = 100 - 40 * drop + rng.normal(0, 4, len(ts))
    return pd.DataFrame({"timestamp": ts, "vessel_count": vessels,
                         "avg_delay_hours": rng.gamma(2, 2, len(ts)),
                         "congestion_index": rng.uniform(0, 1, len(ts))})


def _detect(**config) -> pd.DataFrame:
    frame = _frame()
    agent = ShippingAgent(config=config)
    agent.fit(frame[frame["timestamp"] < "2023-07-01"])
    return agent.detect(agent.preprocess(frame))


def test_off_restores_the_shock_only_aggregation() -> None:
    out = _detect(level_shift=False)
    w_if, w_z = 0.70, 0.30                               # the hand-tuned blend
    pre_f7b87ff = w_if * out["isolation_score"] + w_z * out["max_zscore_norm"]
    assert np.array_equal(out["anomaly_score"].to_numpy(), out["shock_score"].to_numpy())
    np.testing.assert_allclose(out["anomaly_score"], pre_f7b87ff, rtol=0, atol=1e-15)
    assert (out["duration_score"] == 0.0).all()
    assert np.array_equal(out["is_anomaly"].to_numpy(),
                          (out["shock_score"] >= 0.65).to_numpy())


@pytest.mark.parametrize("config", [{}, {"level_shift": True}])
def test_on_is_the_current_eq1_and_the_default(config) -> None:
    out = _detect(**config)
    expected = np.maximum(out["shock_score"], out["duration_score"])
    assert np.array_equal(out["anomaly_score"].to_numpy(), expected.to_numpy())
    assert (out["duration_score"] > out["shock_score"]).any()   # the switch matters here


def test_the_switch_changes_only_the_duration_term() -> None:
    on, off = _detect(), _detect(level_shift=False)
    moved = {c for c in on.columns
             if not np.array_equal(on[c].to_numpy(), off[c].to_numpy())}
    assert moved <= {"duration_score", "anomaly_score", "is_anomaly"}
    assert "duration_score" in moved and "anomaly_score" in moved
