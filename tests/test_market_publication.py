"""FRED values enter the market frame on the day they were first published.

A monthly PPI stamped on the first of its reference month is only released
about six weeks later, so carrying it forward from the reference date gives a
detector data it could not yet have had. The connector re-stamps every FRED
column on its ALFRED first-release date instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion import market_connector as mod
from src.ingestion.market_connector import MarketConnector, as_published


def _frame(days: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=days, freq="D"),
                         "x": np.arange(days, dtype=float)})


def test_a_value_appears_on_its_release_day_not_before() -> None:
    f = _frame()
    out = as_published(f, {"x": f["timestamp"] + pd.Timedelta(days=2)})
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-03")
    # On day t the latest published row is t-2.
    assert out.set_index("timestamp")["x"].loc["2024-01-03":"2024-01-11"].tolist() == \
        list(np.arange(0, 9, dtype=float))


def test_a_row_published_after_a_later_one_never_reappears() -> None:
    f = _frame(4)
    avail = pd.to_datetime(["2024-01-09", "2024-01-03", "2024-01-05", "2024-01-06"])
    out = as_published(f, {"x": pd.Series(avail)}).set_index("timestamp")["x"]
    # Row 0 (published 01-09) is superseded by row 1 (published 01-03) and
    # must never show up; values never step back in observation time.
    assert 0.0 not in out.tolist()
    assert out.loc["2024-01-03"] == 1.0 and out.loc["2024-01-05"] == 2.0
    assert out.loc["2024-01-06":].eq(3.0).all()


def test_availability_is_never_before_the_day_itself() -> None:
    f = _frame(5)
    out = as_published(f, {"x": f["timestamp"] - pd.Timedelta(days=3)})   # bad metadata
    merged = out.merge(f, on="x", suffixes=("_pub", "_obs"))
    assert (merged["timestamp_pub"] >= merged["timestamp_obs"]).all()


def test_columns_keep_their_own_release_schedules() -> None:
    f = _frame(10).assign(y=lambda d: d["x"] * 10)
    out = as_published(f, {"x": f["timestamp"] + pd.Timedelta(days=1),
                           "y": f["timestamp"] + pd.Timedelta(days=4)}).set_index("timestamp")
    # The frame starts once every column has a value.
    assert out.index[0] == pd.Timestamp("2024-01-05")
    assert out.loc["2024-01-05", "x"] == 3.0 and out.loc["2024-01-05", "y"] == 0.0


# ------------------------------------------------------------ fetch_from_api
def _monthly(start: str, months: int, lag: int, base: float) -> list[dict]:
    dates = pd.date_range(start, periods=months, freq="MS")
    return [{"date": d.strftime("%Y-%m-%d"),
             "realtime_start": (d + pd.Timedelta(days=lag)).strftime("%Y-%m-%d"),
             "value": f"{base + i}"} for i, d in enumerate(dates)]


def _daily(start: str, days: int, lag: int) -> list[dict]:
    dates = pd.date_range(start, periods=days, freq="D")
    return [{"date": d.strftime("%Y-%m-%d"),
             "realtime_start": (d + pd.Timedelta(days=lag)).strftime("%Y-%m-%d"),
             "value": f"{60 + (i % 30)}"} for i, d in enumerate(dates)]


def _api(monkeypatch, seen: list[dict]):
    payloads = {"DCOILBRENTEU": _daily("2023-01-01", 600, lag=6),
                "PCU4831114831115": _monthly("2023-01-01", 20, lag=42, base=100.0),
                "TSIFRGHTC": _monthly("2023-01-01", 20, lag=72, base=1.0)}

    class _Resp:
        def __init__(self, sid):
            self._sid = sid

        def raise_for_status(self) -> None: ...

        def json(self) -> dict:
            return {"observations": payloads[self._sid]}

    def fake_get(_url, params=None, **_kw):
        seen.append(dict(params or {}))
        return _Resp(params["series_id"])

    monkeypatch.setattr(mod.requests, "get", fake_get)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    return MarketConnector(source_mode="api", config={"api": {"fred_key": "k"}})


def test_requests_first_release_vintages(monkeypatch) -> None:
    seen: list[dict] = []
    _api(monkeypatch, seen).fetch_from_api()
    assert len(seen) == 3
    for p in seen:
        assert p["output_type"] == 4                     # initial release only
        assert p["realtime_start"] == "1776-07-04" and p["realtime_end"] == "9999-12-31"


def test_monthly_ppi_steps_on_its_release_date(monkeypatch) -> None:
    df = _api(monkeypatch, []).fetch().set_index("timestamp")
    ppi = df["freight_rate_index"]
    # March 2024's value (released 2024-04-12) is not visible on 2024-04-11.
    assert ppi.loc["2024-04-11"] != ppi.loc["2024-04-12"]
    assert ppi.loc["2024-04-01"] == ppi.loc["2024-04-11"]        # February's, carried
    # The frame starts with Brent; a slower series is NaN until its first
    # release instead of cutting the whole frame short.
    assert df.index[0] == pd.Timestamp("2023-01-07")             # Brent's first release
    assert df.loc[:"2023-02-11", "freight_rate_index"].isna().all()
    assert df.loc["2023-02-12":, "freight_rate_index"].notna().all()   # Jan PPI out
    assert df.loc[:"2023-03-13", "freight_services_pct_change"].isna().all()
    assert df["brent_crude_usd"].notna().all()


def test_brent_is_lagged_by_its_publication_delay(monkeypatch) -> None:
    df = _api(monkeypatch, []).fetch().set_index("timestamp")
    obs = pd.DataFrame(_daily("2023-01-01", 600, lag=6))
    obs_value = obs.set_index(pd.to_datetime(obs["date"]))["value"].astype(float)
    t = pd.Timestamp("2024-03-15")
    assert df.loc[t, "brent_crude_usd"] == obs_value.loc[t - pd.Timedelta(days=6)]


def test_provenance_travels_with_the_frame(monkeypatch) -> None:
    df = _api(monkeypatch, []).fetch()
    src = df.attrs["market_source"]
    assert src["dating"].startswith("first-release")
    assert set(src["median_release_lag_days"]) == {
        "DCOILBRENTEU", "PCU4831114831115", "TSIFRGHTC"}
    assert src["median_release_lag_days"]["PCU4831114831115"] == 42


def test_missing_release_dates_are_an_error(monkeypatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None: ...

        @staticmethod
        def json() -> dict:
            return {"observations": [{"date": "2024-01-01", "value": "80"}]}

    monkeypatch.setattr(mod.requests, "get", lambda *_a, **_kw: _Resp())
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="first-release"):
        MarketConnector(source_mode="api", config={"api": {"fred_key": "k"}}).fetch_from_api()


def test_start_column_leaves_other_columns_nan_until_their_release() -> None:
    f = _frame(10).assign(y=lambda d: d["x"] * 10)
    out = as_published(f, {"x": f["timestamp"] + pd.Timedelta(days=1),
                           "y": f["timestamp"] + pd.Timedelta(days=4)},
                       start_column="x").set_index("timestamp")
    assert out.index[0] == pd.Timestamp("2024-01-02")
    assert out.loc[:"2024-01-04", "y"].isna().all() and out.loc["2024-01-05", "y"] == 0.0


def test_release_lag_ignores_history_stamped_with_the_first_vintage(monkeypatch) -> None:
    seen: list[dict] = []
    connector = _api(monkeypatch, seen)
    # Brent history before ALFRED's first vintage: all stamped 2023-01-07.
    early = [{"date": d.strftime("%Y-%m-%d"), "realtime_start": "2023-01-07",
              "value": "70"} for d in pd.date_range("2015-01-01", "2022-12-31", freq="D")]
    real_get = mod.requests.get

    def with_history(url, params=None, **kw):
        response = real_get(url, params=params, **kw)
        if params["series_id"] == "DCOILBRENTEU":
            obs = early + response.json()["observations"]

            class _R:
                def raise_for_status(self) -> None: ...

                def json(self) -> dict:
                    return {"observations": obs}
            return _R()
        return response

    monkeypatch.setattr(mod.requests, "get", with_history)
    lags = connector.fetch().attrs["market_source"]["median_release_lag_days"]
    assert lags["DCOILBRENTEU"] == 6                      # not years
