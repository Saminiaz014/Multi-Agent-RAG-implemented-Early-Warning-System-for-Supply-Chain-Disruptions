"""Hybrid market-data connector for the Strait of Hormuz corridor.

Supports three ingestion modes selected via ``source_mode``:

- ``"csv"``: Load real FRED data from local CSVs — Brent crude (daily),
  deep-sea freight PPI (monthly), and freight transport services
  % change (monthly). Monthly series are resampled to daily via
  forward-fill; trade volume is derived from inverse Brent volatility;
  the ground-truth ``is_disruption`` label is computed from co-occurring
  Brent and freight spikes.
- ``"synthetic"``: Generate the legacy daily synthetic dataset with three
  disruption scenarios aligned with the shipping connector. Retained as
  a fallback so detection agents have ground-truth labels for testing.
- ``"api"``: Live FRED. Every value is stamped on the day it was first
  published (ALFRED first release), not on its reference date: a monthly
  PPI dated the 1st of its month is released about six weeks later, so the
  reference-date frame carries it forward before anyone could have known
  it. The CSV exports carry no release dates and stay on reference dates.

The connector emits a unified daily-frequency schema that downstream
agents can consume regardless of which mode produced the data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import requests

# Shared with the extractors rather than reimplemented: '${VAR}' placeholders
# must resolve identically on both sides of the pipeline.
from src.extractors.base_extractor import resolve_env_value
from src.ingestion.base_connector import BaseConnector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic-mode constants (preserved from the original implementation).
# ---------------------------------------------------------------------------

_BASELINE_BRENT_USD: float = 78.5
_BASELINE_TRADE_VOLUME: float = 0.85
_BASELINE_FREIGHT_INDEX: float = 110.0

_BRENT_NOISE_SD: float = 0.6
_TRADE_NOISE_SD: float = 0.02
_FREIGHT_NOISE_SD: float = 1.5

_BRENT_PEAK_PCT: float = 0.20
_TRADE_PEAK_DROP: float = 0.45
_FREIGHT_PEAK_PCT: float = 0.50

_PERSISTENCE: float = 0.70
_DEFAULT_LAG_DAYS: int = 2


@dataclass(frozen=True)
class _DefaultPeriod:
    """Default disruption period aligned with the shipping connector."""

    start_day: int
    end_day: int
    severity: float


_DEFAULT_DISRUPTION_PERIODS: tuple[_DefaultPeriod, ...] = (
    _DefaultPeriod(start_day=60, end_day=74, severity=0.45),
    _DefaultPeriod(start_day=150, end_day=170, severity=1.00),
    _DefaultPeriod(start_day=280, end_day=290, severity=0.25),
)


# ---------------------------------------------------------------------------
# CSV-mode constants.
# ---------------------------------------------------------------------------

_BRENT_DEFAULT_PATH: str = "data/raw/brent_crude.csv"
_FREIGHT_PPI_DEFAULT_PATH: str = "data/raw/freight_ppi.csv"
_FREIGHT_SERVICES_DEFAULT_PATH: str = "data/raw/freight_services.csv"

_BRENT_VALUE_COLUMN: str = "DCOILBRENTEU"
_FREIGHT_PPI_VALUE_COLUMN: str = "PCU4831114831115"
_FREIGHT_SERVICES_VALUE_COLUMN: str = "TSIFRGHTC"

# FRED encodes missing values either as a "." sentinel (older exports) or as
# an empty cell (newer exports). Treat both as NaN at load time.
_FRED_NA_VALUES: tuple[str, ...] = (".",)

_PPI_REBASE_WINDOW_DAYS: int = 365 * 2  # last ~2 years → rebased to 100
_ROLLING_WINDOW: int = 30
_BRENT_SIGMA: float = 2.0
_FREIGHT_SIGMA: float = 1.5

# Live-FRED request tuning. The _*_VALUE_COLUMN constants above double as the
# series ids: a FRED CSV export names its column after the series it came from.
_API_TIMEOUT: int = 60
_API_ATTEMPTS: int = 3
_API_BACKOFF_SECONDS: float = 2.0

# ALFRED: the whole real-time period, initial release only (output_type=4),
# so each observation comes with the date it was first published.
_FRED_REALTIME_ALL: tuple[str, str] = ("1776-07-04", "9999-12-31")
_FRED_INITIAL_RELEASE: int = 4


def as_published(frame: pd.DataFrame, available: dict[str, pd.Series],
                 start_column: str | None = None) -> pd.DataFrame:
    """Re-stamp reference-dated columns on the days they were public.

    Args:
        frame: Daily frame on reference dates, sorted by ``timestamp``.
        available: Column name -> the date each row's value became public,
            aligned with ``frame``'s rows. A row is never treated as public
            before its own day.
        start_column: Start the frame when this column first has a value;
            other columns are NaN (False for a boolean column) until their
            own first release. ``None`` starts once every column has one.

    Returns:
        A daily frame indexed by publication date: on day ``t`` each column
        holds its value from the latest row that was public by ``t``. It
        ends on the last publication. A row overtaken by a later row before
        it was published never appears.
    """
    ts = pd.to_datetime(frame["timestamp"]).to_numpy(dtype="datetime64[ns]")
    effective: dict[str, np.ndarray] = {}
    for column, when in available.items():
        when = pd.to_datetime(pd.Series(when).to_numpy()).to_numpy(dtype="datetime64[ns]")
        when = np.maximum(when, ts)
        # The latest row public by t is the last row whose suffix-minimum
        # availability is <= t; the suffix minimum is non-decreasing.
        effective[column] = np.minimum.accumulate(when[::-1])[::-1]
    start = (effective[start_column][0] if start_column is not None
             else max(e[0] for e in effective.values()))
    end = max(e[-1] for e in effective.values())
    grid = pd.date_range(start, end, freq="D")
    out = pd.DataFrame({"timestamp": grid})
    for column, eff in effective.items():
        pos = np.searchsorted(eff, grid.to_numpy(dtype="datetime64[ns]"), side="right") - 1
        values = frame[column].to_numpy()
        if values.dtype == bool:
            out[column] = np.where(pos >= 0, values[np.maximum(pos, 0)], False)
        else:
            out[column] = np.where(pos >= 0, values[np.maximum(pos, 0)].astype(float),
                                   np.nan)
    return out


def _row_release(ts: pd.Series, obs: pd.DataFrame) -> pd.Series:
    """For each reference-dated row, when every observation up to it was public.

    A daily row's value comes from the last observation on or before it
    (forward-filled), and a rolling feature reads all earlier ones, so the
    row is public once all observations up to that day are.
    """
    left = pd.DataFrame({"timestamp": pd.to_datetime(ts).astype("datetime64[ns]")})
    right = obs[["timestamp", "released"]].assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"]).astype("datetime64[ns]"),
        released=lambda d: pd.to_datetime(d["released"]).astype("datetime64[ns]"),
    ).sort_values("timestamp")
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    return merged["released"].bfill().cummax()


class MarketConnector(BaseConnector):
    """Hybrid Brent / freight market-data connector.

    Loads real FRED CSV exports (Brent crude, deep-sea freight PPI,
    freight transport services index) and produces a daily-frequency
    dataset aligned to the shipping connector. Falls back to a synthetic
    generator with three injected disruption scenarios when real CSVs are
    unavailable, and stubs out a future FRED-API integration.

    Args:
        source_mode: Ingestion mode — one of ``"csv"``, ``"synthetic"``,
            or ``"api"``. When ``None``, the value is taken from
            ``config["source_mode"]`` and defaults to ``"csv"``.
        config: Connector-specific configuration block — typically
            ``settings["ingestion"]["market"]`` from ``settings.yaml``.
    """

    LOCATION: str = "Global/Persian Gulf"
    SOURCE: str = "market"
    FEATURE_COLUMNS_CSV: tuple[str, ...] = (
        "brent_crude_usd",
        "trade_volume_index",
        "freight_rate_index",
        "freight_services_pct_change",
    )
    FEATURE_COLUMNS_SYNTHETIC: tuple[str, ...] = (
        "brent_crude_usd",
        "trade_volume_index",
        "freight_rate_index",
    )
    # Backwards-compatible alias used by older callers / tests.
    FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS_SYNTHETIC

    def __init__(
        self,
        source_mode: str | None = None,
        config: dict | None = None,
    ) -> None:
        cfg: dict = dict(config) if config is not None else {}
        super().__init__(cfg)
        self.source_mode: str = (
            source_mode or cfg.get("source_mode") or "csv"
        ).lower()
        self.brent_crude_path: str = cfg.get(
            "brent_crude_path", _BRENT_DEFAULT_PATH
        )
        self.freight_ppi_path: str = cfg.get(
            "freight_ppi_path", _FREIGHT_PPI_DEFAULT_PATH
        )
        self.freight_services_path: str = cfg.get(
            "freight_services_path", _FREIGHT_SERVICES_DEFAULT_PATH
        )
        api_cfg: dict = cfg.get("api", {}) or {}
        self.fred_api_key: str | None = resolve_env_value(api_cfg.get("fred_key") or "")
        self.alpha_vantage_key: str | None = api_cfg.get("alpha_vantage_key")
        self.fred_endpoint: str | None = api_cfg.get(
            "fred_endpoint", "https://api.stlouisfed.org/fred"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self) -> pd.DataFrame:
        """Route to the configured ingestion mode and validate the result.

        Returns:
            Validated, cleaned daily-frequency DataFrame.

        Raises:
            ValueError: If ``self.source_mode`` is not recognised.
            NotImplementedError: If ``source_mode == "api"``.
            FileNotFoundError: If CSV mode is selected but a file is missing.
        """
        mode = self.source_mode
        if mode == "csv":
            df = self.load_from_csv()
        elif mode == "synthetic":
            days = int(self.config.get("days", 365))
            seed = int(self.config.get("seed", 42))
            lag_days = int(self.config.get("lag_days", _DEFAULT_LAG_DAYS))
            df = self.generate_synthetic(days=days, seed=seed, lag_days=lag_days)
        elif mode == "api":
            df = self.fetch_from_api()
        else:
            raise ValueError(
                f"Unknown market source_mode={self.source_mode!r}; "
                "expected 'csv', 'synthetic', or 'api'."
            )
        attrs = dict(df.attrs)
        df = self.validate(df)
        df.attrs.update(attrs)
        self._log_summary(df, mode)
        return df

    def fetch_and_validate(self) -> pd.DataFrame:
        """Override base implementation — :meth:`fetch` already validates."""
        return self.fetch()

    def load_from_csv(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load, merge, and derive features from the three FRED CSV exports.

        Args:
            start_date: Optional ISO date or Timestamp to filter from.
            end_date: Optional ISO date or Timestamp to filter to.

        Returns:
            Daily-frequency DataFrame with columns ``timestamp``,
            ``brent_crude_usd``, ``trade_volume_index``,
            ``freight_rate_index``, ``freight_services_pct_change``, and
            the ground-truth label ``is_disruption``.

        Raises:
            FileNotFoundError: If any required CSV is missing.
        """
        brent = self._load_brent(Path(self.brent_crude_path))
        ppi = self._load_freight_ppi(Path(self.freight_ppi_path))
        services = self._load_freight_services(
            Path(self.freight_services_path)
        )
        return self._merge_and_derive(brent, ppi, services, start_date, end_date)

    def _merge_and_derive(
        self,
        brent: pd.DataFrame,
        ppi: pd.DataFrame,
        services: pd.DataFrame,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Merge the three FRED series onto a daily index and derive features.

        Shared by :meth:`load_from_fred_csvs` and :meth:`fetch_from_api` so a
        live pull and a CSV export produce identical numbers. The derivations
        here (freight rebasing, trade volume from inverse Brent volatility,
        the co-spike disruption label) are modelling choices; defining them
        per-source would let the two paths silently disagree.

        Args:
            brent: ``timestamp``, ``brent_crude_usd`` (daily).
            ppi: ``timestamp``, ``freight_rate_index_raw`` (monthly).
            services: ``timestamp``, ``freight_services_pct_change`` (monthly).
            start_date: Optional inclusive lower bound.
            end_date: Optional inclusive upper bound.

        Returns:
            Daily-frequency frame in the connector's standard schema.
        """
        # Build a daily index spanning the brent series and merge all three.
        idx_min = brent["timestamp"].min()
        idx_max = brent["timestamp"].max()
        daily_index = pd.date_range(start=idx_min, end=idx_max, freq="D")

        df = pd.DataFrame({"timestamp": daily_index})
        df = df.merge(brent, on="timestamp", how="left")
        df = df.merge(ppi, on="timestamp", how="left")
        df = df.merge(services, on="timestamp", how="left")

        # Forward-fill: weekends/holidays for brent, month-to-day for PPI/services.
        df["brent_crude_usd"] = df["brent_crude_usd"].ffill().bfill()
        df["freight_rate_index_raw"] = df["freight_rate_index_raw"].ffill()
        df["freight_services_pct_change"] = df["freight_services_pct_change"].ffill()

        # Rebase the freight PPI so the trailing ~2 years average to 100,
        # producing a freight_rate_index in a band comparable to the synthetic
        # series (~100 baseline, ~150 during disruption).
        cutoff = df["timestamp"].max() - pd.Timedelta(days=_PPI_REBASE_WINDOW_DAYS)
        recent = df.loc[df["timestamp"] >= cutoff, "freight_rate_index_raw"]
        rebase_anchor = float(recent.mean()) if not recent.empty else float(
            df["freight_rate_index_raw"].mean()
        )
        if not np.isfinite(rebase_anchor) or rebase_anchor <= 0:
            rebase_anchor = 1.0
        df["freight_rate_index"] = (
            df["freight_rate_index_raw"] / rebase_anchor * 100.0
        )

        # trade_volume_index ∈ [0, 1]: inverse of normalised 30-day Brent
        # volatility. Higher volatility ⇒ lower trade volume.
        brent_std = (
            df["brent_crude_usd"]
            .rolling(window=_ROLLING_WINDOW, min_periods=5)
            .std()
        )
        overall_std = float(brent_std.std(ddof=0))
        brent_std_filled = brent_std.fillna(brent_std.median())
        s_min = float(brent_std_filled.min())
        s_max = float(brent_std_filled.max())
        if s_max - s_min > 1e-9:
            normalised_std = (brent_std_filled - s_min) / (s_max - s_min)
        else:
            normalised_std = pd.Series(0.0, index=brent_std_filled.index)
        df["trade_volume_index"] = (1.0 - normalised_std).clip(0.0, 1.0)

        # Ground-truth disruption label: simultaneous Brent + freight spike.
        brent_mean = df["brent_crude_usd"].rolling(
            window=_ROLLING_WINDOW, min_periods=5
        ).mean()
        brent_sd = df["brent_crude_usd"].rolling(
            window=_ROLLING_WINDOW, min_periods=5
        ).std()
        freight_mean = df["freight_rate_index"].rolling(
            window=_ROLLING_WINDOW, min_periods=5
        ).mean()
        freight_sd = df["freight_rate_index"].rolling(
            window=_ROLLING_WINDOW, min_periods=5
        ).std()
        brent_spike = df["brent_crude_usd"] > (
            brent_mean + _BRENT_SIGMA * brent_sd
        )
        freight_spike = df["freight_rate_index"] > (
            freight_mean + _FREIGHT_SIGMA * freight_sd
        )
        df["is_disruption"] = (brent_spike & freight_spike).fillna(False)

        if start_date is not None:
            df = df.loc[df["timestamp"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df.loc[df["timestamp"] <= pd.Timestamp(end_date)]
        df = df.reset_index(drop=True)

        df = df[
            [
                "timestamp",
                "brent_crude_usd",
                "trade_volume_index",
                "freight_rate_index",
                "freight_services_pct_change",
                "is_disruption",
            ]
        ]

        self._report_csv_stats(df, overall_std)
        return df

    def generate_synthetic(
        self,
        days: int = 365,
        seed: int = 42,
        disruption_periods: Iterable[Sequence[float]] | None = None,
        lag_days: int = _DEFAULT_LAG_DAYS,
    ) -> pd.DataFrame:
        """Generate the legacy daily synthetic market dataset.

        Retained as the testing fallback when real CSVs are unavailable.
        Disruption windows are aligned with — but slightly lagged behind —
        the shipping connector's three scenarios, and mean-reverting tails
        keep post-disruption dynamics realistic.

        Args:
            days: Number of consecutive days to simulate.
            seed: NumPy random seed for reproducibility.
            disruption_periods: Iterable of ``(start_day, end_day, severity)``
                tuples. ``None`` uses the three defaults aligned with
                :class:`ShippingConnector`.
            lag_days: Days the market response lags the shipping disruption.

        Returns:
            DataFrame with columns ``timestamp``, ``brent_crude_usd``,
            ``trade_volume_index``, ``freight_rate_index``, and the
            ground-truth label ``is_disruption``.
        """
        if days <= 0:
            raise ValueError("days must be positive.")
        if lag_days < 0:
            raise ValueError("lag_days must be non-negative.")

        rng = np.random.default_rng(seed)
        timestamps = pd.date_range(start="2025-01-01", periods=days, freq="D")
        periods = self._resolve_periods(disruption_periods)

        is_disruption = np.zeros(days, dtype=bool)
        market_envelope = np.zeros(days)
        for start, end, severity in periods:
            for day in range(max(start, 0), min(end, days - 1) + 1):
                is_disruption[day] = True
            self._stamp_envelope(
                envelope=market_envelope,
                start=start + lag_days,
                end=end + lag_days,
                severity=float(severity),
                total_days=days,
            )
        market_envelope = self._apply_persistence(market_envelope, days=days)

        brent = rng.normal(_BASELINE_BRENT_USD, _BRENT_NOISE_SD, size=days)
        trade = rng.normal(_BASELINE_TRADE_VOLUME, _TRADE_NOISE_SD, size=days)
        freight = rng.normal(_BASELINE_FREIGHT_INDEX, _FREIGHT_NOISE_SD, size=days)

        brent *= 1.0 + _BRENT_PEAK_PCT * market_envelope
        trade -= _TRADE_PEAK_DROP * market_envelope
        freight *= 1.0 + _FREIGHT_PEAK_PCT * market_envelope

        trade = np.clip(trade, 0.0, 1.0)
        brent = np.clip(brent, 30.0, 250.0)
        freight = np.clip(freight, 50.0, 400.0)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "brent_crude_usd": np.round(brent, 2),
                "trade_volume_index": np.round(trade, 4),
                "freight_rate_index": np.round(freight, 2),
                "is_disruption": is_disruption,
            }
        )

        self._report_separation(df)
        return df

    def generate_dataset(
        self,
        days: int = 365,
        seed: int = 42,
        disruption_periods: Iterable[Sequence[float]] | None = None,
        lag_days: int = _DEFAULT_LAG_DAYS,
    ) -> pd.DataFrame:
        """Backwards-compatible alias for :meth:`generate_synthetic`."""
        return self.generate_synthetic(
            days=days,
            seed=seed,
            disruption_periods=disruption_periods,
            lag_days=lag_days,
        )

    def _fetch_fred_series(self, series_id: str, value_column: str) -> pd.DataFrame:
        """Fetch one FRED series, as first published, with its release dates.

        Requests ALFRED's initial-release vintage (``output_type=4`` over the
        whole real-time period): each observation's value as first published
        and ``realtime_start``, the day it was published.

        Args:
            series_id: FRED series identifier, e.g. ``"DCOILBRENTEU"``.
            value_column: Name to give the observation column.

        Returns:
            ``timestamp`` (reference date), ``value_column`` and ``released``
            (first-release date); missing observations dropped.

        Raises:
            RuntimeError: If the series cannot be fetched after retrying,
                comes back empty, or has no first-release dates. A short
                series would quietly change the rebasing and volatility
                baselines derived from it.
        """
        for attempt in range(1, _API_ATTEMPTS + 1):
            try:
                response = requests.get(
                    f"{self.fred_endpoint}/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": self.fred_api_key,
                        # FRED defaults to XML; without this, .json() raises.
                        "file_type": "json",
                        "output_type": _FRED_INITIAL_RELEASE,
                        "realtime_start": _FRED_REALTIME_ALL[0],
                        "realtime_end": _FRED_REALTIME_ALL[1],
                    },
                    timeout=_API_TIMEOUT,
                )
                response.raise_for_status()
                observations = response.json().get("observations", [])
                break
            except Exception as exc:
                if attempt == _API_ATTEMPTS:
                    raise RuntimeError(
                        f"FRED series {series_id} failed after "
                        f"{_API_ATTEMPTS} attempts: {exc}"
                    ) from exc
                logger.warning(
                    "[MarketConnector/FRED] %s attempt %d/%d failed (%s) — retrying",
                    series_id, attempt, _API_ATTEMPTS, exc,
                )
                time.sleep(_API_BACKOFF_SECONDS * attempt)

        usable = [o for o in observations
                  # FRED writes "." for a missing observation, not null.
                  if o.get("value") not in (".", "", None)]
        if not usable:
            raise RuntimeError(
                f"FRED series {series_id} returned no usable observations."
            )
        if any(not o.get("realtime_start") for o in usable):
            raise RuntimeError(
                f"FRED series {series_id} came back without first-release "
                "dates (realtime_start); cannot stamp values on publication."
            )
        return pd.DataFrame([
            {"timestamp": pd.Timestamp(o["date"]), value_column: float(o["value"]),
             "released": pd.Timestamp(o["realtime_start"])}
            for o in usable
        ])

    @staticmethod
    def _median_release_lag(frame: pd.DataFrame) -> int:
        """Median days from reference date to first release.

        ALFRED stamps every observation older than a series' first vintage
        with that vintage's date, so those are left out: they measure when
        the archive began, not the publication delay.
        """
        later = frame[frame["released"] > frame["released"].min()]
        lags = (later if not later.empty else frame)
        return int((lags["released"] - lags["timestamp"]).dt.days.median())

    def fetch_from_api(self) -> pd.DataFrame:
        """Fetch the three FRED series live, dated by first publication.

        Same series the CSV path reads — ``DCOILBRENTEU`` (Brent, daily),
        ``PCU4831114831115`` (deep-sea freight PPI, monthly) and
        ``TSIFRGHTC`` (freight transport services % change, monthly) — pulled
        from ``api.stlouisfed.org`` as first published. They pass through the
        same :meth:`_merge_and_derive` transform on reference dates, and each
        derived column is then re-stamped by :func:`as_published` on the day
        its inputs were public: Brent-derived columns on Brent's release
        dates (EIA publishes weekly), the freight index on the PPI's,
        ``freight_services_pct_change`` on the TSI's, and the co-spike label
        once both Brent and the PPI were out. ``timestamp`` is therefore the
        day a forecaster could have known the value, and a live pull no
        longer matches a reference-dated CSV export day for day.

        Alpha Vantage is not used: FRED already serves every series this
        connector needs, and adding a second vendor for intraday prices would
        buy nothing for a daily-frequency pipeline.

        Returns:
            Daily-frequency frame in the connector's standard schema.

        Raises:
            ValueError: If no FRED API key is configured.
            RuntimeError: If any series cannot be fetched.
        """
        if not self.fred_api_key:
            raise ValueError(
                "No FRED API key configured. Set api_keys.fred / FRED_API_KEY, "
                "or use source_mode='csv'."
            )

        # The _*_VALUE_COLUMN constants are the FRED series ids.
        brent = self._fetch_fred_series(_BRENT_VALUE_COLUMN, "brent_crude_usd")
        ppi = self._fetch_fred_series(_FREIGHT_PPI_VALUE_COLUMN, "freight_rate_index_raw")
        services = self._fetch_fred_series(
            _FREIGHT_SERVICES_VALUE_COLUMN, "freight_services_pct_change"
        )

        reference = self._merge_and_derive(
            brent.drop(columns="released"), ppi.drop(columns="released"),
            services.drop(columns="released"),
        )
        on_brent = _row_release(reference["timestamp"], brent)
        on_ppi = _row_release(reference["timestamp"], ppi)
        df = as_published(reference, {
            "brent_crude_usd": on_brent,
            "trade_volume_index": on_brent,
            "freight_rate_index": on_ppi,
            "freight_services_pct_change": _row_release(reference["timestamp"], services),
            "is_disruption": np.maximum(on_brent, on_ppi),
        }, start_column="brent_crude_usd")
        lags = {sid: self._median_release_lag(frame)
                for sid, frame in ((_BRENT_VALUE_COLUMN, brent),
                                   (_FREIGHT_PPI_VALUE_COLUMN, ppi),
                                   (_FREIGHT_SERVICES_VALUE_COLUMN, services))}
        df.attrs["market_source"] = {
            "source": "FRED / ALFRED",
            "dating": ("first-release date (ALFRED output_type=4): each value "
                       "enters on the day it was first published, carried "
                       "forward until the next release"),
            "values": "as first published (later revisions not used)",
            "median_release_lag_days": lags,
        }
        logger.info(
            "[MarketConnector/FRED] %d days [%s .. %s] brent mean=$%.2f "
            "disruption_days=%d",
            len(df), df["timestamp"].min().date(), df["timestamp"].max().date(),
            df["brent_crude_usd"].mean(), int(df["is_disruption"].sum()),
        )
        return df

    # Synthetic-schema CSV columns produced by generate_synthetic / save_raw.
    _SYNTHETIC_SCHEMA: tuple[str, ...] = (
        "timestamp",
        "brent_crude_usd",
        "trade_volume_index",
        "freight_rate_index",
        "is_disruption",
    )

    def load_csv(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load a *synthetic-schema* market CSV (the :meth:`save_raw` format).

        Round-trip counterpart to :meth:`save_raw`, distinct from
        :meth:`load_from_csv` (which merges the three real FRED exports).

        Args:
            path: CSV location. When ``None``, ``config["csv_path"]`` is used,
                defaulting to ``data/raw/market_data.csv``.

        Returns:
            DataFrame with the five synthetic-schema columns, ``timestamp``
            parsed to datetime and ``is_disruption`` cast to ``bool``.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing or ``timestamp`` /
                ``brent_crude_usd`` contain NaN.
        """
        csv_path = Path(path) if path is not None else Path(
            self.config.get("csv_path", "data/raw/market_data.csv")
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"Market CSV not found at {csv_path}.")

        df = pd.read_csv(csv_path)
        missing = [c for c in self._SYNTHETIC_SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(
                f"Market CSV {csv_path} missing required columns: {missing}"
            )
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["is_disruption"] = df["is_disruption"].astype(bool)
        for col in ("brent_crude_usd", "trade_volume_index", "freight_rate_index"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df["timestamp"].isna().any() or df["brent_crude_usd"].isna().any():
            raise ValueError(
                f"Market CSV {csv_path} has NaN in critical columns "
                "('timestamp' / 'brent_crude_usd')."
            )
        return df[list(self._SYNTHETIC_SCHEMA)]

    def fetch_api(self) -> pd.DataFrame:
        """Graceful API fallback — warn and return a synthetic dataset.

        Unlike :meth:`fetch_from_api` (which hard-raises for the planned live
        FRED / Alpha Vantage integration), this hook logs a warning and
        returns :meth:`generate_dataset` so ``api`` mode degrades gracefully.

        Returns:
            A freshly generated synthetic market DataFrame.
        """
        logger.warning("API mode not configured — using synthetic")
        return self.generate_dataset(
            days=int(self.config.get("days", 365)),
            seed=int(self.config.get("seed", 42)),
            lag_days=int(self.config.get("lag_days", _DEFAULT_LAG_DAYS)),
        )

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Schema and domain checks; returns a sorted, cleaned DataFrame.

        Args:
            df: DataFrame returned by :meth:`load_from_csv` or
                :meth:`generate_synthetic`.

        Returns:
            Cleaned DataFrame sorted by timestamp.

        Raises:
            AssertionError: If a non-recoverable schema or domain check fails.
        """
        assert "timestamp" in df.columns, "Missing 'timestamp' column."
        assert "brent_crude_usd" in df.columns, "Missing 'brent_crude_usd' column."
        assert "trade_volume_index" in df.columns, (
            "Missing 'trade_volume_index' column."
        )

        df = df.sort_values("timestamp").reset_index(drop=True).copy()

        assert df["timestamp"].notna().all(), "NaN values in 'timestamp'."
        assert df["brent_crude_usd"].notna().all(), (
            "NaN values in 'brent_crude_usd' after forward-fill."
        )
        assert (df["brent_crude_usd"] > 0).all(), (
            "Non-positive 'brent_crude_usd' values detected."
        )
        assert df["trade_volume_index"].between(0.0, 1.0).all(), (
            "'trade_volume_index' out of [0, 1]."
        )
        if "freight_rate_index" in df.columns:
            present = df["freight_rate_index"].dropna()
            assert (present > 0).all(), (
                "'freight_rate_index' has non-positive values."
            )
        assert df["timestamp"].is_monotonic_increasing, (
            "Timestamps must be monotonically increasing."
        )
        return df

    def align_with_shipping(
        self,
        shipping_df: pd.DataFrame,
        market_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Align market data onto the shipping connector's timestamp grid.

        Args:
            shipping_df: DataFrame produced by :class:`ShippingConnector`.
                Its ``timestamp`` column defines the target index.
            market_df: Optional pre-loaded market DataFrame. When ``None``,
                :meth:`fetch` is invoked.

        Returns:
            Market DataFrame reindexed to every timestamp in
            ``shipping_df``, with weekend/holiday gaps forward-filled.
        """
        if "timestamp" not in shipping_df.columns:
            raise ValueError("shipping_df must contain a 'timestamp' column.")
        if market_df is None:
            market_df = self.fetch()

        target_dates = pd.to_datetime(shipping_df["timestamp"]).sort_values()
        start, end = target_dates.min(), target_dates.max()

        market = market_df.copy()
        market["timestamp"] = pd.to_datetime(market["timestamp"])
        market = market.loc[
            (market["timestamp"] >= start) & (market["timestamp"] <= end)
        ]

        aligned = (
            market.set_index("timestamp")
            .reindex(target_dates.unique())
            .ffill()
            .bfill()
            .rename_axis("timestamp")
            .reset_index()
        )
        if "is_disruption" in aligned.columns:
            aligned["is_disruption"] = aligned["is_disruption"].fillna(False).astype(bool)
        return aligned

    def save_raw(
        self,
        df: pd.DataFrame | None = None,
        path: str | Path = "data/raw/market_processed.csv",
    ) -> Path:
        """Persist a processed market dataset as CSV.

        Args:
            df: DataFrame to write. When ``None``, :meth:`fetch` is invoked
                so the connector loads from its configured source.
            path: Destination CSV path. Parent directories are created.

        Returns:
            Resolved absolute path of the written file.
        """
        if df is None:
            df = self.fetch()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Wrote %d rows to %s", len(df), out)
        return out.resolve()

    def to_unified_signals(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Convert a dataset into the unified signal-record schema.

        Args:
            df: DataFrame returned by :meth:`load_from_csv` or
                :meth:`generate_synthetic`.

        Returns:
            List of dicts with keys ``timestamp``, ``source``, ``feature``,
            ``value``, ``location``. NaN values are skipped so output is
            always JSON-serialisable.
        """
        feature_cols = [
            c for c in df.columns
            if c not in {"timestamp", "is_disruption"}
        ]
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["timestamp"]).isoformat()
            for feature in feature_cols:
                val = row[feature]
                if pd.isna(val):
                    continue
                records.append(
                    {
                        "timestamp": ts,
                        "source": self.SOURCE,
                        "feature": feature,
                        "value": float(val),
                        "location": self.LOCATION,
                    }
                )
        return records

    def to_signal_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Backwards-compatible alias for :meth:`to_unified_signals`."""
        return self.to_unified_signals(df)

    # ------------------------------------------------------------------
    # CSV loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _read_fred_csv(path: Path, value_column: str) -> pd.DataFrame:
        """Read a FRED CSV regardless of date-column naming convention."""
        if not path.exists():
            raise FileNotFoundError(
                f"FRED CSV not found at {path}. "
                "Set the corresponding path under ingestion.market in "
                "config/settings.yaml or place the file at the default location."
            )
        raw = pd.read_csv(path, na_values=list(_FRED_NA_VALUES))
        date_col_candidates = ["observation_date", "DATE", "date"]
        date_col = next(
            (c for c in date_col_candidates if c in raw.columns), None
        )
        if date_col is None:
            raise ValueError(
                f"{path}: expected one of {date_col_candidates} as the date column."
            )
        if value_column not in raw.columns:
            raise ValueError(
                f"{path}: expected value column {value_column!r}, "
                f"got {list(raw.columns)}."
            )
        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(raw[date_col])
        out["__value__"] = pd.to_numeric(raw[value_column], errors="coerce")
        return out

    def _load_brent(self, path: Path) -> pd.DataFrame:
        out = self._read_fred_csv(path, _BRENT_VALUE_COLUMN)
        return out.rename(columns={"__value__": "brent_crude_usd"})

    def _load_freight_ppi(self, path: Path) -> pd.DataFrame:
        out = self._read_fred_csv(path, _FREIGHT_PPI_VALUE_COLUMN)
        # Keep the raw monthly value; daily rebasing happens after merge.
        return out.rename(columns={"__value__": "freight_rate_index_raw"})

    def _load_freight_services(self, path: Path) -> pd.DataFrame:
        out = self._read_fred_csv(path, _FREIGHT_SERVICES_VALUE_COLUMN)
        return out.rename(columns={"__value__": "freight_services_pct_change"})

    # ------------------------------------------------------------------
    # Synthetic-mode helpers (preserved unchanged from prior implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_periods(
        disruption_periods: Iterable[Sequence[float]] | None,
    ) -> list[tuple[int, int, float]]:
        """Normalise the disruption_periods argument into validated tuples."""
        if disruption_periods is None:
            return [
                (p.start_day, p.end_day, p.severity)
                for p in _DEFAULT_DISRUPTION_PERIODS
            ]
        out: list[tuple[int, int, float]] = []
        for tup in disruption_periods:
            tup = tuple(tup)
            if len(tup) != 3:
                raise ValueError(
                    "Each disruption period must be (start_day, end_day, severity)."
                )
            start, end, severity = tup
            if int(start) > int(end):
                raise ValueError("start_day must not exceed end_day.")
            out.append((int(start), int(end), float(severity)))
        return out

    @staticmethod
    def _stamp_envelope(
        envelope: np.ndarray,
        start: int,
        end: int,
        severity: float,
        total_days: int,
    ) -> None:
        """Stamp a triangular ramp/plateau/decay envelope into ``envelope``."""
        if start >= total_days or severity <= 0.0:
            return
        clipped_start = max(start, 0)
        clipped_end = min(end, total_days - 1)
        if clipped_end < clipped_start:
            return

        window_len = clipped_end - clipped_start + 1
        ramp = max(window_len // 4, 1)
        decay = max(window_len // 4, 1)

        for offset, day_idx in enumerate(range(clipped_start, clipped_end + 1)):
            if offset < ramp:
                intensity = (offset + 1) / (ramp + 1)
            elif offset >= window_len - decay:
                tail_offset = offset - (window_len - decay)
                intensity = 1.0 - (tail_offset + 1) / (decay + 1)
            else:
                intensity = 1.0
            envelope[day_idx] = max(envelope[day_idx], severity * intensity)

    @staticmethod
    def _apply_persistence(envelope: np.ndarray, days: int) -> np.ndarray:
        """Carry residual disruption forward for mean-reverting tails."""
        out = np.zeros_like(envelope)
        if days == 0:
            return out
        out[0] = envelope[0]
        for i in range(1, days):
            out[i] = max(envelope[i], _PERSISTENCE * out[i - 1])
        return out

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _log_summary(self, df: pd.DataFrame, mode: str) -> None:
        n = len(df)
        n_dis = (
            int(df["is_disruption"].sum())
            if "is_disruption" in df.columns else 0
        )
        ts_min = pd.Timestamp(df["timestamp"].min()).date()
        ts_max = pd.Timestamp(df["timestamp"].max()).date()
        msg = (
            f"[MarketConnector] mode='{mode}' rows={n} "
            f"range=[{ts_min} .. {ts_max}] disruption_days={n_dis}"
        )
        logger.info(msg)
        print(msg)

    @staticmethod
    def _report_csv_stats(df: pd.DataFrame, overall_std: float) -> None:
        """Print descriptive stats for the loaded FRED dataset."""
        brent = df["brent_crude_usd"].to_numpy(dtype=float)
        ts_min = pd.Timestamp(df["timestamp"].min()).date()
        ts_max = pd.Timestamp(df["timestamp"].max()).date()
        n_disrupt = int(df["is_disruption"].sum())
        msg = (
            f"[MarketConnector/CSV] rows={len(df)} "
            f"range=[{ts_min} .. {ts_max}] "
            f"brent mean=${brent.mean():.2f} min=${brent.min():.2f} "
            f"max=${brent.max():.2f} 30d-std={overall_std:.2f} "
            f"disruption_days={n_disrupt}"
        )
        logger.info(msg)
        print(msg)

    @staticmethod
    def _report_separation(df: pd.DataFrame) -> None:
        """Log mean trade-volume index in normal vs disruption windows."""
        normal = df.loc[~df["is_disruption"], "trade_volume_index"].to_numpy(dtype=float)
        disrupted = df.loc[df["is_disruption"], "trade_volume_index"].to_numpy(dtype=float)
        if len(disrupted) == 0 or len(normal) == 0:
            return
        msg = (
            f"[MarketConnector] trade_volume_index — "
            f"normal mean={normal.mean():.3f} (n={len(normal)}), "
            f"disruption mean={disrupted.mean():.3f} (n={len(disrupted)})"
        )
        logger.info(msg)
        print(msg)
