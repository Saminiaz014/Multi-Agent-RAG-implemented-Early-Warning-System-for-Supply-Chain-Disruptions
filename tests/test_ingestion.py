"""Tests for the shipping and market ingestion connectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion import MarketConnector, ShippingConnector

_CSV_PATH = Path("data/raw/shuaiba_arrivals.csv")


# ---------------------------------------------------------------------------
# Shipping connector — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def connector() -> ShippingConnector:
    """Default shipping connector — pinned to synthetic mode for unit tests."""
    return ShippingConnector(source_mode="synthetic", config={})


@pytest.fixture()
def df(connector: ShippingConnector) -> pd.DataFrame:
    return connector.generate_synthetic(days=365, seed=42)


@pytest.fixture()
def csv_connector() -> ShippingConnector:
    return ShippingConnector(
        source_mode="csv",
        config={"csv_path": str(_CSV_PATH)},
    )


# ---------------------------------------------------------------------------
# Shipping connector — CSV mode
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _CSV_PATH.exists(),
    reason=f"Shuaiba arrivals CSV not present at {_CSV_PATH}",
)
class TestShippingCsvMode:
    """Validate the IMF PortWatch Shuaiba CSV ingestion path."""

    def test_csv_mode_has_expected_columns(
        self, csv_connector: ShippingConnector
    ) -> None:
        df = csv_connector.load_from_csv()
        expected = {
            "timestamp",
            "vessel_count",
            "tanker_count",
            "vessel_count_7dma",
            "avg_delay_hours",
            "congestion_index",
            "oil_price_usd",
            "is_disruption",
        }
        assert expected.issubset(set(df.columns))

    def test_csv_mode_loads_full_history(
        self, csv_connector: ShippingConnector
    ) -> None:
        df = csv_connector.load_from_csv()
        assert len(df) > 2500, "Expected the full 2019-2026 Shuaiba history."
        assert pd.Timestamp(df["timestamp"].min()) <= pd.Timestamp("2019-01-31")
        assert pd.Timestamp(df["timestamp"].max()) >= pd.Timestamp("2026-04-01")

    def test_csv_mode_flags_2026_shutdown(
        self, csv_connector: ShippingConnector
    ) -> None:
        """The known April-May 2026 Hormuz shutdown must be flagged."""
        df = csv_connector.load_from_csv()
        shutdown = df.loc[
            df["timestamp"].between(
                pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-15")
            )
        ]
        assert not shutdown.empty
        assert shutdown["is_disruption"].all(), (
            "All April-May 2026 rows must be flagged as disruption."
        )
        # The CSV records a near-total shutdown — a stray vessel on one day
        # is acceptable, but the mean must be well below the normal baseline.
        assert shutdown["vessel_count"].mean() < 0.2, (
            f"Shutdown vessel_count mean should be ~0; got {shutdown['vessel_count'].mean():.2f}"
        )

    def test_csv_mode_derived_columns_in_range(
        self, csv_connector: ShippingConnector
    ) -> None:
        df = csv_connector.load_from_csv()
        assert df["congestion_index"].between(0.0, 1.0).all()
        assert df["avg_delay_hours"].between(1.0, 72.0).all()
        # Normal-baseline rows should produce moderate (≤ 10h) delays.
        normal = df.loc[~df["is_disruption"]]
        assert normal["avg_delay_hours"].median() < 10.0

    def test_csv_mode_vessel_count_sums_types(
        self, csv_connector: ShippingConnector
    ) -> None:
        raw = pd.read_csv(_CSV_PATH)
        expected_sum = raw[
            ["Container", "Dry Bulk", "General Cargo", "Roll-on/roll-off", "Tanker"]
        ].sum(axis=1).astype(float).sum()
        df = csv_connector.load_from_csv()
        assert df["vessel_count"].sum() == pytest.approx(expected_sum)

    def test_csv_mode_oil_price_is_nan(
        self, csv_connector: ShippingConnector
    ) -> None:
        """The market connector fills oil_price_usd later; CSV must leave it NaN."""
        df = csv_connector.load_from_csv()
        assert df["oil_price_usd"].isna().all()

    def test_csv_mode_ttest_significant(
        self, csv_connector: ShippingConnector
    ) -> None:
        """Welch t-test on vessel_count: disruption vs normal must be highly significant."""
        from scipy import stats as _stats

        df = csv_connector.load_from_csv()
        normal = df.loc[~df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        disrupted = df.loc[df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        result = _stats.ttest_ind(normal, disrupted, equal_var=False)
        t_stat = float(result.statistic)
        p_val = float(result.pvalue)
        print(
            f"\n[test/csv] vessel_count Welch t={t_stat:.2f}, "
            f"p={p_val:.2e}, n_normal={len(normal)}, n_dis={len(disrupted)}"
        )
        assert p_val < 0.05

    def test_csv_mode_fetch_dispatches_to_csv(
        self, csv_connector: ShippingConnector
    ) -> None:
        df = csv_connector.fetch()
        assert "tanker_count" in df.columns
        assert "vessel_count_7dma" in df.columns


# ---------------------------------------------------------------------------
# Shipping connector — synthetic mode (no-regression checks)
# ---------------------------------------------------------------------------


class TestShippingSyntheticMode:
    """Verify that legacy synthetic generation still produces the same dataset."""

    def test_synthetic_dataset_shape_and_columns(self, df: pd.DataFrame) -> None:
        assert len(df) == 365
        assert set(df.columns) == {
            "timestamp",
            "vessel_count",
            "avg_delay_hours",
            "congestion_index",
            "oil_price_usd",
            "is_disruption",
        }

    def test_synthetic_no_nan(self, df: pd.DataFrame) -> None:
        assert not df.isna().any().any()

    def test_synthetic_congestion_within_unit_interval(
        self, df: pd.DataFrame
    ) -> None:
        assert df["congestion_index"].between(0.0, 1.0).all()

    def test_synthetic_vessel_count_non_negative(self, df: pd.DataFrame) -> None:
        assert (df["vessel_count"] >= 0).all()

    def test_synthetic_disruption_day_count_roughly_46(
        self, df: pd.DataFrame
    ) -> None:
        n_disrupt = int(df["is_disruption"].sum())
        assert 40 <= n_disrupt <= 55, f"Expected ~46 disruption days, got {n_disrupt}"

    def test_synthetic_normal_vs_disruption_distinguishable(
        self, df: pd.DataFrame
    ) -> None:
        normal = df.loc[~df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        disrupted = df.loc[df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        diff = normal.mean() - disrupted.mean()
        se = np.sqrt(
            normal.var(ddof=1) / len(normal)
            + disrupted.var(ddof=1) / len(disrupted)
        )
        t_stat = diff / se
        print(f"\n[test/synthetic] vessel_count Welch t-statistic: {t_stat:.2f}")
        assert t_stat > 5

    def test_synthetic_disruptions_cover_expected_windows(
        self, df: pd.DataFrame
    ) -> None:
        flags = df["is_disruption"].to_numpy()
        assert flags[65] and flags[160] and flags[285]
        assert not flags[0]

    def test_synthetic_seed_is_reproducible(
        self, connector: ShippingConnector
    ) -> None:
        a = connector.generate_synthetic(seed=42)
        b = connector.generate_synthetic(seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_synthetic_different_seeds_diverge(
        self, connector: ShippingConnector
    ) -> None:
        a = connector.generate_synthetic(seed=42)
        b = connector.generate_synthetic(seed=7)
        assert not a["vessel_count"].equals(b["vessel_count"])

    def test_synthetic_fetch_dispatches_to_synthetic(
        self, connector: ShippingConnector
    ) -> None:
        out = connector.fetch()
        assert set(out.columns) == {
            "timestamp",
            "vessel_count",
            "avg_delay_hours",
            "congestion_index",
            "oil_price_usd",
            "is_disruption",
        }
        assert len(out) == 365


# ---------------------------------------------------------------------------
# Shipping connector — API mode
# ---------------------------------------------------------------------------


class TestShippingApiMode:
    def test_api_mode_raises_not_implemented_via_fetch(self) -> None:
        connector = ShippingConnector(source_mode="api", config={})
        with pytest.raises(NotImplementedError, match="aisstream"):
            connector.fetch()

    def test_api_mode_raises_not_implemented_direct(self) -> None:
        connector = ShippingConnector(source_mode="api", config={})
        with pytest.raises(NotImplementedError):
            connector.fetch_from_api()


# ---------------------------------------------------------------------------
# Shipping connector — validate()
# ---------------------------------------------------------------------------


class TestShippingValidate:
    """validate() must clean small gaps and assert on hard schema/domain breaks."""

    def _baseline_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=10, freq="D"),
                "vessel_count": np.arange(1, 11, dtype=float),
                "congestion_index": np.linspace(0.1, 0.9, 10),
            }
        )

    def test_validate_returns_cleaned_dataframe(
        self, connector: ShippingConnector
    ) -> None:
        out = connector.validate(self._baseline_df())
        assert isinstance(out, pd.DataFrame)
        assert out["timestamp"].is_monotonic_increasing
        assert len(out) == 10

    def test_validate_rejects_nan_in_vessel_count(
        self, connector: ShippingConnector
    ) -> None:
        bad = self._baseline_df()
        bad.loc[3, "vessel_count"] = np.nan
        with pytest.raises(AssertionError, match="vessel_count"):
            connector.validate(bad)

    def test_validate_rejects_negative_vessel_count(
        self, connector: ShippingConnector
    ) -> None:
        bad = self._baseline_df()
        bad.loc[3, "vessel_count"] = -1.0
        with pytest.raises(AssertionError, match="Negative"):
            connector.validate(bad)

    def test_validate_rejects_out_of_range_congestion(
        self, connector: ShippingConnector
    ) -> None:
        bad = self._baseline_df()
        bad.loc[3, "congestion_index"] = 1.5
        with pytest.raises(AssertionError, match="congestion_index"):
            connector.validate(bad)

    def test_validate_forward_fills_small_gap(
        self, connector: ShippingConnector
    ) -> None:
        df = self._baseline_df()
        df.loc[5, "congestion_index"] = np.nan
        out = connector.validate(df)
        assert not out["congestion_index"].isna().any(), (
            "validate must ffill small (1-2 day) NaN gaps."
        )


# ---------------------------------------------------------------------------
# Cross-mode synthetic fallback (named per the integration test spec)
# ---------------------------------------------------------------------------


def test_synthetic_fallback() -> None:
    """source_mode='synthetic' must reproduce the original (pre-CSV) behaviour.

    Verifies both connectors fall back cleanly to their synthetic generators
    when explicitly configured with ``source_mode="synthetic"`` — no CSV
    access, the legacy column schemas, and the expected 365-day length.
    """
    shipping = ShippingConnector(source_mode="synthetic", config={})
    market = MarketConnector(source_mode="synthetic", config={})

    s_df = shipping.fetch()
    m_df = market.fetch()

    assert len(s_df) == 365
    assert len(m_df) == 365
    assert set(s_df.columns) == {
        "timestamp",
        "vessel_count",
        "avg_delay_hours",
        "congestion_index",
        "oil_price_usd",
        "is_disruption",
    }
    assert set(m_df.columns) == {
        "timestamp",
        "brent_crude_usd",
        "trade_volume_index",
        "freight_rate_index",
        "is_disruption",
    }
    # Round-trip guarantee: ground truths align so cross-agent validation works.
    np.testing.assert_array_equal(
        s_df["is_disruption"].to_numpy(),
        m_df["is_disruption"].to_numpy(),
    )

    def test_validate_passes_on_synthetic(
        self, connector: ShippingConnector, df: pd.DataFrame
    ) -> None:
        out = connector.validate(df)
        assert len(out) == len(df)


# ---------------------------------------------------------------------------
# Shipping connector — unified signals + save_raw
# ---------------------------------------------------------------------------


def test_unified_signals_match_unified_schema(
    connector: ShippingConnector,
) -> None:
    small = connector.generate_synthetic(days=10)
    records = connector.to_unified_signals(small)
    expected_records = 10 * len(connector.FEATURE_COLUMNS_SYNTHETIC)
    assert len(records) == expected_records
    sample = records[0]
    assert set(sample.keys()) == {"timestamp", "source", "feature", "value", "location"}
    assert sample["source"] == "shipping"
    assert sample["location"] == "Shuaiba Port, Persian Gulf"
    assert isinstance(sample["value"], float)
    json.dumps(records)


def test_save_raw_to_tmp(
    connector: ShippingConnector, tmp_path: Path
) -> None:
    target = tmp_path / "raw" / "shipping_processed.csv"
    written = connector.save_raw(path=target)
    assert written.exists()
    reloaded = pd.read_csv(written)
    assert len(reloaded) == 365
    assert "is_disruption" in reloaded.columns


def test_save_raw_accepts_explicit_dataframe(
    connector: ShippingConnector, df: pd.DataFrame, tmp_path: Path
) -> None:
    target = tmp_path / "raw" / "explicit.csv"
    written = connector.save_raw(df=df, path=target)
    assert written.exists()
    reloaded = pd.read_csv(written)
    assert len(reloaded) == len(df)


# ---------------------------------------------------------------------------
# Market connector tests
# ---------------------------------------------------------------------------


_BRENT_PATH = Path("data/raw/brent_crude.csv")
_FREIGHT_PPI_PATH = Path("data/raw/freight_ppi.csv")
_FREIGHT_SERVICES_PATH = Path("data/raw/freight_services.csv")
_MARKET_CSVS_PRESENT = all(
    p.exists() for p in (_BRENT_PATH, _FREIGHT_PPI_PATH, _FREIGHT_SERVICES_PATH)
)


@pytest.fixture()
def market_connector() -> MarketConnector:
    """Default market connector — pinned to synthetic mode for unit tests."""
    return MarketConnector(source_mode="synthetic", config={})


@pytest.fixture()
def market_csv_connector() -> MarketConnector:
    return MarketConnector(
        source_mode="csv",
        config={
            "brent_crude_path": str(_BRENT_PATH),
            "freight_ppi_path": str(_FREIGHT_PPI_PATH),
            "freight_services_path": str(_FREIGHT_SERVICES_PATH),
        },
    )


@pytest.fixture()
def market_df(market_connector: MarketConnector) -> pd.DataFrame:
    return market_connector.generate_dataset(days=365, seed=42)


def test_market_dataset_shape_and_columns(market_df: pd.DataFrame) -> None:
    assert len(market_df) == 365
    assert set(market_df.columns) == {
        "timestamp",
        "brent_crude_usd",
        "trade_volume_index",
        "freight_rate_index",
        "is_disruption",
    }


def test_market_no_nan(market_df: pd.DataFrame) -> None:
    assert not market_df.isna().any().any()


def test_market_trade_volume_in_unit_interval(market_df: pd.DataFrame) -> None:
    assert market_df["trade_volume_index"].between(0.0, 1.0).all()


def test_market_brent_in_realistic_range(market_df: pd.DataFrame) -> None:
    normal = market_df.loc[~market_df["is_disruption"], "brent_crude_usd"]
    assert normal.between(70.0, 85.0).all(), (
        f"Normal Brent should sit in 75-82 ± noise; got [{normal.min():.2f}, {normal.max():.2f}]"
    )


def test_market_freight_in_realistic_baseline(market_df: pd.DataFrame) -> None:
    normal = market_df.loc[~market_df["is_disruption"], "freight_rate_index"]
    assert normal.between(95.0, 130.0).all(), (
        f"Normal freight index should sit in 100-120 ± noise; got [{normal.min():.2f}, {normal.max():.2f}]"
    )


def test_market_disruption_count_matches_shipping(
    market_df: pd.DataFrame, df: pd.DataFrame
) -> None:
    """Market ground truth must align with the shipping connector's labels."""
    np.testing.assert_array_equal(
        market_df["is_disruption"].to_numpy(),
        df["is_disruption"].to_numpy(),
    )


def test_market_seed_reproducible(market_connector: MarketConnector) -> None:
    a = market_connector.generate_dataset(seed=42)
    b = market_connector.generate_dataset(seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_market_different_seeds_diverge(market_connector: MarketConnector) -> None:
    a = market_connector.generate_dataset(seed=42)
    b = market_connector.generate_dataset(seed=7)
    assert not a["brent_crude_usd"].equals(b["brent_crude_usd"])


def test_market_response_lags_shipping(market_connector: MarketConnector) -> None:
    """Brent peak should arrive after the shipping disruption begins."""
    market = market_connector.generate_dataset(days=365, seed=42, lag_days=2)
    window = market.iloc[150:175]
    peak_day = int(window["brent_crude_usd"].idxmax())
    assert peak_day >= 152, f"Expected lagged peak >= day 152, got {peak_day}"


def test_market_mean_reversion_after_window(
    market_connector: MarketConnector,
) -> None:
    """Freight index must decay back toward baseline after the window ends."""
    market = market_connector.generate_dataset(days=365, seed=42)
    end_day = 170
    elevated = float(market.loc[end_day, "freight_rate_index"])
    settled = float(market.loc[end_day + 15, "freight_rate_index"])
    baseline_band_top = 130.0
    assert elevated > settled, "Freight index should decay after the window closes."
    assert settled < baseline_band_top, (
        f"Freight index should mean-revert below {baseline_band_top}; got {settled:.2f}"
    )


def test_market_validate_returns_dataframe_on_generated(
    market_connector: MarketConnector, market_df: pd.DataFrame
) -> None:
    out = market_connector.validate(market_df)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(market_df)


def test_market_validate_rejects_out_of_range_volume(
    market_connector: MarketConnector, market_df: pd.DataFrame
) -> None:
    bad = market_df.copy()
    bad.loc[0, "trade_volume_index"] = 1.5
    with pytest.raises(AssertionError, match="trade_volume_index"):
        market_connector.validate(bad)


def test_market_signal_records_match_unified_schema(
    market_connector: MarketConnector,
) -> None:
    small = market_connector.generate_dataset(days=10)
    records = market_connector.to_signal_records(small)
    assert len(records) == 10 * len(market_connector.FEATURE_COLUMNS)
    sample = records[0]
    assert set(sample.keys()) == {"timestamp", "source", "feature", "value", "location"}
    assert sample["source"] == "market"
    assert sample["location"] == "Global/Persian Gulf"
    assert isinstance(sample["value"], float)
    json.dumps(records)


def test_market_save_raw_to_tmp(
    market_connector: MarketConnector, tmp_path: Path
) -> None:
    target = tmp_path / "raw" / "market_processed.csv"
    written = market_connector.save_raw(path=target)
    assert written.exists()
    reloaded = pd.read_csv(written)
    assert len(reloaded) == 365
    assert "is_disruption" in reloaded.columns


def test_market_save_raw_to_canonical_location(tmp_path: Path) -> None:
    """Persist a canonical artefact for downstream agents."""
    connector = MarketConnector(source_mode="synthetic", config={})
    target = tmp_path / "market_data.csv"
    written = connector.save_raw(path=target)
    assert written.exists()
    assert written.name == "market_data.csv"


def test_market_ingestion_correlates_with_shipping(
    market_connector: MarketConnector, connector: ShippingConnector
) -> None:
    """Pearson r between vessel_count and trade_volume_index > 0.5 in disruption windows."""
    shipping = connector.generate_synthetic(days=365, seed=42)
    market = market_connector.generate_dataset(days=365, seed=42)

    mask = shipping["is_disruption"].to_numpy()
    vc = shipping.loc[mask, "vessel_count"].to_numpy(dtype=float)
    tvi = market.loc[mask, "trade_volume_index"].to_numpy(dtype=float)

    corr = float(np.corrcoef(vc, tvi)[0, 1])
    print(
        f"\n[test] vessel_count <-> trade_volume_index Pearson r "
        f"(disruption days, n={int(mask.sum())}): {corr:.3f}"
    )
    assert corr > 0.5, f"Expected r > 0.5 during disruption windows; got {corr:.3f}"


# ---------------------------------------------------------------------------
# Market connector — CSV mode
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MARKET_CSVS_PRESENT,
    reason="FRED CSVs not present in data/raw/",
)
class TestMarketCsvMode:
    """Validate the FRED CSV ingestion path (Brent + freight PPI + services)."""

    def test_csv_mode_has_expected_columns(
        self, market_csv_connector: MarketConnector
    ) -> None:
        df = market_csv_connector.load_from_csv()
        expected = {
            "timestamp",
            "brent_crude_usd",
            "trade_volume_index",
            "freight_rate_index",
            "freight_services_pct_change",
            "is_disruption",
        }
        assert expected.issubset(set(df.columns))

    def test_csv_mode_brent_in_historical_range(
        self, market_csv_connector: MarketConnector
    ) -> None:
        df = market_csv_connector.load_from_csv()
        # Brent crude history: trough ~$9 (1998 collapse), peak ~$147 (2008).
        assert df["brent_crude_usd"].min() >= 5.0
        assert df["brent_crude_usd"].max() <= 200.0

    def test_csv_mode_trade_volume_in_unit_interval(
        self, market_csv_connector: MarketConnector
    ) -> None:
        df = market_csv_connector.load_from_csv()
        assert df["trade_volume_index"].between(0.0, 1.0).all()

    def test_csv_mode_freight_rebased(
        self, market_csv_connector: MarketConnector
    ) -> None:
        """Trailing 2-year freight index should sit close to 100 after rebasing."""
        df = market_csv_connector.load_from_csv()
        recent_cutoff = df["timestamp"].max() - pd.Timedelta(days=365 * 2)
        recent = df["freight_rate_index"][df["timestamp"] >= recent_cutoff].dropna()
        assert recent.mean() == pytest.approx(100.0, rel=0.05)

    def test_csv_mode_date_range_filter(
        self, market_csv_connector: MarketConnector
    ) -> None:
        df = market_csv_connector.load_from_csv(
            start_date="2024-01-01", end_date="2025-12-31"
        )
        assert df["timestamp"].min() >= pd.Timestamp("2024-01-01")
        assert df["timestamp"].max() <= pd.Timestamp("2025-12-31")

    def test_csv_mode_disruption_label_populated(
        self, market_csv_connector: MarketConnector
    ) -> None:
        """Across 30+ years the co-spike rule must fire at least once."""
        df = market_csv_connector.load_from_csv()
        assert df["is_disruption"].sum() > 0

    def test_csv_mode_fetch_dispatches_to_csv(
        self, market_csv_connector: MarketConnector
    ) -> None:
        df = market_csv_connector.fetch()
        assert "freight_services_pct_change" in df.columns


# ---------------------------------------------------------------------------
# Market connector — API mode
# ---------------------------------------------------------------------------


class TestMarketApiMode:
    def test_api_mode_raises_not_implemented_via_fetch(self) -> None:
        c = MarketConnector(source_mode="api", config={})
        with pytest.raises(NotImplementedError, match="FRED"):
            c.fetch()

    def test_api_mode_raises_not_implemented_direct(self) -> None:
        c = MarketConnector(source_mode="api", config={})
        with pytest.raises(NotImplementedError):
            c.fetch_from_api()


# ---------------------------------------------------------------------------
# Market connector — synthetic-mode dispatch via fetch()
# ---------------------------------------------------------------------------


class TestMarketSyntheticMode:
    """No-regression: synthetic generation still produces the legacy schema."""

    def test_synthetic_fetch_dispatches_to_synthetic(
        self, market_connector: MarketConnector
    ) -> None:
        out = market_connector.fetch()
        assert set(out.columns) == {
            "timestamp",
            "brent_crude_usd",
            "trade_volume_index",
            "freight_rate_index",
            "is_disruption",
        }
        assert len(out) == 365


# ---------------------------------------------------------------------------
# Market connector — alignment with shipping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_MARKET_CSVS_PRESENT and _CSV_PATH.exists()),
    reason="Real CSVs (shipping + market) not present in data/raw/",
)
class TestMarketAlignment:
    def test_alignment_matches_shipping_dates(
        self,
        csv_connector: ShippingConnector,
        market_csv_connector: MarketConnector,
    ) -> None:
        shipping = csv_connector.load_from_csv()
        market = market_csv_connector.load_from_csv()
        aligned = market_csv_connector.align_with_shipping(
            shipping, market_df=market
        )
        np.testing.assert_array_equal(
            pd.to_datetime(aligned["timestamp"]).to_numpy(),
            pd.to_datetime(shipping["timestamp"]).to_numpy(),
        )

    def test_alignment_no_remaining_gaps_in_brent(
        self,
        csv_connector: ShippingConnector,
        market_csv_connector: MarketConnector,
    ) -> None:
        shipping = csv_connector.load_from_csv()
        aligned = market_csv_connector.align_with_shipping(shipping)
        assert aligned["brent_crude_usd"].notna().all(), (
            "Aligned market data must be gap-free (weekends ffilled)."
        )

    def test_alignment_loads_internally_when_market_df_omitted(
        self,
        csv_connector: ShippingConnector,
        market_csv_connector: MarketConnector,
    ) -> None:
        shipping = csv_connector.load_from_csv()
        aligned = market_csv_connector.align_with_shipping(shipping)
        assert len(aligned) == len(shipping)


# ---------------------------------------------------------------------------
# Cross-source correlation on real CSV data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_MARKET_CSVS_PRESENT and _CSV_PATH.exists()),
    reason="Real CSVs (shipping + market) not present in data/raw/",
)
def test_cross_correlation_real_data(
    csv_connector: ShippingConnector,
    market_csv_connector: MarketConnector,
) -> None:
    """Brent crude should anti-correlate with shipping vessel count.

    When the Strait of Hormuz is disrupted, vessel arrivals drop while oil
    prices spike — the Pearson r between brent_crude_usd and vessel_count
    on the overlapping date range should be negative.
    """
    shipping = csv_connector.load_from_csv()
    aligned = market_csv_connector.align_with_shipping(shipping)

    overlap = shipping.merge(
        aligned[["timestamp", "brent_crude_usd", "trade_volume_index"]],
        on="timestamp",
        how="inner",
    )
    mask = overlap["brent_crude_usd"].notna() & overlap["vessel_count"].notna()
    overlap = overlap.loc[mask]

    r_brent_vc = float(
        np.corrcoef(overlap["brent_crude_usd"], overlap["vessel_count"])[0, 1]
    )
    r_tvi_vc = float(
        np.corrcoef(overlap["trade_volume_index"], overlap["vessel_count"])[0, 1]
    )
    print(
        f"\n[test/csv] Real-data Pearson r over n={len(overlap)} days: "
        f"brent_crude_usd <-> vessel_count = {r_brent_vc:+.3f}; "
        f"trade_volume_index <-> vessel_count = {r_tvi_vc:+.3f}"
    )
    assert r_brent_vc < 0, (
        f"Expected negative Brent ↔ vessel_count correlation; got {r_brent_vc:+.3f}"
    )


# ---------------------------------------------------------------------------
# Synthetic-schema CSV round-trip (load_csv) + graceful API fallback (fetch_api)
# ---------------------------------------------------------------------------
def test_shipping_load_csv_round_trip(tmp_path: Path) -> None:
    """save_raw(synthetic) → load_csv() reproduces schema, length, and labels."""
    conn = ShippingConnector(source_mode="synthetic")
    original = conn.generate_dataset(days=365, seed=42)
    csv_path = tmp_path / "shipping_hormuz.csv"
    conn.save_raw(original, path=csv_path)

    loaded = conn.load_csv(path=csv_path)
    assert list(loaded.columns) == [
        "timestamp", "vessel_count", "avg_delay_hours",
        "congestion_index", "oil_price_usd", "is_disruption",
    ]
    assert len(loaded) == len(original)
    assert loaded["is_disruption"].dtype == bool
    assert int(loaded["is_disruption"].sum()) == int(original["is_disruption"].sum())
    assert not loaded[["timestamp", "vessel_count"]].isna().any().any()


def test_market_load_csv_round_trip(tmp_path: Path) -> None:
    """save_raw(synthetic) → load_csv() reproduces schema, length, and labels."""
    conn = MarketConnector(source_mode="synthetic")
    original = conn.generate_dataset(days=365, seed=42)
    csv_path = tmp_path / "market_data.csv"
    conn.save_raw(original, path=csv_path)

    loaded = conn.load_csv(path=csv_path)
    assert list(loaded.columns) == [
        "timestamp", "brent_crude_usd", "trade_volume_index",
        "freight_rate_index", "is_disruption",
    ]
    assert len(loaded) == len(original)
    assert int(loaded["is_disruption"].sum()) == int(original["is_disruption"].sum())


def test_shipping_load_csv_rejects_bad_schema(tmp_path: Path) -> None:
    """load_csv raises ValueError when required columns are absent."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2025-01-01"], "foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        ShippingConnector(source_mode="synthetic").load_csv(path=bad)


def test_load_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ShippingConnector(source_mode="synthetic").load_csv(path=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        MarketConnector(source_mode="synthetic").load_csv(path=tmp_path / "nope.csv")


def test_shipping_fetch_api_falls_back_to_synthetic(caplog) -> None:
    """fetch_api warns and returns a usable synthetic frame instead of raising."""
    import logging

    conn = ShippingConnector(source_mode="synthetic")
    with caplog.at_level(logging.WARNING):
        df = conn.fetch_api()
    assert "API mode not configured" in caplog.text
    assert len(df) == 365
    assert {"vessel_count", "is_disruption"}.issubset(df.columns)


def test_market_fetch_api_falls_back_to_synthetic(caplog) -> None:
    import logging

    conn = MarketConnector(source_mode="synthetic")
    with caplog.at_level(logging.WARNING):
        df = conn.fetch_api()
    assert "API mode not configured" in caplog.text
    assert len(df) == 365
    assert {"brent_crude_usd", "is_disruption"}.issubset(df.columns)
