"""Tests for the shipping ingestion connector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion import MarketConnector, ShippingConnector


@pytest.fixture()
def connector() -> ShippingConnector:
    return ShippingConnector(config={})


@pytest.fixture()
def df(connector: ShippingConnector) -> pd.DataFrame:
    return connector.generate_dataset(days=365, seed=42)


def test_dataset_has_expected_shape_and_columns(df: pd.DataFrame) -> None:
    assert len(df) == 365
    assert set(df.columns) == {
        "timestamp",
        "vessel_count",
        "avg_delay_hours",
        "congestion_index",
        "oil_price_usd",
        "is_disruption",
    }


def test_no_nan_values(df: pd.DataFrame) -> None:
    assert not df.isna().any().any()


def test_congestion_within_unit_interval(df: pd.DataFrame) -> None:
    assert df["congestion_index"].between(0.0, 1.0).all()


def test_vessel_count_non_negative(df: pd.DataFrame) -> None:
    assert (df["vessel_count"] >= 0).all()


def test_disruption_day_count_roughly_46(df: pd.DataFrame) -> None:
    n_disrupt = int(df["is_disruption"].sum())
    assert 40 <= n_disrupt <= 55, f"Expected ~46 disruption days, got {n_disrupt}"


def test_normal_vs_disruption_distinguishable(df: pd.DataFrame) -> None:
    normal = df.loc[~df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
    disrupted = df.loc[df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
    diff = normal.mean() - disrupted.mean()
    se = np.sqrt(
        normal.var(ddof=1) / len(normal) + disrupted.var(ddof=1) / len(disrupted)
    )
    t_stat = diff / se
    print(f"\n[test] vessel_count Welch t-statistic: {t_stat:.2f}")
    assert t_stat > 5, "Normal and disruption distributions should differ strongly."


def test_disruptions_cover_expected_windows(df: pd.DataFrame) -> None:
    flags = df["is_disruption"].to_numpy()
    # Sample one day inside each scenario's core window.
    assert flags[65] and flags[160] and flags[285]
    # Day 0 is normal.
    assert not flags[0]


def test_seed_is_reproducible(connector: ShippingConnector) -> None:
    a = connector.generate_dataset(seed=42)
    b = connector.generate_dataset(seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_diverge(connector: ShippingConnector) -> None:
    a = connector.generate_dataset(seed=42)
    b = connector.generate_dataset(seed=7)
    assert not a["vessel_count"].equals(b["vessel_count"])


def test_validate_passes_on_generated(
    connector: ShippingConnector, df: pd.DataFrame
) -> None:
    assert connector.validate(df) is True


def test_validate_rejects_out_of_range_congestion(
    connector: ShippingConnector, df: pd.DataFrame
) -> None:
    bad = df.copy()
    bad.loc[0, "congestion_index"] = 1.5
    assert connector.validate(bad) is False


def test_signal_records_match_unified_schema(
    connector: ShippingConnector,
) -> None:
    small = connector.generate_dataset(days=10)
    records = connector.to_signal_records(small)
    assert len(records) == 10 * len(connector.FEATURE_COLUMNS)
    sample = records[0]
    assert set(sample.keys()) == {"timestamp", "source", "feature", "value", "location"}
    assert sample["source"] == "shipping"
    assert sample["location"] == "Strait of Hormuz"
    assert isinstance(sample["value"], float)
    json.dumps(records)


def test_save_raw_to_tmp(
    connector: ShippingConnector, tmp_path: Path
) -> None:
    target = tmp_path / "raw" / "shipping_hormuz.csv"
    written = connector.save_raw(target)
    assert written.exists()
    reloaded = pd.read_csv(written)
    assert len(reloaded) == 365
    assert "is_disruption" in reloaded.columns


def test_save_raw_to_canonical_location() -> None:
    """Persist the canonical artefact under data/raw/ for downstream agents."""
    connector = ShippingConnector(config={})
    written = connector.save_raw()
    assert written.exists()
    assert written.name == "shipping_hormuz.csv"


# ---------------------------------------------------------------------------
# Market connector tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def market_connector() -> MarketConnector:
    return MarketConnector(config={})


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
    # Window 2 (Major Blockage) starts at day 150 in shipping;
    # market peak must appear no earlier than day 152.
    window = market.iloc[150:175]
    peak_day = int(window["brent_crude_usd"].idxmax())
    assert peak_day >= 152, f"Expected lagged peak >= day 152, got {peak_day}"


def test_market_mean_reversion_after_window(
    market_connector: MarketConnector,
) -> None:
    """Freight index must decay back toward baseline after the window ends."""
    market = market_connector.generate_dataset(days=365, seed=42)
    end_day = 170  # end of Major Blockage scenario
    elevated = market.loc[end_day, "freight_rate_index"]
    settled = market.loc[end_day + 15, "freight_rate_index"]
    baseline_band_top = 130.0
    assert elevated > settled, "Freight index should decay after the window closes."
    assert settled < baseline_band_top, (
        f"Freight index should mean-revert below {baseline_band_top}; got {settled:.2f}"
    )


def test_market_validate_passes_on_generated(
    market_connector: MarketConnector, market_df: pd.DataFrame
) -> None:
    assert market_connector.validate(market_df) is True


def test_market_validate_rejects_out_of_range_volume(
    market_connector: MarketConnector, market_df: pd.DataFrame
) -> None:
    bad = market_df.copy()
    bad.loc[0, "trade_volume_index"] = 1.5
    assert market_connector.validate(bad) is False


def test_market_signal_records_match_unified_schema(
    market_connector: MarketConnector,
) -> None:
    small = market_connector.generate_dataset(days=10)
    records = market_connector.to_signal_records(small)
    assert len(records) == 10 * len(market_connector.FEATURE_COLUMNS)
    sample = records[0]
    assert set(sample.keys()) == {"timestamp", "source", "feature", "value", "location"}
    assert sample["source"] == "market"
    assert sample["location"] == "Strait of Hormuz"
    assert isinstance(sample["value"], float)
    json.dumps(records)


def test_market_save_raw_to_tmp(
    market_connector: MarketConnector, tmp_path: Path
) -> None:
    target = tmp_path / "raw" / "market_data.csv"
    written = market_connector.save_raw(target)
    assert written.exists()
    reloaded = pd.read_csv(written)
    assert len(reloaded) == 365
    assert "is_disruption" in reloaded.columns


def test_market_save_raw_to_canonical_location() -> None:
    """Persist the canonical artefact under data/raw/ for downstream agents."""
    connector = MarketConnector(config={})
    written = connector.save_raw()
    assert written.exists()
    assert written.name == "market_data.csv"


def test_market_ingestion_correlates_with_shipping(
    market_connector: MarketConnector, connector: ShippingConnector
) -> None:
    """Pearson r between vessel_count and trade_volume_index > 0.5 in disruption windows."""
    shipping = connector.generate_dataset(days=365, seed=42)
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
