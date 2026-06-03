"""End-to-end scenario tests for the DSS pipeline.

These tests exercise the orchestrator with synthetic signal data that
mimics plausible Strait of Hormuz disruption scenarios.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.base_agent import BaseAgent, DetectionResult
from src.agents.market_agent import MarketAgent
from src.agents.shipping_agent import ShippingAgent
from src.aggregation.risk_engine import RiskLevel
from src.orchestrator import Orchestrator

_CONFIG = {
    "weights": {"shipping": 0.4, "market": 0.3, "geopolitical": 0.3},
    "thresholds": {"risk_critical": 0.8, "risk_high": 0.7, "risk_medium": 0.4},
}


class _ThresholdAgent(BaseAgent):
    """Agent that scores each row as the mean of its numeric features."""

    def fit(self, df: pd.DataFrame) -> None:
        self._is_fitted = True

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        scores = df.mean(axis=1).to_numpy()
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=scores,
            anomaly_flags=scores > 0.5,
            feature_names=list(df.columns),
        )


@pytest.fixture()
def orchestrator() -> Orchestrator:
    return Orchestrator(config=_CONFIG)


def _normal_signals() -> pd.DataFrame:
    """Synthetic data representing a calm, undisrupted period."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "vessel_count": rng.uniform(0.1, 0.3, 30),
            "oil_price_change": rng.uniform(0.05, 0.15, 30),
            "incident_index": rng.uniform(0.0, 0.2, 30),
        }
    )


def _disrupted_signals() -> pd.DataFrame:
    """Synthetic data representing an active disruption event."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "vessel_count": rng.uniform(0.7, 1.0, 30),
            "oil_price_change": rng.uniform(0.75, 0.95, 30),
            "incident_index": rng.uniform(0.8, 1.0, 30),
        }
    )


def test_no_agents_returns_low_risk(orchestrator: Orchestrator) -> None:
    result = orchestrator.run(_normal_signals())
    assert result["risk_level"] == "LOW"
    assert result["composite_score"] == 0.0


def test_normal_signals_produce_low_risk(orchestrator: Orchestrator) -> None:
    orchestrator.register_agent(_ThresholdAgent("shipping", _CONFIG))
    result = orchestrator.run(_normal_signals())
    assert result["risk_level"] in (RiskLevel.LOW, RiskLevel.MEDIUM)


def test_disrupted_signals_produce_critical_risk(orchestrator: Orchestrator) -> None:
    orchestrator.register_agent(_ThresholdAgent("shipping", _CONFIG))
    orchestrator.register_agent(_ThresholdAgent("market", _CONFIG))
    orchestrator.register_agent(_ThresholdAgent("geopolitical", _CONFIG))
    result = orchestrator.run(_disrupted_signals())
    assert result["risk_level"] == RiskLevel.CRITICAL
    assert result["composite_score"] >= 0.8


def test_pipeline_output_has_required_keys(orchestrator: Orchestrator) -> None:
    orchestrator.register_agent(_ThresholdAgent("shipping", _CONFIG))
    result = orchestrator.run(_normal_signals())
    for key in ("composite_score", "risk_level", "agent_scores", "shap", "context"):
        assert key in result


# --------------------------------------------------------------------------- #
# Hybrid ingest pipeline — exercises run_full_pipeline / run_timeseries.      #
# --------------------------------------------------------------------------- #


_SHUAIBA_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "shuaiba_arrivals.csv"
)
_BRENT_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "brent_crude.csv"
)
_FREIGHT_PPI_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "freight_ppi.csv"
)
_FREIGHT_SERVICES_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "freight_services.csv"
)
_REAL_DATA_PRESENT = all(
    p.exists() for p in (
        _SHUAIBA_CSV, _BRENT_CSV, _FREIGHT_PPI_CSV, _FREIGHT_SERVICES_CSV
    )
)


def _hybrid_config(shipping_mode: str = "csv", market_mode: str = "csv") -> dict:
    """Build a config dict that wires up real or synthetic ingestion."""
    return {
        "ingestion": {
            "shipping": {
                "source_mode": shipping_mode,
                "csv_path": str(_SHUAIBA_CSV),
            },
            "market": {
                "source_mode": market_mode,
                "brent_crude_path": str(_BRENT_CSV),
                "freight_ppi_path": str(_FREIGHT_PPI_CSV),
                "freight_services_path": str(_FREIGHT_SERVICES_CSV),
            },
        },
        "weights": {"shipping": 0.4, "market": 0.3, "geopolitical": 0.3},
        "thresholds": {
            "risk_critical": 0.8,
            "risk_high": 0.6,
            "risk_medium": 0.4,
        },
    }


def test_orchestrator_initialises_connectors_from_config() -> None:
    """Connectors must come up in the configured source mode."""
    config = _hybrid_config(shipping_mode="synthetic", market_mode="synthetic")
    orc = Orchestrator(config=config)
    assert orc._shipping_mode == "synthetic"
    assert orc._market_mode == "synthetic"
    assert orc._shipping_connector.source_mode == "synthetic"
    assert orc._market_connector.source_mode == "synthetic"


def test_orchestrator_handles_missing_ingestion_config() -> None:
    """Without an ingestion block both connectors default to synthetic."""
    orc = Orchestrator(config=_CONFIG)
    assert orc._shipping_mode == "synthetic"
    assert orc._market_mode == "synthetic"


def test_orchestrator_ingest_synthetic_merges_on_timestamp() -> None:
    """Ingest must merge shipping + market on a single timestamp axis."""
    orc = Orchestrator(
        config=_hybrid_config(shipping_mode="synthetic", market_mode="synthetic")
    )
    combined = orc.ingest()
    assert "timestamp" in combined.columns
    assert "vessel_count" in combined.columns
    assert "brent_crude_usd" in combined.columns
    # Synthetic both sides produce 365 daily rows starting 2025-01-01.
    assert len(combined) == 365
    # oil_price_usd should be backfilled from brent_crude_usd.
    assert combined["oil_price_usd"].notna().all()


def test_orchestrator_full_pipeline_synthetic_returns_required_keys() -> None:
    """run_full_pipeline must return aggregated risk + ingest summary."""
    config = _hybrid_config(shipping_mode="synthetic", market_mode="synthetic")
    orc = Orchestrator(config=config)
    orc.register_agent(ShippingAgent(config={}))
    orc.register_agent(MarketAgent(config={"z_threshold": 1.2, "threshold": 0.4}))

    result = orc.run_full_pipeline()
    for key in ("composite_score", "risk_level", "agent_scores", "shap", "context", "data"):
        assert key in result
    assert {"shipping", "market"}.issubset(result["agent_scores"].keys())
    assert 0.0 <= result["composite_score"] <= 1.0
    assert result["data"]["rows"] == 365


def test_orchestrator_run_timeseries_returns_daily_risk_series() -> None:
    """run_timeseries_analysis must emit a per-day composite score frame."""
    config = _hybrid_config(shipping_mode="synthetic", market_mode="synthetic")
    orc = Orchestrator(config=config)
    orc.register_agent(ShippingAgent(config={}))
    orc.register_agent(MarketAgent(config={"z_threshold": 1.2, "threshold": 0.4}))

    ts = orc.run_timeseries_analysis()
    assert {
        "timestamp", "shipping_score", "market_score",
        "composite_score", "risk_level",
    }.issubset(ts.columns)
    assert ts["composite_score"].between(0.0, 1.0).all()
    assert set(ts["risk_level"].unique()).issubset(
        {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    )
    # Synthetic data should surface at least one HIGH/CRITICAL day during
    # the injected Major Blockage window.
    assert (ts["risk_level"].isin(["HIGH", "CRITICAL"])).any()


def test_orchestrator_falls_back_to_synthetic_when_csv_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing CSV must trigger a logged fallback to synthetic, not a crash."""
    bogus_path = tmp_path / "does_not_exist.csv"
    config = {
        "ingestion": {
            "shipping": {"source_mode": "csv", "csv_path": str(bogus_path)},
            "market": {"source_mode": "synthetic"},
        },
        "weights": {"shipping": 0.4, "market": 0.3, "geopolitical": 0.3},
        "thresholds": {
            "risk_critical": 0.8, "risk_high": 0.6, "risk_medium": 0.4,
        },
    }
    orc = Orchestrator(config=config)
    with caplog.at_level(logging.WARNING):
        combined = orc.ingest()
    assert orc._shipping_connector.source_mode == "synthetic"
    assert any("falling back to synthetic" in r.message for r in caplog.records)
    assert "vessel_count" in combined.columns


def test_orchestrator_warns_when_market_coverage_short(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Market data that doesn't span shipping range must log a coverage warning."""

    class _StubMarket:
        source_mode = "synthetic"

        def fetch(self) -> pd.DataFrame:
            # Cover only the first half of the shipping window.
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range("2025-01-01", periods=100, freq="D"),
                    "brent_crude_usd": np.full(100, 80.0),
                    "trade_volume_index": np.full(100, 0.9),
                    "freight_rate_index": np.full(100, 110.0),
                    "is_disruption": np.zeros(100, dtype=bool),
                }
            )

        def align_with_shipping(self, shipping_df, market_df):
            ts = pd.to_datetime(shipping_df["timestamp"]).sort_values()
            out = (
                market_df.set_index("timestamp")
                .reindex(ts.unique())
                .ffill()
                .bfill()
                .rename_axis("timestamp")
                .reset_index()
            )
            return out

    config = _hybrid_config(shipping_mode="synthetic", market_mode="synthetic")
    orc = Orchestrator(config=config)
    orc._market_connector = _StubMarket()  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING):
        orc.ingest()
    assert any(
        "does not fully cover shipping range" in r.message
        for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# Real-data integration (skipped when CSVs absent).                           #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _REAL_DATA_PRESENT,
    reason="Real Shuaiba + FRED CSVs not all present in data/raw/",
)
def test_orchestrator_full_pipeline_real_data() -> None:
    """End-to-end ingest + detect + aggregate on real Shuaiba + FRED CSVs."""
    config = _hybrid_config(shipping_mode="csv", market_mode="csv")
    orc = Orchestrator(config=config)
    orc.register_agent(
        ShippingAgent(
            config={"contamination": 0.05, "threshold": 0.55, "z_threshold": 2.0}
        )
    )
    orc.register_agent(
        MarketAgent(
            config={"z_threshold": 1.5, "threshold": 0.50, "baseline_years": 5}
        )
    )
    result = orc.run_full_pipeline()
    assert result["data"]["rows"] == 2699
    assert result["data"]["start"] == "2019-01-01"
    assert {"shipping", "market"}.issubset(result["agent_scores"].keys())
    assert 0.0 <= result["composite_score"] <= 1.0


@pytest.mark.skipif(
    not _REAL_DATA_PRESENT,
    reason="Real Shuaiba + FRED CSVs not all present in data/raw/",
)
def test_orchestrator_timeseries_real_data_flags_2026_shutdown() -> None:
    """Time-series analysis on real data must escalate during the 2026 shutdown."""
    config = _hybrid_config(shipping_mode="csv", market_mode="csv")
    orc = Orchestrator(config=config)
    orc.register_agent(
        ShippingAgent(
            config={"contamination": 0.05, "threshold": 0.55, "z_threshold": 2.0}
        )
    )
    orc.register_agent(
        MarketAgent(
            config={"z_threshold": 1.5, "threshold": 0.50, "baseline_years": 5}
        )
    )
    ts = orc.run_timeseries_analysis()
    shutdown = ts.loc[
        ts["timestamp"].between(
            pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-15")
        )
    ]
    assert not shutdown.empty
    assert (
        shutdown["risk_level"].isin(["HIGH", "CRITICAL"]).any()
    ), "Expected HIGH/CRITICAL risk during the April-May 2026 Hormuz shutdown."


# --------------------------------------------------------------------------- #
# Historical-event scenarios — verify the orchestrator behaves sanely across  #
# known windows in the real Shuaiba + FRED record.                            #
#                                                                             #
# These tests use conservative agent thresholds tuned to the production       #
# discrimination characteristics demonstrated in test_agents.py — see the     #
# probe in commit fa28629 for the empirical risk-level distribution.          #
# --------------------------------------------------------------------------- #


def _historical_orchestrator() -> Orchestrator:
    """Build an orchestrator wired for real CSV ingestion + conservative agents.

    The market agent's ``baseline_years=10`` is needed so the 2019 and 2020
    test windows actually have a rolling baseline (the default 5-year clip
    would exclude them from the time series entirely).
    """
    config = _hybrid_config(shipping_mode="csv", market_mode="csv")
    orc = Orchestrator(config=config)
    orc.register_agent(
        ShippingAgent(
            config={"contamination": 0.05, "threshold": 0.60, "z_threshold": 2.5}
        )
    )
    orc.register_agent(
        MarketAgent(
            config={
                "z_threshold": 3.0,
                "threshold": 0.70,
                "baseline_years": 10,
            }
        )
    )
    return orc


def _window(ts: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return ts.loc[
        ts["timestamp"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]


@pytest.fixture(scope="module")
def historical_timeseries() -> pd.DataFrame:
    if not _REAL_DATA_PRESENT:
        pytest.skip("Real Shuaiba + FRED CSVs not all present in data/raw/")
    orc = _historical_orchestrator()
    ts = orc.run_timeseries_analysis()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    return ts


@pytest.mark.skipif(
    not _REAL_DATA_PRESENT,
    reason="Real Shuaiba + FRED CSVs not all present in data/raw/",
)
def test_2026_hormuz_shutdown(historical_timeseries: pd.DataFrame) -> None:
    """March-May 2026 Hormuz shutdown must surface as HIGH-level risk."""
    window = _window(historical_timeseries, "2026-03-01", "2026-05-22")
    assert not window.empty
    escalated = window["risk_level"].isin(["HIGH", "CRITICAL"]).sum()
    print(
        f"\n[scenario/2026-hormuz] n={len(window)} avg_score="
        f"{window['composite_score'].mean():.3f} levels="
        f"{dict(window['risk_level'].value_counts())}"
    )
    # At least a quarter of the window must reach HIGH or CRITICAL — the
    # Hormuz shutdown is the canonical high-risk event in the dataset.
    assert escalated / len(window) >= 0.25, (
        f"Expected >=25% of Mar-May 2026 days at HIGH/CRITICAL; got "
        f"{escalated}/{len(window)}."
    )
    assert window["composite_score"].max() >= 0.6, (
        f"Expected peak composite_score >= 0.6 during the shutdown; got "
        f"{window['composite_score'].max():.3f}."
    )


@pytest.mark.skipif(
    not _REAL_DATA_PRESENT,
    reason="Real Shuaiba + FRED CSVs not all present in data/raw/",
)
def test_2019_tanker_attacks(historical_timeseries: pd.DataFrame) -> None:
    """June-July 2019 Iranian-corridor tanker incidents must show >= MEDIUM risk."""
    window = _window(historical_timeseries, "2019-06-01", "2019-07-31")
    assert not window.empty
    elevated = window["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"]).sum()
    print(
        f"\n[scenario/2019-tanker] n={len(window)} avg_score="
        f"{window['composite_score'].mean():.3f} levels="
        f"{dict(window['risk_level'].value_counts())}"
    )
    assert elevated >= 5, (
        f"Expected >=5 days at MEDIUM+ during Jun-Jul 2019 tanker attacks; "
        f"got {elevated}/{len(window)}."
    )


@pytest.mark.skipif(
    not _REAL_DATA_PRESENT,
    reason="Real Shuaiba + FRED CSVs not all present in data/raw/",
)
def test_normal_period_2023(historical_timeseries: pd.DataFrame) -> None:
    """Jan-Jun 2023 (no major Hormuz incident) must stay predominantly LOW."""
    window = _window(historical_timeseries, "2023-01-01", "2023-06-30")
    assert not window.empty
    low_fraction = (window["risk_level"] == "LOW").sum() / len(window)
    critical_fraction = (window["risk_level"] == "CRITICAL").sum() / len(window)
    print(
        f"\n[scenario/2023-normal] n={len(window)} avg_score="
        f"{window['composite_score'].mean():.3f} levels="
        f"{dict(window['risk_level'].value_counts())}"
    )
    assert low_fraction >= 0.5, (
        f"Expected >=50% of Jan-Jun 2023 days at LOW risk (calm baseline); "
        f"got {low_fraction:.2%}."
    )
    assert critical_fraction <= 0.05, (
        f"Expected <=5% of 2023 days at CRITICAL; got {critical_fraction:.2%}."
    )


@pytest.mark.skipif(
    not _REAL_DATA_PRESENT,
    reason="Real Shuaiba + FRED CSVs not all present in data/raw/",
)
def test_covid_impact(historical_timeseries: pd.DataFrame) -> None:
    """March-April 2020 (COVID demand shock) must surface as elevated risk."""
    window = _window(historical_timeseries, "2020-03-01", "2020-04-30")
    assert not window.empty
    elevated = window["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"]).sum()
    print(
        f"\n[scenario/covid] n={len(window)} avg_score="
        f"{window['composite_score'].mean():.3f} levels="
        f"{dict(window['risk_level'].value_counts())}"
    )
    assert elevated >= 5, (
        f"Expected >=5 days at MEDIUM+ during Mar-Apr 2020 COVID shock; "
        f"got {elevated}/{len(window)}."
    )
