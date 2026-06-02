"""Unit tests for the detection agent contracts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.agents.base_agent import BaseAgent, DetectionResult
from src.agents.market_agent import MarketAgent
from src.agents.shipping_agent import ShippingAgent
from src.ingestion import MarketConnector, ShippingConnector


class _ConcreteAgent(BaseAgent):
    """Minimal concrete agent that flags everything above 0.5 as something that
     deviates from the standard, normal behavior."""

    def fit(self, df: pd.DataFrame) -> None:
        self._is_fitted = True

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        scores = df.iloc[:, 0].to_numpy().astype(float)
        flags = scores > 0.5
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=scores,
            anomaly_flags=flags,
            feature_names=list(df.columns),
        )

    """SO ITS JUST AGENTS LOGIC IS TESTED THAT IF ITS ABOVE =:% THEN TRUE ELSE FALSE,
    AND THE SHAPE OF THE SCORES AND FLAGS ARE CORRECT, AND THE DETECT FUNCTION RETURNS
    THE RIGHT DATACLASS, AND THE FIT FUNCTION SETS THE FITTED FLAG TO TRUE. ALSO TESTS 
    THAT THE BASE AGENT IS ABSTRACT AND CANNOT BE INSTANTIATED."""
@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"signal": [0.1, 0.6, 0.9, 0.3, 0.8]})

@pytest.fixture()
def agent() -> _ConcreteAgent:
    return _ConcreteAgent(name="test_agent", config={})


def test_fit_sets_fitted_flag(agent: _ConcreteAgent, sample_df: pd.DataFrame) -> None:
    agent.fit(sample_df)
    assert agent._is_fitted is True


"""  Downstream code (the cross-agent aggregator, the LLM-facing summarizer) will assume 
    every agent hands back the same dataclass. If a future agent author 
    returns something else, this test fails immediately instead of breaking 
    the pipeline at runtime. """
def test_detect_returns_detection_result(
    agent: _ConcreteAgent, sample_df: pd.DataFrame
) -> None:
    agent.fit(sample_df)
    result = agent.detect(sample_df)
    assert isinstance(result, DetectionResult)

"""  The downstream code will also assume the shapes of the scores and flags. 
    If an agent author returns something else, this test fails immediately instead
    of breaking the pipeline at runtime. """
def test_detection_result_shapes(
    agent: _ConcreteAgent, sample_df: pd.DataFrame
) -> None:
    result = agent.fit_detect(sample_df)
    assert result.anomaly_scores.shape == (len(sample_df),)
    assert result.anomaly_flags.shape == (len(sample_df),)

"""  Finally, we test that the logic of the agent is correct. 
    If an agent author makes a mistake in their logic, this test fails
    immediately instead of breaking the pipeline at runtime. """
def test_detection_flags_correct_rows(
    agent: _ConcreteAgent, sample_df: pd.DataFrame
) -> None:
    result = agent.fit_detect(sample_df)
    expected_flags = np.array([False, True, True, False, True]) 
    np.testing.assert_array_equal(result.anomaly_flags, expected_flags)


def test_base_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseAgent(name="x", config={})  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# ShippingAgent — end-to-end evaluation against the synthetic Hormuz dataset. #
# Loads the CSV produced by the Phase-1 ShippingConnector, fits the agent on  #
# non-disruption rows only (no leakage), runs the full preprocess → detect →  #
# validate pipeline, then prints a confusion matrix, classification report,   #
# and TPR/FPR metrics. The contract: TPR >= 0.8 and FPR <= 0.15.              #
# --------------------------------------------------------------------------- #


_SHIPPING_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "shipping_hormuz.csv"
)


@pytest.fixture(scope="module")
def shipping_df() -> pd.DataFrame:
    if not _SHIPPING_CSV.exists():
        pytest.skip(
            f"shipping CSV not found at {_SHIPPING_CSV} — generate it via "
            "ShippingConnector(config={}).save_raw() first."
        )
    df = pd.read_csv(_SHIPPING_CSV, parse_dates=["timestamp"])
    df["is_disruption"] = df["is_disruption"].astype(bool)
    return df


@pytest.fixture()
def shipping_agent() -> ShippingAgent:
    return ShippingAgent(
        config={"contamination": 0.13, "threshold": 0.55, "z_threshold": 3.0}
    )


def test_shipping_agent_evaluation(
    shipping_agent: ShippingAgent, shipping_df: pd.DataFrame
) -> None:
    """End-to-end evaluation: TPR >= 0.8, FPR <= 0.15 on synthetic data."""

    shipping_agent.fit(shipping_df)
    validated = shipping_agent.run_dataframe(shipping_df)
    reports = shipping_agent.run(shipping_df)

    y_true = shipping_df["is_disruption"].astype(bool).to_numpy()
    y_pred = validated["validated"].astype(bool).to_numpy()
    assert y_true.shape == y_pred.shape, "row alignment broken"

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    print("\n" + "=" * 70)
    print("ShippingAgent — End-to-End Evaluation on Synthetic Hormuz Dataset")
    print("=" * 70)
    print(f"Total rows               : {len(shipping_df)}")
    print(f"Ground-truth disruption  : {int(y_true.sum())} days")
    print(f"Predicted (validated)    : {int(y_pred.sum())} days")
    print(f"Anomaly windows reported : {len(reports)}")
    print()
    print("Confusion Matrix")
    print("                 pred=Normal   pred=Disruption")
    print(f"true=Normal      {tn:>11d}   {fp:>15d}")
    print(f"true=Disruption  {fn:>11d}   {tp:>15d}")
    print()
    print(f"TPR (recall)     : {tpr:.3f}")
    print(f"FPR              : {fpr:.3f}")
    print(f"Precision        : {precision:.3f}")
    print()
    print("Classification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Disruption"],
            digits=3,
            zero_division=0,
        )
    )
    if reports:
        print("Sample anomaly window report:")
        sample = reports[0]
        for k, v in sample.items():
            print(f"  {k}: {v}")
    print("=" * 70)

    assert tpr >= 0.80, f"TPR {tpr:.3f} below 0.80 threshold"
    assert fpr <= 0.15, f"FPR {fpr:.3f} above 0.15 threshold"


def test_shipping_agent_run_output_schema(
    shipping_agent: ShippingAgent, shipping_df: pd.DataFrame
) -> None:
    """Each anomaly window must conform to the documented dict schema."""
    reports = shipping_agent.run(shipping_df)
    assert reports, "expected at least one anomaly window on synthetic data"
    required = {
        "agent",
        "anomaly_score",
        "confidence",
        "signals",
        "start_timestamp",
        "end_timestamp",
        "location",
    }
    for r in reports:
        assert required.issubset(r.keys())
        assert r["agent"] == "shipping"
        assert r["location"] == "Strait of Hormuz"
        assert 0.0 <= r["anomaly_score"] <= 1.0
        assert 0.0 <= r["confidence"] <= 1.0
        assert set(r["signals"].keys()) == {
            "vessel_count_zscore",
            "delay_zscore",
            "congestion_zscore",
        }


def test_shipping_agent_determinism(shipping_df: pd.DataFrame) -> None:
    """Same input + config must produce identical anomaly scores."""
    cfg = {"contamination": 0.13, "threshold": 0.55, "z_threshold": 3.0}
    a = ShippingAgent(config=cfg)
    b = ShippingAgent(config=cfg)
    a_df = a.run_dataframe(shipping_df)
    b_df = b.run_dataframe(shipping_df)
    np.testing.assert_array_almost_equal(
        a_df["anomaly_score"].to_numpy(),
        b_df["anomaly_score"].to_numpy(),
    )
    np.testing.assert_array_equal(
        a_df["validated"].to_numpy(),
        b_df["validated"].to_numpy(),
    )


def test_shipping_agent_no_leakage(shipping_df: pd.DataFrame) -> None:
    """Fit must drop disruption rows when is_disruption is present."""
    agent = ShippingAgent(config={"contamination": 0.13, "threshold": 0.55})
    agent.fit(shipping_df)
    # The scaler's mean should approximate the non-disruption mean, not the
    # full-dataset mean — confirms only normal rows informed the fit.
    normal = shipping_df.loc[~shipping_df["is_disruption"], ShippingAgent(config={})._feature_columns]
    full = shipping_df[ShippingAgent(config={})._feature_columns]
    fitted_mean = agent._scaler.mean_  # type: ignore[union-attr]
    np.testing.assert_allclose(fitted_mean, normal.mean().to_numpy(), rtol=1e-6)
    assert not np.allclose(fitted_mean, full.mean().to_numpy(), rtol=1e-3)


# --------------------------------------------------------------------------- #
# ShippingAgent — end-to-end evaluation against the REAL Shuaiba PortWatch    #
# dataset (CSV mode). Verifies the agent picks up the optional tanker_count   #
# and vessel_count_trend features, emits the Shuaiba location, and meets the  #
# relaxed TPR >= 0.7 / FPR <= 0.2 target appropriate for noisier real data.   #
# --------------------------------------------------------------------------- #


_SHUAIBA_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "shuaiba_arrivals.csv"
)


@pytest.fixture(scope="module")
def real_shipping_df() -> pd.DataFrame:
    if not _SHUAIBA_CSV.exists():
        pytest.skip(
            f"Shuaiba arrivals CSV not found at {_SHUAIBA_CSV} — required for "
            "real-data shipping agent evaluation."
        )
    connector = ShippingConnector(
        source_mode="csv", config={"csv_path": str(_SHUAIBA_CSV)}
    )
    return connector.fetch()


def test_shipping_agent_real_data_evaluation(
    real_shipping_df: pd.DataFrame,
) -> None:
    """End-to-end on Shuaiba real data: TPR >= 0.7, FPR <= 0.2."""
    agent = ShippingAgent(
        config={"contamination": 0.05, "threshold": 0.55, "z_threshold": 2.0}
    )
    agent.fit(real_shipping_df)
    validated = agent.run_dataframe(real_shipping_df)
    reports = agent.run(real_shipping_df)

    # preprocess() drops warm-up rows where the derived trend feature is NaN
    # (the first 6 days before vessel_count_7dma is defined). Re-align ground
    # truth onto the validated frame's timestamps before scoring.
    y_true = (
        real_shipping_df.set_index("timestamp")["is_disruption"]
        .reindex(pd.to_datetime(validated["timestamp"]))
        .astype(bool)
        .to_numpy()
    )
    y_pred = validated["validated"].astype(bool).to_numpy()
    assert y_true.shape == y_pred.shape

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    print("\n" + "=" * 70)
    print("ShippingAgent — End-to-End Evaluation on REAL Shuaiba Dataset")
    print("=" * 70)
    print(f"Total rows               : {len(real_shipping_df)}")
    print(f"Active features          : {agent._active_features()}")
    print(f"Ground-truth disruption  : {int(y_true.sum())} days")
    print(f"Predicted (validated)    : {int(y_pred.sum())} days")
    print(f"Anomaly windows reported : {len(reports)}")
    print()
    print("Confusion Matrix")
    print("                 pred=Normal   pred=Disruption")
    print(f"true=Normal      {tn:>11d}   {fp:>15d}")
    print(f"true=Disruption  {fn:>11d}   {tp:>15d}")
    print()
    print(f"TPR (recall)     : {tpr:.3f}")
    print(f"FPR              : {fpr:.3f}")
    print(f"Precision        : {precision:.3f}")
    print("=" * 70)

    assert tpr >= 0.70, f"TPR {tpr:.3f} below 0.70 threshold on real data"
    assert fpr <= 0.20, f"FPR {fpr:.3f} above 0.20 threshold on real data"


def test_shipping_agent_real_data_discovers_optional_features(
    real_shipping_df: pd.DataFrame,
) -> None:
    """Agent must auto-discover tanker_count and derive vessel_count_trend."""
    agent = ShippingAgent(config={})
    agent.fit(real_shipping_df)
    assert "tanker_count" in agent._extra_features
    assert "vessel_count_trend" in agent._extra_features


def test_shipping_agent_real_data_emits_shuaiba_location(
    real_shipping_df: pd.DataFrame,
) -> None:
    """Output reports must label the location as Shuaiba Port for real data."""
    agent = ShippingAgent(
        config={"contamination": 0.05, "threshold": 0.55, "z_threshold": 2.0}
    )
    reports = agent.run(real_shipping_df)
    assert reports, "expected at least one anomaly window on real data"
    assert all(r["location"] == "Shuaiba Port, Persian Gulf" for r in reports)
    # The 2026 April-May shutdown must surface in at least one window.
    end_dates = [r["end_timestamp"] for r in reports]
    assert any(ed and ed >= "2026-04-01" for ed in end_dates), (
        "Expected an anomaly window covering the 2026 Hormuz shutdown."
    )


def test_shipping_agent_real_data_signals_include_extras(
    real_shipping_df: pd.DataFrame,
) -> None:
    """Signals dict must include tanker_zscore + trend_zscore on real data."""
    agent = ShippingAgent(
        config={"contamination": 0.05, "threshold": 0.55, "z_threshold": 2.0}
    )
    reports = agent.run(real_shipping_df)
    assert reports
    sample = reports[0]
    assert "tanker_zscore" in sample["signals"]
    assert "trend_zscore" in sample["signals"]
    # Base z-scores must still be present (backwards compatibility).
    assert {
        "vessel_count_zscore", "delay_zscore", "congestion_zscore",
    }.issubset(sample["signals"].keys())


def test_shipping_agent_location_override_via_config(
    real_shipping_df: pd.DataFrame,
) -> None:
    """Explicit config['location'] must take precedence over auto-detection."""
    agent = ShippingAgent(
        config={
            "contamination": 0.05,
            "threshold": 0.55,
            "z_threshold": 2.0,
            "location": "Custom Corridor",
        }
    )
    reports = agent.run(real_shipping_df)
    assert all(r["location"] == "Custom Corridor" for r in reports)


# --------------------------------------------------------------------------- #
# MarketAgent — end-to-end evaluation against the synthetic market dataset.   #
# Loads the CSV produced by the Phase-1 MarketConnector, fits the agent       #
# (schema check only — rolling stats are computed inline), runs the full      #
# preprocess → detect → validate pipeline, and prints metrics. Contract:      #
# TPR >= 0.7 and FPR <= 0.2 (market signals are noisier — looser bar).        #
# --------------------------------------------------------------------------- #


_MARKET_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "market_data.csv"
)


@pytest.fixture(scope="module")
def market_df() -> pd.DataFrame:
    if not _MARKET_CSV.exists():
        pytest.skip(
            f"market CSV not found at {_MARKET_CSV} — generate it via "
            "MarketConnector(config={}).save_raw() first."
        )
    df = pd.read_csv(_MARKET_CSV, parse_dates=["timestamp"])
    df["is_disruption"] = df["is_disruption"].astype(bool)
    return df


@pytest.fixture()
def market_agent() -> MarketAgent:
    return MarketAgent(
        config={"z_threshold": 1.2, "threshold": 0.40, "window": 30}
    )


def test_market_agent_evaluation(
    market_agent: MarketAgent, market_df: pd.DataFrame
) -> None:
    """End-to-end evaluation: TPR >= 0.7, FPR <= 0.2 on synthetic data."""

    market_agent.fit(market_df)
    validated = market_agent.run_dataframe(market_df)
    reports = market_agent.run(market_df)

    # Align ground truth to validated frame (preprocess drops warm-up rows).
    aligned_truth = (
        market_df.set_index("timestamp")["is_disruption"]
        .reindex(pd.to_datetime(validated["timestamp"]))
        .astype(bool)
        .to_numpy()
    )
    y_pred = validated["validated"].astype(bool).to_numpy()
    assert aligned_truth.shape == y_pred.shape, "row alignment broken"

    tn, fp, fn, tp = confusion_matrix(
        aligned_truth, y_pred, labels=[False, True]
    ).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    print("\n" + "=" * 70)
    print("MarketAgent — End-to-End Evaluation on Synthetic Market Dataset")
    print("=" * 70)
    print(f"Scored rows              : {len(validated)} (warm-up rows dropped)")
    print(f"Ground-truth disruption  : {int(aligned_truth.sum())} days")
    print(f"Predicted (validated)    : {int(y_pred.sum())} days")
    print(f"Anomaly windows reported : {len(reports)}")
    print()
    print("Confusion Matrix")
    print("                 pred=Normal   pred=Disruption")
    print(f"true=Normal      {tn:>11d}   {fp:>15d}")
    print(f"true=Disruption  {fn:>11d}   {tp:>15d}")
    print()
    print(f"TPR (recall)     : {tpr:.3f}")
    print(f"FPR              : {fpr:.3f}")
    print(f"Precision        : {precision:.3f}")
    print()
    print("Classification Report")
    print(
        classification_report(
            aligned_truth,
            y_pred,
            target_names=["Normal", "Disruption"],
            digits=3,
            zero_division=0,
        )
    )
    if reports:
        print("Sample anomaly window report:")
        sample = reports[0]
        for k, v in sample.items():
            print(f"  {k}: {v}")
    print("=" * 70)

    assert tpr >= 0.70, f"TPR {tpr:.3f} below 0.70 threshold"
    assert fpr <= 0.20, f"FPR {fpr:.3f} above 0.20 threshold"


def test_market_agent_run_output_schema(
    market_agent: MarketAgent, market_df: pd.DataFrame
) -> None:
    """Each market window dict must conform to the documented schema."""
    reports = market_agent.run(market_df)
    assert reports, "expected at least one anomaly window on synthetic data"
    required = {
        "agent",
        "anomaly_score",
        "confidence",
        "signals",
        "start_timestamp",
        "end_timestamp",
        "location",
    }
    for r in reports:
        assert required.issubset(r.keys())
        assert r["agent"] == "market"
        assert r["location"] == "Strait of Hormuz"
        assert 0.0 <= r["anomaly_score"] <= 1.0
        assert 0.0 <= r["confidence"] <= 1.0
        assert set(r["signals"].keys()) == {
            "oil_zscore",
            "trade_volume_zscore",
            "freight_zscore",
        }


def test_market_agent_determinism(market_df: pd.DataFrame) -> None:
    """Same input + config must produce identical anomaly scores."""
    cfg = {"z_threshold": 1.2, "threshold": 0.40, "window": 30}
    a = MarketAgent(config=cfg)
    b = MarketAgent(config=cfg)
    a_df = a.run_dataframe(market_df)
    b_df = b.run_dataframe(market_df)
    np.testing.assert_array_almost_equal(
        a_df["anomaly_score"].to_numpy(),
        b_df["anomaly_score"].to_numpy(),
    )
    np.testing.assert_array_equal(
        a_df["validated"].to_numpy(),
        b_df["validated"].to_numpy(),
    )


def test_market_agent_oil_led_validation(market_df: pd.DataFrame) -> None:
    """Validation must require oil + at least one other feature elevated."""
    agent = MarketAgent(config={"z_threshold": 1.2, "threshold": 0.40})
    validated = agent.run_dataframe(market_df)
    flagged = validated[validated["validated"]]
    z_threshold = agent._z_threshold
    # Every validated row must have |oil_z| > z_threshold ...
    assert (flagged["oil_zscore"].abs() > z_threshold).all()
    # ... AND at least one other feature also elevated.
    other_elevated = (flagged["trade_volume_zscore"].abs() > z_threshold) | (
        flagged["freight_zscore"].abs() > z_threshold
    )
    assert other_elevated.all()


# --------------------------------------------------------------------------- #
# MarketAgent — end-to-end evaluation against the REAL FRED dataset (CSV).    #
# Verifies the agent picks up the optional freight_services_pct_change        #
# feature, applies the recent-baseline filter, emits the Global/Persian Gulf  #
# location, and meets the relaxed TPR >= 0.7 / FPR <= 0.2 target.             #
# --------------------------------------------------------------------------- #


_REAL_MARKET_CSVS = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "brent_crude.csv",
    Path(__file__).resolve().parent.parent / "data" / "raw" / "freight_ppi.csv",
    Path(__file__).resolve().parent.parent / "data" / "raw" / "freight_services.csv",
)


@pytest.fixture(scope="module")
def real_market_df() -> pd.DataFrame:
    if not all(p.exists() for p in _REAL_MARKET_CSVS):
        pytest.skip(
            "Real FRED CSVs not all present in data/raw/ — required for "
            "real-data market agent evaluation."
        )
    connector = MarketConnector(source_mode="csv", config={})
    return connector.fetch()


def test_market_agent_real_data_evaluation(
    real_market_df: pd.DataFrame,
) -> None:
    """End-to-end on FRED real data: TPR >= 0.7, FPR <= 0.2."""
    agent = MarketAgent(
        config={
            "z_threshold": 1.5,
            "threshold": 0.50,
            "window": 30,
            "baseline_years": 5,
        }
    )
    agent.fit(real_market_df)
    validated = agent.run_dataframe(real_market_df)
    reports = agent.run(real_market_df)

    # preprocess() clips to the last 5 years then drops a 1-row rolling
    # warm-up; re-align truth onto the validated frame's timestamps.
    aligned_truth = (
        real_market_df.set_index("timestamp")["is_disruption"]
        .reindex(pd.to_datetime(validated["timestamp"]))
        .astype(bool)
        .to_numpy()
    )
    y_pred = validated["validated"].astype(bool).to_numpy()
    assert aligned_truth.shape == y_pred.shape

    tn, fp, fn, tp = confusion_matrix(
        aligned_truth, y_pred, labels=[False, True]
    ).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    print("\n" + "=" * 70)
    print("MarketAgent — End-to-End Evaluation on REAL FRED Dataset")
    print("=" * 70)
    print(f"Scored rows              : {len(validated)} (warm-up dropped)")
    print(f"Active features          : {agent._active_features()}")
    print(f"Ground-truth disruption  : {int(aligned_truth.sum())} days")
    print(f"Predicted (validated)    : {int(y_pred.sum())} days")
    print(f"Anomaly windows reported : {len(reports)}")
    print()
    print("Confusion Matrix")
    print("                 pred=Normal   pred=Disruption")
    print(f"true=Normal      {tn:>11d}   {fp:>15d}")
    print(f"true=Disruption  {fn:>11d}   {tp:>15d}")
    print()
    print(f"TPR (recall)     : {tpr:.3f}")
    print(f"FPR              : {fpr:.3f}")
    print(f"Precision        : {precision:.3f}")
    print("=" * 70)

    assert tpr >= 0.70, f"TPR {tpr:.3f} below 0.70 threshold on real data"
    assert fpr <= 0.20, f"FPR {fpr:.3f} above 0.20 threshold on real data"


def test_market_agent_real_data_discovers_freight_services(
    real_market_df: pd.DataFrame,
) -> None:
    """Agent must auto-discover the FRED freight_services_pct_change column."""
    agent = MarketAgent(config={})
    agent.fit(real_market_df)
    assert "freight_services_pct_change" in agent._extra_features
    assert len(agent._active_features()) == 4


def test_market_agent_real_data_weight_redistribution(
    real_market_df: pd.DataFrame,
) -> None:
    """4-feature regime must redistribute weights — oil 0.35 not 0.40."""
    agent = MarketAgent(config={})
    agent.fit(real_market_df)
    weights = agent._feature_weights()
    assert weights["brent_crude_usd"] == pytest.approx(0.35)
    assert weights["trade_volume_index"] == pytest.approx(0.30)
    assert weights["freight_rate_index"] == pytest.approx(0.20)
    assert weights["freight_services_pct_change"] == pytest.approx(0.15)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_market_agent_real_data_emits_global_location(
    real_market_df: pd.DataFrame,
) -> None:
    """Output reports must label location as Global/Persian Gulf for FRED data."""
    agent = MarketAgent(
        config={
            "z_threshold": 1.5,
            "threshold": 0.50,
            "baseline_years": 5,
        }
    )
    reports = agent.run(real_market_df)
    assert reports, "expected at least one anomaly window on real data"
    assert all(r["location"] == "Global/Persian Gulf" for r in reports)


def test_market_agent_real_data_signals_include_freight_services(
    real_market_df: pd.DataFrame,
) -> None:
    """Signals dict must include freight_services_zscore in FRED mode."""
    agent = MarketAgent(
        config={
            "z_threshold": 1.5,
            "threshold": 0.50,
            "baseline_years": 5,
        }
    )
    reports = agent.run(real_market_df)
    assert reports
    sample = reports[0]
    assert "freight_services_zscore" in sample["signals"]
    # The three base z-scores must still be present (backwards compatibility).
    assert {
        "oil_zscore", "trade_volume_zscore", "freight_zscore",
    }.issubset(sample["signals"].keys())


def test_market_agent_real_data_baseline_filter_clips_history(
    real_market_df: pd.DataFrame,
) -> None:
    """preprocess() must clip to the trailing baseline_years on long history."""
    agent = MarketAgent(config={"baseline_years": 5})
    agent.fit(real_market_df)
    prepped = agent.preprocess(real_market_df)
    span_days = (prepped["timestamp"].max() - prepped["timestamp"].min()).days
    assert span_days <= 5 * 366 + 30, (
        f"Expected ~5 years of data after clip; got {span_days} days."
    )


def test_market_agent_location_override_via_config(
    real_market_df: pd.DataFrame,
) -> None:
    """Explicit config['location'] must take precedence over auto-detection."""
    agent = MarketAgent(
        config={
            "z_threshold": 1.5,
            "threshold": 0.50,
            "baseline_years": 5,
            "location": "Custom Region",
        }
    )
    reports = agent.run(real_market_df)
    assert all(r["location"] == "Custom Region" for r in reports)
