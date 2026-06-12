"""
Covers, per agent (geopolitical, natural_disaster, routing, news_sentiment):

1. Connector synthetic mode — schema, ranges, disruption-day count.
2. Connector CSV mode — round-trip via ``save_raw`` / ``load_csv``.
3. Connector API mode — ``NotImplementedError``.
4. Agent detection — anomaly_score during scenarios > anomaly_score baseline.

Plus a 6-agent integration test verifying Scenario B aggregates higher
than Scenario A which aggregates higher than normal periods.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aggregation.risk_engine import RiskEngine
from src.ingestion import (
    DisasterConnector,
    GeopoliticalConnector,
    MarketConnector,
    NewsConnector,
    RoutingConnector,
    ShippingConnector,
)
from src.agents import (
    DisasterAgent,
    GeopoliticalAgent,
    MarketAgent,
    NewsAgent,
    RoutingAgent,
    ShippingAgent,
)


# ---------------------------------------------------------------------------
# Geopolitical
# ---------------------------------------------------------------------------


class TestGeopoliticalConnector:
    def test_synthetic_shape_and_columns(self) -> None:
        df = GeopoliticalConnector(config={"data_mode": "synthetic"}).fetch()
        assert len(df) == 365
        expected = {
            "timestamp",
            "sanctions_severity",
            "military_activity_index",
            "diplomatic_incident_score",
            "regime_stability_index",
            "composite_geopolitical_risk",
            "flagged_incidents",
            "is_disruption",
        }
        assert expected.issubset(set(df.columns))

    def test_synthetic_ranges(self) -> None:
        df = GeopoliticalConnector().fetch()
        for col in (
            "sanctions_severity",
            "military_activity_index",
            "diplomatic_incident_score",
            "regime_stability_index",
            "composite_geopolitical_risk",
        ):
            assert df[col].between(0.0, 1.0).all(), col
            assert not df[col].isna().any(), col

    def test_synthetic_disruption_days_present(self) -> None:
        df = GeopoliticalConnector().fetch()
        # 3 scenarios × ~15-30 days = 50+ disruption days, leading by 3 days.
        assert df["is_disruption"].sum() >= 50

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "geo.csv"
        connector = GeopoliticalConnector(
            config={"data_mode": "synthetic", "csv_path": str(target)}
        )
        connector.save_raw(target)
        loader = GeopoliticalConnector(
            config={"data_mode": "csv", "csv_path": str(target)}
        )
        df = loader.fetch()
        assert len(df) == 365
        assert df["composite_geopolitical_risk"].between(0.0, 1.0).all()

    def test_api_mode_raises(self) -> None:
        connector = GeopoliticalConnector(config={"data_mode": "api"})
        with pytest.raises(NotImplementedError):
            connector.fetch()


class TestGeopoliticalAgent:
    def test_detection_lifts_scores_during_disruption(self) -> None:
        df = GeopoliticalConnector().fetch()
        agent = GeopoliticalAgent()
        agent.fit(df)
        validated = agent.run_dataframe(df)
        normal_mean = validated.loc[
            ~df["is_disruption"], "anomaly_score"].mean()
        disrupted_mean = validated.loc[
            df["is_disruption"], "anomaly_score"].mean()
        print(
            f"\n[geo] anomaly_score: normal={normal_mean:.3f} "
            f"disruption={disrupted_mean:.3f}"
        )
        assert disrupted_mean > normal_mean * 2

    def test_run_returns_window_dicts(self) -> None:
        df = GeopoliticalConnector().fetch()
        windows = GeopoliticalAgent().run(df)
        assert windows
        for w in windows:
            assert {
                "agent", "anomaly_score", "confidence", "signals",
                "flagged_incidents", "start_timestamp", "end_timestamp",
                "location",
            }.issubset(w.keys())
            assert w["agent"] == "geopolitical"
            assert 0.0 <= w["anomaly_score"] <= 1.0


# ---------------------------------------------------------------------------
# Natural Disaster
# ---------------------------------------------------------------------------


class TestDisasterConnector:
    def test_synthetic_shape_and_columns(self) -> None:
        df = DisasterConnector().fetch()
        assert len(df) == 365
        expected = {
            "timestamp",
            "earthquake_severity",
            "tsunami_risk",
            "cyclone_severity",
            "severe_weather_index",
            "composite_disaster_risk",
            "active_events",
            "is_disruption",
        }
        assert expected.issubset(set(df.columns))

    def test_synthetic_scenario_b_only(self) -> None:
        """Only the Scenario-B window (days 148+) should carry the earthquake."""
        df = DisasterConnector().fetch()
        # convert to numpy array to satisfy type checkers that expect ArrayLike
        disrupt_idx = np.where(df["is_disruption"].to_numpy())[0]
        assert len(disrupt_idx) > 0
        # All disruption days fall in [_QUAKE_DAY, _QUAKE_DAY + 7]
        assert disrupt_idx.min() >= 145
        assert disrupt_idx.max() <= 160

    def test_synthetic_scenarios_a_and_c_clean(self) -> None:
        df = DisasterConnector().fetch()
        for window in [(60, 74), (280, 290)]:
            sub = df.iloc[window[0]:window[1] + 1]
            assert not sub["is_disruption"].any()
            assert sub["composite_disaster_risk"].max() < 0.30

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "disaster.csv"
        DisasterConnector(
            config={"data_mode": "synthetic", "csv_path": str(target)}
        ).save_raw(target)
        df = DisasterConnector(
            config={"data_mode": "csv", "csv_path": str(target)}
        ).fetch()
        assert len(df) == 365

    def test_api_mode_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            DisasterConnector(config={"data_mode": "api"}).fetch()


class TestDisasterAgent:
    def test_detection_fires_during_quake(self) -> None:
        df = DisasterConnector().fetch()
        agent = DisasterAgent()
        validated = agent.run_dataframe(df)
        flagged = validated.loc[validated["validated"]]
        print(
            f"\n[disaster] flagged_days={len(flagged)} "
            f"max_score={validated['anomaly_score'].max():.3f}"
        )
        assert not flagged.empty
        # Single-day validation acceptable — magnitude alone must clear.
        assert flagged["max_single_severity"].max() >= 0.40

    def test_run_returns_window_dicts(self) -> None:
        df = DisasterConnector().fetch()
        windows = DisasterAgent().run(df)
        assert windows
        for w in windows:
            assert w["agent"] == "natural_disaster"
            assert {"signals", "active_events", "location"}.issubset(w.keys())


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRoutingConnector:
    def test_synthetic_shape_and_columns(self) -> None:
        df = RoutingConnector().fetch()
        assert len(df) == 365
        expected = {
            "timestamp",
            "rerouting_percentage",
            "avg_route_deviation_km",
            "transit_volume_ratio",
            "vessels_holding",
            "alternative_route_traffic",
            "composite_routing_risk",
            "is_disruption",
        }
        assert expected.issubset(set(df.columns))

    def test_synthetic_ranges(self) -> None:
        df = RoutingConnector().fetch()
        assert df["rerouting_percentage"].between(0.0, 100.0).all()
        assert df["transit_volume_ratio"].between(0.0, 1.0).all()
        assert df["composite_routing_risk"].between(0.0, 1.0).all()
        assert (df["vessels_holding"] >= 0).all()

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "routing.csv"
        RoutingConnector(
            config={"data_mode": "synthetic", "csv_path": str(target)}
        ).save_raw(target)
        df = RoutingConnector(
            config={"data_mode": "csv", "csv_path": str(target)}
        ).fetch()
        assert len(df) == 365

    def test_api_mode_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            RoutingConnector(config={"data_mode": "api"}).fetch()


class TestRoutingAgent:
    def test_detection_lifts_scores_during_disruption(self) -> None:
        df = RoutingConnector().fetch()
        agent = RoutingAgent()
        agent.fit(df)
        validated = agent.run_dataframe(df)
        mask = df["is_disruption"].to_numpy(dtype=bool)
        scores = validated["anomaly_score"]
        normal = float(scores[~mask].mean())
        disrupt = float(scores[mask].mean())
        print(f"\n[routing] anomaly_score: normal={normal:.3f} disruption={disrupt:.3f}")
        assert disrupt > normal * 1.5

    def test_run_returns_window_dicts(self) -> None:
        windows = RoutingAgent().run(RoutingConnector().fetch())
        assert windows
        for w in windows:
            assert w["agent"] == "routing"
            assert w["model_version"] == "hormuz_v1.0"


# ---------------------------------------------------------------------------
# News sentiment
# ---------------------------------------------------------------------------


class TestNewsConnector:
    def test_synthetic_shape_and_columns(self) -> None:
        df = NewsConnector().fetch()
        assert len(df) == 365
        expected = {
            "timestamp",
            "sentiment_score",
            "sentiment_magnitude",
            "source_consensus",
            "article_volume",
            "dominant_narrative",
            "recency_weighted_score",
            "composite_news_risk",
            "is_disruption",
        }
        assert expected.issubset(set(df.columns))

    def test_synthetic_sentiment_drops_during_disruption(self) -> None:
        df = NewsConnector().fetch()
        normal_sent = df.loc[~df["is_disruption"], "sentiment_score"].mean()
        disrupt_sent = df.loc[df["is_disruption"], "sentiment_score"].mean()
        print(
            f"\n[news] sentiment: normal={normal_sent:.3f} "
            f"disruption={disrupt_sent:.3f}"
        )
        assert disrupt_sent < normal_sent
        assert disrupt_sent < -0.1

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "news.csv"
        NewsConnector(
            config={"data_mode": "synthetic", "csv_path": str(target)}
        ).save_raw(target)
        df = NewsConnector(
            config={"data_mode": "csv", "csv_path": str(target)}
        ).fetch()
        assert len(df) == 365

    def test_api_mode_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            NewsConnector(config={"data_mode": "api"}).fetch()


class TestNewsAgent:
    def test_detection_lifts_scores_during_disruption(self) -> None:
        df = NewsConnector().fetch()
        agent = NewsAgent()
        agent.fit(df)
        validated = agent.run_dataframe(df)
        mask = df["is_disruption"].to_numpy(dtype=bool)
        scores = validated["anomaly_score"]
        normal = float(scores[~mask].mean())
        disrupt = float(scores[mask].mean())
        print(f"\n[news] anomaly_score: normal={normal:.3f} disruption={disrupt:.3f}")
        assert disrupt > normal * 1.5

    def test_run_returns_window_dicts(self) -> None:
        windows = NewsAgent().run(NewsConnector().fetch())
        assert windows
        for w in windows:
            assert w["agent"] == "news_sentiment"
            assert "dominant_narrative" in w


# ---------------------------------------------------------------------------
# 6-Agent integration test — Scenario B > Scenario A > Normal
# ---------------------------------------------------------------------------


def _per_day_composite(
    *, agent_scores: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Compute a per-day weighted composite over only the configured agents."""
    total_w = 0.0
    composite = np.zeros_like(next(iter(agent_scores.values())))
    for name, scores in agent_scores.items():
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        composite = composite + w * scores
        total_w += w
    return composite / total_w if total_w else composite


def test_six_agent_integration_scenarios_rank_correctly() -> None:
    """Composite risk: Scenario B > Scenario A > normal days, all 6 agents firing."""
    import yaml

    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    # Build all six dataframes (use synthetic everywhere for a consistent 365-day window).
    ship_df = ShippingConnector(source_mode="synthetic", config={}).fetch()
    mkt_df = MarketConnector(source_mode="synthetic", config={}).fetch()
    geo_df = GeopoliticalConnector().fetch()
    dis_df = DisasterConnector().fetch()
    rt_df = RoutingConnector().fetch()
    nw_df = NewsConnector().fetch()

    agents = {
        "shipping": (ShippingAgent(), ship_df),
        "market": (MarketAgent(config={"z_threshold": 1.2, "threshold": 0.4}), mkt_df),
        "geopolitical": (GeopoliticalAgent(), geo_df),
        "natural_disaster": (DisasterAgent(), dis_df),
        "routing": (RoutingAgent(), rt_df),
        "news_sentiment": (NewsAgent(), nw_df),
    }

    detection_results = []
    score_series: dict[str, np.ndarray] = {}
    for name, (agent, df) in agents.items():
        agent.fit(df)
        validated = agent.run_dataframe(df)
        detection_results.append(agent.to_detection_result(validated))
        # Align each agent's per-row scores to the 365-day grid by timestamp.
        ts = pd.to_datetime(validated["timestamp"]) if "timestamp" in validated.columns else None
        if ts is not None:
            series = pd.Series(
                validated["anomaly_score"].to_numpy(),
                index=ts,
            ).reindex(pd.date_range("2025-01-01", periods=365, freq="D")).ffill().fillna(0.0)
            score_series[name] = series.to_numpy()

    # RiskEngine aggregation works (returns valid output).
    engine = RiskEngine(config)
    agg = engine.aggregate(detection_results)
    assert "composite_score" in agg
    assert {"shipping", "market", "geopolitical", "natural_disaster", "routing", "news_sentiment"}.issubset(
        agg["agent_scores"].keys()
    )

    weights = config["weights"]
    composite = _per_day_composite(agent_scores=score_series, weights=weights)
    assert composite.shape == (365,)

    # Scenario B (days 150-170) — all six agents fire, including disaster.
    scenario_b = composite[150:171].mean()
    # Scenario A (days 60-74) — five agents fire (no disaster).
    scenario_a = composite[60:75].mean()
    # Normal baseline (days 200-260) — no scenarios.
    normal = composite[200:260].mean()

    print(
        f"\n[integration] composite: normal={normal:.3f} "
        f"scenario_A={scenario_a:.3f} scenario_B={scenario_b:.3f}"
    )
    assert scenario_b > scenario_a, (
        f"Expected Scenario B ({scenario_b:.3f}) > Scenario A ({scenario_a:.3f})"
    )
    assert scenario_a > normal, (
        f"Expected Scenario A ({scenario_a:.3f}) > normal ({normal:.3f})"
    )
    assert scenario_b > 0.4, "Scenario B should comfortably reach MEDIUM+ risk."
