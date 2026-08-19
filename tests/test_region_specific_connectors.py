"""Tests for region-specific connector settings (Phase 11.4).

Two layers, and the second is the one that matters:

1. Each connector reads its region setting out of its own config block and
   degrades sensibly when it is absent.
2. ``load_config_for_region`` actually *puts* that setting where the connector
   looks. The Orchestrator hands each connector a sub-block
   (``ingestion.shipping``, ``agents.geopolitical``, ...), not the whole
   config, so a setting projected to the wrong path would leave every
   connector on Hormuz defaults while the tests above still passed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.config_manager import load_config_for_region
from src.ingestion import (
    DisasterConnector,
    GeopoliticalConnector,
    MarketConnector,
    NewsConnector,
    RoutingConnector,
    ShippingConnector,
)
from src.core.regions import list_regions


def _shipping_block(region: str) -> dict:
    """The block the Orchestrator hands ShippingConnector for ``region``."""
    return load_config_for_region(region)["ingestion"]["shipping"]


def _agent_block(region: str, agent: str) -> dict:
    """The block the Orchestrator hands a domain connector for ``region``."""
    return load_config_for_region(region)["agents"][agent]


class TestShippingConnectorRegion:
    """ShippingConnector reads region-specific AIS bounds."""

    @pytest.mark.parametrize("region", list_regions())
    def test_reads_ais_bounds_from_merged_config(self, region: str) -> None:
        connector = ShippingConnector(
            source_mode="csv", config=_shipping_block(region)
        )
        assert connector.ais_bounds is not None, f"{region}: no AIS bounds"
        (lat_min, lon_min), (lat_max, lon_max) = connector.ais_bounds
        assert lat_min < lat_max
        assert lon_min < lon_max

    def test_bounds_differ_between_regions(self) -> None:
        """Panama's box is west of Greenwich; Hormuz's is east."""
        hormuz = ShippingConnector(config=_shipping_block("hormuz")).ais_bounds
        panama = ShippingConnector(config=_shipping_block("panama")).ais_bounds

        assert hormuz != panama
        assert hormuz[0][1] > 0  # Persian Gulf, positive longitude
        assert panama[0][1] < 0  # Panama, negative longitude

    def test_bounds_match_the_region_overlay(self) -> None:
        """The projected value is the overlay's, not a re-derived guess."""
        config = load_config_for_region("malacca")
        assert config["ingestion"]["shipping"]["ais_bounds"] == [
            [0.5, 99.0],
            [4.0, 105.0],
        ]
        connector = ShippingConnector(config=config["ingestion"]["shipping"])
        assert connector.ais_bounds == [[0.5, 99.0], [4.0, 105.0]]

    def test_falls_back_to_legacy_api_bounding_box(self) -> None:
        """settings.yaml's pre-Phase-11 dict form still resolves."""
        connector = ShippingConnector(
            config={
                "api": {
                    "bounding_box": {
                        "lat_min": 28.95,
                        "lat_max": 29.20,
                        "lon_min": 48.05,
                        "lon_max": 48.25,
                    }
                }
            }
        )
        assert connector.ais_bounds == [[28.95, 48.05], [29.20, 48.25]]

    def test_missing_bounds_is_none_and_csv_still_works(self) -> None:
        """No bounds must not break the mode the pipeline actually runs in."""
        connector = ShippingConnector(source_mode="synthetic", config={})
        assert connector.ais_bounds is None

        df = connector.fetch()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_malformed_bounds_degrade_to_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bad box warns and disables API mode rather than raising."""
        with caplog.at_level("WARNING"):
            connector = ShippingConnector(config={"ais_bounds": [1, 2, 3]})
        assert connector.ais_bounds is None
        assert "malformed ais_bounds" in caplog.text

    def test_api_mode_reports_resolved_bounds(self) -> None:
        """The stub names the bounds it would have used."""
        connector = ShippingConnector(config=_shipping_block("panama"))
        with pytest.raises(NotImplementedError, match=r"Region bounds resolved"):
            connector.fetch_from_api()


class TestGeopoliticalConnectorRegion:
    """GeopoliticalConnector reads region-specific ACLED countries."""

    @pytest.mark.parametrize("region", list_regions())
    def test_reads_acled_countries_from_merged_config(self, region: str) -> None:
        connector = GeopoliticalConnector(
            config=_agent_block(region, "geopolitical")
        )
        assert connector.acled_countries, f"{region}: no ACLED countries"
        assert all(isinstance(c, str) for c in connector.acled_countries)

    def test_countries_differ_between_regions(self) -> None:
        hormuz = GeopoliticalConnector(
            config=_agent_block("hormuz", "geopolitical")
        )
        panama = GeopoliticalConnector(
            config=_agent_block("panama", "geopolitical")
        )
        assert "Iran" in hormuz.acled_countries
        assert panama.acled_countries == ["Panama"]

    def test_missing_countries_is_empty_and_synthetic_still_works(self) -> None:
        connector = GeopoliticalConnector(config={})
        assert connector.acled_countries == []

        df = connector.fetch()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_api_mode_reports_resolved_countries(self) -> None:
        connector = GeopoliticalConnector(
            config=_agent_block("bab_el_mandeb", "geopolitical")
        )
        with pytest.raises(NotImplementedError, match=r"Region ACLED countries"):
            connector.fetch_api()


class TestNewsConnectorRegion:
    """NewsConnector resolves region-specific NewsAPI keywords."""

    @pytest.mark.parametrize("region", list_regions())
    def test_derives_keywords_from_region_location_context(
        self, region: str
    ) -> None:
        config = _agent_block(region, "news_sentiment")
        connector = NewsConnector(config=config)

        assert connector.newsapi_keywords
        assert (
            config["location_context"]["primary_location"]
            in connector.newsapi_keywords
        )

    def test_keywords_differ_between_regions(self) -> None:
        hormuz = NewsConnector(config=_agent_block("hormuz", "news_sentiment"))
        panama = NewsConnector(config=_agent_block("panama", "news_sentiment"))

        assert "Strait of Hormuz" in hormuz.newsapi_keywords
        assert "Panama Canal" in panama.newsapi_keywords
        assert "Gatun Lake" in panama.newsapi_keywords
        assert "Gatun Lake" not in hormuz.newsapi_keywords

    def test_explicit_keywords_override_derivation(self) -> None:
        connector = NewsConnector(
            config={
                "newsapi_keywords": ["custom term"],
                "location_context": {"primary_location": "Ignored"},
            }
        )
        assert connector.newsapi_keywords == ["custom term"]

    def test_keywords_are_deduplicated_in_order(self) -> None:
        connector = NewsConnector(
            config={
                "location_context": {
                    "primary_location": "Suez Canal",
                    "region": "Suez Canal",
                    "topics": ["shipping", "shipping", "transit"],
                }
            }
        )
        assert connector.newsapi_keywords == ["Suez Canal", "shipping", "transit"]

    def test_absent_context_keeps_the_connector_default(self) -> None:
        """An absent/empty context falls back to the pre-Phase-11 Hormuz block.

        That fallback is the connector's existing contract, so keywords are
        still derived rather than empty.
        """
        connector = NewsConnector(config={})
        assert "Strait of Hormuz" in connector.newsapi_keywords

    def test_blank_context_is_empty_and_synthetic_still_works(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A context present but carrying no usable values warns, not raises."""
        with caplog.at_level("WARNING"):
            connector = NewsConnector(
                config={
                    "location_context": {
                        "primary_location": "",
                        "region": None,
                        "topics": [],
                    }
                }
            )
        assert connector.newsapi_keywords == []
        assert "unfiltered" in caplog.text

        df = connector.fetch()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_api_mode_reports_resolved_keywords(self) -> None:
        connector = NewsConnector(config=_agent_block("malacca", "news_sentiment"))
        with pytest.raises(NotImplementedError, match=r"Region keywords"):
            connector.fetch_api()


class TestRegionProjection:
    """The merged config puts region settings where connectors look for them."""

    @pytest.mark.parametrize("region", list_regions())
    def test_chokepoint_entry_is_populated(self, region: str) -> None:
        """Extractors read extraction.chokepoints[key]; every region needs one."""
        config = load_config_for_region(region)
        key = config["extraction"]["chokepoint_key"]
        entry = config["extraction"]["chokepoints"][key]

        assert entry["countries"] == config["extraction"]["countries"]
        assert entry["bounding_box"] == config["extraction"]["bounding_box"]

    def test_panama_chokepoint_is_added_not_just_hormuz(self) -> None:
        """settings.yaml has no panama entry — the projection must create it."""
        config = load_config_for_region("panama")
        assert config["extraction"]["chokepoints"]["panama"]["countries"] == [
            "Panama"
        ]

    def test_bab_el_mandeb_maps_onto_the_red_sea_entry(self) -> None:
        """chokepoint_key, not region name, decides the destination."""
        config = load_config_for_region("bab_el_mandeb")
        assert config["extraction"]["chokepoint_key"] == "red_sea"
        assert "Yemen" in config["extraction"]["chokepoints"]["red_sea"]["countries"]

    @pytest.mark.parametrize("region", list_regions())
    def test_aisstream_monitor_region_is_replaced_not_appended(
        self, region: str
    ) -> None:
        """A run monitors one region; Hormuz's box must not leak into others."""
        config = load_config_for_region(region)
        monitor_regions = config["aisstream"]["monitor_regions"]

        assert len(monitor_regions) == 1
        assert monitor_regions[0]["name"] == region
        assert monitor_regions[0]["bbox"] == config["aisstream"]["bbox"]

    def test_untouched_chokepoints_survive(self) -> None:
        """Projecting one region must not drop settings.yaml's other entries."""
        config = load_config_for_region("panama")
        for key in ("hormuz", "red_sea", "malacca", "suez"):
            assert key in config["extraction"]["chokepoints"]


class TestConnectorsWithMergedConfig:
    """Every connector builds from every region's merged config."""

    @pytest.mark.parametrize("region", list_regions())
    def test_all_six_connectors_initialise(self, region: str) -> None:
        config = load_config_for_region(region)
        agents = config["agents"]

        connectors = [
            ShippingConnector(
                source_mode="csv", config=config["ingestion"]["shipping"]
            ),
            MarketConnector(
                source_mode="csv", config=config["ingestion"]["market"]
            ),
            GeopoliticalConnector(config=agents["geopolitical"]),
            DisasterConnector(config=agents["natural_disaster"]),
            RoutingConnector(config=agents["routing"]),
            NewsConnector(config=agents["news_sentiment"]),
        ]
        assert all(c is not None for c in connectors)

    @pytest.mark.parametrize("region", list_regions())
    def test_orchestrator_connectors_carry_region_settings(
        self, region: str
    ) -> None:
        """End-to-end: the settings survive the Orchestrator's own wiring.

        This is the assertion the per-connector tests cannot make — it uses the
        real construction path rather than hand-picked config blocks.
        """
        from src.orchestrator import Orchestrator

        config = load_config_for_region(region)
        orchestrator = Orchestrator(config=config)

        assert (
            orchestrator._shipping_connector.ais_bounds
            == config["aisstream"]["bbox"]
        )
        geo = orchestrator._domain_connectors.get("geopolitical")
        if geo is not None:  # passive in panama
            assert geo.acled_countries == config["extraction"]["countries"]
        news = orchestrator._domain_connectors["news_sentiment"]
        assert (
            config["agents"]["news_sentiment"]["location_context"][
                "primary_location"
            ]
            in news.newsapi_keywords
        )
