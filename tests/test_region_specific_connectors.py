"""Tests for region-specific connector settings (Phase 11.4).

The Orchestrator hands each connector a *sub-block* (``ingestion.shipping``,
``agents.geopolitical``, ...) rather than the whole config, so these tests read
the same sub-blocks. A setting projected to the wrong path would otherwise
leave every connector on Hormuz defaults while still passing.

Regions are looped inside tests rather than parametrized, to keep one concept
to one test case.
"""

from __future__ import annotations

import pytest

from src.core.config_manager import load_config_for_region
from src.core.regions import list_regions
from src.ingestion import GeopoliticalConnector, NewsConnector, ShippingConnector


class TestRegionProjection:
    """The merged config puts region settings where connectors look."""

    def test_chokepoint_entries_are_populated_for_every_region(self) -> None:
        """Extractors read extraction.chokepoints[key]; every region needs one.

        Covers the two non-obvious cases: panama has no entry in settings.yaml
        and must be created, and bab_el_mandeb maps onto the existing red_sea
        entry rather than a key of its own.
        """
        for region in list_regions():
            config = load_config_for_region(region)
            key = config["extraction"]["chokepoint_key"]
            entry = config["extraction"]["chokepoints"][key]

            assert entry["countries"] == config["extraction"]["countries"]
            assert entry["bounding_box"] == config["extraction"]["bounding_box"]
            # Projecting one region must not drop settings.yaml's other entries.
            for preexisting in ("hormuz", "red_sea", "malacca", "suez"):
                assert preexisting in config["extraction"]["chokepoints"]

        assert load_config_for_region("panama")["extraction"]["chokepoints"][
            "panama"
        ]["countries"] == ["Panama"]
        assert (
            load_config_for_region("bab_el_mandeb")["extraction"]["chokepoint_key"]
            == "red_sea"
        )

    def test_aisstream_monitor_region_is_replaced_not_appended(self) -> None:
        """A run monitors one region; Hormuz's box must not leak into others."""
        for region in list_regions():
            config = load_config_for_region(region)
            monitor_regions = config["aisstream"]["monitor_regions"]

            assert len(monitor_regions) == 1, f"{region}: expected one region"
            assert monitor_regions[0]["name"] == region
            assert monitor_regions[0]["bbox"] == config["aisstream"]["bbox"]


class TestShippingConnectorRegion:
    """ShippingConnector reads region-specific AIS bounds."""

    def test_reads_distinct_bounds_for_every_region(self) -> None:
        seen: list[list] = []
        for region in list_regions():
            config = load_config_for_region(region)
            connector = ShippingConnector(config=config["ingestion"]["shipping"])

            assert connector.ais_bounds == config["aisstream"]["bbox"], region
            (lat_min, lon_min), (lat_max, lon_max) = connector.ais_bounds
            assert lat_min < lat_max and lon_min < lon_max, region
            assert connector.ais_bounds not in seen, f"{region}: duplicate bounds"
            seen.append(connector.ais_bounds)

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

    def test_absent_or_malformed_bounds_degrade_to_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Bad bounds must not break the modes the pipeline actually runs in."""
        with caplog.at_level("WARNING"):
            assert ShippingConnector(config={"ais_bounds": [1, 2, 3]}).ais_bounds is None
        assert "malformed ais_bounds" in caplog.text

        connector = ShippingConnector(source_mode="synthetic", config={})
        assert connector.ais_bounds is None
        assert not connector.fetch().empty


class TestGeopoliticalConnectorRegion:
    """GeopoliticalConnector reads region-specific ACLED countries."""

    def test_reads_region_countries(self) -> None:
        for region in list_regions():
            config = load_config_for_region(region)
            connector = GeopoliticalConnector(config=config["agents"]["geopolitical"])

            assert connector.acled_countries == config["extraction"]["countries"]
            assert all(isinstance(c, str) for c in connector.acled_countries)

        hormuz = load_config_for_region("hormuz")["agents"]["geopolitical"]
        panama = load_config_for_region("panama")["agents"]["geopolitical"]
        assert "Iran" in GeopoliticalConnector(config=hormuz).acled_countries
        assert GeopoliticalConnector(config=panama).acled_countries == ["Panama"]

    def test_missing_countries_leaves_synthetic_mode_working(self) -> None:
        connector = GeopoliticalConnector(config={})
        assert connector.acled_countries == []
        assert not connector.fetch().empty


class TestNewsConnectorRegion:
    """NewsConnector resolves region-specific NewsAPI keywords."""

    def test_derives_keywords_from_region_location_context(self) -> None:
        for region in list_regions():
            config = load_config_for_region(region)["agents"]["news_sentiment"]
            connector = NewsConnector(config=config)

            assert (
                config["location_context"]["primary_location"]
                in connector.newsapi_keywords
            ), region

        panama = NewsConnector(
            config=load_config_for_region("panama")["agents"]["news_sentiment"]
        )
        hormuz = NewsConnector(
            config=load_config_for_region("hormuz")["agents"]["news_sentiment"]
        )
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

    def test_derived_keywords_are_deduplicated_in_order(self) -> None:
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

    def test_blank_context_leaves_synthetic_mode_working(
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
        assert not connector.fetch().empty


def test_api_stubs_report_the_settings_they_resolved() -> None:
    """Each stub names its region setting, so a config gap surfaces there."""
    config = load_config_for_region("malacca")

    with pytest.raises(NotImplementedError, match=r"Region bounds resolved"):
        ShippingConnector(config=config["ingestion"]["shipping"]).fetch_from_api()
    with pytest.raises(NotImplementedError, match=r"Region ACLED countries"):
        GeopoliticalConnector(config=config["agents"]["geopolitical"]).fetch_api()
    with pytest.raises(NotImplementedError, match=r"Region keywords"):
        NewsConnector(config=config["agents"]["news_sentiment"]).fetch_api()


def test_orchestrator_wiring_carries_region_settings() -> None:
    """End-to-end: settings survive the Orchestrator's own construction path.

    This is the assertion the per-connector tests cannot make — they pick their
    config blocks by hand, and this one does not.
    """
    from src.orchestrator import Orchestrator

    for region in list_regions():
        config = load_config_for_region(region)
        orchestrator = Orchestrator(config=config)

        assert (
            orchestrator._shipping_connector.ais_bounds == config["aisstream"]["bbox"]
        ), region

        geo = orchestrator._domain_connectors.get("geopolitical")
        if geo is not None:  # passive in panama
            assert geo.acled_countries == config["extraction"]["countries"], region

        news = orchestrator._domain_connectors["news_sentiment"]
        assert (
            config["agents"]["news_sentiment"]["location_context"]["primary_location"]
            in news.newsapi_keywords
        ), region


def test_all_six_connectors_build_from_every_merged_config() -> None:
    """No connector chokes on a region config, including the passive ones."""
    from src.ingestion import DisasterConnector, MarketConnector, RoutingConnector

    for region in list_regions():
        config = load_config_for_region(region)
        agents = config["agents"]
        built = [
            ShippingConnector(config=config["ingestion"]["shipping"]),
            MarketConnector(config=config["ingestion"]["market"]),
            GeopoliticalConnector(config=agents["geopolitical"]),
            DisasterConnector(config=agents["natural_disaster"]),
            RoutingConnector(config=agents["routing"]),
            NewsConnector(config=agents["news_sentiment"]),
        ]
        assert all(c is not None for c in built), region


def test_connectors_accept_only_a_config_dict() -> None:
    """Phase 11 keeps region awareness in config, not in constructor signatures."""
    import inspect

    for cls in (ShippingConnector, GeopoliticalConnector, NewsConnector):
        params = set(inspect.signature(cls.__init__).parameters)
        assert "region" not in params, f"{cls.__name__} grew a region parameter"


def test_projection_is_idempotent_across_repeated_loads() -> None:
    """Loading a region twice must not accumulate monitor regions or entries."""
    first = load_config_for_region("panama")
    second = load_config_for_region("panama")

    assert first["aisstream"]["monitor_regions"] == second["aisstream"]["monitor_regions"]
    assert first["extraction"]["chokepoints"] == second["extraction"]["chokepoints"]
    assert len(second["aisstream"]["monitor_regions"]) == 1


def test_regions_do_not_share_mutable_config_state() -> None:
    """Two regions' configs must be independent objects, not aliases.

    _deep_merge copies shallowly per level, so a shared nested dict would let
    one region's projection rewrite another's.
    """
    hormuz = load_config_for_region("hormuz")
    panama = load_config_for_region("panama")

    assert hormuz["agents"] is not panama["agents"]
    assert hormuz["extraction"]["chokepoints"] is not panama["extraction"]["chokepoints"]
    assert hormuz["ingestion"]["shipping"]["ais_bounds"] != panama["ingestion"][
        "shipping"
    ]["ais_bounds"]
