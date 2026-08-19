"""Region config completeness for the P11.4 connectors (Phase 11.5).

P11.4 gave three connectors a region-specific setting each. These tests assert
every region actually supplies all three, and that the values are distinct
per region rather than copy-pasted from Hormuz.

Deliberately *not* re-tested here (already covered elsewhere, and duplicating
it would give two places to update when a region changes):

- YAML existence, structure, agent keys, activation-vs-registry, coordinates,
  bounding-box coherence — ``test_region_configs.py``
- how each connector consumes its setting — ``test_region_specific_connectors.py``
- merge and projection mechanics — ``test_config_manager.py``
"""

from __future__ import annotations

from src.core.config_manager import load_config_for_region
from src.core.regions import get_region, list_regions
from src.ingestion import NewsConnector

# Where each P11.4 setting lives in the merged config. The connector reads the
# block, not the whole config, so the path is the contract.
_SHIPPING_BOUNDS = ("ingestion", "shipping", "ais_bounds")
_ACLED_COUNTRIES = ("agents", "geopolitical", "acled_countries")


def _dig(config: dict, path: tuple[str, ...]):
    """Walk ``path`` through ``config``, returning ``None`` if it breaks."""
    node = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _keywords(config: dict) -> list[str]:
    """The keywords NewsConnector would resolve for this merged config."""
    return NewsConnector(config=config["agents"]["news_sentiment"]).newsapi_keywords


def test_every_region_supplies_all_three_connector_settings() -> None:
    """The completeness checklist: no region may be missing a P11.4 setting."""
    missing: list[str] = []
    for region in list_regions():
        config = load_config_for_region(region)
        if not _dig(config, _SHIPPING_BOUNDS):
            missing.append(f"{region}: ais_bounds")
        if not _dig(config, _ACLED_COUNTRIES):
            missing.append(f"{region}: acled_countries")
        if not _keywords(config):
            missing.append(f"{region}: newsapi_keywords")
    assert not missing, f"incomplete region configs: {missing}"


def test_settings_are_distinct_across_regions() -> None:
    """Guards the copy-paste failure: a region left on Hormuz's values.

    Config completeness alone would not catch it — every field would be
    populated, just populated wrongly.
    """
    bounds: dict[str, str] = {}
    countries: dict[str, str] = {}
    keywords: dict[str, str] = {}

    for region in list_regions():
        config = load_config_for_region(region)
        for seen, value in (
            (bounds, _dig(config, _SHIPPING_BOUNDS)),
            (countries, _dig(config, _ACLED_COUNTRIES)),
            (keywords, _keywords(config)),
        ):
            key = repr(value)
            assert key not in seen, (
                f"{region} shares a value with {seen[key]}: {value}"
            )
            seen[key] = region


def test_country_and_keyword_lists_are_clean() -> None:
    """No blanks and no duplicates — both silently waste an API query."""
    for region in list_regions():
        config = load_config_for_region(region)
        for label, values in (
            ("acled_countries", _dig(config, _ACLED_COUNTRIES)),
            ("newsapi_keywords", _keywords(config)),
        ):
            assert all(isinstance(v, str) for v in values), f"{region}: {label} not str"
            assert all(v.strip() for v in values), f"{region}: {label} has a blank"
            assert len(values) == len(set(values)), f"{region}: {label} has duplicates"


def test_keywords_name_the_chokepoint() -> None:
    """A keyword set that never names its own chokepoint is not region-specific."""
    for region in list_regions():
        config = load_config_for_region(region)
        assert get_region(region).display_name in _keywords(config), (
            f"{region}: keywords omit the chokepoint's display name"
        )


def test_ais_bounds_surround_the_registry_centre_point() -> None:
    """The connector's box must actually contain the strait it claims to watch.

    ``test_region_configs.py`` checks this on the raw YAML; this checks the
    value that survived the merge and projection into ingestion.shipping,
    which is the one ShippingConnector reads.
    """
    for region in list_regions():
        config = load_config_for_region(region)
        (lat_min, lon_min), (lat_max, lon_max) = _dig(config, _SHIPPING_BOUNDS)
        registry = get_region(region)

        assert lat_min <= registry.latitude <= lat_max, f"{region}: centre lat outside"
        assert lon_min <= registry.longitude <= lon_max, f"{region}: centre lon outside"
