"""Tests for the Phase 11 region YAML overlays (``config/regions/*.yaml``).

These files are inert config data — nothing loads them until Phase 11.3 — so
these tests check structural validity and, more importantly, that the YAMLs
agree with the :mod:`src.core.regions` registry. Two independent sources of
truth for the same activation flags will drift otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.regions import AGENT_KEYS, get_region, list_regions

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGION_CONFIG_DIR = _PROJECT_ROOT / "config" / "regions"
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_region_yaml(region: str) -> dict:
    path = _REGION_CONFIG_DIR / f"{region}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_all_region_configs_exist() -> None:
    """One YAML per registered region, named for its region key."""
    for region in list_regions():
        assert (_REGION_CONFIG_DIR / f"{region}.yaml").exists(), f"missing {region}.yaml"


@pytest.mark.parametrize("region", list_regions())
def test_region_yaml_structure(region: str) -> None:
    """Each YAML parses and carries the expected top-level blocks."""
    data = _load_region_yaml(region)
    assert data is not None
    for block in ("region", "agents", "extraction", "aisstream"):
        assert block in data, f"{region}.yaml missing '{block}' block"

    assert data["region"]["name"] == region
    assert isinstance(data["region"]["latitude"], (int, float))
    assert isinstance(data["region"]["longitude"], (int, float))


@pytest.mark.parametrize("region", list_regions())
def test_yaml_agent_keys_are_real_config_keys(region: str) -> None:
    """Each YAML covers exactly the six real pipeline agent keys.

    Guards against ``disaster``/``news`` (the plan document's names) slipping
    in for ``natural_disaster``/``news_sentiment``.
    """
    data = _load_region_yaml(region)
    assert set(data["agents"]) == set(AGENT_KEYS)
    for agent_key, block in data["agents"].items():
        assert "enabled" in block, f"{region}.yaml: {agent_key} has no 'enabled' flag"
        assert isinstance(block["enabled"], bool)


@pytest.mark.parametrize("region", list_regions())
def test_yaml_activation_matches_registry(region: str) -> None:
    """The YAML's enabled flags agree with src/core/regions.py's registry.

    These are two independent sources of truth for the same decision; this
    test is what keeps them from drifting apart.
    """
    data = _load_region_yaml(region)
    registry = get_region(region)
    for agent_key in AGENT_KEYS:
        assert data["agents"][agent_key]["enabled"] is registry.agents[agent_key], (
            f"{region}.yaml disagrees with the registry on {agent_key}"
        )


@pytest.mark.parametrize("region", list_regions())
def test_yaml_coordinates_match_registry(region: str) -> None:
    """Centre coordinates agree between YAML and registry."""
    data = _load_region_yaml(region)
    registry = get_region(region)
    assert data["region"]["latitude"] == pytest.approx(registry.latitude)
    assert data["region"]["longitude"] == pytest.approx(registry.longitude)


def test_bab_el_mandeb_routing_active() -> None:
    """Routing is enabled only in Bab el-Mandeb (85% documented diversion)."""
    assert _load_region_yaml("bab_el_mandeb")["agents"]["routing"]["enabled"] is True
    for region in ("hormuz", "panama", "malacca"):
        assert _load_region_yaml(region)["agents"]["routing"]["enabled"] is False


def test_evidence_based_exclusions_in_yaml() -> None:
    """The two evidence-excluded domains are disabled in their YAMLs."""
    bab = _load_region_yaml("bab_el_mandeb")
    assert bab["agents"]["natural_disaster"]["enabled"] is False

    panama = _load_region_yaml("panama")
    assert panama["agents"]["geopolitical"]["enabled"] is False

    malacca = _load_region_yaml("malacca")
    assert malacca["agents"]["market"]["enabled"] is False


def test_bounding_boxes_are_coherent() -> None:
    """Every bounding box has min < max, and contains its own centre point."""
    for region in list_regions():
        data = _load_region_yaml(region)
        box = data["extraction"]["bounding_box"]
        assert box["lat_min"] < box["lat_max"], f"{region}: lat_min >= lat_max"
        assert box["lon_min"] < box["lon_max"], f"{region}: lon_min >= lon_max"

        # The aisstream bbox is the same extent in nested-list form.
        (lat_min, lon_min), (lat_max, lon_max) = data["aisstream"]["bbox"]
        assert lat_min < lat_max and lon_min < lon_max, f"{region}: bad aisstream bbox"

        # The strait's own centre point must fall inside the aisstream box.
        lat, lon = data["region"]["latitude"], data["region"]["longitude"]
        assert lat_min <= lat <= lat_max, f"{region}: centre lat outside aisstream bbox"
        assert lon_min <= lon <= lon_max, f"{region}: centre lon outside aisstream bbox"


def test_hormuz_overlay_matches_live_settings() -> None:
    """Hormuz's overlay reproduces settings.yaml, except the routing flag.

    Hormuz is the default region, so its overlay must be a faithful copy of
    current behaviour — otherwise applying it in Phase 11.3 would silently
    change the default pipeline. routing.enabled is the one intended
    deviation (settings.yaml has it true; see hormuz.yaml's regression-gate
    warning).
    """
    settings = yaml.safe_load(_SETTINGS_PATH.read_text(encoding="utf-8"))
    overlay = _load_region_yaml("hormuz")

    live_context = settings["agents"]["news_sentiment"]["location_context"]
    assert overlay["agents"]["news_sentiment"]["location_context"] == live_context

    live_chokepoint = settings["extraction"]["chokepoints"]["hormuz"]
    assert overlay["extraction"]["countries"] == live_chokepoint["countries"]
    assert overlay["extraction"]["bounding_box"] == live_chokepoint["bounding_box"]

    assert overlay["aisstream"]["bbox"] == settings["aisstream"]["monitor_regions"][0]["bbox"]

    # The intended deviation, asserted explicitly so it can't change silently.
    assert settings["agents"]["routing"]["enabled"] is True
    assert overlay["agents"]["routing"]["enabled"] is False
