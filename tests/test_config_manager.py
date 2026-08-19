"""Tests for region-aware config loading and merging (Phase 11.3).

Covers the four pieces that can silently go wrong: the recursive merge itself,
the two loaders' failure modes, region resolution from the environment, and the
end-to-end shape of a merged config for every registered region.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.config_manager import (
    DEFAULT_REGION,
    REGION_ENV_VAR,
    _deep_merge,
    available_regions,
    load_base_config,
    load_config_for_region,
    load_region_overlay,
    resolve_active_region,
)
from src.core.regions import AGENT_KEYS, get_region, list_regions

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clear_region_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test independent of the developer's own SUPPLY_CHAIN_REGION.

    Autouse and monkeypatch-based so the variable is restored even when a test
    fails partway — an ambient region leaking into the rest of the suite would
    be a maddening failure to chase.
    """
    monkeypatch.delenv(REGION_ENV_VAR, raising=False)


class TestDeepMerge:
    """Tests for :func:`_deep_merge`."""

    def test_merge_simple_dicts(self) -> None:
        assert _deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {
            "a": 1,
            "b": 3,
            "c": 4,
        }

    def test_merge_nested_dicts_is_recursive(self) -> None:
        """Sibling keys inside a nested dict survive; only named keys change."""
        base = {"agents": {"shipping": {"enabled": True, "threshold": 0.65}}}
        overlay = {"agents": {"shipping": {"threshold": 0.70}}}
        result = _deep_merge(base, overlay)
        assert result["agents"]["shipping"] == {"enabled": True, "threshold": 0.70}

    def test_merge_does_not_mutate_inputs(self) -> None:
        """Neither argument is modified — including nested dicts."""
        base = {"agents": {"routing": {"enabled": True}}}
        overlay = {"agents": {"routing": {"enabled": False}}}
        result = _deep_merge(base, overlay)

        assert base == {"agents": {"routing": {"enabled": True}}}
        assert overlay == {"agents": {"routing": {"enabled": False}}}
        assert result["agents"]["routing"]["enabled"] is False

    def test_scalar_replaces_dict_and_vice_versa(self) -> None:
        """A type change replaces outright rather than attempting a merge."""
        assert _deep_merge({"a": {"b": 1}}, {"a": 5}) == {"a": 5}
        assert _deep_merge({"a": 5}, {"a": {"b": 1}}) == {"a": {"b": 1}}

    def test_lists_are_replaced_not_concatenated(self) -> None:
        result = _deep_merge({"bbox": [[1, 2], [3, 4]]}, {"bbox": [[5, 6]]})
        assert result["bbox"] == [[5, 6]]

    def test_merge_empty_dicts(self) -> None:
        assert _deep_merge({}, {}) == {}
        assert _deep_merge({"a": 1}, {}) == {"a": 1}
        assert _deep_merge({}, {"a": 1}) == {"a": 1}


class TestLoadBaseConfig:
    """Tests for :func:`load_base_config`."""

    def test_loads_settings_yaml(self) -> None:
        config = load_base_config()
        assert isinstance(config, dict)
        assert "agents" in config
        assert "weights" in config

    def test_resolves_relative_path_from_any_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The default path is project-root-relative, not CWD-relative.

        The dashboard and pytest do not necessarily run from the repo root, so
        this is the property that keeps the loader usable from all call sites.
        """
        monkeypatch.chdir(tmp_path)
        assert "agents" in load_base_config()

    def test_absolute_path_is_used_as_given(self) -> None:
        config = load_base_config(_PROJECT_ROOT / "config" / "settings.yaml")
        assert "agents" in config

    def test_missing_config_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_base_config("nonexistent/settings.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("agents:\n  - [unclosed\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_base_config(bad)


class TestLoadRegionOverlay:
    """Tests for :func:`load_region_overlay`."""

    @pytest.mark.parametrize("region", list_regions())
    def test_every_registered_region_has_a_loadable_overlay(
        self, region: str
    ) -> None:
        overlay = load_region_overlay(region)
        assert isinstance(overlay, dict)
        assert overlay["region"]["name"] == region
        assert set(overlay["agents"]) == set(AGENT_KEYS)

    def test_region_key_is_case_insensitive(self) -> None:
        assert load_region_overlay("HORMUZ") == load_region_overlay("hormuz")

    def test_invalid_region_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_region_overlay("invalid_region")


class TestResolveActiveRegion:
    """Tests for :func:`resolve_active_region`."""

    def test_defaults_to_hormuz(self) -> None:
        assert resolve_active_region() == DEFAULT_REGION == "hormuz"

    def test_reads_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REGION_ENV_VAR, "panama")
        assert resolve_active_region() == "panama"

    def test_env_var_is_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REGION_ENV_VAR, "  BAB_EL_MANDEB ")
        assert resolve_active_region() == "bab_el_mandeb"

    def test_invalid_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo'd env var fails loudly instead of silently running Hormuz."""
        monkeypatch.setenv(REGION_ENV_VAR, "hormuzz")
        with pytest.raises(ValueError, match=REGION_ENV_VAR):
            resolve_active_region()

    def test_empty_env_var_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(REGION_ENV_VAR, "")
        assert resolve_active_region() == DEFAULT_REGION


class TestLoadConfigForRegion:
    """Tests for :func:`load_config_for_region` — the module entry point."""

    @pytest.mark.parametrize("region", list_regions())
    def test_merged_config_for_every_region(self, region: str) -> None:
        config = load_config_for_region(region)

        assert config["_active_region"] == region
        assert config["region"]["name"] == region
        # Base-config sections untouched by any overlay must survive intact.
        for section in ("agents", "weights", "thresholds", "ingestion", "logging"):
            assert section in config, f"{section} lost from merged {region} config"

    @pytest.mark.parametrize("region", list_regions())
    def test_all_six_agent_keys_present(self, region: str) -> None:
        """The overlay must not drop or rename any agent block."""
        agents = load_config_for_region(region)["agents"]
        for agent_key in AGENT_KEYS:
            assert agent_key in agents, f"'{agent_key}' missing from {region} config"

    @pytest.mark.parametrize("region", list_regions())
    def test_enabled_flags_match_the_region_registry(self, region: str) -> None:
        """The merged config is what actually drives Orchestrator activation.

        src/core/regions.py and config/regions/*.yaml already agree with each
        other (test_region_configs.py); this asserts the *merge* carries that
        agreement through to the dict the Orchestrator reads.
        """
        agents = load_config_for_region(region)["agents"]
        registry = get_region(region)
        for agent_key in AGENT_KEYS:
            assert agents[agent_key]["enabled"] is registry.agents[agent_key], (
                f"{region}: '{agent_key}' enabled flag disagrees with the registry"
            )

    @pytest.mark.parametrize("region", list_regions())
    def test_routing_disabled_everywhere(self, region: str) -> None:
        """Phase 11's temporary global routing muting reaches the merged config."""
        config = load_config_for_region(region)
        assert config["agents"]["routing"]["enabled"] is False

    @pytest.mark.parametrize("region", list_regions())
    def test_overlay_preserves_unoverridden_agent_settings(
        self, region: str
    ) -> None:
        """Overlaying ``enabled`` must not wipe an agent's tuning parameters."""
        config = load_config_for_region(region)
        base = load_base_config()

        shipping = config["agents"]["shipping"]
        assert shipping["detection_method"] == base["agents"]["shipping"][
            "detection_method"
        ]
        assert shipping["threshold"] == base["agents"]["shipping"]["threshold"]
        # settings.yaml's monitoring_points are deliberately left as-is.
        assert (
            config["agents"]["natural_disaster"]["monitoring_points"]
            == base["agents"]["natural_disaster"]["monitoring_points"]
        )

    def test_none_region_uses_default(self) -> None:
        assert load_config_for_region(None)["_active_region"] == DEFAULT_REGION

    def test_none_region_honours_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(REGION_ENV_VAR, "malacca")
        assert load_config_for_region()["_active_region"] == "malacca"

    def test_explicit_region_beats_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(REGION_ENV_VAR, "malacca")
        assert load_config_for_region("panama")["_active_region"] == "panama"

    def test_invalid_region_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_config_for_region("invalid_region")

    def test_regions_differ_from_each_other(self) -> None:
        """Sanity check that the overlay is doing something region-specific."""
        hormuz = load_config_for_region("hormuz")
        panama = load_config_for_region("panama")
        assert hormuz["agents"]["geopolitical"]["enabled"] is True
        assert panama["agents"]["geopolitical"]["enabled"] is False
        assert (
            hormuz["agents"]["news_sentiment"]["location_context"]["primary_location"]
            != panama["agents"]["news_sentiment"]["location_context"][
                "primary_location"
            ]
        )

    def test_merged_config_is_orchestrator_ready(self) -> None:
        """The merged dict constructs an Orchestrator and honours the flags.

        The point of Phase 11.3 is that merging at the call site needs no
        Orchestrator change, so this asserts against the real class rather than
        the config shape alone.
        """
        from src.orchestrator import Orchestrator

        orchestrator = Orchestrator(config=load_config_for_region("panama"))
        built = set(orchestrator._domain_connectors)
        assert "geopolitical" not in built  # passive in panama
        assert "routing" not in built  # globally muted
        assert "natural_disaster" in built


class TestAvailableRegions:
    """Tests for :func:`available_regions`."""

    def test_matches_the_registry(self) -> None:
        assert available_regions() == list_regions()
        assert set(available_regions()) == {
            "hormuz",
            "panama",
            "bab_el_mandeb",
            "malacca",
        }
