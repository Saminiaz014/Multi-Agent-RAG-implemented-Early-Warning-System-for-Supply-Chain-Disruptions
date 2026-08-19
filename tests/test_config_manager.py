"""Tests for region-aware config loading and merging (Phase 11.3).

Scoped to what only this module can get wrong: merge semantics, region
resolution precedence, and the merged config's agreement with the registry.
The region YAMLs' own structure is covered by test_region_configs.py, and the
connector settings the merge projects are covered by
test_region_specific_connectors.py — neither is repeated here.
"""

from __future__ import annotations

import pytest

from src.core.config_manager import (
    DEFAULT_REGION,
    REGION_ENV_VAR,
    _deep_merge,
    load_base_config,
    load_config_for_region,
    load_region_overlay,
    resolve_active_region,
)
from src.core.regions import AGENT_KEYS, get_region, list_regions


@pytest.fixture(autouse=True)
def _clear_region_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of the developer's own SUPPLY_CHAIN_REGION."""
    monkeypatch.delenv(REGION_ENV_VAR, raising=False)


class TestDeepMerge:
    """The three merge behaviours the overlays depend on."""

    def test_recurses_into_nested_dicts_and_keeps_siblings(self) -> None:
        """An overlay naming one key must not wipe its siblings."""
        base = {"agents": {"shipping": {"enabled": True, "threshold": 0.65}}}
        overlay = {"agents": {"shipping": {"threshold": 0.70}}, "extra": 1}
        result = _deep_merge(base, overlay)

        assert result["agents"]["shipping"] == {"enabled": True, "threshold": 0.70}
        assert result["extra"] == 1

    def test_does_not_mutate_inputs(self) -> None:
        base = {"agents": {"routing": {"enabled": True}}}
        result = _deep_merge(base, {"agents": {"routing": {"enabled": False}}})

        assert base == {"agents": {"routing": {"enabled": True}}}
        assert result["agents"]["routing"]["enabled"] is False

    def test_replaces_rather_than_merges_non_dict_values(self) -> None:
        """Lists are replaced, not concatenated — bounding boxes rely on this."""
        assert _deep_merge({"bbox": [[1, 2]]}, {"bbox": [[5, 6]]})["bbox"] == [[5, 6]]
        assert _deep_merge({"a": {"b": 1}}, {"a": 5})["a"] == 5


class TestLoading:
    """Path resolution and failure modes."""

    def test_base_config_loads_independent_of_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The default path is project-root-relative, not CWD-relative.

        The dashboard and pytest do not necessarily run from the repo root, so
        this is what keeps the loader usable from every call site.
        """
        monkeypatch.chdir(tmp_path)
        assert "agents" in load_base_config()

    def test_missing_base_config_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_base_config("nonexistent/settings.yaml")

    def test_region_overlay_loads(self) -> None:
        overlay = load_region_overlay("hormuz")
        assert overlay["region"]["name"] == "hormuz"
        assert set(overlay["agents"]) == set(AGENT_KEYS)

    def test_invalid_region_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_region_overlay("invalid_region")
        with pytest.raises(ValueError, match="not found"):
            load_config_for_region("invalid_region")


class TestRegionSelection:
    """Which region wins: explicit argument > env var > default."""

    def test_selection_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert resolve_active_region() == DEFAULT_REGION == "hormuz"

        monkeypatch.setenv(REGION_ENV_VAR, "  MALACCA ")
        assert resolve_active_region() == "malacca"
        assert load_config_for_region()["_active_region"] == "malacca"
        assert load_config_for_region("panama")["_active_region"] == "panama"

    def test_invalid_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo'd env var fails loudly instead of silently running Hormuz."""
        monkeypatch.setenv(REGION_ENV_VAR, "hormuzz")
        with pytest.raises(ValueError, match=REGION_ENV_VAR):
            resolve_active_region()


class TestMergedConfig:
    """The dict the Orchestrator actually receives."""

    def test_enabled_flags_match_the_registry_in_every_region(self) -> None:
        """The merge carries region activation through to the Orchestrator's flag.

        src/core/regions.py and config/regions/*.yaml already agree with each
        other (test_region_configs.py); this covers the merge between them, and
        with it Phase 11's global routing muting.
        """
        for region in list_regions():
            agents = load_config_for_region(region)["agents"]
            registry = get_region(region)
            for agent_key in AGENT_KEYS:
                assert agents[agent_key]["enabled"] is registry.agents[agent_key], (
                    f"{region}: '{agent_key}' disagrees with the registry"
                )

    def test_overlay_preserves_base_settings_it_does_not_name(self) -> None:
        """Overlaying ``enabled`` must not wipe tuning parameters or sections."""
        config = load_config_for_region("panama")
        base = load_base_config()

        for section in ("weights", "thresholds", "logging"):
            assert config[section] == base[section]
        assert config["agents"]["shipping"] == base["agents"]["shipping"]
        assert (
            config["agents"]["natural_disaster"]["monitoring_points"]
            == base["agents"]["natural_disaster"]["monitoring_points"]
        )

    def test_carries_region_identity(self) -> None:
        config = load_config_for_region("panama")
        assert config["_active_region"] == "panama"
        assert config["region"]["display_name"] == "Panama Canal"
