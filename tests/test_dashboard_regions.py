"""Multi-region dashboard: selector, caching, integration (Phase 12.1/12.2/12.4).

Streamlit pages cannot be driven headlessly here, so these tests target the
layer the page reads from — ``AVAILABLE_REGIONS``, ``load_app_config`` and
``src.dashboard.cache`` — plus the render path's behaviour on a region with no
mapped routes, which is the one branch the selector newly makes reachable.
"""

from __future__ import annotations

import pytest

from src.core.regions import get_region, list_regions
from src.dashboard.cache import (
    cache_info,
    clear_region_caches,
    get_cached_config,
    get_cached_orchestrator,
)
from src.dashboard.core import AVAILABLE_REGIONS, load_app_config


@pytest.fixture(autouse=True)
def _clean_caches() -> None:
    """Each test starts from cold caches and leaves them cold."""
    clear_region_caches()
    yield
    clear_region_caches()


class TestRegionSelector:
    """What the sidebar dropdown offers (P12.1)."""

    def test_offers_every_registered_region(self) -> None:
        """The selector is built from the registry, so it cannot drift from it."""
        assert set(AVAILABLE_REGIONS.values()) == set(list_regions())
        assert len(AVAILABLE_REGIONS) == 4

    def test_labels_are_display_names_in_registry_order(self) -> None:
        """Users pick 'Panama Canal', not 'panama'; Hormuz stays first."""
        assert list(AVAILABLE_REGIONS.items())[0] == ("Strait of Hormuz", "hormuz")
        for display_name, key in AVAILABLE_REGIONS.items():
            assert display_name == get_region(key).display_name

    def test_labels_are_unique(self) -> None:
        """Duplicate labels would make two options indistinguishable."""
        assert len(set(AVAILABLE_REGIONS)) == len(AVAILABLE_REGIONS)


class TestLoadAppConfig:
    """The dashboard's config entry point is region-aware (P12.1)."""

    def test_returns_the_selected_regions_config(self) -> None:
        for key in AVAILABLE_REGIONS.values():
            config = load_app_config(key)
            assert config["_active_region"] == key
            assert config["agents"]["routing"]["enabled"] is False

    def test_defaults_to_hormuz(self) -> None:
        """A no-argument call matches the CLI's default, not an empty config."""
        assert load_app_config()["_active_region"] == "hormuz"

    def test_activation_differs_by_region(self) -> None:
        """Switching region must actually change which agents run."""
        assert load_app_config("hormuz")["agents"]["geopolitical"]["enabled"] is True
        assert load_app_config("panama")["agents"]["geopolitical"]["enabled"] is False


class TestRegionCaching:
    """One config and one Orchestrator per region (P12.2)."""

    def test_repeated_calls_return_the_same_objects(self) -> None:
        assert get_cached_config("hormuz") is get_cached_config("hormuz")
        assert get_cached_orchestrator("hormuz") is get_cached_orchestrator("hormuz")

    def test_regions_get_separate_instances(self) -> None:
        assert get_cached_orchestrator("hormuz") is not get_cached_orchestrator("panama")
        assert get_cached_config("hormuz") is not get_cached_config("panama")

    def test_no_region_evicts_another(self) -> None:
        """maxsize covers every region, so a full sweep leaves the first cached."""
        first = get_cached_orchestrator("hormuz")
        for key in list_regions():
            get_cached_orchestrator(key)
        assert get_cached_orchestrator("hormuz") is first
        assert cache_info()["orchestrator"]["currsize"] == len(list_regions())

    def test_switching_back_is_a_cache_hit(self) -> None:
        """The property that makes a region switch fast."""
        get_cached_orchestrator("hormuz")
        get_cached_orchestrator("panama")
        get_cached_orchestrator("hormuz")
        assert cache_info()["orchestrator"]["hits"] >= 1
        assert cache_info()["orchestrator"]["misses"] == 2

    def test_clearing_drops_both_caches(self) -> None:
        before = get_cached_orchestrator("hormuz")
        clear_region_caches()
        assert cache_info()["orchestrator"]["currsize"] == 0
        assert get_cached_orchestrator("hormuz") is not before

    def test_invalid_region_raises_and_is_not_cached(self) -> None:
        """A bad key must not occupy a slot or be served from cache later."""
        with pytest.raises(ValueError, match="not found"):
            get_cached_orchestrator("atlantis")
        assert cache_info()["orchestrator"]["currsize"] == 0


class TestDashboardIntegration:
    """The dashboard's region path, end to end (P12.4)."""

    def test_every_region_builds_a_runnable_orchestrator(self) -> None:
        for key in AVAILABLE_REGIONS.values():
            orchestrator = get_cached_orchestrator(key)
            assert orchestrator.config["_active_region"] == key
            expected = set(get_region(key).active_agents())
            built = set(orchestrator._domain_connectors)
            # Domain connectors cover four of the six agents; shipping and
            # market are always built and live outside this dict.
            assert built == expected - {"shipping", "market"}, key

    def test_cached_orchestrator_matches_a_fresh_pipeline_run(self) -> None:
        """Caching must not change results — only skip the rebuild."""
        from src.core.config_manager import load_config_for_region
        from src.orchestrator import Orchestrator

        cached = get_cached_orchestrator("panama").run_full_pipeline()
        fresh = Orchestrator(config=load_config_for_region("panama")).run_full_pipeline()

        assert cached["composite_score"] == fresh["composite_score"]
        assert cached["agent_scores"] == fresh["agent_scores"]

    def test_region_without_routes_renders_its_activation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Selecting Panama shows its real agent activation, not a dead end.

        The Decision view draws corridor geometry for Hormuz only, so this
        branch is what every other region now reaches.
        """
        import src.dashboard.decision_view as view

        written: list[str] = []
        monkeypatch.setattr(view.st, "info", lambda text, **_: written.append(text))
        monkeypatch.setattr(view.st, "markdown", lambda text, **_: written.append(text))

        view._render_region_without_routes("panama")
        rendered = "\n".join(written)

        assert "Panama Canal" in rendered
        assert "Natural Disaster" in rendered  # active here
        # Passive agents appear with the registry's recorded reason.
        assert "Geopolitical" in rendered
        assert "hydrological" in rendered
