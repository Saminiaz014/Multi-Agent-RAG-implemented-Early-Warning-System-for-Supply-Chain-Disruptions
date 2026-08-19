"""Region isolation: each region runs independently (Phase 11.6).

Where ``test_config_manager.py`` checks the *config* agrees with the registry,
these tests check the *pipeline run* does — a passive agent could still be
built, scored, or weighted despite its ``enabled: false`` flag, and only an
end-to-end run shows that.

The four runs are shared through a module-scoped fixture: ``run_full_pipeline``
takes several seconds per region, and every test here asks a different question
of the same four results.
"""

from __future__ import annotations

import pytest

from src.core.config_manager import load_config_for_region
from src.core.regions import AGENT_KEYS, get_region, list_regions
from src.orchestrator import Orchestrator


def _run(region: str) -> dict:
    """Run one region's full pipeline from a freshly merged config."""
    return Orchestrator(config=load_config_for_region(region)).run_full_pipeline()


@pytest.fixture(scope="module")
def region_results() -> dict[str, dict]:
    """Run every region once, sequentially, and share the results.

    Sequential execution is the point as well as an optimisation: if one
    region's run polluted global state, the regions after it would show it.
    """
    return {region: _run(region) for region in list_regions()}


def test_full_pipeline_runs_for_every_region(region_results: dict[str, dict]) -> None:
    """Every region completes and produces a usable composite score."""
    for region, result in region_results.items():
        assert result["agent_scores"], f"{region}: no agent produced a score"
        for key in ("risk_score", "composite_score", "risk_level"):
            assert key in result, f"{region}: result missing '{key}'"
        assert 0.0 <= float(result["composite_score"]) <= 1.0, region


def test_only_the_registry_active_agents_score(
    region_results: dict[str, dict]
) -> None:
    """The registry's activation decision holds at runtime, not just in config.

    This is the assertion that would catch a passive agent still being built
    and scored — the failure mode a config-only test cannot see.
    """
    for region, result in region_results.items():
        expected = set(get_region(region).active_agents())
        assert set(result["agent_scores"]) == expected, (
            f"{region}: scored {sorted(result['agent_scores'])}, "
            f"expected {sorted(expected)}"
        )


def test_routing_is_passive_in_every_region(region_results: dict[str, dict]) -> None:
    """Phase 11's global routing muting, verified end-to-end.

    Asserted separately from the registry check above because the muting is a
    deliberate, temporary, cross-region decision rather than a per-region
    evidence call — it should fail loudly and by name if it is ever undone by
    accident. See the ROUTING AGENT block in src/core/regions.py.
    """
    for region, result in region_results.items():
        assert "routing" not in result["agent_scores"], f"{region}: routing scored"
        weights_used = result.get("metadata", {}).get("weights_used", {}) or {}
        assert not weights_used.get("routing"), f"{region}: routing carried weight"


def test_passive_agents_are_never_weighted(region_results: dict[str, dict]) -> None:
    """A passive agent must not dilute the composite by holding weight."""
    for region, result in region_results.items():
        passive = set(get_region(region).passive_agents())
        weights_used = result.get("metadata", {}).get("weights_used", {}) or {}
        for agent in passive:
            assert not weights_used.get(agent), f"{region}: passive '{agent}' weighted"


def test_regions_produce_different_results(region_results: dict[str, dict]) -> None:
    """Different agent sets and inputs must not collapse to one shared answer."""
    signatures = {
        region: (
            tuple(sorted(result["agent_scores"])),
            round(float(result["composite_score"]), 6),
        )
        for region, result in region_results.items()
    }
    assert len(set(signatures.values())) == len(signatures), (
        f"regions produced identical results: {signatures}"
    )


def test_rerunning_a_region_reproduces_it_exactly(
    region_results: dict[str, dict]
) -> None:
    """Hormuz re-run *after* all four regions must match its isolated run.

    Combines the determinism check with the contamination check: the fixture
    already ran Panama, Bab el-Mandeb and Malacca in between, so a difference
    here means earlier runs left state behind.
    """
    first = region_results["hormuz"]
    second = _run("hormuz")

    assert second["composite_score"] == first["composite_score"]
    assert second["agent_scores"] == first["agent_scores"]


def test_every_agent_key_is_active_somewhere_or_documented(
    region_results: dict[str, dict]
) -> None:
    """No agent silently vanishes from all four regions without a reason.

    Routing is the one agent passive everywhere; that is intentional and
    documented, so it is the sole permitted exception.
    """
    scored_anywhere = {
        agent for result in region_results.values() for agent in result["agent_scores"]
    }
    never_scored = set(AGENT_KEYS) - scored_anywhere
    assert never_scored == {"routing"}, (
        f"unexpected agents inactive in every region: {sorted(never_scored - {'routing'})}"
    )
