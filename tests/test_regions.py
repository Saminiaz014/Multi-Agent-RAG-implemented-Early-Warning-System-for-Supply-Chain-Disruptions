"""Tests for the Phase 11 region registry (``src/core/regions.py``).

Covers registry lookup, the evidence-grounded agent activation flags, and —
critically — that the registry's agent keys are the pipeline's *real* config
keys. That last guard exists because the Phase 11 plan document specified
``disaster``/``news``, while the pipeline actually uses
``natural_disaster``/``news_sentiment``; a mismatch there would silently
disable the wrong agents.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.regions import (
    AGENT_KEYS,
    get_region,
    is_agent_active,
    list_regions,
)

_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def test_list_all_regions() -> None:
    """All four Phase 11 chokepoints are registered."""
    regions = list_regions()
    assert len(regions) == 4
    assert set(regions) == {"hormuz", "panama", "bab_el_mandeb", "malacca"}


def test_get_region_returns_config() -> None:
    """Lookup returns the right region, and is case-insensitive."""
    hormuz = get_region("hormuz")
    assert hormuz.name == "hormuz"
    assert hormuz.display_name == "Strait of Hormuz"
    assert get_region("HORMUZ").name == "hormuz"


def test_invalid_region_raises() -> None:
    """An unknown region fails loudly and lists the valid options."""
    with pytest.raises(ValueError, match="not found"):
        get_region("atlantis")


def test_routing_passive_in_every_region() -> None:
    """Routing is globally muted — passive in all four regions.

    This is a deliberate temporary muting, not an evidence judgement; see the
    "ROUTING AGENT: GLOBALLY PASSIVE" block in src/core/regions.py. Bab
    el-Mandeb's routing evidence is the strongest in the benchmark and would
    justify activation, so its muting carries an explicit reason recording
    that — asserted here so re-enabling is a conscious change.
    """
    for region in list_regions():
        assert is_agent_active(region, "routing") is False, (
            f"routing should be passive in {region}"
        )

    bab_reason = get_region("bab_el_mandeb").passive_reasons["routing"]
    assert "not an evidence judgement" in bab_reason.lower()


def test_market_passive_only_in_malacca() -> None:
    """Market is passive solely in Malacca (all four events have a null market field)."""
    assert is_agent_active("malacca", "market") is False
    for region in ("hormuz", "panama", "bab_el_mandeb"):
        assert is_agent_active(region, "market") is True, (
            f"market should be active in {region}"
        )


def test_evidence_based_exclusions() -> None:
    """The two exclusions carried over from the EVAL01 evidence review.

    Both are affirmative absences found in the evidence, not missing data —
    Bab el-Mandeb's event is a security campaign with no natural-hazard
    driver, and Panama's is purely hydrological with no geopolitical driver.
    """
    assert is_agent_active("bab_el_mandeb", "natural_disaster") is False
    assert is_agent_active("panama", "geopolitical") is False

    # Each exclusion carries its stated reason alongside the flag.
    assert "natural_disaster" in get_region("bab_el_mandeb").passive_reasons
    assert "geopolitical" in get_region("panama").passive_reasons


def test_agent_keys_match_settings_yaml() -> None:
    """Registry agent keys are the pipeline's real config keys.

    Guards against reintroducing ``disaster``/``news`` (the Phase 11 plan's
    names) in place of ``natural_disaster``/``news_sentiment``, which would
    make every activation flag a silent no-op.
    """
    settings = yaml.safe_load(_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(AGENT_KEYS) == set(settings["agents"])
    assert set(AGENT_KEYS) == set(settings["weights"])

    for region_key in list_regions():
        assert set(get_region(region_key).agents) == set(AGENT_KEYS), (
            f"{region_key} does not cover exactly the six pipeline agents"
        )


def test_unknown_agent_raises() -> None:
    """An unrecognised agent name raises rather than reporting False.

    Returning False would read as "passive" and quietly drop a real agent.
    """
    with pytest.raises(ValueError, match="Unknown agent"):
        is_agent_active("hormuz", "disaster")  # real key is natural_disaster


def test_active_and_passive_partition() -> None:
    """active_agents() and passive_agents() partition all six agents."""
    for region_key in list_regions():
        region = get_region(region_key)
        assert set(region.active_agents()) | set(region.passive_agents()) == set(
            AGENT_KEYS
        )
        assert not set(region.active_agents()) & set(region.passive_agents())

    # Hormuz keeps five of six (routing passive); shipping/news are universal.
    assert len(get_region("hormuz").active_agents()) == 5
    for region_key in list_regions():
        assert is_agent_active(region_key, "shipping") is True
        assert is_agent_active(region_key, "news_sentiment") is True
