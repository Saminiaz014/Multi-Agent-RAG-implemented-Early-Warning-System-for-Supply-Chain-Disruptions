"""Region registry — which chokepoints the pipeline runs against (Phase 11).

Defines the four live-pipeline regions and, per region, which of the six
agents are *active* (contribute a real signal) versus *passive* (excluded
from the run entirely).

**Agent keys are the pipeline's own config keys**, not display names — i.e.
``natural_disaster`` and ``news_sentiment``, matching ``config/settings.yaml``'s
``agents`` block and :meth:`src.orchestrator.Orchestrator._build_enabled_agents`.
A region's ``agents`` mapping is designed to drive the *existing*
``config["agents"][<name>]["enabled"]`` flag rather than introduce a second,
parallel enable/disable mechanism.

Activation is grounded in the per-region evidence review carried out for the
EVAL01 benchmark (``config/benchmark/{region}.yaml``'s ``active_domains``,
archived under ``docs/eval01-archived/``). Where a domain has no documented
real-world driver for a region, it is passive here — the same
evidence-over-plausibility standard that removed Malacca's ``market`` domain.
Each exclusion below cites its reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The six pipeline agents, in config-key form. Any region's ``agents`` mapping
# must cover exactly these — enforced by :func:`_validate_registry` at import.
AGENT_KEYS: tuple[str, ...] = (
    "shipping",
    "market",
    "geopolitical",
    "natural_disaster",
    "routing",
    "news_sentiment",
)


@dataclass(frozen=True)
class RegionConfig:
    """Static description of one monitored chokepoint.

    Attributes:
        name: Canonical region key (``"hormuz"``, ``"panama"``, ...). Matches
            the region's YAML filename under ``config/regions/``.
        display_name: Human-readable label for dashboards and logs.
        latitude: Centre-point latitude, decimal degrees.
        longitude: Centre-point longitude, decimal degrees (negative = West).
        agents: ``{agent_config_key: is_active}`` for all six agents. An agent
            mapped to ``False`` is passive — it is not built, not run, and not
            weighted for this region.
        passive_reasons: ``{agent_config_key: why_it_is_passive}``, for the
            agents mapped ``False``. Kept alongside the flags so the rationale
            travels with the data rather than living only in commit history.
    """

    name: str
    display_name: str
    latitude: float
    longitude: float
    agents: dict[str, bool]
    passive_reasons: dict[str, str] = field(default_factory=dict)

    def active_agents(self) -> list[str]:
        """Return the config keys of this region's active agents, in AGENT_KEYS order."""
        return [key for key in AGENT_KEYS if self.agents.get(key, False)]

    def passive_agents(self) -> list[str]:
        """Return the config keys of this region's passive agents, in AGENT_KEYS order."""
        return [key for key in AGENT_KEYS if not self.agents.get(key, False)]


REGIONS: dict[str, RegionConfig] = {
    "hormuz": RegionConfig(
        name="hormuz",
        display_name="Strait of Hormuz",
        latitude=26.5,
        longitude=56.5,
        agents={
            "shipping": True,
            "market": True,
            "geopolitical": True,
            "natural_disaster": True,
            "routing": False,
            "news_sentiment": True,
        },
        passive_reasons={
            "routing": (
                "No viable sea alternate into the Persian Gulf "
                "(config/benchmark/hormuz.yaml: reroutable: false), so "
                "avg_route_deviation_km has no real-world analogue here — "
                "routing_connector.py's own docstring calls it a modeled "
                "proxy, not a measured quantity."
            ),
        },
    ),
    "panama": RegionConfig(
        name="panama",
        display_name="Panama Canal",
        latitude=9.08,
        longitude=-79.68,
        agents={
            "shipping": True,
            "market": True,
            # Evidence-excluded — see passive_reasons.
            "geopolitical": False,
            "natural_disaster": True,
            "routing": False,
            "news_sentiment": True,
        },
        passive_reasons={
            "geopolitical": (
                "No geopolitical driver applies — the documented Panama "
                "disruption is purely hydrological/climate-driven (Gatun Lake "
                "drought). Stated outright in the region's evidence file; "
                "geopolitical is absent from panama.yaml's active_domains."
            ),
            "routing": (
                "WEAK evidence: real but minority diversion is documented "
                "(Stolt-Nielsen via Suez, Maersk's rail land bridge), but the "
                "dominant observed response was queueing and paying auction "
                "premiums — a queued vessel's path hasn't changed, so the "
                "AIS-deviation model cannot detect it."
            ),
        },
    ),
    "bab_el_mandeb": RegionConfig(
        name="bab_el_mandeb",
        display_name="Bab el-Mandeb",
        latitude=12.58,
        longitude=43.33,
        agents={
            "shipping": True,
            "market": True,
            "geopolitical": True,
            # Evidence-excluded — see passive_reasons.
            "natural_disaster": False,
            # The one region where routing is active — see module docstring.
            "routing": True,
            "news_sentiment": True,
        },
        passive_reasons={
            "natural_disaster": (
                "No natural-hazard driver applies: the documented event is a "
                "security/geopolitical campaign, not a natural disaster "
                "(bab_el_mandeb.yaml: disaster_relevance: none — an "
                "affirmative absence found in the evidence, not missing data)."
            ),
        },
    ),
    "malacca": RegionConfig(
        name="malacca",
        display_name="Strait of Malacca",
        latitude=2.50,
        longitude=101.80,
        agents={
            "shipping": True,
            # Evidence-excluded — see passive_reasons.
            "market": False,
            "geopolitical": True,
            "natural_disaster": True,
            "routing": False,
            "news_sentiment": True,
        },
        passive_reasons={
            "market": (
                "All four documented Malacca events carry a null market field "
                "— no quantified or qualitative market signal was located. "
                "Originally included on plausibility grounds and removed as a "
                "standards violation; this registry preserves that removal."
            ),
            "routing": (
                "WEAK evidence: a targeted search for documented Sunda/Lombok "
                "diversion due to piracy or haze found none. Both straits are "
                "real but draft-constrained alternatives; no observed "
                "diversion response to either driver is documented."
            ),
        },
    ),
}


def _validate_registry() -> None:
    """Fail loudly at import if any region's agent mapping is malformed.

    Guards the two mistakes most likely to slip in when adding a region: a
    typo'd or renamed agent key (e.g. ``news`` instead of ``news_sentiment``),
    and a missing agent leaving activation implicitly undefined.

    Raises:
        ValueError: If a region's ``agents`` keys don't exactly match
            :data:`AGENT_KEYS`, or a ``passive_reasons`` entry names an agent
            that isn't actually passive.
    """
    for key, region in REGIONS.items():
        if key != region.name:
            raise ValueError(
                f"Region registry key {key!r} does not match its "
                f"RegionConfig.name {region.name!r}."
            )
        missing = set(AGENT_KEYS) - set(region.agents)
        unknown = set(region.agents) - set(AGENT_KEYS)
        if missing or unknown:
            raise ValueError(
                f"Region {key!r} agent mapping is malformed. "
                f"Missing: {sorted(missing)}. Unknown: {sorted(unknown)}. "
                f"Valid agent keys: {list(AGENT_KEYS)}."
            )
        mislabelled = [
            name for name in region.passive_reasons if region.agents.get(name, False)
        ]
        if mislabelled:
            raise ValueError(
                f"Region {key!r} lists passive_reasons for active agent(s) "
                f"{sorted(mislabelled)}."
            )


_validate_registry()


def get_region(key: str) -> RegionConfig:
    """Return the :class:`RegionConfig` for ``key``.

    Args:
        key: Canonical region key, e.g. ``"hormuz"``. Case-insensitive.

    Returns:
        The matching :class:`RegionConfig`.

    Raises:
        ValueError: If ``key`` is not a known region. Lists valid options.
    """
    normalized = str(key or "").strip().lower()
    if normalized not in REGIONS:
        raise ValueError(
            f"Region {key!r} not found. Valid regions: {list_regions()}."
        )
    return REGIONS[normalized]


def list_regions() -> list[str]:
    """Return every known region key, in registry order."""
    return list(REGIONS)


def is_agent_active(region_key: str, agent_name: str) -> bool:
    """Return whether ``agent_name`` contributes a signal in ``region_key``.

    Args:
        region_key: Canonical region key (see :func:`get_region`).
        agent_name: Agent *config* key — one of :data:`AGENT_KEYS`, i.e.
            ``natural_disaster``/``news_sentiment``, not ``disaster``/``news``.

    Returns:
        ``True`` if the agent is active in that region, ``False`` if passive.

    Raises:
        ValueError: If ``region_key`` is unknown, or ``agent_name`` is not a
            recognised agent key — an unknown agent returns loudly rather than
            silently reporting ``False``, which would read as "passive" and
            quietly drop the agent from every region.
    """
    region = get_region(region_key)
    if agent_name not in AGENT_KEYS:
        raise ValueError(
            f"Unknown agent {agent_name!r}. Valid agent keys: {list(AGENT_KEYS)}."
        )
    return region.agents[agent_name]
