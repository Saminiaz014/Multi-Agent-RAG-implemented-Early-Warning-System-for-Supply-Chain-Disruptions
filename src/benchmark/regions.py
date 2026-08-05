"""Region registry for ChokeBench-16.

A :class:`Region` captures everything about a chokepoint that is *not*
scenario-specific: where it is, how busy it normally is, which detection
agents (:mod:`src.agents`) apply there, and how losses scale if it is
disrupted. Region-specific numbers live in ``config/benchmark/{name}.yaml``;
this module only knows how to read and validate that shape.

For R1 only ``hormuz`` is populated. Additional regions are added by
dropping a new ``config/benchmark/{name}.yaml`` — no code changes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "benchmark"

# The six detection domains ChokeBench-16 scenarios can drive, one per
# agent in src/agents (shipping_agent, market_agent, geopolitical_agent,
# routing_agent, news_agent, disaster_agent).
KNOWN_DOMAINS: tuple[str, ...] = (
    "shipping",
    "market",
    "geopolitical",
    "routing",
    "news",
    "disaster",
)


@dataclass
class Region:
    """A ChokeBench-16 region spec, loaded from ``config/benchmark/{name}.yaml``.

    Attributes:
        name: Region key (matches the YAML filename and top-level key).
        center_lat: Latitude of the chokepoint's geographic center.
        center_lng: Longitude of the chokepoint's geographic center.
        baseline_transits_per_day: ``[min, max]`` (or a single value) daily
            vessel-transit range under normal conditions.
        active_domains: Subset of :data:`KNOWN_DOMAINS` this region's
            scenarios are allowed to drive. Domains outside this list are
            *disabled* — materialized scenarios never populate them.
        reroutable: Whether traffic can divert to an alternate corridor.
        loss_scaling: ``"linear"`` or ``"superlinear"`` — how disruption
            severity scales with duration/closure.
        disaster_relevance: ``"low"`` / ``"none"`` / ``"high"`` — how much
            the natural-disaster domain matters for this region.
    """

    name: str
    center_lat: float
    center_lng: float
    baseline_transits_per_day: list
    active_domains: list[str]
    reroutable: bool
    loss_scaling: str
    disaster_relevance: str


def _region_yaml_path(name: str) -> Path:
    return _CONFIG_DIR / f"{name}.yaml"


def load_region(name: str) -> Region:
    """Load and validate a region spec from ``config/benchmark/{name}.yaml``.

    Args:
        name: Region key, e.g. ``"hormuz"``.

    Returns:
        Populated :class:`Region`.

    Raises:
        FileNotFoundError: If no YAML spec exists for ``name``.
        ValueError: If the spec is missing required fields or lists an
            ``active_domains`` entry outside :data:`KNOWN_DOMAINS`.
    """
    path = _region_yaml_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No region spec at {path}. Expected config/benchmark/{name}.yaml."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    spec = raw.get(name)
    if spec is None:
        raise ValueError(
            f"{path} must have a top-level '{name}:' key matching its filename."
        )

    active_domains = list(spec.get("active_domains", []))
    unknown = set(active_domains) - set(KNOWN_DOMAINS)
    if unknown:
        raise ValueError(
            f"Region '{name}' has unknown active_domains {sorted(unknown)}; "
            f"expected a subset of {KNOWN_DOMAINS}."
        )

    center = spec.get("center", {}) or {}
    try:
        return Region(
            name=name,
            center_lat=float(center["lat"]),
            center_lng=float(center["lng"]),
            baseline_transits_per_day=list(spec["baseline_transits_per_day"]),
            active_domains=active_domains,
            reroutable=bool(spec["reroutable"]),
            loss_scaling=str(spec["loss_scaling"]),
            disaster_relevance=str(spec["disaster_relevance"]),
        )
    except KeyError as exc:
        raise ValueError(f"Region '{name}' spec at {path} is missing {exc}.") from exc


def _discover_region_names() -> list[str]:
    """Region keys with a YAML spec directly under config/benchmark/."""
    if not _CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))


# Populated at import time: {region_name: Region}. A region YAML with a
# broken spec is skipped rather than crashing every import that touches the
# registry — load_region(name) still raises directly for callers who need
# the failure surfaced.
REGION_REGISTRY: dict[str, Region] = {}
for _name in _discover_region_names():
    try:
        REGION_REGISTRY[_name] = load_region(_name)
    except (ValueError, FileNotFoundError):
        continue
