"""Region registry for EVAL01.

A :class:`Region` captures everything about a chokepoint that is *not*
scenario-specific: where it is, how busy it normally is, which detection
agents (:mod:`src.agents`) apply there, and how losses scale if it is
disrupted. Region-specific numbers live in ``config/benchmark/{name}.yaml``;
this module only knows how to read and validate that shape.

For R1 only ``hormuz`` is populated. Additional regions are added by
dropping a new ``config/benchmark/{name}.yaml`` — no code changes here.

Two independent readiness flags, not one — do not conflate them:
    :data:`REGION_STATUS` — live-pipeline readiness. The only flag
        src/dashboard/core.py's ``AVAILABLE_REGIONS`` reads.
    :data:`BENCHMARK_STATUS` — EVAL01 benchmark-spec readiness. What
        tests/test_benchmark_regions.py's parametrized ``load_region``
        coverage reads. A region can be "populated" here with no live
        ingestion behind it at all (e.g. bab_el_mandeb).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "benchmark"

# The six detection domains EVAL01 scenarios can drive, one per
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

# ---------------------------------------------------------------------------
# Canonical region vocabulary
#
# Before this, three parts of the codebase each maintained their own,
# unreconciled region vocabulary: this module (hormuz only), the dashboard's
# AVAILABLE_REGIONS (hormuz only, a literal), and config/settings.yaml's
# extraction.chokepoints (hormuz/red_sea/malacca/suez, for RAG backfill only).
# CANONICAL_REGIONS below is the single source of truth every subsystem
# should read from going forward — see docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md
# and STATE_OF_PROJECT_MULTIREGION.md for the full audit.
# ---------------------------------------------------------------------------

#: Canonical region key -> display name, in priority order.
CANONICAL_REGIONS: dict[str, str] = {
    "hormuz": "Strait of Hormuz",
    "bab_el_mandeb": "Bab el-Mandeb",
    "panama": "Panama Canal",
    "suez": "Suez Canal",
    "malacca": "Strait of Malacca",
}

#: LIVE-PIPELINE readiness per canonical region key — "does this region have
#: real ingestion (connectors/agents/orchestrator) behind it, such that a
#: manager-facing surface (the dashboard) should offer it?" ``"populated"``
#: = real live data exists today (only ``hormuz``, until Phase B wires
#: ingestion for another region). ``"planned"`` = reserved in the canonical
#: vocabulary but not live yet. This is the ONLY status flag
#: src/dashboard/core.py's ``AVAILABLE_REGIONS`` reads — see
#: :data:`BENCHMARK_STATUS` below for the separate, EVAL01-only concern.
#:
#: These two flags were one field through A4; A4 surfaced that "this region
#: has validated benchmark scenarios" (an EVAL01/offline concern) and "this
#: region should appear in the live dashboard" (a live-ingestion/UI concern)
#: are genuinely different questions that can diverge — bab_el_mandeb is
#: benchmark-ready (a validated Region spec) but has no live ingestion, so
#: it belongs in BENCHMARK_STATUS as "populated" while staying "planned"
#: here. Flipping a key here to "populated" without real ingestion behind
#: it would surface an empty-data region in the dashboard; flipping it in
#: BENCHMARK_STATUS only affects EVAL01 test coverage.
REGION_STATUS: dict[str, str] = {
    "hormuz": "populated",
    "bab_el_mandeb": "planned",
    "panama": "planned",
    "suez": "planned",
    "malacca": "planned",
}

#: BENCHMARK readiness per canonical region key — "does this region have a
#: validated config/benchmark/{key}.yaml Region spec (and, once A5 lands,
#: passing scenario YAMLs)?" Same "populated"/"planned" vocabulary as
#: :data:`REGION_STATUS`, but a separate concern: this gates EVAL01 test
#: coverage (see tests/test_benchmark_regions.py's
#: ``test_load_region_succeeds_for_every_populated_region``, parametrized
#: off this dict), not anything user-facing. Nothing outside the benchmark
#: harness and its tests should read this — the dashboard reads
#: :data:`REGION_STATUS` only.
BENCHMARK_STATUS: dict[str, str] = {
    "hormuz": "populated",
    "bab_el_mandeb": "populated",
    "panama": "populated",
    "suez": "populated",
    "malacca": "populated",
}

#: Alternate spellings that resolve to a canonical key. ``red_sea`` is the
#: key config/settings.yaml's extraction.chokepoints and monitoring_points
#: blocks already use (pre-existing RAG/live-monitoring config) — aliased
#: here rather than renamed there, since renaming would invalidate the
#: existing 553-document knowledge base keyed on ``red_sea``.
REGION_ALIASES: dict[str, str] = {
    "red_sea": "bab_el_mandeb",
}


def resolve_region_key(value: str) -> str:
    """Resolve a canonical key, alias, or display name to its canonical key.

    Matching is case-insensitive and accepts any of: a canonical key
    (``"hormuz"``), an alias (``"red_sea"``), or a display name
    (``"Strait of Hormuz"``).

    Args:
        value: The region reference to resolve.

    Returns:
        The canonical region key (always one of :data:`CANONICAL_REGIONS`'s
        keys).

    Raises:
        ValueError: If ``value`` doesn't match any canonical key, alias, or
            display name. Lists every valid option.
    """
    text = str(value or "").strip()
    lowered = text.lower()

    if lowered in CANONICAL_REGIONS:
        return lowered
    if lowered in REGION_ALIASES:
        return REGION_ALIASES[lowered]
    for key, display_name in CANONICAL_REGIONS.items():
        if display_name.lower() == lowered:
            return key

    valid = sorted(set(CANONICAL_REGIONS) | set(REGION_ALIASES) |
                   {d for d in CANONICAL_REGIONS.values()})
    raise ValueError(f"Unknown region {value!r}. Valid options: {valid}.")


@dataclass
class Region:
    """A EVAL01 region spec, loaded from ``config/benchmark/{name}.yaml``.

    Attributes:
        name: Region key (matches the YAML filename and top-level key).
        center_lat: Latitude of the chokepoint's geographic center.
        center_lng: Longitude of the chokepoint's geographic center.
        baseline_transits_per_day: ``[min, max]`` daily vessel-transit range
            under normal conditions, or a single number — normalized
            internally to a one-element list. Any other type raises
            ``ValueError`` at load time (see :func:`load_region`).
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
        raw_transits = spec["baseline_transits_per_day"]
        if isinstance(raw_transits, (list, tuple)):
            baseline_transits_per_day = list(raw_transits)
        elif isinstance(raw_transits, (int, float)) and not isinstance(raw_transits, bool):
            # A single number is accepted and normalized to a one-element
            # list, per the Region docstring's "(or a single value)" note —
            # this used to crash with an uncaught TypeError instead.
            baseline_transits_per_day = [raw_transits]
        else:
            raise ValueError(
                f"Region '{name}' baseline_transits_per_day must be a "
                f"[min, max] list or a single number; got {raw_transits!r} "
                f"({type(raw_transits).__name__})."
            )

        return Region(
            name=name,
            center_lat=float(center["lat"]),
            center_lng=float(center["lng"]),
            baseline_transits_per_day=baseline_transits_per_day,
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
