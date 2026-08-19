"""Config loading and merging for the region-aware pipeline (Phase 11.3).

Loads base ``config/settings.yaml`` and overlays a region's YAML from
``config/regions/<region>.yaml``, returning a single merged dict ready to hand
to :class:`~src.orchestrator.Orchestrator`.

**Merging happens at the call site** (``main.py``, the dashboard), never inside
the Orchestrator — ``Orchestrator(config: dict)`` stays untouched and unaware
of regions.

Path resolution
---------------
Relative paths are resolved against the *project root* (two levels above this
file), not the process CWD, so ``load_config_for_region()`` behaves the same
from ``main.py``, from pytest, and from the Streamlit dashboard. Absolute paths
are used as given.

Two-step assembly
-----------------
``load_config_for_region`` does a plain recursive merge and then a *projection*
pass, because the overlay YAMLs are flat where settings.yaml is keyed:

1. **Merge** (:func:`_deep_merge`) — lands each overlay block at the key path it
   occupies in the region YAML. ``agents.<key>.enabled`` overlays
   ``config["agents"][<key>]["enabled"]``, exactly the flag
   :meth:`Orchestrator.__init__` reads to decide whether to build a domain
   connector. The ``region`` block arrives as a new top-level key.

2. **Projection** (:func:`_project_region_settings`, Phase 11.4) — copies the
   overlay's flat ``extraction`` / ``aisstream`` values to the key paths the
   extractors and connectors actually read. A plain merge cannot reach these,
   so without this step they would sit in the config inert:

   ==================================  ==========================================
   Overlay key                         Projected to
   ==================================  ==========================================
   ``extraction.countries``            ``extraction.chokepoints.<key>.countries``
   ``extraction.bounding_box``         ``extraction.chokepoints.<key>.bounding_box``
   ``extraction.countries``            ``agents.geopolitical.acled_countries``
   ``aisstream.bbox``                  ``aisstream.monitor_regions[0]``
   ``aisstream.bbox``                  ``ingestion.shipping.ais_bounds``
   ==================================  ==========================================

   The flat overlay keys are left in place as well — the projection copies
   rather than moves, so the merged config stays a superset of the overlay.

News keywords are not projected: :class:`~src.ingestion.NewsConnector` derives
them from ``agents.news_sentiment.location_context``, which the merge already
makes region-specific.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from src.core.regions import get_region, list_regions

logger = logging.getLogger(__name__)

# Two levels up from src/core/ — the repo root that holds config/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: Region used when nothing else selects one. Hormuz is the thesis's primary
#: chokepoint and the region settings.yaml is written around.
DEFAULT_REGION = "hormuz"

#: Environment variable consulted by :func:`resolve_active_region`.
REGION_ENV_VAR = "SUPPLY_CHAIN_REGION"


def _resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute.

    Args:
        path: Absolute path, or a path relative to the repo root
            (e.g. ``"config/settings.yaml"``).

    Returns:
        An absolute :class:`~pathlib.Path`.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _PROJECT_ROOT / candidate


def _resolve_region_key(region: str) -> str:
    """Normalise ``region`` to its canonical registry key.

    Thin wrapper over :func:`src.core.regions.get_region`, which already
    lower-cases and strips its argument and raises on an unknown region.

    Args:
        region: Region key, any casing (``"Hormuz"``, ``" panama "``).

    Returns:
        The canonical key (``"hormuz"``, ``"panama"``, ...).

    Raises:
        ValueError: If ``region`` is not in the registry.
    """
    return get_region(region).name


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base``.

    Overlay values take precedence. Where both sides hold a dict the merge
    recurses; any other type is replaced wholesale (notably lists, which are
    replaced rather than concatenated).

    Args:
        base: Base dictionary to merge into.
        overlay: Dictionary whose values are laid over ``base``.

    Returns:
        A new dict. Neither ``base`` nor ``overlay`` is modified at the top
        level, and nested dicts reached by the recursion are likewise copied
        rather than mutated.
    """
    result = dict(base)

    for key, overlay_value in overlay.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            result[key] = _deep_merge(base_value, overlay_value)
        else:
            result[key] = overlay_value

    return result


def load_base_config(config_path: str | Path = "config/settings.yaml") -> dict:
    """Load the base application config.

    Args:
        config_path: Path to settings.yaml, absolute or relative to the project
            root.

    Returns:
        The parsed config dict (``{}`` for an empty file).

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = _resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Base config file not found: {config_path} (resolved to {path})"
        )

    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Failed to parse {path}: {exc}") from exc

    logger.debug("[config_manager] loaded base config from %s", path)
    return content


def load_region_overlay(region: str) -> dict:
    """Load the region overlay from ``config/regions/<region>.yaml``.

    A missing overlay file is non-fatal: it logs a warning and returns ``{}``,
    so the caller falls back to base config rather than failing a run. An
    *invalid region key* is fatal — that is a caller mistake, not a missing
    optional file.

    Args:
        region: Region key, e.g. ``"hormuz"``. Case-insensitive.

    Returns:
        The overlay dict, or ``{}`` if the file is absent or empty.

    Raises:
        ValueError: If ``region`` is not in :mod:`src.core.regions`.
        yaml.YAMLError: If the overlay YAML is malformed.
    """
    region_key = _resolve_region_key(region)
    region_path = _resolve_path(f"config/regions/{region_key}.yaml")

    if not region_path.exists():
        logger.warning(
            "[config_manager] region overlay not found: %s — using base config only",
            region_path,
        )
        return {}

    try:
        content = yaml.safe_load(region_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(
            f"Failed to parse region overlay {region_path}: {exc}"
        ) from exc

    logger.debug("[config_manager] loaded region overlay from %s", region_path)
    return content


def resolve_active_region() -> str:
    """Resolve which region should be active when the caller names none.

    Checks, in order:

    1. the ``SUPPLY_CHAIN_REGION`` environment variable;
    2. :data:`DEFAULT_REGION` (``"hormuz"``).

    Returns:
        A canonical region key.

    Raises:
        ValueError: If the environment variable is set to an unknown region.
            A typo'd env var fails loudly rather than silently running the
            default region under the wrong name.
    """
    env_region = os.getenv(REGION_ENV_VAR)

    if env_region and env_region.strip():
        try:
            resolved = _resolve_region_key(env_region)
        except ValueError as exc:
            raise ValueError(
                f"{REGION_ENV_VAR}={env_region!r} is not a known region: {exc}"
            ) from exc
        logger.info(
            "[config_manager] region from %s: %s", REGION_ENV_VAR, resolved
        )
        return resolved

    logger.debug(
        "[config_manager] %s unset — defaulting to region '%s'",
        REGION_ENV_VAR,
        DEFAULT_REGION,
    )
    return DEFAULT_REGION


def _project_region_settings(
    merged: dict, overlay: dict, region_key: str
) -> None:
    """Copy the overlay's flat region settings to the paths consumers read.

    Mutates ``merged`` in place. See this module's docstring for the full
    source → destination table. Each projection is independently guarded: an
    overlay missing ``extraction.countries`` simply leaves the ACLED country
    list alone rather than writing an empty one over whatever settings.yaml
    already had.

    Args:
        merged: The deep-merged config, modified in place.
        overlay: The raw region overlay, read for its flat keys.
        region_key: Canonical region key, used to name the aisstream monitor
            region and as the chokepoint key fallback.
    """
    extraction_overlay = overlay.get("extraction") or {}
    countries = extraction_overlay.get("countries")
    bounding_box = extraction_overlay.get("bounding_box")
    # Regions may share a chokepoint entry with a different name — bab_el_mandeb
    # maps onto settings.yaml's existing "red_sea" entry.
    chokepoint_key = extraction_overlay.get("chokepoint_key") or region_key
    bbox = (overlay.get("aisstream") or {}).get("bbox")

    if countries or bounding_box:
        chokepoints = merged.setdefault("extraction", {}).setdefault(
            "chokepoints", {}
        )
        entry = dict(chokepoints.get(chokepoint_key) or {})
        if countries:
            entry["countries"] = countries
        if bounding_box:
            entry["bounding_box"] = bounding_box
        chokepoints[chokepoint_key] = entry
        logger.debug(
            "[config_manager] projected extraction.chokepoints['%s'] for region '%s'",
            chokepoint_key,
            region_key,
        )

    if countries:
        # GeopoliticalConnector receives only its agents.geopolitical block, so
        # the ACLED country list has to live there to reach it.
        merged.setdefault("agents", {}).setdefault("geopolitical", {})[
            "acled_countries"
        ] = countries

    if bbox:
        # Replaced wholesale, not appended: a run monitors one region, and
        # settings.yaml's single Hormuz entry would otherwise leak into every
        # other region's live monitoring.
        merged.setdefault("aisstream", {})["monitor_regions"] = [
            {"name": region_key, "bbox": bbox}
        ]
        # Likewise, ShippingConnector only receives ingestion.shipping.
        merged.setdefault("ingestion", {}).setdefault("shipping", {})[
            "ais_bounds"
        ] = bbox


def load_config_for_region(
    region: str | None = None,
    base_config_path: str | Path = "config/settings.yaml",
) -> dict:
    """Load settings.yaml, overlay one region's config, and project it.

    Deep-merges the region overlay onto the base config, then runs
    :func:`_project_region_settings` so the overlay's flat ``extraction`` /
    ``aisstream`` values reach the key paths the extractors and connectors
    actually read (see the module docstring's table).

    This is the module's entry point and the intended call-site replacement for
    a bare ``yaml.safe_load("config/settings.yaml")``::

        config = load_config_for_region("panama")
        orchestrator = Orchestrator(config=config)

    Args:
        region: Canonical region key — one of ``hormuz``, ``panama``,
            ``bab_el_mandeb``, ``malacca``. ``None`` defers to
            :func:`resolve_active_region`.
        base_config_path: Path to settings.yaml, absolute or relative to the
            project root.

    Returns:
        The merged config dict, plus an ``"_active_region"`` key naming the
        region it was built for. The underscore marks it as loader metadata
        rather than user-authored settings; nothing in the pipeline consumes
        it, and it exists so a run's region is visible in logs and in any
        dumped config.

    Raises:
        ValueError: If ``region`` (or ``SUPPLY_CHAIN_REGION``) is unknown.
        FileNotFoundError: If the base config does not exist.
    """
    region_key = (
        resolve_active_region() if region is None else _resolve_region_key(region)
    )

    base = load_base_config(base_config_path)
    overlay = load_region_overlay(region_key)
    merged = _deep_merge(base, overlay)
    _project_region_settings(merged, overlay, region_key)
    merged["_active_region"] = region_key

    logger.info(
        "[config_manager] config for region '%s' (base=%s, overlay=config/regions/%s.yaml)",
        region_key,
        base_config_path,
        region_key,
    )
    return merged


def available_regions() -> list[str]:
    """Return every region :func:`load_config_for_region` accepts.

    Convenience re-export of :func:`src.core.regions.list_regions` so callers
    building a CLI or dashboard selector need only import this module.
    """
    return list_regions()
