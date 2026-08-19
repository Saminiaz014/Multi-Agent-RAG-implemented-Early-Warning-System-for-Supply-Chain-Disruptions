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

What the overlay actually reaches
---------------------------------
The merge is a plain recursive dict merge. That lands each overlay block at the
same key path it occupies in the region YAML:

- ``agents.<key>.enabled`` — **effective today.** Overlays
  ``config["agents"][<key>]["enabled"]``, which is exactly the flag
  :meth:`Orchestrator.__init__` already reads to decide whether to build a
  domain connector. This is the one block that changes pipeline behaviour, and
  it is the point of Phase 11.3.
- ``region`` — added as a new top-level block (name / display_name / latitude /
  longitude). Nothing reads it yet; it is there for logging and for Phase 11.4.
- ``extraction.{chokepoint_key,countries,bounding_box}`` — lands as *siblings*
  of ``extraction.chokepoints``, **not** merged into
  ``extraction["chokepoints"][<key>]``. settings.yaml keys those by chokepoint
  name, and the overlay is flat, so no plain dict merge can reach that path.
- ``aisstream.bbox`` — likewise a sibling of ``aisstream.monitor_regions``, not
  an update to ``monitor_regions[0]["bbox"]``.

The last two are therefore inert: they are carried in the merged config but no
connector reads those key paths. Translating them into the shapes the
extractors and the aisstream client expect is Phase 11.4's job ("Connectors
Read Region-Specific Settings"); doing it here would mean rewriting connector
inputs before any connector is ready to read them. The values are correct and
present — they simply do not take effect yet.
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


def load_config_for_region(
    region: str | None = None,
    base_config_path: str | Path = "config/settings.yaml",
) -> dict:
    """Load settings.yaml and overlay one region's config onto it.

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
