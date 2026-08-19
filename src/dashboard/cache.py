"""Per-region config and Orchestrator caching for the dashboard (Phase 12.2).

Streamlit re-runs a page top to bottom on every widget interaction, so an
uncached ``Orchestrator(load_config_for_region(region))`` would rebuild six
connectors on every click. These helpers keep one config and one Orchestrator
per region, so the first visit to a region pays the cost and switching back
is free.

Two properties worth knowing before relying on these:

* **The cache is process-global**, not per-session — ``lru_cache`` lives on the
  module. Every Streamlit session in the process shares one Orchestrator per
  region. That suits this single-analyst thesis dashboard; a multi-user
  deployment would want session state instead.
* **Orchestrator is stateful.** It keeps ``_agents`` and ``_last_agent_frames``
  from the last run, so a cached instance carries its previous run's frames.
  That is what makes reuse cheap, and it is safe here because each region gets
  its own instance and runs are sequential — but do not hand the same instance
  to concurrent runs.

Use :func:`clear_region_caches` to drop both caches (tests, or after editing a
region YAML in a running dashboard).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from src.core.config_manager import load_config_for_region
from src.core.regions import list_regions

logger = logging.getLogger(__name__)

#: One entry per region, so no region is ever evicted by another.
_CACHE_SIZE = len(list_regions())


@lru_cache(maxsize=_CACHE_SIZE)
def get_cached_config(region: str) -> dict:
    """Return the merged config for ``region``, built once per process.

    Args:
        region: Canonical region key.

    Returns:
        The merged config dict. **Shared, not copied** — callers must treat it
        as read-only, since mutating it would corrupt every later cache hit.

    Raises:
        ValueError: If ``region`` is not a known region.
    """
    logger.info("[dashboard.cache] loading config for region '%s'", region)
    return load_config_for_region(region)


@lru_cache(maxsize=_CACHE_SIZE)
def get_cached_orchestrator(region: str) -> Any:
    """Return the Orchestrator for ``region``, built once per process.

    The Orchestrator builds its own connectors and agents from the config, so
    there is nothing to register afterwards — call ``run_full_pipeline()``
    directly on the result.

    Args:
        region: Canonical region key.

    Returns:
        A cached :class:`~src.orchestrator.Orchestrator` for that region.

    Raises:
        ValueError: If ``region`` is not a known region.
    """
    from src.orchestrator import Orchestrator

    logger.info("[dashboard.cache] initialising Orchestrator for region '%s'", region)
    return Orchestrator(config=get_cached_config(region))


def clear_region_caches() -> None:
    """Drop both caches — for tests, or after editing a region YAML."""
    get_cached_config.cache_clear()
    get_cached_orchestrator.cache_clear()
    logger.info("[dashboard.cache] caches cleared")


def cache_info() -> dict[str, Any]:
    """Return hit/miss counts for both caches, for debugging a slow switch."""
    return {
        "config": get_cached_config.cache_info()._asdict(),
        "orchestrator": get_cached_orchestrator.cache_info()._asdict(),
    }
