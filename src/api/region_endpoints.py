"""Region management endpoints (Phase 12.3).

Exposes the Phase 11 region registry over HTTP so a dashboard or frontend can
discover the available chokepoints and change which one the service scores:

- ``GET  /api/regions/list`` — every region, with its activation summary
- ``GET  /api/regions/current`` — the region the service is scoring now
- ``GET  /api/regions/info/{region}`` — one region in detail
- ``POST /api/regions/switch`` — change the active region

``switch`` mutates process-global state in :mod:`src.api.endpoints` (the same
``_config`` / ``_orchestrator`` globals the rest of the API already uses) and
resets the cached Orchestrator so the next scoring request rebuilds against
the new region. That makes it a genuine switch rather than a validation echo —
but it also means the active region is **per-process, not per-client**: one
caller switching changes what every other caller sees. Fine for the single
-analyst thesis deployment this serves; a multi-tenant service would take the
region as a request parameter instead.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.regions import AGENT_KEYS, get_region, list_regions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/regions", tags=["regions"])


class RegionInfo(BaseModel):
    """One region's identity and agent activation."""

    key: str = Field(..., description="Canonical region key, e.g. 'panama'.")
    display_name: str = Field(..., description="Human-readable chokepoint name.")
    latitude: float = Field(..., description="Centre-point latitude, decimal degrees.")
    longitude: float = Field(
        ..., description="Centre-point longitude, decimal degrees (negative = West)."
    )
    active_agents: list[str] = Field(
        ..., description="Agent config keys that contribute a signal here."
    )
    passive_agents: list[str] = Field(
        ..., description="Agent config keys excluded from this region's run."
    )
    passive_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="Why each passive agent is excluded — evidence, not omission.",
    )
    is_active: bool = Field(
        ..., description="Whether this is the region the service is currently scoring."
    )


class RegionListResponse(BaseModel):
    """Response body for ``GET /api/regions/list``."""

    regions: list[RegionInfo]
    current_region: str
    count: int


class RegionSwitchRequest(BaseModel):
    """Request body for ``POST /api/regions/switch``."""

    region: str = Field(..., description="Region key to switch to. Case-insensitive.")


class RegionSwitchResponse(BaseModel):
    """Response body for ``POST /api/regions/switch``."""

    success: bool
    previous_region: str
    region: str
    display_name: str
    active_agents: list[str]
    detail: str


def _describe(region_key: str, current: str) -> RegionInfo:
    """Build a :class:`RegionInfo` from the registry.

    Args:
        region_key: Canonical region key.
        current: The service's currently active region key.

    Returns:
        The populated response model.
    """
    cfg = get_region(region_key)
    return RegionInfo(
        key=cfg.name,
        display_name=cfg.display_name,
        latitude=cfg.latitude,
        longitude=cfg.longitude,
        active_agents=cfg.active_agents(),
        passive_agents=cfg.passive_agents(),
        passive_reasons=dict(cfg.passive_reasons),
        is_active=(cfg.name == current),
    )


@router.get("/list", response_model=RegionListResponse)
def list_all_regions() -> RegionListResponse:
    """List every registered region and flag the active one."""
    from src.api.endpoints import get_active_region

    current = get_active_region()
    return RegionListResponse(
        regions=[_describe(key, current) for key in list_regions()],
        current_region=current,
        count=len(list_regions()),
    )


@router.get("/current", response_model=RegionInfo)
def current_region() -> RegionInfo:
    """Return the region the service is scoring right now."""
    from src.api.endpoints import get_active_region

    current = get_active_region()
    return _describe(current, current)


@router.get("/info/{region}", response_model=RegionInfo)
def region_info(region: str) -> RegionInfo:
    """Return one region's detail.

    Args:
        region: Region key, case-insensitive.

    Raises:
        HTTPException: 404 if the region is not registered. The message lists
            the valid keys so a caller can recover without a second request.
    """
    from src.api.endpoints import get_active_region

    try:
        return _describe(region, get_active_region())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/switch", response_model=RegionSwitchResponse)
def switch_region(request: RegionSwitchRequest) -> RegionSwitchResponse:
    """Change the region the service scores.

    Resets the cached Orchestrator, so the next scoring request rebuilds
    against the new region's merged config. Switching to the already-active
    region is a no-op that still succeeds — callers can be idempotent.

    Args:
        request: Body carrying the target ``region``.

    Raises:
        HTTPException: 400 if the region is not registered. Unknown regions are
            rejected *before* any global state changes, so a bad request cannot
            leave the service half-switched.
    """
    from src.api.endpoints import get_active_region, set_active_region

    previous = get_active_region()
    try:
        cfg = get_region(request.region)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    set_active_region(cfg.name)
    logger.info("[api.regions] active region: %s -> %s", previous, cfg.name)

    return RegionSwitchResponse(
        success=True,
        previous_region=previous,
        region=cfg.name,
        display_name=cfg.display_name,
        active_agents=cfg.active_agents(),
        detail=(
            f"Now scoring {cfg.display_name}. "
            f"{len(cfg.active_agents())}/{len(AGENT_KEYS)} agents active."
        ),
    )
