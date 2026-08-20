"""Vessel-level view of corridor risk for the Decision map (Phase 12.5).

**Where the risk number comes from, and where it does not.** The pipeline has
no per-vessel signal: the shipping connector ingests daily aggregate arrival
counts, ``aisstream.enabled`` is false, and ``ShippingConnector.fetch_from_api``
is still a stub. So a vessel's risk here is *its corridor's* risk for that day —
the pipeline's own weighted aggregation scoped to that corridor's agents
(:func:`src.dashboard.core.route_risk_series`), the same number the route status
list shows. Vessels on one corridor therefore share a score.

That is deliberate. Spreading the corridor score across vessels with a
per-ship jitter would look more informative while being entirely invented, and
would let a reader believe one ship was measurably riskier than the one beside
it. Identity fields (name, flag, ETA) are deterministic synthetic labels; every
record carries ``synthetic: True``, and callers are expected to say so in the UI.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.dashboard.core import (
    STATUS_COLORS,
    get_routes,
    get_vessels,
    route_risk_series,
)

logger = logging.getLogger(__name__)

#: Risk bands for vessel colouring, per the Phase 12.5 spec.
_GREEN_BELOW = 0.3
_YELLOW_BELOW = 0.6

#: Common flag states, used only as synthetic identity labels.
_FLAGS = ("Panama", "Liberia", "Marshall Islands", "Singapore", "Malta",
          "Bahamas", "Greece")

_STATUS_BY_COLOR = {"green": "normal", "yellow": "delayed", "red": "critical"}


def risk_to_color(risk_score: float) -> str:
    """Map a risk score to ``"green"`` / ``"yellow"`` / ``"red"``."""
    value = float(risk_score)
    if value < _GREEN_BELOW:
        return "green"
    if value < _YELLOW_BELOW:
        return "yellow"
    return "red"


def risk_to_status(risk_score: float) -> str:
    """Map a risk score to ``"normal"`` / ``"delayed"`` / ``"critical"``."""
    return _STATUS_BY_COLOR[risk_to_color(risk_score)]


def color_to_hex(color: str) -> str:
    """Hex for a band colour, reusing the dashboard's status palette.

    Keeps vessel markers, route lines and status pills the same three colours
    rather than introducing a fourth green somewhere on the page.
    """
    return {
        "green": STATUS_COLORS["Low"],
        "yellow": STATUS_COLORS["High"],
        "red": STATUS_COLORS["Critical"],
    }.get(color, STATUS_COLORS["Low"])


def get_vessels_for_region(
    region: str,
    day: int,
    ts: pd.DataFrame,
    params: dict,
    routes: list[dict] | None = None,
) -> list[dict]:
    """Vessels across every corridor in ``region``, carrying corridor risk.

    Args:
        region: Region key.
        day: 1-based day index into the risk series.
        ts: Frame from :func:`~src.dashboard.core.compute_timeseries`.
        params: Weight/threshold params from ``resolve_mode_params``.
        routes: Corridors to place vessels on; defaults to the region's own.

    Returns:
        Vessel dicts with ``id``, ``name``, ``lat``/``lng``, ``risk_score``,
        ``status``, ``color``, ``flag``, ``type``, ``route``, ``eta``, and
        ``synthetic: True``. Empty when the region has no corridors.
    """
    corridors = get_routes(region) if routes is None else routes
    if not corridors:
        return []

    index = max(0, min(int(day) - 1, len(ts) - 1)) if len(ts) else 0
    vessels: list[dict] = []

    for corridor in corridors:
        try:
            series = route_risk_series(ts, params, corridor, region)
            corridor_risk = float(series.iloc[index]) if len(series) else 0.0
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[vessel_data] risk unavailable for corridor '%s': %s",
                corridor.get("key"), exc,
            )
            corridor_risk = 0.0

        color = risk_to_color(corridor_risk)
        for position, raw in enumerate(get_vessels(corridor, day, region=region)):
            # Seeded per vessel id so identity labels are stable across reruns.
            rng = np.random.default_rng(abs(hash((raw["id"], corridor["key"]))) % 2**32)
            vessels.append({
                **raw,
                "name": f"{raw['type']} {chr(65 + position % 26)}",
                "risk_score": round(corridor_risk, 4),
                "status": risk_to_status(corridor_risk),
                "color": color,
                "color_hex": color_to_hex(color),
                "flag": _FLAGS[int(rng.integers(0, len(_FLAGS)))],
                "route": corridor["name"],
                "route_key": corridor["key"],
                "eta": f"+{int(rng.integers(1, 6))}d",
                "region": region,
                "synthetic": True,
            })

    return vessels


def get_vessel_details(vessel_id: str, vessels: list[dict]) -> dict | None:
    """Look one vessel up by id.

    Takes the vessel list rather than recomputing it: the caller already has
    the list it rendered markers from, and rebuilding risks returning a vessel
    generated for a different day than the one on screen.

    Args:
        vessel_id: The ``id`` field to match.
        vessels: The list from :func:`get_vessels_for_region`.

    Returns:
        The matching vessel, or ``None``.
    """
    for vessel in vessels:
        if vessel.get("id") == vessel_id:
            return vessel
    return None
