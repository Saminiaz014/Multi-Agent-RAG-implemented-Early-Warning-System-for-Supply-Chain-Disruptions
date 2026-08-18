"""Cross-cutting core abstractions shared by the pipeline.

Currently holds the region registry (:mod:`src.core.regions`), which defines
which chokepoints the pipeline can run against and which agents are active
in each.
"""

from src.core.regions import (
    REGIONS,
    RegionConfig,
    get_region,
    is_agent_active,
    list_regions,
)

__all__ = [
    "REGIONS",
    "RegionConfig",
    "get_region",
    "is_agent_active",
    "list_regions",
]
