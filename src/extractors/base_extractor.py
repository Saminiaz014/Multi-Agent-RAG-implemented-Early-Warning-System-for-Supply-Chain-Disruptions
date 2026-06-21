"""Base class for live API extractors that populate the RAG knowledge base.

Each extractor normalizes its source's output to a common document schema
suitable for ChromaDB embedding::

    {
        "id": str,
        "text": str,
        "metadata": {
            "source_api": str,
            "event_date": str,
            "region": str,
            "countries": str,          # comma-joined (ChromaDB metadata is scalar-only)
            "primary_agents": str,     # comma-joined agent names
            "event_type": str,
            "severity": str,           # "low" | "medium" | "high"
        },
    }
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional
    pass

_ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def resolve_env_value(value: str | None) -> str:
    """Resolve a ``"${VAR_NAME}"`` placeholder from settings.yaml against the
    environment (after ``.env`` has been loaded). Plain strings pass through
    unchanged; missing env vars resolve to ``""``.
    """
    if not value:
        return ""
    match = _ENV_PLACEHOLDER.match(value.strip())
    if match:
        return os.environ.get(match.group(1), "")
    return value


class BaseExtractor(ABC):
    """Abstract base for API extractors."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.rate_limit = (
            config.get("extraction", {}).get("rate_limits", {}).get(self.source_name, 60)
        )
        self._last_request_time = 0.0

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of this data source (e.g. ``'newsapi'``)."""

    @abstractmethod
    def extract_historical(self, region: str, **kwargs) -> list[dict]:
        """Extract historical data for a chokepoint region.

        Args:
            region: Chokepoint name (``"hormuz"``, ``"red_sea"``, ``"malacca"``, ``"suez"``).

        Returns:
            List of normalized document dicts ready for ChromaDB.
        """

    def _api_key(self, key_name: str) -> str:
        """Resolve an ``api_keys.<key_name>`` config value via the environment."""
        raw = self.config.get("api_keys", {}).get(key_name, "")
        return resolve_env_value(raw)

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting between API calls."""
        if self.rate_limit <= 0:
            return
        min_interval = 60.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _normalize_document(
        self,
        doc_id: str,
        text: str,
        event_date: str,
        region: str,
        countries: list[str],
        primary_agents: list[str],
        event_type: str = "",
        severity: str = "medium",
        extra_metadata: dict | None = None,
    ) -> dict:
        """Build a normalized document dict for ChromaDB insertion."""
        metadata: dict = {
            "source_api": self.source_name,
            "event_date": event_date,
            "region": region,
            "countries": ",".join(countries),
            "primary_agents": ",".join(primary_agents),
            "event_type": event_type,
            "severity": severity,
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                metadata[k] = v if isinstance(v, (str, int, float, bool)) else str(v)

        return {
            "id": f"{self.source_name}_{doc_id}",
            "text": text,
            "metadata": metadata,
        }
