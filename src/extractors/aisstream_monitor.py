"""aisstream.io WebSocket monitor for real-time vessel tracking.

Covers: shipping + routing agents (LIVE monitoring only). Historical RAG
population does not use aisstream — there is no historical API; the
shipping/routing connectors fall back to their synthetic baselines for
backtesting (see ``src/ingestion``).
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any, Callable

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class AISStreamMonitor(BaseExtractor):
    """Real-time AIS vessel position monitor via WebSocket.

    Usage::

        monitor = AISStreamMonitor(config)
        asyncio.run(monitor.start_monitoring(callback=my_handler))
    """

    @property
    def source_name(self) -> str:
        return "aisstream"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.api_key = self._api_key("aisstream")
        self.ws_url = config.get("aisstream", {}).get("websocket_url", "wss://stream.aisstream.io/v0/stream")
        self.monitor_regions = config.get("aisstream", {}).get("monitor_regions", [])
        self.metrics_config = config.get("aisstream", {}).get("metrics", {})

        self._vessel_positions: dict[str, dict] = {}
        self._vessel_counts: defaultdict[str, int] = defaultdict(int)

        if not self.api_key:
            logger.warning("aisstream API key not configured — set api_keys.aisstream / AISSTREAM_API_KEY.")

    def extract_historical(self, region: str, **kwargs) -> list[dict]:
        """aisstream has no historical API — always returns ``[]``."""
        logger.info("aisstream has no historical API; historical shipping data uses synthetic baselines.")
        return []

    def _build_subscription_message(self) -> dict:
        bboxes = [r["bbox"] for r in self.monitor_regions if "bbox" in r]
        return {"APIKey": self.api_key, "BoundingBoxes": bboxes}

    def compute_current_metrics(self) -> dict:
        """Compute shipping metrics from current in-memory vessel tracking state."""
        now = time.time()
        window_hours = self.metrics_config.get("vessel_count_window_hours", 24)
        cutoff = now - (window_hours * 3600)

        active_vessels = {
            mmsi: pos for mmsi, pos in self._vessel_positions.items()
            if pos.get("last_seen", 0) > cutoff
        }
        congestion_baseline = self.metrics_config.get("congestion_baseline", 0.3)
        vessel_count = len(active_vessels)
        slow_vessels = sum(1 for v in active_vessels.values() if v.get("speed", 0) < 2.0)
        congestion = slow_vessels / max(vessel_count, 1)

        return {
            "vessel_count": vessel_count,
            "unique_mmsi_count": len(active_vessels),
            "slow_vessels": slow_vessels,
            "congestion_index": round(congestion, 3),
            "congestion_vs_baseline": round(congestion - congestion_baseline, 3),
            "timestamp": now,
        }

    async def start_monitoring(
        self, callback: Callable[[dict], None] | None = None, duration_seconds: int = 0,
    ) -> None:
        """Start WebSocket monitoring of AIS data."""
        if not self.api_key:
            logger.error("aisstream API key not set — cannot start monitoring")
            return
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. Run: pip install websockets")
            return

        sub_msg = self._build_subscription_message()
        logger.info("aisstream: connecting to %s (%d region(s))", self.ws_url, len(sub_msg["BoundingBoxes"]))
        start_time = time.time()

        try:
            async with websockets.connect(self.ws_url) as ws:
                await ws.send(json.dumps(sub_msg))
                async for message in ws:
                    data = json.loads(message)
                    if data.get("MessageType") == "PositionReport":
                        position = data.get("Message", {}).get("PositionReport", {})
                        meta = data.get("MetaData", {})
                        mmsi = str(meta.get("MMSI", ""))
                        if mmsi:
                            self._vessel_positions[mmsi] = {
                                "mmsi": mmsi,
                                "ship_name": meta.get("ShipName", "").strip(),
                                "latitude": position.get("Latitude", 0),
                                "longitude": position.get("Longitude", 0),
                                "speed": position.get("Sog", 0),
                                "course": position.get("Cog", 0),
                                "last_seen": time.time(),
                            }
                            if callback:
                                callback(self._vessel_positions[mmsi])

                    if duration_seconds > 0 and time.time() - start_time > duration_seconds:
                        logger.info("aisstream: monitoring duration reached (%ds)", duration_seconds)
                        break
        except Exception as exc:
            logger.error("aisstream WebSocket error: %s", exc)
