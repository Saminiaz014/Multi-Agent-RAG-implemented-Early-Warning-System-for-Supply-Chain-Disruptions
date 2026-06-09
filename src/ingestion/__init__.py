"""Data ingestion layer — connectors for raw signal sources."""

from src.ingestion.base_connector import BaseConnector
from src.ingestion.disaster_connector import DisasterConnector
from src.ingestion.geopolitical_connector import GeopoliticalConnector
from src.ingestion.market_connector import MarketConnector
from src.ingestion.news_connector import NewsConnector
from src.ingestion.routing_connector import RoutingConnector
from src.ingestion.shipping_connector import ShippingConnector

__all__ = [
    "BaseConnector",
    "DisasterConnector",
    "GeopoliticalConnector",
    "MarketConnector",
    "NewsConnector",
    "RoutingConnector",
    "ShippingConnector",
]
