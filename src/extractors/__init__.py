"""Live API extractors for RAG knowledge base population."""

from src.extractors.acled_extractor import ACLEDExtractor
from src.extractors.aisstream_monitor import AISStreamMonitor
from src.extractors.ambee_extractor import AmbeeExtractor
from src.extractors.fred_extractor import FREDExtractor
from src.extractors.knowledge_base_builder import KnowledgeBaseBuilder
from src.extractors.newsapi_extractor import NewsAPIExtractor
from src.extractors.reliefweb_extractor import ReliefWebExtractor
from src.extractors.serpapi_extractor import SerpAPIExtractor

__all__ = [
    "ACLEDExtractor",
    "AISStreamMonitor",
    "AmbeeExtractor",
    "FREDExtractor",
    "KnowledgeBaseBuilder",
    "NewsAPIExtractor",
    "ReliefWebExtractor",  # kept as fallback once an approved appname is obtained
    "SerpAPIExtractor",
]
