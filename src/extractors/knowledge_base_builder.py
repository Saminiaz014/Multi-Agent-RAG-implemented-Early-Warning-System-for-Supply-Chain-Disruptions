"""Knowledge base builder — orchestrates all extractors and populates ChromaDB.

Pipeline: extract from each enabled API source -> deduplicate by id ->
save a JSON backup -> upsert into the ``live_extracted_context`` ChromaDB
collection (queried alongside the static ``disruption_cases`` collection by
:class:`~src.rag.context_retriever.ContextRetriever`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.extractors.acled_extractor import ACLEDExtractor
from src.extractors.ambee_extractor import AmbeeExtractor
from src.extractors.base_extractor import BaseExtractor
from src.extractors.disaster_combined_extractor import DisasterCombinedExtractor
from src.extractors.fred_extractor import FREDExtractor
from src.extractors.gdacs_extractor import GDACSExtractor
from src.extractors.newsapi_extractor import NewsAPIExtractor
from src.extractors.reliefweb_extractor import ReliefWebExtractor
from src.extractors.serpapi_extractor import SerpAPIExtractor
from src.extractors.usgs_extractor import USGSExtractor

logger = logging.getLogger(__name__)

_EXTRACTOR_CLASSES: dict[str, type] = {
    "newsapi": NewsAPIExtractor,
    "serpapi": SerpAPIExtractor,    # historical news for RAG backfill (NewsAPI is current-only)
    # GDACS + USGS behind one wrapper, so the per-month cap covers their
    # combined output. Both remain registered for a targeted, uncapped run.
    "disaster_combined": DisasterCombinedExtractor,
    "gdacs": GDACSExtractor,        # primary natural_disaster source (replaced ambee)
    "usgs": USGSExtractor,          # seismic detail below GDACS's alert bar
    "ambee": AmbeeExtractor,        # retained, disabled: valid key, zero documents
    "reliefweb": ReliefWebExtractor,  # fallback once an approved appname is obtained
    "fred": FREDExtractor,
    "acled": ACLEDExtractor,
}


class KnowledgeBaseBuilder:
    """Orchestrate all extractors and populate ChromaDB.

    Usage::

        builder = KnowledgeBaseBuilder(config)
        stats = builder.build()
    """

    def __init__(self, config: dict, region_scoped: bool = False) -> None:
        """Build the extractor set from ``config``.

        Args:
            config: Full application config. ``extraction.chokepoints`` decides
                which regions are extracted — narrow that dict to scope a run.
            region_scoped: ``True`` when the caller has narrowed
                ``extraction.chokepoints`` to a single region. This changes two
                behaviours so a partial run cannot damage what it did not
                fetch — see :meth:`_extract_all_regions` and :meth:`build`.
        """
        self.config = config
        self.region_scoped = region_scoped
        self.rag_config = config.get("rag", {}) or {}

        self.extractors: dict[str, BaseExtractor] = {}
        enabled = config.get("extraction", {}).get("enabled_extractors", [])
        for name in enabled:
            cls = _EXTRACTOR_CLASSES.get(name)
            if cls is None:
                continue
            try:
                self.extractors[name] = cls(config)
                logger.info("Initialized extractor: %s", name)
            except Exception as exc:
                logger.error("Failed to initialize extractor %s: %s", name, exc)

    def _get_chromadb_collection(self):
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        client = chromadb.PersistentClient(path="data/knowledge_base/.chromadb")
        collection_name = self.rag_config.get("collections", {}).get(
            "live_context", "live_extracted_context"
        )
        return client.get_or_create_collection(
            name=collection_name,
            embedding_function=DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

    def _extract_all_regions(self) -> list[dict]:
        all_documents: list[dict] = []
        regions = list(self.config.get("extraction", {}).get("chokepoints", {}).keys())

        for name, extractor in self.extractors.items():
            for region in regions:
                try:
                    docs = extractor.extract_historical(region)
                    all_documents.extend(docs)
                    logger.info("  %s/%s: %d documents", name, region, len(docs))
                except Exception as exc:
                    logger.error("  %s/%s failed: %s", name, region, exc)

            # These methods extract curated historical cases with the region
            # hardcoded (Hormuz / Red Sea / Suez), ignoring the region loop
            # above. On a region-scoped run they would spend rate-limited API
            # calls re-fetching documents from other regions, so skip them.
            if self.region_scoped:
                continue

            for method_name in ("extract_specific_events", "extract_specific_scenarios", "extract_historical_events"):
                method = getattr(extractor, method_name, None)
                if method is None:
                    continue
                try:
                    docs = method()
                    all_documents.extend(docs)
                    logger.info("  %s/%s: %d documents", name, method_name, len(docs))
                except Exception as exc:
                    logger.error("  %s/%s failed: %s", name, method_name, exc)

        return all_documents

    @staticmethod
    def _deduplicate(documents: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for doc in documents:
            if doc["id"] not in seen:
                seen.add(doc["id"])
                unique.append(doc)
        return unique

    @staticmethod
    def _upsert_to_chromadb(collection, documents: list[dict], batch_size: int = 50) -> int:
        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                collection.upsert(
                    ids=[d["id"] for d in batch],
                    documents=[d["text"] for d in batch],
                    metadatas=[d["metadata"] for d in batch],
                )
                total += len(batch)
            except Exception as exc:
                logger.error("ChromaDB upsert failed for batch starting at %d: %s", i, exc)
        return total

    def _write_backup(self, path: Path, documents: list[dict]) -> int:
        """Persist the JSON backup, preserving regions this run did not fetch.

        A full run rewrites the file: it fetched every configured region, so
        its output is authoritative. A **region-scoped run merges** instead —
        overwriting would silently delete every other region's documents, and
        the dashboard's news feed reads this file directly
        (``core.get_news``), so a Panama-only run would blank the Hormuz
        headlines. Existing entries are replaced by id where they overlap.

        Args:
            path: Destination JSON file.
            documents: This run's deduplicated documents.

        Returns:
            Number of documents in the file after writing.
        """
        merged = documents
        if self.region_scoped and path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                # Never let an unreadable backup cost us this run's documents.
                logger.warning(
                    "Could not read existing backup %s (%s) — writing this "
                    "run's documents only.", path, exc,
                )
                existing = []
            fresh_ids = {doc["id"] for doc in documents}
            kept = [doc for doc in existing if doc.get("id") not in fresh_ids]
            merged = kept + documents
            logger.info(
                "Merging region-scoped run into backup: %d existing kept + "
                "%d from this run = %d total.", len(kept), len(documents), len(merged),
            )

        path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
        logger.info("Saved backup to %s (%d documents)", path, len(merged))
        return len(merged)

    def build(self) -> dict:
        """Run the full knowledge base population pipeline.

        Returns:
            Stats dict with extraction and storage metrics.
        """
        stats: dict = {
            "extractors_run": list(self.extractors.keys()),
            "documents_extracted": 0,
            "documents_deduplicated": 0,
            "documents_stored": 0,
            "errors": [],
        }

        all_documents = self._extract_all_regions()
        stats["documents_extracted"] = len(all_documents)

        unique_documents = self._deduplicate(all_documents)
        stats["documents_deduplicated"] = len(unique_documents)

        backup_path = Path("data/knowledge_base/live_extracted_backup.json")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        written = self._write_backup(backup_path, unique_documents)
        stats["documents_in_backup"] = written

        try:
            collection = self._get_chromadb_collection()
            stats["documents_stored"] = self._upsert_to_chromadb(collection, unique_documents)
        except Exception as exc:
            stats["errors"].append(f"ChromaDB storage failed: {exc}")
            logger.error(stats["errors"][-1])

        logger.info("Knowledge base build complete: %s", json.dumps(stats))
        return stats
