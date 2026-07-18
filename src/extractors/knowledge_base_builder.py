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
from src.extractors.fred_extractor import FREDExtractor
from src.extractors.newsapi_extractor import NewsAPIExtractor
from src.extractors.reliefweb_extractor import ReliefWebExtractor
from src.extractors.serpapi_extractor import SerpAPIExtractor

logger = logging.getLogger(__name__)

_EXTRACTOR_CLASSES: dict[str, type] = {
    "newsapi": NewsAPIExtractor,
    "serpapi": SerpAPIExtractor,    # historical news for RAG backfill (NewsAPI is current-only)
    "ambee": AmbeeExtractor,        # primary natural_disaster source
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

    def __init__(self, config: dict) -> None:
        self.config = config
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
    def _merge_with_existing_backup(backup_path: Path, new_documents: list[dict]) -> list[dict]:
        """Merge this run's documents over the existing snapshot by id.

        A subset run (e.g. the daily Airflow ``--extractors fred,ambee``
        pull) must not clobber documents only a full extraction produces —
        the snapshot doubles as the committed seed for fresh deploys and the
        dashboard news feed. This run's documents win on id collision; an
        unreadable existing snapshot degrades to this run's documents only.
        """
        by_id: dict[str, dict] = {}
        try:
            if backup_path.exists():
                existing = json.loads(backup_path.read_text(encoding="utf-8"))
                by_id = {d["id"]: d for d in existing if isinstance(d, dict) and "id" in d}
        except Exception as exc:
            logger.warning(
                "Existing backup unreadable (%s) — writing this run's documents only", exc
            )
        for doc in new_documents:
            by_id[doc["id"]] = doc
        return list(by_id.values())

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
        merged_documents = self._merge_with_existing_backup(backup_path, unique_documents)
        backup_path.write_text(
            json.dumps(merged_documents, indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "Saved backup to %s (%d docs total: %d from this run)",
            backup_path,
            len(merged_documents),
            len(unique_documents),
        )

        try:
            collection = self._get_chromadb_collection()
            stats["documents_stored"] = self._upsert_to_chromadb(collection, unique_documents)
        except Exception as exc:
            stats["errors"].append(f"ChromaDB storage failed: {exc}")
            logger.error(stats["errors"][-1])

        logger.info("Knowledge base build complete: %s", json.dumps(stats))
        return stats
