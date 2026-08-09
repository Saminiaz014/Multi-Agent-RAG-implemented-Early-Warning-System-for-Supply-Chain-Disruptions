#!/usr/bin/env python
"""Export the live ChromaDB collection to the portable backup JSON.

Schema: [{"id": "...", "text": "...", "metadata": {...}}, ...] — matches
what ``ContextRetriever.seed_live_collection_from_backup`` expects when
seeding a fresh deploy's empty ``live_extracted_context`` collection.
"""

import json
import logging
from pathlib import Path

import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_kb_backup() -> int:
    """Export the live ChromaDB store to the portable JSON backup."""
    try:
        client = chromadb.PersistentClient(path="data/knowledge_base/.chromadb")
        collection = client.get_collection("live_extracted_context")
        logger.info("Connected to live collection: %d docs", collection.count())

        all_docs = collection.get(include=["documents", "metadatas"])
        backup_docs = [
            {"id": doc_id, "text": doc_text, "metadata": doc_meta}
            for doc_id, doc_text, doc_meta in zip(
                all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
            )
        ]

        backup_path = Path("data/knowledge_base/live_extracted_backup.json")
        backup_path.write_text(
            json.dumps(backup_docs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info("Exported %d docs to %s", len(backup_docs), backup_path)
        return len(backup_docs)
    except Exception:
        logger.error("Export failed", exc_info=True)
        return 0


if __name__ == "__main__":
    count = export_kb_backup()
    raise SystemExit(0 if count > 0 else 1)
