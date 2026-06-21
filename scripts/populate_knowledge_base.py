"""Entry point script to populate the RAG knowledge base from live APIs.

Run from project root::

    python scripts/populate_knowledge_base.py
    python scripts/populate_knowledge_base.py --extractors reliefweb,fred
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extractors.knowledge_base_builder import KnowledgeBaseBuilder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate RAG knowledge base from live APIs")
    parser.add_argument(
        "--extractors", type=str, default="",
        help="Comma-separated extractor names to run (default: all enabled in config)",
    )
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to settings.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if args.extractors:
        config.setdefault("extraction", {})["enabled_extractors"] = args.extractors.split(",")
        logger.info("Running extractors: %s", args.extractors)

    builder = KnowledgeBaseBuilder(config)
    stats = builder.build()

    print("\n" + "=" * 50)
    print("KNOWLEDGE BASE POPULATION COMPLETE")
    print("=" * 50)
    print(f"  Extractors run:     {', '.join(stats['extractors_run'])}")
    print(f"  Documents found:    {stats['documents_extracted']}")
    print(f"  After dedup:        {stats['documents_deduplicated']}")
    print(f"  Stored in ChromaDB: {stats['documents_stored']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for err in stats["errors"]:
            print(f"    - {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
