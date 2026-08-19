"""Entry point script to populate the RAG knowledge base from live APIs.

Run from project root::

    python scripts/populate_knowledge_base.py                      # every chokepoint in settings.yaml
    python scripts/populate_knowledge_base.py --region panama      # one region only
    python scripts/populate_knowledge_base.py --extractors reliefweb,fred

``--region`` loads the region's *merged* config
(:func:`src.core.config_manager.load_config_for_region`) and narrows the run to
that region's chokepoint. This is the only way to reach a chokepoint that
settings.yaml does not define on its own — Panama's ``extraction.chokepoints``
entry exists only after the Phase 11.4 projection.

A region-scoped run merges into the JSON backup rather than overwriting it, and
skips the extractors' curated historical-case methods (whose regions are
hardcoded), so it neither deletes nor re-fetches other regions' documents.

Note that the extractors are not equally region-driven. ACLED and NewsAPI
derive their queries from config and work for any registered region; SerpAPI,
FRED and Ambee key off hand-authored tables of historical cases, series periods
and monitoring points that currently cover only the four original chokepoints,
and will extract nothing for a region absent from them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_manager import load_config_for_region  # noqa: E402
from src.core.regions import list_regions  # noqa: E402
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
    parser.add_argument(
        "--region", type=str, default=None, choices=list_regions(),
        help="Populate one region only, using its merged config. Required to "
             "reach a chokepoint settings.yaml does not define (e.g. panama).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    if args.region:
        config = load_config_for_region(args.region, base_config_path=str(config_path))
        chokepoints = config.get("extraction", {}).get("chokepoints", {}) or {}
        # bab_el_mandeb maps onto settings.yaml's existing "red_sea" entry, so
        # the chokepoint key is not always the region key.
        key = config.get("extraction", {}).get("chokepoint_key") or args.region
        if key not in chokepoints:
            logger.error(
                "Region '%s' resolved to chokepoint '%s', which is not in the "
                "merged config's extraction.chokepoints (%s).",
                args.region, key, sorted(chokepoints),
            )
            return 1
        config["extraction"]["chokepoints"] = {key: chokepoints[key]}
        logger.info(
            "Region-scoped run: %s -> chokepoint '%s', countries=%s",
            args.region, key, chokepoints[key].get("countries", []),
        )
    else:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if args.extractors:
        config.setdefault("extraction", {})["enabled_extractors"] = args.extractors.split(",")
        logger.info("Running extractors: %s", args.extractors)

    builder = KnowledgeBaseBuilder(config, region_scoped=bool(args.region))
    stats = builder.build()

    print("\n" + "=" * 50)
    print("KNOWLEDGE BASE POPULATION COMPLETE")
    print("=" * 50)
    if args.region:
        print(f"  Region:             {args.region}")
    print(f"  Extractors run:     {', '.join(stats['extractors_run'])}")
    print(f"  Documents found:    {stats['documents_extracted']}")
    print(f"  After dedup:        {stats['documents_deduplicated']}")
    print(f"  Stored in ChromaDB: {stats['documents_stored']}")
    print(f"  Backup file total:  {stats.get('documents_in_backup', '?')}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for err in stats["errors"]:
            print(f"    - {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
