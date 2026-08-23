"""Populate the knowledge base with GDACS + USGS hazard events.

Runs :class:`~src.extractors.disaster_combined_extractor.DisasterCombinedExtractor`
over every configured chokepoint and upserts the result into the live ChromaDB
collection.

Separate from ``scripts/populate_knowledge_base.py`` because that script runs
*every* enabled extractor and spends rate-limited quota on NewsAPI, SerpAPI,
FRED and ACLED. The two disaster sources need no credentials and are the only
ones worth re-running on their own.

Storage goes through :class:`~src.extractors.knowledge_base_builder.KnowledgeBaseBuilder`
rather than talking to ChromaDB directly, so the collection name, batch size,
deduplication and backup-merge behaviour stay defined in exactly one place.
The run is treated as region-scoped: it fetches two sources out of six, so it
merges into the JSON backup instead of overwriting the other extractors' work.

Usage::

    python scripts/run_disaster_extraction_2018_2026.py
    python scripts/run_disaster_extraction_2018_2026.py --dry-run
    python scripts/run_disaster_extraction_2018_2026.py --region malacca
    python scripts/run_disaster_extraction_2018_2026.py --start-year 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config_manager import load_base_config  # noqa: E402
from src.extractors.disaster_combined_extractor import (  # noqa: E402
    DisasterCombinedExtractor,
)
from src.extractors.knowledge_base_builder import KnowledgeBaseBuilder  # noqa: E402

DEFAULT_START_YEAR = 2018
DEFAULT_END_YEAR = 2026


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--region", action="append", dest="regions", metavar="KEY",
        help="Limit to one chokepoint (repeatable). Default: all configured.",
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and report counts without writing to ChromaDB or the backup.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def _resolve_regions(config: dict, requested: list[str] | None) -> list[str]:
    """Validate ``requested`` against the configured chokepoints.

    An unknown key is an error rather than an empty result: a typo would
    otherwise look like a region with no hazards.
    """
    configured = list(config.get("extraction", {}).get("chokepoints", {}))
    if not requested:
        return configured

    unknown = [r for r in requested if r not in configured]
    if unknown:
        raise SystemExit(
            f"Unknown region(s): {', '.join(unknown)}. "
            f"Configured: {', '.join(configured)}"
        )
    return requested


def _summarize(documents: list[dict]) -> str:
    """One-line source/severity breakdown for a region's documents."""
    sources = Counter(
        str((d.get("metadata") or {}).get("source_api", "?")) for d in documents
    )
    severities = Counter(
        str((d.get("metadata") or {}).get("severity", "?")) for d in documents
    )
    parts = [f"{name}={count}" for name, count in sorted(sources.items())]
    parts += [f"{name}={severities[name]}" for name in ("high", "medium", "low")
              if severities.get(name)]
    return " ".join(parts) or "no documents"


def _year_coverage(documents: list[dict]) -> str:
    """Which years actually produced documents — a gap is worth seeing."""
    years = sorted(
        {str((d.get("metadata") or {}).get("event_date", ""))[:4] for d in documents}
        - {""}
    )
    return ", ".join(years) if years else "none"


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_base_config()
    regions = _resolve_regions(config, args.regions)
    extractor = DisasterCombinedExtractor(config)

    print(f"GDACS + USGS extraction | {args.start_year}-{args.end_year} | "
          f"cap {DisasterCombinedExtractor.DOCS_PER_MONTH} docs/region/month")
    print(f"Regions: {', '.join(regions)}")
    if args.dry_run:
        print("DRY RUN — nothing will be written.")
    print()

    all_documents: list[dict] = []
    per_region: dict[str, int] = {}
    started = time.monotonic()

    for region in regions:
        print(f"  {region} ... ", end="", flush=True)
        region_start = time.monotonic()
        documents = extractor.extract_historical(
            region, start_year=args.start_year, end_year=args.end_year
        )
        elapsed = time.monotonic() - region_start
        per_region[region] = len(documents)
        all_documents.extend(documents)
        print(f"{len(documents):>5} docs in {elapsed:>5.0f}s  [{_summarize(documents)}]")
        print(f"          years: {_year_coverage(documents)}")

    total_elapsed = time.monotonic() - started
    print(f"\nExtracted {len(all_documents)} documents in {total_elapsed:.0f}s")

    if not all_documents:
        print("Nothing extracted — leaving the knowledge base untouched.")
        return 1

    if args.dry_run:
        for region, count in per_region.items():
            print(f"  {region}: {count}")
        return 0

    # region_scoped=True: this run fetched two of six sources, so the JSON
    # backup must be merged rather than overwritten.
    builder = KnowledgeBaseBuilder(
        {**config, "extraction": {**config.get("extraction", {}),
                                  "enabled_extractors": []}},
        region_scoped=True,
    )
    unique = builder._deduplicate(all_documents)
    print(f"Deduplicated: {len(unique)} unique documents")

    backup_path = _PROJECT_ROOT / "data" / "knowledge_base" / "live_extracted_backup.json"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    in_backup = builder._write_backup(backup_path, unique)
    print(f"Backup: {in_backup} documents in {backup_path.name}")

    collection = builder._get_chromadb_collection()
    before = collection.count()
    stored = builder._upsert_to_chromadb(collection, unique)
    after = collection.count()

    print(f"\nChromaDB '{collection.name}': {before} -> {after} (+{after - before})")
    print(f"Upserted {stored} documents ({len(unique) - stored} failed)")
    for region, count in per_region.items():
        print(f"  {region}: {count}")

    return 0 if stored else 1


if __name__ == "__main__":
    sys.exit(run())
