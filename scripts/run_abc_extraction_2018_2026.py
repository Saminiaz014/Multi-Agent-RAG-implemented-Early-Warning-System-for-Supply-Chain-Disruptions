"""Populate the knowledge base with ACLED (geopolitical) + FRED (market) data.

Sibling of ``run_disaster_extraction_2018_2026.py``, same shape: fetch per
region, report per-source counts and year coverage, then upsert through
:class:`~src.extractors.knowledge_base_builder.KnowledgeBaseBuilder` so the
collection name, batching, deduplication and backup-merge stay defined once.

Vessel history is deliberately not sourced here. The AIS services available to
this project (``aisstream``, and the commercial vessel APIs surveyed) report
live positions with no historical query, so a "2018-2026 backfill" against one
would stamp a single present-day snapshot with historical dates. Shipping
evidence therefore comes from the live pipeline
(``src/ingestion/shipping_connector.py``), not from the RAG backfill.

Usage::

    python scripts/run_abc_extraction_2018_2026.py --dry-run
    python scripts/run_abc_extraction_2018_2026.py
    python scripts/run_abc_extraction_2018_2026.py --extractor fred
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
from src.extractors.acled_extractor import ACLEDExtractor  # noqa: E402
from src.extractors.fred_extractor import FREDExtractor  # noqa: E402
from src.extractors.knowledge_base_builder import KnowledgeBaseBuilder  # noqa: E402

DEFAULT_START_YEAR = 2018
DEFAULT_END_YEAR = 2026

_EXTRACTORS = {"acled": ACLEDExtractor, "fred": FREDExtractor}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--extractor", action="append", dest="extractors", choices=sorted(_EXTRACTORS),
        help="Limit to one source (repeatable). Default: both.",
    )
    parser.add_argument("--region", action="append", dest="regions", metavar="KEY")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def _resolve_regions(config: dict, requested: list[str] | None) -> list[str]:
    configured = list(config.get("extraction", {}).get("chokepoints", {}))
    if not requested:
        return configured
    unknown = [r for r in requested if r not in configured]
    if unknown:
        raise SystemExit(
            f"Unknown region(s): {', '.join(unknown)}. Configured: {', '.join(configured)}"
        )
    return requested


def _year_coverage(documents: list[dict]) -> str:
    years = sorted(
        {str((d.get("metadata") or {}).get("event_date", ""))[:4] for d in documents} - {""}
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
    names = args.extractors or sorted(_EXTRACTORS)
    extractors = {name: _EXTRACTORS[name](config) for name in names}

    print(f"ACLED + FRED extraction | {args.start_year}-{args.end_year}")
    print(f"Sources: {', '.join(names)}")
    print(f"Regions: {', '.join(regions)}")
    if args.dry_run:
        print("DRY RUN — nothing will be written.")
    print()

    all_documents: list[dict] = []
    started = time.monotonic()

    for name, extractor in extractors.items():
        for region in regions:
            print(f"  {name:>6} / {region:<14} ... ", end="", flush=True)
            region_start = time.monotonic()
            try:
                documents = extractor.extract_historical(
                    region, start_year=args.start_year, end_year=args.end_year
                )
            except Exception as exc:
                # One source failing must not cost the other its results.
                print(f"FAILED: {exc}")
                logging.exception("%s/%s failed", name, region)
                continue
            elapsed = time.monotonic() - region_start
            all_documents.extend(documents)
            sev = Counter(
                str((d.get("metadata") or {}).get("severity", "?")) for d in documents
            )
            print(
                f"{len(documents):>4} docs in {elapsed:>5.0f}s  "
                f"[{' '.join(f'{k}={v}' for k, v in sorted(sev.items()))}]"
            )
            if documents:
                print(f"{'':>26}years: {_year_coverage(documents)}")

    print(f"\nExtracted {len(all_documents)} documents in {time.monotonic() - started:.0f}s")
    if not all_documents:
        print("Nothing extracted — leaving the knowledge base untouched.")
        return 1

    if args.dry_run:
        return 0

    builder = KnowledgeBaseBuilder(
        {**config, "extraction": {**config.get("extraction", {}),
                                  "enabled_extractors": []}},
        region_scoped=True,   # partial run: merge the backup, never overwrite it
    )
    unique = builder._deduplicate(all_documents)
    dropped = len(all_documents) - len(unique)
    print(f"Deduplicated: {len(unique)} unique ({dropped} duplicate ids)")
    if dropped:
        print("  NOTE: duplicate ids across regions usually mean an id is not "
              "region-qualified — check before accepting the loss.")

    backup_path = _PROJECT_ROOT / "data" / "knowledge_base" / "live_extracted_backup.json"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Backup: {builder._write_backup(backup_path, unique)} documents")

    collection = builder._get_chromadb_collection()
    before = collection.count()
    stored = builder._upsert_to_chromadb(collection, unique)
    after = collection.count()

    print(f"\nChromaDB '{collection.name}': {before} -> {after} (+{after - before})")
    print(f"Upserted {stored} documents ({len(unique) - stored} failed)")
    return 0 if stored else 1


if __name__ == "__main__":
    sys.exit(run())
