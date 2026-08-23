"""Retag knowledge-base documents that carry retired region keys.

When the extraction vocabulary was realigned to :mod:`src.core.regions`,
``red_sea`` and ``suez`` stopped being regions. Documents already written with
those keys kept them, and region-filtered retrieval cannot reach any of them —
they are present in the collection but invisible to every query that scopes by
region.

This retags them in place using :data:`src.core.regions.RETIRED_REGION_ALIASES`,
touching both stores that hold documents:

* the ChromaDB ``live_extracted_context`` collection, and
* ``data/knowledge_base/live_extracted_backup.json``, which the dashboard's
  news feed reads directly.

Only the ``region`` metadata field changes. Document text, ids, embeddings and
every other metadata field are left alone — the text of a Suez document still
names Suez, which is what keeps the remapping honest and legible to a reader.

Defaults to a dry run. Pass ``--apply`` to write.

Usage::

    python scripts/migrate_retired_region_keys.py
    python scripts/migrate_retired_region_keys.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config_manager import load_base_config  # noqa: E402
from src.core.regions import RETIRED_REGION_ALIASES, list_regions  # noqa: E402

_BACKUP_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "live_extracted_backup.json"
_CHROMA_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / ".chromadb"
_BATCH = 200


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the changes. Without it, report what would change and exit.",
    )
    return parser.parse_args(argv)


def _collection():
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    config = load_base_config()
    name = (config.get("rag", {}).get("collections", {}) or {}).get(
        "live_context", "live_extracted_context"
    )
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    return client.get_or_create_collection(
        name=name,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def _report(counts: Counter, label: str) -> int:
    total = sum(counts.values())
    if not total:
        print(f"  {label}: nothing to migrate")
        return 0
    print(f"  {label}: {total} document(s)")
    for (source, old), n in sorted(counts.items()):
        print(f"      {source:>10} {old:<9} -> {RETIRED_REGION_ALIASES[old]:<14} {n:>5}")
    return total


def migrate_chromadb(apply: bool) -> int:
    """Retag the ChromaDB collection. Returns the number of affected documents."""
    collection = _collection()
    stored = collection.get(include=["metadatas"])

    stale_ids: list[str] = []
    updated: list[dict] = []
    counts: Counter = Counter()

    for doc_id, metadata in zip(stored["ids"], stored["metadatas"]):
        old = metadata.get("region")
        if old not in RETIRED_REGION_ALIASES:
            continue
        counts[(metadata.get("source_api", "?"), old)] += 1
        stale_ids.append(doc_id)
        # Copy: never mutate the dict Chroma handed back.
        updated.append({**metadata, "region": RETIRED_REGION_ALIASES[old]})

    total = _report(counts, f"ChromaDB '{collection.name}'")
    if not total or not apply:
        return total

    # update() rewrites metadata only — documents and embeddings are untouched,
    # so this costs no re-embedding.
    for i in range(0, len(stale_ids), _BATCH):
        collection.update(
            ids=stale_ids[i : i + _BATCH],
            metadatas=updated[i : i + _BATCH],
        )
    print(f"      retagged {len(stale_ids)} document(s)")
    return total


def migrate_backup(apply: bool) -> int:
    """Retag the JSON backup. Returns the number of affected documents."""
    if not _BACKUP_PATH.exists():
        print(f"  {_BACKUP_PATH.name}: not present, skipping")
        return 0

    documents = json.loads(_BACKUP_PATH.read_text(encoding="utf-8"))
    counts: Counter = Counter()
    for doc in documents:
        metadata = doc.get("metadata") or {}
        old = metadata.get("region")
        if old not in RETIRED_REGION_ALIASES:
            continue
        counts[(metadata.get("source_api", "?"), old)] += 1
        metadata["region"] = RETIRED_REGION_ALIASES[old]

    total = _report(counts, _BACKUP_PATH.name)
    if not total or not apply:
        return total

    # Keep the pre-migration file: this is the only copy of the original
    # tagging, and the mapping is a judgement call worth being able to undo.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = _BACKUP_PATH.with_name(f"{_BACKUP_PATH.stem}.pre_region_migration_{stamp}.json")
    shutil.copy2(_BACKUP_PATH, archive)
    _BACKUP_PATH.write_text(json.dumps(documents, indent=2, default=str), encoding="utf-8")
    print(f"      retagged {total} document(s); original saved as {archive.name}")
    return total


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print("Retired region keys:")
    for old, new in sorted(RETIRED_REGION_ALIASES.items()):
        print(f"  {old} -> {new}")
    print(f"Registry regions: {', '.join(list_regions())}")
    print("DRY RUN — pass --apply to write.\n" if not args.apply else "APPLYING\n")

    affected = migrate_chromadb(args.apply) + migrate_backup(args.apply)

    if not affected:
        print("\nNothing to migrate.")
        return 0
    if not args.apply:
        print(f"\n{affected} document(s) would be retagged. Re-run with --apply.")
        return 0

    # Verify rather than assume: a silent partial update is the failure worth
    # catching here.
    remaining = sum(
        1 for m in _collection().get(include=["metadatas"])["metadatas"]
        if m.get("region") in RETIRED_REGION_ALIASES
    )
    print(f"\nChromaDB documents still carrying a retired key: {remaining}")
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
