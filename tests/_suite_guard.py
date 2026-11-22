"""Helpers for the suite-wide guard on real optimizer outputs (tests/conftest.py).

Kept out of conftest.py so tests can import them without re-executing it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Real optimizer outputs no test may create, change or delete.
REAL_OPTIMIZER_OUTPUTS: tuple[Path, ...] = (
    ROOT / "data" / "processed" / "optimization_results.json",
    ROOT / "config" / "optimized_weights.yaml",
)


def fingerprint(path: Path) -> tuple[str, int] | None:
    """(sha256, mtime in ns) of the file, or None if it does not exist.

    The mtime catches a rewrite with identical bytes: a deterministic 5-trial
    smoke run reproduces the same JSON, and a hash alone would miss it.
    """
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def describe_change(before: tuple[str, int] | None, after: tuple[str, int] | None) -> str:
    """How a guarded file changed, in words, or '' if it did not."""
    if before == after:
        return ""
    if before is None:
        return "created"
    if after is None:
        return "deleted"
    if before[0] != after[0]:
        return "content changed"
    return "rewritten with identical bytes (mtime changed)"
