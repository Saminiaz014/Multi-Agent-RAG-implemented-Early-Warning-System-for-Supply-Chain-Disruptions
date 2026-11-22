"""The suite-wide guard in conftest.py: what counts as touching a real file."""

from __future__ import annotations

import os

from tests._suite_guard import (REAL_OPTIMIZER_OUTPUTS, ROOT, describe_change,
                                fingerprint)


def test_guard_covers_both_real_optimizer_outputs() -> None:
    names = {p.relative_to(ROOT).as_posix() for p in REAL_OPTIMIZER_OUTPUTS}
    assert names == {"data/processed/optimization_results.json",
                     "config/optimized_weights.yaml"}


def test_rewrite_with_identical_bytes_is_detected(tmp_path) -> None:
    path = tmp_path / "optimization_results.json"
    path.write_text('{"best_trial": 63}', encoding="utf-8")
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))     # an old mtime
    before = fingerprint(path)
    path.write_text('{"best_trial": 63}', encoding="utf-8")  # same bytes, new write
    after = fingerprint(path)
    assert after != before
    assert describe_change(before, after) == "rewritten with identical bytes (mtime changed)"


def test_content_change_creation_and_deletion_are_detected(tmp_path) -> None:
    path = tmp_path / "optimized_weights.yaml"
    absent = fingerprint(path)
    assert absent is None
    path.write_text("x: 1\n", encoding="utf-8")
    created = fingerprint(path)
    assert describe_change(absent, created) == "created"
    path.write_text("x: 2\n", encoding="utf-8")
    assert describe_change(created, fingerprint(path)) == "content changed"
    edited = fingerprint(path)
    path.unlink()
    assert describe_change(edited, fingerprint(path)) == "deleted"


def test_unchanged_file_reports_no_change(tmp_path) -> None:
    path = tmp_path / "optimization_results.json"
    path.write_text("{}", encoding="utf-8")
    assert describe_change(fingerprint(path), fingerprint(path)) == ""
