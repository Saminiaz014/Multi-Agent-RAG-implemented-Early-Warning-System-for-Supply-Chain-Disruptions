"""docs/ is ignored, except the tracked thesis brief, which must stay visible to git."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True)


def _ignored(path: str) -> bool:
    # --no-index: judge the rules alone (tracked files are otherwise never reported).
    return _git("check-ignore", "--no-index", "-q", path).returncode == 0


@pytest.fixture(scope="module", autouse=True)
def _in_a_git_checkout() -> None:
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not a git checkout")


def test_the_thesis_brief_is_tracked_and_not_matched_by_the_ignore_rules() -> None:
    assert _git("ls-files", "--error-unmatch", "docs/THESIS_BRIEF.md").returncode == 0
    assert not _ignored("docs/THESIS_BRIEF.md")


def test_everything_else_under_docs_stays_ignored() -> None:
    assert _ignored("docs/other.md") and _ignored("docs/sub/deeper/x.md")
    assert _ignored("docs/eval01-archived/CROSS_REGION_RESULTS.md")
