"""The data-gap notes in the Chapter 3 case metrics come from the manifest, not a constant."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_spec = importlib.util.spec_from_file_location("case_metrics_under_test", _ROOT / "scripts" / "case_metrics.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


def _manifest(tmp_path: Path, region: str, coverage: dict | None) -> None:
    body = {"region": region} if coverage is None else {"region": region, "coverage": coverage}
    (tmp_path / f"{region}.manifest.json").write_text(json.dumps(body), encoding="utf-8")


def test_the_acled_end_is_the_last_day_the_manifest_says_has_conflict_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cm, "LIVE_DIR", tmp_path)
    _manifest(tmp_path, "hormuz", {"geopolitical": ["2019-01-01", "2025-11-24"], "shipping": ["2019-01-01", "2026-10-25"]})
    assert cm.acled_end("hormuz") == "2025-11-24"


def test_the_acled_end_says_unknown_when_the_manifest_has_no_coverage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cm, "LIVE_DIR", tmp_path)
    _manifest(tmp_path, "hormuz", None)
    assert cm.acled_end("hormuz") == "unknown"
    assert cm.acled_end("panama") == "unknown"                 # no manifest at all


def test_the_acled_end_is_not_the_old_hardcoded_day() -> None:
    assert not hasattr(cm, "ACLED_END")


def test_the_gap_note_names_the_manifest_end_day(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cm, "LIVE_DIR", tmp_path)
    _manifest(tmp_path, "hormuz", {"geopolitical": ["2019-01-01", "2025-11-24"]})
    risk = pd.DataFrame({"timestamp": pd.to_datetime(["2026-04-15"]), "shipping": [0.5],
                         "geopolitical": [float("nan")], "news_sentiment": [0.4]})
    notes = cm.gap_notes("hormuz", ["geopolitical"], risk)
    assert notes == ["geopolitical: ACLED conflict data end 2025-11-24"]


def test_no_gap_note_when_no_agent_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cm, "LIVE_DIR", tmp_path)
    _manifest(tmp_path, "hormuz", {"geopolitical": ["2019-01-01", "2025-11-24"]})
    risk = pd.DataFrame({"timestamp": pd.to_datetime(["2024-04-15"]), "news_sentiment": [0.4]})
    assert cm.gap_notes("hormuz", [], risk) == []


def test_a_missing_news_series_is_still_noted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cm, "LIVE_DIR", tmp_path)
    risk = pd.DataFrame({"timestamp": pd.to_datetime(["2024-04-15"]), "shipping": [0.4]})
    assert any("news_sentiment" in n for n in cm.gap_notes("hormuz", [], risk))
