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


# ------------------------------------------- the metrics file is committed, with its hash
import hashlib  # noqa: E402

_capture_spec = importlib.util.spec_from_file_location(
    "capture_under_test", _ROOT / "scripts" / "capture_demo_screenshots.py")
capture = importlib.util.module_from_spec(_capture_spec)
_capture_spec.loader.exec_module(capture)


def test_sha256_file_is_the_hex_digest_of_the_bytes(tmp_path) -> None:
    f = tmp_path / "x.json"
    f.write_bytes(b'{"a": 1}\n')
    assert cm.sha256_file(f) == hashlib.sha256(b'{"a": 1}\n').hexdigest()


def test_provenance_hashes_the_live_files_the_dashboard_reads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cm, "LIVE_DIR", tmp_path)
    (tmp_path / "hormuz.csv").write_bytes(b"timestamp\n2026-10-25\n")
    (tmp_path / "hormuz_risk.csv").write_bytes(b"timestamp,risk_score\n2026-10-25,0.5\n")
    _manifest(tmp_path, "hormuz", {"geopolitical": ["2019-01-01", "2025-11-24"]})
    prov = cm.provenance(["hormuz", "panama"])
    assert set(prov["hormuz"]) == {"frame", "risk", "manifest"}
    assert prov["hormuz"]["frame"] == hashlib.sha256(b"timestamp\n2026-10-25\n").hexdigest()
    assert prov["panama"] == {"frame": None, "risk": None, "manifest": None}     # absent, said so


def test_write_metrics_writes_identical_bytes_everywhere_and_returns_the_hash(tmp_path) -> None:
    a, b = tmp_path / "a" / "m.json", tmp_path / "b" / "m.json"
    digest = cm.write_metrics({"cases": [1, 2], "x": 0.5}, (a, b))
    assert a.read_bytes() == b.read_bytes()
    assert b"\r\n" not in a.read_bytes()                       # LF on Windows too: the hash is portable
    assert digest == hashlib.sha256(a.read_bytes()).hexdigest()
    assert json.loads(a.read_text(encoding="utf-8")) == {"cases": [1, 2], "x": 0.5}


def test_the_committed_metrics_file_has_the_four_cases_and_the_input_hashes() -> None:
    committed = json.loads(cm.COMMITTED.read_text(encoding="utf-8"))
    assert [c["case"] for c in committed["cases"]] == [1, 2, 3, 4]
    assert set(committed["provenance"]) == {"hormuz", "bab_el_mandeb", "panama", "malacca"}
    assert all(len(v["frame"]) == 64 for v in committed["provenance"].values())


def test_the_committed_metrics_file_is_not_git_ignored() -> None:
    import subprocess
    ignored = subprocess.run(["git", "check-ignore", "-q", "eval/case_metrics.json"], cwd=_ROOT)
    assert ignored.returncode != 0


def test_the_screenshot_manifest_records_the_hash_of_the_committed_file() -> None:
    manifest = _ROOT / "thesis" / "figures" / "ch3_demo" / "manifest.json"
    if not manifest.exists():
        import pytest
        pytest.skip("thesis/ is git-ignored and not present")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["case_metrics_sha256"]
    assert recorded == cm.sha256_file(cm.COMMITTED)


def test_the_capture_script_hashes_the_metrics_file_it_read(tmp_path) -> None:
    f = tmp_path / "case_metrics.json"
    f.write_bytes(b"{}\n")
    assert capture.metrics_sha256(f) == hashlib.sha256(b"{}\n").hexdigest()


def test_stale_captures_are_the_ones_taken_against_other_numbers() -> None:
    manifest = {"case_metrics_sha256": "bbb", "captures": [
        {"file": "a.png", "case_metrics_sha256": "bbb"},
        {"file": "b.png", "case_metrics_sha256": "aaa"},
        {"file": "c.png"}]}
    assert capture.stale_captures(manifest, "bbb") == ["b.png", "c.png"]
    assert capture.stale_captures({"captures": []}, "bbb") == []


def test_every_current_capture_was_taken_against_the_committed_numbers() -> None:
    manifest = _ROOT / "thesis" / "figures" / "ch3_demo" / "manifest.json"
    if not manifest.exists():
        import pytest
        pytest.skip("thesis/ is git-ignored and not present")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert capture.stale_captures(data, cm.sha256_file(cm.COMMITTED)) == []
