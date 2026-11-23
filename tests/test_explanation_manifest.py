"""Issue 13: traceable explanation outputs — summary of alerts, manifest of artefacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "explanation_manifest", _ROOT / "scripts" / "explanation_manifest.py")
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)


def _alert(path: Path, region: str, date: str, shap: dict[str, float]) -> None:
    features = [{"feature": f, "shap_value": v} for f, v in shap.items()]
    path.joinpath(f"{region}_{date}.json").write_text(
        json.dumps({"region": region, "date": date, "complete": True, "features": features}),
        encoding="utf-8")


def test_top_features_from_alert_files_rank_by_mean_abs_shap(tmp_path) -> None:
    _alert(tmp_path, "x", "2024-01-01", {"a": 0.4, "b": -0.3, "c": 0.1})
    _alert(tmp_path, "x", "2024-01-02", {"b": -0.5, "c": 0.2})
    _alert(tmp_path, "y", "2024-01-01", {"z": 9.0})                # another region
    top = em.top_features_from_alerts(tmp_path, "x", k=2)
    # Mean over the region's 2 alerts, absent = 0: b 0.4, a 0.2, c 0.15.
    assert [t["feature"] for t in top] == ["b", "a"]
    assert top[0]["mean_abs_shap"] == pytest.approx(0.4)
    assert top[1]["mean_abs_shap"] == pytest.approx(0.2)
    assert top[0]["top1_share"] == pytest.approx(0.5)               # leads one alert of two


def test_tree_digest_depends_on_content_not_order(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for d, order in ((a, ("1", "2")), (b, ("2", "1"))):
        d.mkdir()
        for name in order:
            (d / f"{name}.json").write_text(name, encoding="utf-8")
    assert em.tree_digest(a)["sha256"] == em.tree_digest(b)["sha256"]
    assert em.tree_digest(a)["files"] == 2
    (b / "2.json").write_text("changed", encoding="utf-8")
    assert em.tree_digest(a)["sha256"] != em.tree_digest(b)["sha256"]
    assert em.tree_digest(a, exclude={"2.json"})["files"] == 1


def test_text_is_hashed_with_lf_line_endings(tmp_path) -> None:
    crlf, lf = tmp_path / "crlf.md", tmp_path / "lf.md"
    crlf.write_bytes(b"a\r\nb\r\n")
    lf.write_bytes(b"a\nb\n")
    e1, e2 = em.file_entry(crlf, root=tmp_path), em.file_entry(lf, root=tmp_path)
    assert e1["sha256"] == e2["sha256"] == hashlib.sha256(b"a\nb\n").hexdigest()
    assert e1["git_blob"] == e2["git_blob"]
    binary = tmp_path / "m.joblib"
    binary.write_bytes(b"x\r\ny")
    assert em.file_entry(binary, root=tmp_path)["sha256"] == hashlib.sha256(b"x\r\ny").hexdigest()


def test_blob_ids_match_git_for_a_committed_file() -> None:
    rel = "config/regions/hormuz.yaml"
    diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=_ROOT)
    if diff.returncode != 0:
        pytest.skip(f"{rel} differs from HEAD in this working tree")
    at_head = subprocess.run(["git", "ls-tree", "HEAD", "--", rel], cwd=_ROOT,
                             capture_output=True, text=True).stdout.split()[2]
    assert em.file_entry(_ROOT / rel)["git_blob"] == at_head


def test_outputs_gitignore_ignores_artefacts_but_not_coverage() -> None:
    rules = (_ROOT / "outputs" / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "*.joblib" in rules
    assert "explanations/*" in rules and "!explanations/coverage.json" in rules
