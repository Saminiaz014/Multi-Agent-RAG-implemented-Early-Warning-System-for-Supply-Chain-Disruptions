"""The suite-wide guard in conftest.py: what counts as touching a real file."""

from __future__ import annotations

import os

from tests._suite_guard import (GUARDED_OUTPUTS, REAL_OPTIMIZER_OUTPUTS, REAL_SHAP_PLOTS,
                                ROOT, describe_change, fingerprint)


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


def test_guard_covers_the_two_shap_comparison_plots_the_dashboard_reads() -> None:
    names = {p.relative_to(ROOT).as_posix() for p in REAL_SHAP_PLOTS}
    assert names == {"data/processed/shap_comparison_waterfall.png",
                     "data/processed/shap_comparison_importance.png"}
    assert set(REAL_SHAP_PLOTS) | set(REAL_OPTIMIZER_OUTPUTS) == set(GUARDED_OUTPUTS)


def test_the_guarded_plot_names_are_the_ones_the_code_writes_and_the_dashboard_shows() -> None:
    writer = (ROOT / "src" / "explainability" / "shap_explainer.py").read_text(encoding="utf-8")
    reader = (ROOT / "src" / "dashboard" / "analysis_view.py").read_text(encoding="utf-8")
    for plot in REAL_SHAP_PLOTS:
        assert plot.name in writer and plot.name in reader, plot.name


def test_the_plot_test_writes_to_tmp_path_not_to_data_processed() -> None:
    source = (ROOT / "tests" / "test_phase4_depth.py").read_text(encoding="utf-8")
    assert 'save_dir="data/processed' not in source


def test_conftest_watches_every_guarded_output_not_just_the_optimizer_files() -> None:
    import ast

    tree = ast.parse((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    fixture = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "real_optimizer_outputs_unchanged")
    names = {n.id for n in ast.walk(fixture) if isinstance(n, ast.Name)}
    assert "GUARDED_OUTPUTS" in names and "REAL_OPTIMIZER_OUTPUTS" not in names
