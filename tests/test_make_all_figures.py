"""scripts/make_all_figures.py: every figure from committed results, and eval/FIGURES.md says so."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mf = _load("make_all_figures", "scripts/make_all_figures.py")
d1 = _load("fig_tiers_2x2", "eval/ch4/fig_tiers_2x2.py")


def _registered_names() -> set[str]:
    return {Path(p).name for f in mf.FIGURES for p in f.files}


# ------------------------------------------------------------- the script
def test_it_produces_exactly_the_registered_figures(tmp_path) -> None:
    written = mf.run(thesis_dir=tmp_path / "thesis", graph_dirs=(tmp_path / "gm", tmp_path / "gt"))
    names = {p.name for p in written}
    assert names == _registered_names()
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_every_source_exists_and_is_not_git_ignored_and_every_script_exists() -> None:
    for fig in mf.FIGURES:
        for source in fig.sources:
            path = _ROOT / source
            assert path.exists(), (fig.figure, source)
            ignored = subprocess.run(["git", "check-ignore", "-q", source], cwd=_ROOT).returncode == 0
            assert not ignored, f"{source} is git-ignored: not a committed result"
        assert (_ROOT / fig.script.split(" ")[0]).exists(), fig.script


def test_figures_md_is_generated_from_the_registry() -> None:
    doc = (_ROOT / "eval" / "FIGURES.md").read_text(encoding="utf-8")
    assert doc == mf.render_figures_md()
    for fig in mf.FIGURES:
        for path in fig.files:
            assert path in doc


def test_no_committed_report_graph_is_missing_from_the_registry() -> None:
    on_disk = {p.name for folder in ("graphs_method_comparison", "graphs_ablation_tiers")
               for p in (_ROOT / "eval" / folder).glob("*.png")}
    assert on_disk <= _registered_names(), on_disk - _registered_names()


def test_figures_md_says_what_it_cannot_regenerate() -> None:
    doc = mf.render_figures_md()
    assert "Not regenerated here" in doc
    assert "capture_demo_screenshots.py" in doc and "path_d" in doc


def test_a_missing_lead_time_file_says_how_to_make_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mf, "LEAD_CSV", tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="detection_figures.py"):
        mf.run(thesis_dir=tmp_path, graph_dirs=(tmp_path / "a", tmp_path / "b"))


# ------------------------------------------------- the 2x2 tier figure
def _results(t1: float, t5: float) -> pd.DataFrame:
    rows = []
    for region in ("hormuz", "bab_el_mandeb", "panama", "malacca"):
        for k in range(1, 6):
            auc = float("nan") if region == "malacca" else t1 + (t5 - t1) * (k - 1) / 4
            rows.append({"region": region, "method": f"Tier {k}", "kind": "multi-agent",
                         "auc": auc, "alert_rate": 0.2 + 0.05 * k, "test_start": "2023-07-01",
                         "test_end": "2025-11-21"})
        rows.append({"region": region, "method": "M3 persistence", "kind": "unsupervised",
                     "auc": 0.9, "alert_rate": 0.05, "test_start": "2023-07-01",
                     "test_end": "2025-11-21"})
    return pd.DataFrame(rows)


def _with_deltas(results: pd.DataFrame, ci_high: dict[str, float]) -> pd.DataFrame:
    """Add the paired Tier 5 - Tier 1 interval's upper bound per region (< 0: a clear drop)."""
    out = results.copy()
    out["delta_auc_ci_high"] = float("nan")
    for region, high in ci_high.items():
        out.loc[(out.region == region) & (out.method == "Tier 5"), "delta_auc_ci_high"] = high
    return out


def test_the_tier_headline_claims_only_what_the_intervals_support() -> None:
    lower = _results(0.9, 0.6)
    clear = _with_deltas(lower, {"hormuz": -0.05, "bab_el_mandeb": -0.1, "panama": -0.01})
    assert d1.headline(clear) == "Adding agents lowers discrimination in every evaluable region"
    # Panama's point estimate is lower but its paired interval includes 0.
    unclear = _with_deltas(lower, {"hormuz": -0.05, "bab_el_mandeb": -0.1, "panama": 0.005})
    assert d1.headline(unclear) == ("Adding agents lowers AUC in every evaluable region "
                                    "(clearly in 2 of 3)")
    # No interval columns at all: point estimates only, said as such ("AUC", no "clearly").
    assert d1.headline(lower) == "Adding agents lowers AUC in every evaluable region"


def test_the_tier_headline_does_not_say_lowers_when_a_region_rises() -> None:
    assert d1.headline(_results(0.6, 0.9)).startswith("Adding agents does not lift")
    mixed = _results(0.9, 0.6)
    mixed.loc[(mixed.region == "panama") & (mixed.method == "Tier 5"), "auc"] = 0.95
    assert d1.headline(mixed).startswith("Adding agents does not lift")


def test_the_tier_footer_states_the_test_window_from_the_results() -> None:
    text = d1.footer(_results(0.9, 0.6))
    assert "2023-07-01 to 2025-11-21" in text and "2025-08-30" not in text
    assert "aggregation path" not in text


def test_malacca_alert_rates_are_never_clipped(tmp_path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    seen = []
    real_close = plt.close
    monkeypatch.setattr(d1.plt, "close", lambda fig=None: (seen.append(fig), real_close(fig)))
    results = _results(0.9, 0.6)
    results.loc[(results.region == "malacca") & (results.method == "Tier 4"), "alert_rate"] = 0.9
    d1.draw(results, tmp_path)
    malacca_axes = seen[0].axes[3]
    assert malacca_axes.get_ylim()[1] >= 0.9


def test_an_unknown_option_is_an_error_not_a_silent_redraw(monkeypatch) -> None:
    drawn = []
    monkeypatch.setattr(mf, "run", lambda *a, **k: drawn.append(1) or [])
    with pytest.raises(SystemExit) as exc:
        mf.main(["--write-docs"])                             # a typo of --write-doc
    assert exc.value.code == 2 and not drawn


def test_write_doc_writes_the_generated_file_and_draws_nothing(tmp_path, monkeypatch) -> None:
    drawn = []
    monkeypatch.setattr(mf, "run", lambda *a, **k: drawn.append(1) or [])
    monkeypatch.setattr(mf, "FIGURES_MD", tmp_path / "FIGURES.md")
    assert mf.main(["--write-doc"]) == 0
    assert (tmp_path / "FIGURES.md").read_text(encoding="utf-8") == mf.render_figures_md()
    assert not drawn


def test_figures_md_names_the_f_series_that_exists_and_not_one_that_does_not() -> None:
    doc = mf.render_figures_md()
    assert "`F1`" in doc and "`F10`" in doc and "`F12`" in doc and "`F16`" in doc
    assert "F1` to `F16" not in doc                           # there is no F11
