"""Regenerate every thesis and report figure that can be drawn from committed results.

One command, no data needed (the evaluation frames in ``data/eval/`` are
git-ignored and are not read)::

    python scripts/make_all_figures.py              # draw every figure
    python scripts/make_all_figures.py --write-doc  # rewrite eval/FIGURES.md
    python scripts/make_all_figures.py --write-analysis  # rewrite eval/FIGURES_ANALYSIS.md

The sources are committed files: ``eval/method_comparison_results.csv`` (the
report graphs and the 2x2 tier figure), ``eval/lead_time.csv`` (the lead-time
figure), and, for R1 to R4 (``eval/ch4/fig_results.py``), ``eval/results_main.csv``,
``eval/method_comparison_no_eq1.csv``, ``eval/agent_ablation.csv`` and
``eval/split_sensitivity.csv``. ``--write-analysis`` writes
``eval/FIGURES_ANALYSIS.md``, the generated analysis of R1 to R4. The lead-time file is
itself computed from the frames by ``eval/ch4/detection_figures.py``; rerun that
script (it needs the frames) to refresh it.

``FIGURES`` is the single list: the script runs exactly these, and
``eval/FIGURES.md`` (figure, file, source CSV, script) is generated from it, so
the two cannot drift. ``thesis/`` is git-ignored: its figures are regenerated,
not versioned.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "method_comparison_results.csv"
LEAD_CSV = ROOT / "eval" / "lead_time.csv"
FIGURES_MD = ROOT / "eval" / "FIGURES.md"
ANALYSIS_MD = ROOT / "eval" / "FIGURES_ANALYSIS.md"

_RESULTS_REL = "eval/method_comparison_results.csv"
_G_METHOD = "eval/graphs_method_comparison"
_G_TIER = "eval/graphs_ablation_tiers"
_THESIS = "thesis/figures/ch4"
_REGIONS = ("hormuz", "bab_el_mandeb", "panama")


@dataclass(frozen=True)
class Figure:
    """One figure (or a set drawn together): what it is, where, from what, by what."""

    figure: str
    files: tuple[str, ...]
    sources: tuple[str, ...]
    script: str


FIGURES: tuple[Figure, ...] = (
    Figure("A1 AUC by method, all regions", (f"{_G_METHOD}/A1_auc_by_method.png",),
           (_RESULTS_REL,), "scripts/report_method_comparison.py (graph_method_auc)"),
    Figure("A2 false-positive harness (Malacca alert rates)",
           (f"{_G_METHOD}/A2_false_positive_harness.png",),
           (_RESULTS_REL,), "scripts/report_method_comparison.py (graph_alert_rate)"),
    Figure("B tier progression per region",
           tuple(f"{_G_TIER}/B_tier_progression_{r}.png" for r in _REGIONS),
           (_RESULTS_REL,), "scripts/report_method_comparison.py (graph_tiers)"),
    Figure("C method ranking per region",
           tuple(f"{_G_METHOD}/C_ranking_{r}.png" for r in _REGIONS),
           (_RESULTS_REL,), "scripts/report_method_comparison.py (graph_region_rankings)"),
    Figure("D AUC heatmap (method by region)", (f"{_G_METHOD}/D_auc_heatmap.png",),
           (_RESULTS_REL,), "scripts/report_method_comparison.py (graph_heatmap)"),
    Figure("E F1 against false-positive rate", (f"{_G_METHOD}/E_f1_vs_fpr.png",),
           (_RESULTS_REL,), "scripts/report_method_comparison.py (graph_f1_fpr)"),
    Figure("Thesis D1: tier progression, 2x2",
           (f"{_THESIS}/D1_tier_progression_2x2.pdf", f"{_THESIS}/D1_tier_progression_2x2.png"),
           (_RESULTS_REL,), "eval/ch4/fig_tiers_2x2.py"),
    Figure("Thesis Fig. 11 replacement: lead time, all tiers and the random control "
           "(not for the paper)",
           (f"{_THESIS}/D2_lead_time_all_tiers.pdf", f"{_THESIS}/D2_lead_time_all_tiers.png",
            f"{_THESIS}/D2_lead_time_all_tiers.caption.txt"),
           ("eval/lead_time.csv",), "eval/ch4/detection_figures.py (fig_lead; the CSV by lead_times)"),
    Figure("R1 change in AUC against Tier 1, forest plot (copied tiers marked)",
           (f"{_THESIS}/R1_delta_auc_forest.pdf", f"{_THESIS}/R1_delta_auc_forest.png"),
           ("eval/results_main.csv",), "eval/ch4/fig_results.py (fig_r1)"),
    Figure("R2 Eq. 1 ablation, duration term on and off",
           (f"{_THESIS}/R2_eq1_ablation.pdf", f"{_THESIS}/R2_eq1_ablation.png"),
           ("eval/method_comparison_no_eq1.csv", "eval/results_main.csv"),
           "eval/ch4/fig_results.py (fig_r2)"),
    Figure("R3 agent contribution: each agent alone and left out",
           (f"{_THESIS}/R3_agent_contribution.pdf", f"{_THESIS}/R3_agent_contribution.png"),
           ("eval/agent_ablation.csv",), "eval/ch4/fig_results.py (fig_r3)"),
    Figure("R4 split sensitivity: Tier 1 and Tier 4 at three split dates",
           (f"{_THESIS}/R4_split_sensitivity.pdf", f"{_THESIS}/R4_split_sensitivity.png"),
           ("eval/split_sensitivity.csv", "eval/results_main.csv"),
           "eval/ch4/fig_results.py (fig_r4)"),
)

_NOT_REGENERATED = """\
## Not regenerated here

- **Chapter 3 dashboard screenshots** (`thesis/figures/ch3_demo/`: `analysis_*`, `decision_*`):
  captured from the running dashboard by `scripts/capture_demo_screenshots.py`, with the numbers
  in the captions from `scripts/case_metrics.py`. They need a browser and the git-ignored
  `data/live/` series, so they cannot be drawn from committed results. The numbers themselves are
  committed: `eval/case_metrics.json` (its `provenance` block holds the sha256 of every live file
  it was computed from, and the screenshot manifest records the sha256 of the file itself).
- **Older Chapter 4 figures** in `thesis/figures/ch4/`: `D1_tier_progression` (1x4), `D2_lead_time`,
  `D2_lead_time_tier5`, `D2_lead_time_tier5_3x1` and the F-series (`F1` to `F10`, `F12` to `F16`).
  No script in the repository produces them. The earlier lead-time script imported
  `src.evaluation.path_d` and `src.viz.thesis_style`, neither of which is in it, and the 3x1 file
  was made by restacking a supplied 1x3 PNG with a local script that is not committed. Of these,
  only `D2_lead_time_tier5_3x1` is included by the thesis (`4_0_Evaluation.tex`). It is superseded
  by `D2_lead_time_all_tiers`, but the `.tex` still points at the old file, and its Timeliness
  paragraph quotes the old lead-time numbers, which the committed `eval/lead_time.csv` does not
  reproduce.
- **Diagnostic figures** (`eval/diagnostics/`, e.g. `geopolitical_baseline.py`): written to the
  git-ignored `data/eval/diagnostics/` from raw ACLED events, not from committed results.
- **Plots made inside `src/`** (`explainability/shap_explainer.py`,
  `optimization/optimization_analysis.py`): drawn from synthetic-data and optimizer outputs, not
  from the result CSVs.
"""


def render_figures_md() -> str:
    """``eval/FIGURES.md``: figure, files, source CSV and script, from :data:`FIGURES`."""
    rows = ["| Figure | File(s) | Source (committed) | Script |", "|---|---|---|---|"]
    for fig in FIGURES:
        files = "<br>".join(f"`{p}`" for p in fig.files)
        sources = "<br>".join(f"`{s}`" for s in fig.sources)
        rows.append(f"| {fig.figure} | {files} | {sources} | `{fig.script}` |")
    return "\n".join([
        "# Figures",
        "",
        "Every thesis and report figure that can be drawn from committed results, with its "
        "source and script. Draw them all with `python scripts/make_all_figures.py` (no data "
        "needed). `thesis/` is git-ignored: those figures are regenerated, not versioned. "
        "This file is generated (`python scripts/make_all_figures.py --write-doc`) from the "
        "list the script itself runs.",
        "",
        "`eval/lead_time.csv` is computed from the git-ignored frames by "
        "`python eval/ch4/detection_figures.py`; the figure is drawn from the committed CSV.",
        "",
        "R1 to R4 are drawn by `eval/ch4/fig_results.py` from the committed result CSVs; "
        "`eval/FIGURES_ANALYSIS.md` holds their analysis, generated by "
        "`python scripts/make_all_figures.py --write-analysis` (every number from those CSVs). "
        "There is no figure for the Malacca false-alarm rates or the Bab el-Mandeb surrogate "
        "error; the analysis file gives them as values.",
        "",
        *rows,
        "",
        _NOT_REGENERATED,
    ])


def _module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                  # @dataclass resolves its module by name
    spec.loader.exec_module(module)
    return module


def run(thesis_dir: Path | None = None,
        graph_dirs: tuple[Path, Path] | None = None) -> list[Path]:
    """Draw every registered figure; returns the files written.

    Args:
        thesis_dir: Where the thesis figures go (default ``thesis/figures/ch4``).
        graph_dirs: ``(method, tier)`` folders for the report graphs (default
            ``eval/graphs_method_comparison`` and ``eval/graphs_ablation_tiers``).

    Raises:
        FileNotFoundError: If ``eval/lead_time.csv`` is missing.
    """
    if not LEAD_CSV.exists():
        raise FileNotFoundError(f"{LEAD_CSV} is missing: run python eval/ch4/detection_figures.py "
                                "(it needs the evaluation frames) to compute it")
    results = pd.read_csv(RESULTS)
    report = _module("report_method_comparison", "scripts/report_method_comparison.py")
    if graph_dirs is not None:
        report._G_METHOD, report._G_TIER = graph_dirs
    report.make_graphs(results)
    graphs = [report._G_METHOD / Path(p).name for f in FIGURES if f.files[0].startswith(_G_METHOD)
              for p in f.files]
    graphs += [report._G_TIER / Path(p).name for f in FIGURES if f.files[0].startswith(_G_TIER)
               for p in f.files]
    tiers = _module("fig_tiers_2x2", "eval/ch4/fig_tiers_2x2.py")
    lead = _module("detection_figures", "eval/ch4/detection_figures.py")
    r = _module("fig_results", "eval/ch4/fig_results.py")
    return (graphs + tiers.draw(results, thesis_dir)
            + lead.fig_lead(pd.read_csv(LEAD_CSV), thesis_dir)
            + r.draw_all(thesis_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--write-doc", action="store_true",
                        help="rewrite eval/FIGURES.md from the registry instead of drawing")
    parser.add_argument("--write-analysis", action="store_true",
                        help="rewrite eval/FIGURES_ANALYSIS.md (generated analysis of R1 to R4)")
    args = parser.parse_args(argv)
    if args.write_doc or args.write_analysis:
        if args.write_doc:
            FIGURES_MD.write_text(render_figures_md(), encoding="utf-8")
            print("wrote", FIGURES_MD)
        if args.write_analysis:
            ANALYSIS_MD.write_text(_module("fig_results", "eval/ch4/fig_results.py")
                                   .render_analysis(), encoding="utf-8")
            print("wrote", ANALYSIS_MD)
        return 0
    for path in run():
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
