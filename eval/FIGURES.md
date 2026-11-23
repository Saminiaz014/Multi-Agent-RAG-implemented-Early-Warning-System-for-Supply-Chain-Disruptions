# Figures

Every thesis and report figure that can be drawn from committed results, with its source and script. Draw them all with `python scripts/make_all_figures.py` (no data needed). `thesis/` is git-ignored: those figures are regenerated, not versioned. This file is generated (`python scripts/make_all_figures.py --write-doc`) from the list the script itself runs.

`eval/lead_time.csv` is computed from the git-ignored frames by `python eval/ch4/detection_figures.py`; the figure is drawn from the committed CSV.

| Figure | File(s) | Source (committed) | Script |
|---|---|---|---|
| A1 AUC by method, all regions | `eval/graphs_method_comparison/A1_auc_by_method.png` | `eval/method_comparison_results.csv` | `scripts/report_method_comparison.py (graph_method_auc)` |
| A2 false-positive harness (Malacca alert rates) | `eval/graphs_method_comparison/A2_false_positive_harness.png` | `eval/method_comparison_results.csv` | `scripts/report_method_comparison.py (graph_alert_rate)` |
| B tier progression per region | `eval/graphs_ablation_tiers/B_tier_progression_hormuz.png`<br>`eval/graphs_ablation_tiers/B_tier_progression_bab_el_mandeb.png`<br>`eval/graphs_ablation_tiers/B_tier_progression_panama.png` | `eval/method_comparison_results.csv` | `scripts/report_method_comparison.py (graph_tiers)` |
| C method ranking per region | `eval/graphs_method_comparison/C_ranking_hormuz.png`<br>`eval/graphs_method_comparison/C_ranking_bab_el_mandeb.png`<br>`eval/graphs_method_comparison/C_ranking_panama.png` | `eval/method_comparison_results.csv` | `scripts/report_method_comparison.py (graph_region_rankings)` |
| D AUC heatmap (method by region) | `eval/graphs_method_comparison/D_auc_heatmap.png` | `eval/method_comparison_results.csv` | `scripts/report_method_comparison.py (graph_heatmap)` |
| E F1 against false-positive rate | `eval/graphs_method_comparison/E_f1_vs_fpr.png` | `eval/method_comparison_results.csv` | `scripts/report_method_comparison.py (graph_f1_fpr)` |
| Thesis D1: tier progression, 2x2 | `thesis/figures/ch4/D1_tier_progression_2x2.pdf`<br>`thesis/figures/ch4/D1_tier_progression_2x2.png` | `eval/method_comparison_results.csv` | `eval/ch4/fig_tiers_2x2.py` |
| Thesis Fig. 11 replacement: lead time, all tiers and the random control (not for the paper) | `thesis/figures/ch4/D2_lead_time_all_tiers.pdf`<br>`thesis/figures/ch4/D2_lead_time_all_tiers.png`<br>`thesis/figures/ch4/D2_lead_time_all_tiers.caption.txt` | `eval/lead_time.csv` | `eval/ch4/detection_figures.py (fig_lead; the CSV by lead_times)` |

## Not regenerated here

- **Chapter 3 dashboard screenshots** (`thesis/figures/ch3_demo/`: `analysis_*`, `decision_*`):
  captured from the running dashboard by `scripts/capture_demo_screenshots.py`, with the numbers
  in the captions from `scripts/case_metrics.py`. They need a browser and the git-ignored
  `data/live/` series, so they cannot be drawn from committed results.
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
