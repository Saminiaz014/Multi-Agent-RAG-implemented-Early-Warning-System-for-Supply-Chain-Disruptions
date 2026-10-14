# Stale claims register (XAI-02)

These are places that state something the current code or measurements contradict. This register **lists only**; `.tex` files are not edited here. The full scan (0.991, 20 features, six agents, P_CRIT/P_HIGH/P_LOW/N_QUIET, the July METRIC 8 numbers) is completed in Phase 8. Entries below were added as they were found.

| # | Where | Stale statement | Corrected value / replacement | Added |
|---|---|---|---|---|
| 1 | `README.md:64` (SRQ table, SRQ3 Verdict) | "Yes, with a stated surrogate caveat", resting on the SHAP surrogate | Replace with the exact-attribution verdict in `CH4_WRITING_NOTES.md` § SRQ3. The surrogate is a comparator with real-data R² −0.32 to −4.39. | Phase 3 |
| 2 | `DEVELOPMENT_LOG.md:1091` | `train_surrogate` "Returns R² (0.991 on the 364-day synthetic dataset)" | Training-set R² of the retired 20-feature model. Current 17-feature surrogate: train 0.983, chronological hold-out 0.528, 5-fold CV 0.868; on real test windows −0.32 to −4.39 (Phase 2 scorecard). | Phase 3 |
| 3 | `DEVELOPMENT_LOG.md:1119` | `"surrogate_r2": 0.991` in the example output | As #2. It is the in-sample fit, not faithfulness. | Phase 3 |
| 4 | `DEVELOPMENT_LOG.md:1070` | "the full 20-feature, 6-agent space" | 17 features, 5 agents (routing removed in `b2306ae`). | Phase 3 |
| 5 | `thesis/1_0_Introduction.tex:367-373` (SRQ3 framing) | Interpretation via SHAP "quantifies the contribution of each input feature" | The primary attribution is the exact agent- and feature-level Shapley decomposition; SHAP-on-surrogate is a comparator. Replacement paragraph in `CH4_WRITING_NOTES.md` § SRQ3. | Phase 3 |
| 6 | `thesis/3_0_Methodology.tex:1108-1113` | Exact analytic attributions "would be strictly better … listed as future work" | Implemented (`src/explainability/exact_shapley.py`, `feature_shapley.py`); the subsection should present it as the method. | Phase 3 |
| 7 | `README.md:512` | Same "future work" statement as #6 | As #6. | Phase 3 |
| 8 | SRQ numbering: `README.md:60-66` vs `thesis/1_0_Introduction.tex` / `5_0_Discussion.tex` | The README numbers decision effectiveness SRQ5; the thesis uses SRQ4 = dashboard and SRQ5 = performance against baselines | Not a value error, but the two numberings disagree. Pick one before Chapter 4 cross-references SRQs. | Phase 3 |
