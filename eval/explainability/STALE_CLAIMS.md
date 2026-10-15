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

## Phase 8 rescan (2026-10-15)

**Patterns searched:**
- 0.991;
- 20-feature, 20 feature, 20 canonical;
- six/6 agents, six-agent, 6-agent, six detection, six domain, "all six", "of six";
- P_CRIT, P_HIGH, P_LOW, N_QUIET, N_DECOY;
- the July METRIC 8 values 0.8658 / 0.866 / 0.4904 / 0.490;
- "R² > 0.99".

**Where:** `thesis/*.tex`, `README.md`, `docs/*.md`, `DEVELOPMENT_LOG.md`, and code and docstrings in `src/`, `scripts/`, `notebooks/` and `main.py`.

**False positives, excluded:**
- `thesis/3_0_Methodology.tex:862` ("a gain of six");
- `README.md:881` and `docs/THESIS_BRIEF.md:278` (0.866 is B6 CUSUM's AUC, not METRIC 8);
- `scripts/run_disaster_extraction_2018_2026.py:15,153` ("two of six sources", which refers to extractors).

**Not found:** P_CRIT / P_HIGH / P_LOW / N_QUIET anywhere outside the archived `docs/eval01-archived/` (not scanned); "R² > 0.99" after the Phase 6 fix.

### Thesis (`.tex`, list only, not edited)

| # | Where | Stale statement | Corrected value |
|---|---|---|---|
| 9 | `thesis/3_0_Methodology.tex:1591-1593` | "Ingestion normalises six heterogeneous public sources … Six domain-specialised agents" | Five agents (shipping, market, geopolitical, natural disaster, news); routing removed in `b2306ae`. The "six layers" in the same sentence is the pipeline's layer count and stays. |
| 10 | `thesis/4_0_Evaluation.tex:76-85` | Explainability Score: "based on SHAP attributions" (+ TODO) | METRIC 9 (E1–E9) on the exact Shapley decomposition; SHAP surrogate as comparator (`CH4_WRITING_NOTES.md` §Performance Metrics). |
| 11 | `thesis/4_0_Evaluation.tex:87-95` | Decision Effectiveness defined by a rubric / participant study (TODO) | Measured as METRIC 8 on the synthetic series (P0-exact canonical), labelled SYNTHETIC; there is no participant study (Outlook). |

### README and docs (list only)

| # | Where | Stale statement | Corrected value |
|---|---|---|---|
| 12 | `README.md:116` | "Five of the six live sources … a system that required all six agents" | Five agents and five domain sources. |
| 13 | `README.md:168` | "all six agents implement the same six-stage lifecycle" | All five agents. |
| 14 | `README.md:211` | "Five of the six agents read a live, free public API. The sixth is dormant" | All five agents have a live source; routing is removed, not dormant. |
| 15 | `README.md:1199` | `test_shap_6agent.py # 20-feature SHAP surrogate` | 17-feature surrogate (the file name is historical). |
| 16 | `README.md:1348` | "exercises all six agents" | All five agents. |
| 17 | `docs/SCORING_REFERENCE.md:502,505` | "20-feature row", "The 20 canonical features" | 17 features. |
| 18 | `docs/SCORING_REFERENCE.md:586` | METRIC 3 "6-agent vs 2-agent vs 1-agent" | 5-agent vs 2 vs 1. |
| 19 | `docs/REGION_USAGE_GUIDE.md:170` | "mapping all six agent keys" | Five agent keys (`AGENT_KEYS`). |

### Dashboard text shown to users (list only)

| # | Where | Stale statement | Corrected value |
|---|---|---|---|
| 20 | `src/dashboard/analysis_view.py:433` | METRIC 8 export caption "Overall accuracy 0.866 vs naive baseline 0.797" | July value. Current: 0.948 (weight × score) / **0.937 (P0-exact, canonical)**. Section 8 itself reads the stale `evaluation_results.json`. |
| 21 | `src/dashboard/analysis_view.py:259,539,570`; `core.py:1429` | "6-agent" labels on the METRIC 3 diversity charts and the raw-signal expander | "5-agent". The underlying METRIC 3 numbers are also from the July file. |

### Code docstrings (list only; behaviour is correct, only the count is stale)

| # | Where | Stale statement | Corrected value |
|---|---|---|---|
| 22 | `src/api/endpoints.py:3,5,7,304,381` | "6-agent pipeline", "all 6 agents" | 5. |
| 23 | `src/orchestrator.py:216,230` | "all six agents", "six-agent composite" | 5. |
| 24 | `src/orchestrator.py:721` | "single-row 20-feature DataFrame" | 17-feature. |
| 25 | `src/optimization/pipeline_evaluator.py:5,77,101,108`; `data_split.py:92` | "six-agent pipeline", "six agents", "six connectors" | 5. |
| 26 | `src/rag/context_retriever.py:279,413` | "6-agent signal profile" | 5. |
| 27 | `src/core/regions.py:57` | "for all six agents" | 5 (`AGENT_KEYS`). |
| 28 | `main.py:136,193,326`; `notebooks/evaluation.py:9,403,406` | "six-agent pipeline / composite", METRIC 3 "6-agent" | 5. |

### `DEVELOPMENT_LOG.md` (32 hits): historical, do not edit, do not cite

These are dated phase entries that were true when written:
- 6-agent integration (lines 712–1052);
- the 20-feature surrogate and `"surrogate_r2": 0.991` (1070–1132);
- the 6-agent API (1367–1405);
- the July METRIC 8 values 0.866 (1503–1544).

Recommendation: add one line at the top of the log saying entries before 2026-10-11 describe the 6-agent, 20-feature system. Entries #2–#4 above are already registered.

**METRIC 8 values to use instead of the July ones:** P0-exact hand-tuned 0.937 (macro-F1 0.708); weight × score 0.948 (0.855) in a footnote; optimized 0.578, below the naive baseline 0.797.

## Status update (2026-10-15)

**Fixed in code:**
- **#20:** METRIC 8 export caption. It now names the stored July run as superseded and gives the canonical values (0.937 / 0.708, SYNTHETIC).
- **#21 (live charts):** the analysis-bundle key and chart label are now "5-agent"; the raw-signal expander is relabelled.
- **#22–#28:** docstrings now say 5 agents / 17 features.

**Deliberately unchanged:**
- **#21 (stored METRIC 3 section, `analysis_view.py:259`).** It still reads the `"6-agent"` key of the July `evaluation_results.json`, whose numbers really are from the six-agent run. Relabelling them would misstate the data.
- **`notebooks/evaluation.py`.** It keeps writing the legacy `"6-agent"` key, now documented, so stored files stay readable.
- **`.tex` files (#5, #6, #9–#11) and `DEVELOPMENT_LOG.md`:** untouched, as instructed.
- **README / docs (#1, #7, #12–#19):** list only.

