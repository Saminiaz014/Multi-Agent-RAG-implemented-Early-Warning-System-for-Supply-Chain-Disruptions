# Explainability report (XAI-02)

Living document. It is updated at the end of every phase, and every number in it is reproducible with:

```
python eval/explainability/make_figures.py
```

That command regenerates the AERs, the scorecard CSVs and the figures. Branch `xai-02`; baseline `main` @ `b2306ae`.

---

## 1. What is explained, and against what

| Item | Choice |
|---|---|
| Explained score (thesis metrics) | **Path D**: the Tier 5 composite of `scripts/run_method_comparison.py`, a renormalised weighted mean of agent scores. It is reproduced bit-for-bit (`tests/test_xai_frozen_scores.py`). |
| Explained score (dashboard) | **Path C**: the same mean, with the agreement bonus and the clip at 1.0 (Phase 6). |
| Alerts | Test-window days where the Tier 5 composite is above its train-p90. This is the CSV's own operating point. |
| Episodes | Consecutive alert days, merged across gaps of at most 3 calendar days. An episode is TP if any day in its span has `y_true = 1`. |
| Primary attribution | Exact Shapley values over the agent scores (32 coalitions), measured from each agent's **calm baseline**: the median train-window score on `y_true = 0` days. |
| Feature attribution | Exact Shapley values inside each agent over its own inputs, with the agent's non-linearity (max / clip / sigmoid / isolation forest) inside the game. |
| Comparators | `surrogate_prod` is the production SHAP surrogate as built: an RF on one synthetic seed-42 year, not retrained. `surrogate_refit` uses the same recipe, fitted on the region's real train window. |
| Unit of evidence | The **episode**. Day values are averaged within an episode. Headline numbers are means over episodes with a 1,000-resample episode-level bootstrap 95% CI (seed 20261014). Day-level means are secondary and labelled as such. |
| Circularity control | Every scorecard row is reported twice: `with_duration`, and `without_duration` (shipping reduced to its shock score, with the threshold, baselines and noise recomputed on train). |

## 2. Deviations recorded

### Deviation 1: Path D uses the CSV's original weight vector

Path D is explained with `CSV_WEIGHTS = {shipping 0.25, market 0.15, geopolitical 0.25, natural_disaster 0.10, news_sentiment 0.10}`, not with the current `settings.yaml` weights (0.294 / 0.176 / 0.294 / 0.118 / 0.118).

- **Why:** `eval/method_comparison_results.csv` was generated with this vector before the routing agent was removed. The current settings are its 3-decimal renormalisation. Tier composites renormalise over their agents, so only the ratios matter, but the rounding moves Tier ≥ 2 AUC by about 1e-4 (`experiments/ablation_v2/ABLATION_V2_REPORT.md`).
- **Result:** with `CSV_WEIGHTS` the CSV is reproduced to **1.1e-16** in all 80 tier metrics. With the settings weights the 1e-9 frozen-score tolerance would fail.
- **Recorded in:** every output records `weight_mode = hand_tuned`, and the weights are stored in each AER.

### Deviation 5: uncertainty is a score-noise band, not a calibrated probability

Calibration is not estimable here: 3 of 4 regions have zero train-window positives, and the label is the shipping level-shift statistic. Each AER therefore carries a **score-noise band** instead:

- Each agent's noise scale is σᵢ = std(day-to-day change of its score on consecutive calm train days) / √2. This is the per-day noise implied by a first difference.
- The composite is re-evaluated under Gaussian perturbations of the agent scores, clipped to [0, 1].
- AERs use 500 draws; the scorecard uses 200. Each draw set is seeded from the alert ID, or the day index in the scorecard.
- The record reports agent dispersion, the 2.5–97.5% band, and the share of draws above the threshold.

It measures how fragile an alert is to plausible input noise. It is **not** a probability that the alert is correct.

### Other deviations (accepted in Phase 1 review)

2. The agents were not refactored. Every intermediate value is already in their `detect` output.
3. AERs carry extra fields: `weights`, `agent_scores`, `agent_baselines`, `circularity_control`.
4. Path C `bonus_share` is split into bonus and clip parts by linearity of the Shapley value.
6. `precedents` (Phase 3) and `narrative` (Phase 6) are empty placeholders.
7. `thesis/`, `docs/` and `data/eval/` are gitignored. The frozen test skips without `data/eval/`. Phase 2 tracks `thesis/figures/ch4/*.caption.txt` (see §6).

## 3. Phase 1: alerts and episodes (Tier 5, Path D, real_eval)

| Region | Alert days TP / FP | Episodes TP / FP | Without duration: days TP / FP | Without duration: episodes TP / FP |
|---|---|---|---|---|
| Hormuz | 22 / 159 | 5 / 34 | 14 / 157 | 7 / 36 |
| Bab el-Mandeb | 163 / 332 | 7 / 10 | 147 / 332 | 8 / 10 |
| Panama | 106 / 91 | 8 / 30 | 106 / 91 | 8 / 30 |
| Malacca | 0 / 188 | 0 / 49 | 0 / 188 | 0 / 49 |

The maximum |efficiency residual| is 1.1e-16 (agent level) and 3.3e-16 (feature level).

## 4. Phase 2: METRIC 9 scorecard

Source: `scorecard.csv`. Values are episode means with episode-bootstrap 95% CIs. **TP episodes: 5 / 7 / 8 (Hormuz / Bab el-Mandeb / Panama)**; Malacca has none.

### 4.1 Exact vs surrogate (`with_duration`)

| Metric | Method | Hormuz | Bab el-Mandeb | Panama | Malacca |
|---|---|---|---|---|---|
| E1 local error \|f−g\| | prod | 0.234 [0.224, 0.243] | 0.232 [0.221, 0.245] | 0.044 [0.038, 0.051] | 0.193 [0.183, 0.203] |
| | refit | 0.065 [0.054, 0.075] | 0.139 [0.109, 0.167] | 0.062 [0.050, 0.075] | 0.060 [0.051, 0.070] |
| | exact | 0 (residual ≤ 1.1e-16) | 0 | 0 | 0 |
| E1 R² of prod on the real test window | prod | −0.37 | −4.39 | −1.10 | −0.32 |
| E1 R² train / out-of-time / CV5 | prod (synthetic year) | 0.983 / 0.528 / 0.868 | (same model) | | |
| | refit | 0.961 / 0.655 / 0.710 | 0.956 / −2.001 / 0.688 | 0.934 / 0.647 / 0.522 | 0.958 / 0.633 / 0.697 |
| E1 \|SHAP\| on agents outside the composite | prod | 15.0% | 15.8% | 0% | 17.1% |
| E2 agent deletion AUC ↓ | exact | **0.152** [0.121, 0.181] | **0.124** [0.087, 0.157] | **0.055** [0.014, 0.091] | **0.167** [0.140, 0.189] |
| | prod | 0.528 [0.490, 0.576] | 0.530 [0.475, 0.592] | 0.590 [0.555, 0.627] | 0.488 [0.441, 0.530] |
| | refit | 0.199 [0.168, 0.231] | 0.335 [0.254, 0.411] | 0.135 [0.073, 0.199] | 0.227 [0.209, 0.243] |
| | random | 0.500 | 0.500 | 0.500 | 0.500 |
| E2 agent insertion AUC ↑ | exact | **0.848** | **0.876** | **0.945** | **0.833** |
| | prod | 0.472 | 0.470 | 0.410 | 0.512 |
| | refit | 0.801 | 0.665 | 0.865 | 0.773 |
| E2 shipping feature deletion ↓ | exact | **0.438** | **0.393** | **0.445** | **0.252** |
| | prod | 0.792 | 0.579 | 0.677 | 0.685 |
| | refit | 0.485 | 0.513 | 0.477 | 0.300 |
| | random | 0.686 | 0.638 | 0.666 | 0.612 |
| E3 k80 agents | exact | 2.00 | 2.05 | 1.65 | 1.94 |
| | prod | 2.35 | 2.26 | 2.59 | 2.42 |
| | refit | 1.81 | 1.83 | 1.07 | 1.85 |
| E3 k80 features | exact | 4.62 | 4.50 | 3.89 | 4.27 |
| | prod | 6.53 | 6.29 | 6.94 | 6.78 |
| | refit | 3.44 | 3.67 | 2.50 | 3.80 |
| E4 day-to-day ρ within episode | exact | 0.765 | 0.810 | 0.838 | 0.710 |
| | prod | 0.785 | 0.824 | 0.617 | 0.808 |
| | refit | 0.533 | 0.379 | 0.713 | 0.492 |
| E4 ρ under score noise σ = 0.05 | exact | 0.968 | 0.969 | 0.862 | 0.954 |
| E5 top agent = exact top agent | prod | 0.339 [0.234, 0.439] | 0.180 [0.070, 0.323] | 0.009 [0.000, 0.026] | 0.421 [0.315, 0.537] |
| | refit | 0.842 | 0.529 | 0.941 | 0.911 |
| E6 feasible single-agent counterfactual | exact | 0.970 | 0.973 | 1.000 | 0.969 |
| E7 AUC FP-vs-TP: band width | exact | 0.38 [0.12, 0.70] | 0.87 [0.66, 1.00] | 0.57 [0.38, 0.73] | n/a |
| E7 AUC FP-vs-TP: dispersion | exact | 0.55 [0.20, 0.84] | 0.26 [0.04, 0.53] | 0.61 [0.39, 0.81] | n/a |
| E7 calibration | — | not estimable (0 train positives in 3 of 4 regions; circular label) | | | |
| E8 / E9 | — | pending Phase 3 / Phase 6 | | | |

**Reading the table:**

- **The production surrogate orders agents no better than chance.** Its agent deletion AUC is 0.49–0.59 against 0.50 for random order; exact order gives 0.06–0.17.
- **Its top agent rarely matches the exact one.** Agreement is 18–42%, and 0.9% in Panama.
- **15–17% of its attribution mass goes to agents that are not in the composite.**
- **On real data it predicts the score worse than a constant** (R² −0.32 to −4.39).
- **Refitting the surrogate on real data closes most of the gap but not all of it.** Exact is still better on every E2 row and needs fewer features (k80).
- **The surrogates' higher day-to-day stability (E4, prod) is not a virtue.** It comes with a top driver that disagrees with the true one.

**E2 notes:**
- E2 is computed only on days where the explained score sits at least 0.01 above its baseline (`MIN_GAP`); the days used are in `n_days_used`. All agent and shipping-feature days qualify. The market feature game qualifies on 124 / 291 / 134 / 121 of 181 / 495 / 197 / 188 alert days (Hormuz / Bab el-Mandeb / Panama / Malacca).
- **Market feature-level AUCs fall outside [0, 1]. This is genuine.** The market game is non-monotone: a series whose term sits below its calm level *raises* the clipped score when replaced by calm, so the curves overshoot. The ordering comparison still holds (exact deletion −0.21 to 0.10 vs random 0.52–0.54), but the values are not bounded proportions.

**E7, uncertainty usefulness:** every CI spans 0.5 except Bab el-Mandeb band width (0.87 [0.66, 1.00], 7 TP / 10 FP episodes). With TP episodes of 5 / 7 / 8, none of these is evidence either way.

**Circularity control:** the `without_duration` rows move by at most a few hundredths. No conclusion above changes when the shipping duration branch is removed (`scorecard.csv`, `variant == without_duration`).

### 4.2 F14: top exact driver of TP vs FP episodes (`fp_driver_composition.csv`)

| Region | TP episodes | FP episodes |
|---|---|---|
| Hormuz | geopolitical 4/5 (80%, CI 40–100%), shipping 1/5 | shipping 21/34 (62%, CI 44–76%), geopolitical 13/34 (38%, CI 24–56%) |
| Bab el-Mandeb | shipping 7/7 | shipping 8/10, geopolitical 2/10 |
| Panama | shipping 8/8 | shipping 30/30 |
| Malacca | — | **geopolitical 29/49 (59%, CI 45–71%)**, shipping 20/49 |

The brief expected geopolitical to lead the FPs of both Hormuz and Malacca. That holds for **Malacca** only.

- **Hormuz FP episodes** are led by shipping, at episode level (21/34) and at FP-day level (82 vs 77 of 159 FP alert days).
- **In Hormuz, geopolitical leads the TP episodes** (4/5). Geopolitical leadership therefore does *not* separate TP from FP there.

Without the duration branch: Hormuz FP geopolitical 12/36, TP 6/7; Malacca unchanged.

### 4.3 Market saturation (`market_saturation.csv`): report only, agent unchanged

| | Hormuz | Bab el-Mandeb | Panama | Malacca |
|---|---|---|---|---|
| Market calm baseline (train, y = 0, median) | 0.773 | 0.776 | 0.776 | 0.776 |
| Test days with market score ≥ 0.95 (day-level) | 29.0% | 29.0% | 29.0% | 29.0% |
| Test days clipped at 1.0 (day-level) | 25.4% | 25.4% | 25.5% | 25.4% |
| Train calm days ≥ 0.95 (day-level) | 31.2% | 31.3% | 31.3% | 31.3% |
| Mean \|φ_market\| on alerts (episode) | 0.043 [0.038, 0.048] | 0.044 [0.038, 0.051] | 0.035 [0.030, 0.039] | 0.038 [0.033, 0.043] |
| Mean share of Σ\|φ\| (episode) | 14.8% | 13.6% | 15.7% | 12.3% |

The market inputs are global FRED series, so the agent's score series is nearly identical across regions; only the alert sets differ. The agent sits at or near its ceiling on about 30% of days, *calm ones included*. Measured against a calm baseline of 0.78, it can add little to an alert. The market agent was never the top driver of any episode.

### 4.4 FP characterisation (`fp_characterisation.csv`, read-only ChromaDB copy)

**Rule:** a knowledge-base document counts if it is region-matched, geopolitical/conflict, and dated between (episode start − 7 days) and the episode end. Geopolitical/conflict means:
- ACLED daily events, or
- news whose title or text matches `GEO_PATTERN` in `run_scorecard.py`.

An episode is tagged `risk_without_traffic_drop` if at least one document counts, otherwise `unexplained`.

| Region | FP episodes | risk_without_traffic_drop | unexplained | KB news docs in region test window | Nearest region news doc (days, min / median) |
|---|---|---|---|---|---|
| Hormuz | 34 | 0 | 34 | 0 | 276 / 634 |
| Bab el-Mandeb | 10 | 0 | 10 | 134 | 43 / 114.5 |
| Panama | 30 | 0 | 30 | 0 | 336 / 566 |
| Malacca | 49 | 0 | 49 | 0 | 263 / 672 |

**All 123 FP episodes are `unexplained`, and the tag measures KB coverage, not the world:**
- Hormuz, Panama and Malacca have zero news documents in the 2023-08 → 2025-08 test window. NewsAPI documents all date from 2026, and SerpApi stops at 2024-05-31.
- Bab el-Mandeb's 134 in-window articles sit 43+ days from any FP episode.
- ACLED holds only **annual** summaries dated YYYY-01-01. These are reported as `acled_annual_military_events` and excluded from the ±7-day rule.

Phase 3 inherits this: precedent grounding for 2023-25 alerts cannot be scored outside Bab el-Mandeb without backfilling the KB.

## 5. Implementation notes

- **Feature-level games.** Each agent's game reruns its own non-linearity. For shipping that is one batched `IsolationForest.decision_function` call per day over all 16 coalitions, with the duration score bundled with `vessel_count`. Every full coalition reproduces the agent's emitted score (assertion).
- **Surrogate mapping.** Surrogate SHAP is mapped to agents and features via `FEATURE_AGENT_MAP`. `tanker_count` (read by the shipping agent) has no surrogate feature and gets 0; the market and shipping players otherwise share names with surrogate features.
- **Stability under noise.** E4 under score noise is exact-only: the noise is in agent-score space, which the surrogates do not take as input.

## 6. Reproducibility and versioning

- `python eval/explainability/make_figures.py` rebuilds everything (≈10 min); add `--reuse` to redraw from the existing CSVs.
- **Tracked:** code, tests, this report, `thesis/figures/ch4/*.caption.txt`.
- **Untracked (regenerated):** `aer/`, `*.csv`, `*.json`, the PNG/PDF figures.
- `.gitignore` ignores the contents of `thesis/` rather than the directory, which lets the caption exception work. The local `.git/info/exclude` entry was narrowed the same way (`thesis/*`).
