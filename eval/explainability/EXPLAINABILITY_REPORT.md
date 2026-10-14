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

### 4.4 FP characterisation, production KB only (superseded by §7.4)

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

- `python eval/explainability/make_figures.py` rebuilds everything (≈15 min); add `--reuse` to redraw from the existing CSVs.
  - The backfill step is served from `eval/explainability/backfill_raw/` and spends no API quota.
  - On a machine without that cache it needs `SERPAPI_API_KEY` and ACLED credentials.
- **Tracked:** code, tests, this report, `thesis/figures/ch4/*.caption.txt`.
- **Untracked (regenerated):** `aer/`, `*.csv`, `*.json`, the PNG/PDF figures, `backfill_raw/`, `data/knowledge_base/.chromadb_xai/`.
- `.gitignore` ignores the contents of `thesis/` rather than the directory, which lets the caption exception work. The local `.git/info/exclude` entry was narrowed the same way (`thesis/*`).

## 7. Phase 3: precedent grounding

Rules are pre-registered in `PREREGISTERED_RULES.md` §A–B (commit `ae8c7b7`; amendments `d8cdc62`, `ea1e5bb`). The data is real_eval; the unit is the Tier 5 alert episode. **TP episodes: 5 / 7 / 8 (Hormuz / Bab el-Mandeb / Panama).**

### 7.1 Source validation (2026-10-14)

| Source | Reaches 2023-08 → 2025-08? | Evidence |
|---|---|---|
| ACLED event-level | **Yes**, account active | Dated events for all four regions, e.g. Yemen 2024-01-10 (Houthi-sponsored demonstrations), Iran 2024-04-11 (Isfahan farmers' rally), Panama 2023-10-03 (SUNTRACS protest), Malaysia 2024-06-18 (Bersih vigil). |
| SerpApi Google News, `tbs=cdr` | **Yes** | 250/250 searches available. Jan-2024 test: "Shipping oil through troubled waters" (ASPI, 2024-01-18); "How Houthi Attacks in the Red Sea Threaten Global Shipping" (CFR, 2024-01-12); "Oil tankers divert from Red Sea after US, UK strikes in Yemen" (Reuters, 2024-01-12). |
| GDELT DOC 2.0 | **No, skipped** | HTTP 429 on the first request, even with spacing. The DOC API is also documented to search only a rolling ~3-month window, so it could not reach 2023-25 regardless. |

### 7.2 Backfill coverage (`backfill_coverage.csv`): news + ACLED documents per region × month

Written to the **eval copy only**: `data/knowledge_base/.chromadb_xai`, collection `backfill_2023_25`, 1,338 documents tagged `backfill: true`. Production `.chromadb` was only read, once, to make that copy.

- **Quota:** 101 SerpApi searches used (100 + 1 validation; 149 left this month), 250 ACLED requests.
- **Fetch status:** every region-month fetched `ok` (no `fetch_failed`).
- **Cap:** the 15-per-month cap is always hit by news, which is capped at 10 because Google returns at most 10 per page. ACLED fills 0–7 slots, depending on how many maritime-filtered events the month had. Hormuz has 91 candidates in 25 months; Bab el-Mandeb has 3,696.

| Month | Hormuz | Bab el-Mandeb | Panama | Malacca |
|---|---|---|---|---|
| 2023-08 | 10 + 4 | 10 + 5 | 10 + 2 | 8 + 7 |
| 2023-09 | 10 + 2 | 10 + 5 | 10 + 1 | 7 + 7 |
| 2023-10 | 10 + 1 | 9 + 6 | 10 + 2 | 10 + 4 |
| 2023-11 | 10 + 2 | 10 + 5 | 10 + 1 | 9 + 6 |
| 2023-12 | 10 + 2 | 10 + 5 | 10 + 5 | 10 + 2 |
| 2024-01 | 10 + 5 | 10 + 5 | 10 + 0 | 10 + 5 |
| 2024-02 | 10 + 2 | 10 + 5 | 10 + 3 | 10 + 0 |
| 2024-03 | 10 + 1 | 10 + 5 | 10 + 1 | 8 + 3 |
| 2024-04 | 10 + 3 | 10 + 5 | 10 + 0 | 9 + 6 |
| 2024-05 | 10 + 1 | 10 + 5 | 10 + 1 | 9 + 4 |
| 2024-06 | 10 + 3 | 10 + 5 | 10 + 2 | 7 + 5 |
| 2024-07 | 10 + 1 | 10 + 5 | 10 + 2 | 9 + 3 |
| 2024-08 | 10 + 2 | 10 + 5 | 10 + 0 | 7 + 6 |
| 2024-09 | 10 + 1 | 10 + 5 | 10 + 1 | 10 + 5 |
| 2024-10 | 10 + 3 | 10 + 5 | 10 + 1 | 9 + 6 |
| 2024-11 | 10 + 0 | 10 + 5 | 10 + 2 | 6 + 5 |
| 2024-12 | 10 + 0 | 10 + 5 | 10 + 3 | 8 + 6 |
| 2025-01 | 10 + 3 | 10 + 5 | 10 + 5 | 9 + 6 |
| 2025-02 | 10 + 3 | 10 + 5 | 10 + 5 | 8 + 7 |
| 2025-03 | 10 + 5 | 10 + 5 | 10 + 2 | 10 + 5 |
| 2025-04 | 10 + 5 | 10 + 5 | 10 + 5 | 10 + 5 |
| 2025-05 | 10 + 5 | 10 + 5 | 10 + 2 | 10 + 5 |
| 2025-06 | 10 + 5 | 10 + 5 | 10 + 3 | 10 + 5 |
| 2025-07 | 10 + 5 | 10 + 5 | 10 + 0 | 10 + 5 |
| 2025-08 | 10 + 4 | 10 + 5 | 10 + 0 | 10 + 5 |
| **Total** | 250 + 68 = **318** | 249 + 126 = **375** | 250 + 49 = **299** | 223 + 123 = **346** |

### 7.3 Retrieval diagnostics (`rag_diagnostics.csv`): one query per episode, on its first day

| Measure | Configuration | Hormuz | Bab el-Mandeb | Panama | Malacca |
|---|---|---|---|---|---|
| Temporal validity (doc date < alert date) | production | 0.97 | 0.92 | 1.00 | 0.99 |
| | wrapper (prod KB) / wrapper + backfill | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Region match | production | **0.00** | 0.75 | **0.00** | **0.00** |
| | wrapper / wrapper + backfill | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Duplicate rate in top 3 | production | **0.33** | **0.33** | **0.36** | **0.33** |
| | wrapper / wrapper + backfill | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Any hit ≥ 0.55 | all three | 1.00 | 1.00 | 1.00 | 1.00 |
| Triggered (composite ≥ 0.65) | production | 0.31 | 0.24 | **0.00** | 0.14 |
| Similarity median [p10, p90] | production | 0.64 [0.60, 0.64] | 0.64 [0.60, 0.64] | 0.60 [0.60, 0.62] | 0.62 [0.60, 0.64] |
| | wrapper | 0.69 [0.66, 0.72] | 0.69 [0.63, 0.71] | 0.71 [0.65, 0.71] | 0.65 [0.64, 0.67] |
| | wrapper + backfill | 0.73 [0.70, 0.76] | 0.69 [0.63, 0.71] | 0.80 [0.75, 0.83] | 0.74 [0.71, 0.75] |

What the production retriever actually returns:
- **Hormuz:** for the 2023-09-12 alert, a 2021 Suez Canal article tagged `bab_el_mandeb`, stored and returned **twice** (the 0.33 duplicate rate), plus the global 2021 COVID port-congestion case.
- **Every alert gets the same answer.** Its query depends only on which agents exceed 0.4, and the text always names the Strait of Hormuz, so different alerts receive identical precedents.
- **Panama is never triggered.** No episode reaches the 0.65 gate.

The wrapper removes all three defects (region, date, duplicates) by construction, and the backfill raises similarity in every region except Bab el-Mandeb, which was already covered.

"Any hit ≥ 0.55" is 1.00 for every configuration, including the defective one, so the 0.55 floor is not selective. Relevance therefore has to come from annotation (§7.5), not from the similarity score.

### 7.4 FP characterisation (`fp_characterisation.csv`): tags per pre-registration §B

| Region | FP episodes | KB | risk_without_traffic_drop | unexplained | no_coverage |
|---|---|---|---|---|---|
| Hormuz | 34 | production | 0 | 0 | 34 |
| | | production + backfill | 31 | 2 | 1 |
| Bab el-Mandeb | 10 | production | 0 | 0 | 10 |
| | | production + backfill | 10 | 0 | 0 |
| Panama | 30 | production | 0 | 0 | 30 |
| | | production + backfill | 21 | 8 | 1 |
| Malacca | 49 | production | 0 | 0 | 49 |
| | | production + backfill | 45 | 3 | 1 |
| **All** | **123** | production + backfill | **107** | **13** | **3** |

By top exact driver (backfill KB):
- geopolitical-led FP episodes: 40 / 3 / 1 (risk / unexplained / no_coverage);
- shipping-led FP episodes: 67 / 10 / 2.

**This tag does not discriminate, so it must not be read as "the FPs were real risk".** An exploratory control (*not* pre-registered; `fp_characterisation_base_rate.csv`) applies the same rule to 8-day windows ending on calm, non-alert days:

| Region | FP episode windows | TP episode windows | Calm-day windows |
|---|---|---|---|
| Hormuz | 0.82 (n = 34) | 1.00 (n = 5) | 0.85 (n = 439) |
| Bab el-Mandeb | 1.00 (10) | 1.00 (7) | 1.00 (155) |
| Panama | 0.63 (30) | 0.63 (8) | 0.51 (490) |
| Malacca | 0.90 (49) | — | 0.87 (543) |

*Values are the share of windows tagged `risk_without_traffic_drop`.*

**Why it fails:**
- With uniform monthly coverage of conflict-heavy coastal states, almost any 8-day window contains a matching document.
- The pre-registered rule is permissive. Any maritime-filtered ACLED event counts, including protests and "strategic developments"; 22 of the 107 tags rest on ACLED alone.
- The keyword pattern also matches non-geopolitical items. Bab el-Mandeb E002 and E003 match "Shark attacks tourist … Red Sea".

What the backfill *does* establish is coverage: `no_coverage` falls from 123/123 to 3/123. The production-only result in §4.4 was a KB artefact.

### 7.5 Annotation set (for the author)

- **Files:** `eval/explainability/rag_annotation.csv` (blind) and `rag_annotation_key.csv` (labels, similarities, query).
- **Content:** 60 (episode, document) pairs from 20 episodes (9 TP / 11 FP): Hormuz 6, Bab el-Mandeb 6, Panama 6, Malacca 2. Malacca has a single FP stratum; it was reduced from 3 to 2 episodes by the pre-registered rule.
- **Sources of the pairs:** SerpApi 52, PortWatch 5, static cases 3. ACLED events never reach the top 3: they are short, factual notes that embed far from a driver-phrased query.
- **To score:** fill `relevance` (0 / 1 / 2, scale in `PREREGISTERED_RULES.md` §B), then run `python eval/explainability/compute_precision_at_3.py`. It refuses to run while any `relevance` cell is blank.

### 7.6 Scorecard rows added (`scorecard.csv`, replacing the "pending" rows)

| Metric | Hormuz | Bab el-Mandeb | Panama | Malacca |
|---|---|---|---|---|
| E8 grounding: ≥ 1 region-matched, pre-alert precedent ≥ 0.55 (episode) | 1.00 | 0.99 [0.96, 1.00] | 1.00 | 1.00 |
| E5 top exact agent ∈ precedents' `primary_agents` (episode) | 0.62 [0.46, 0.77] | 0.41 [0.28, 0.54] | **0.00** | 0.02 [0.00, 0.07] |

- **E8 is near-trivially high** because the floor is not selective (§7.3). Its informative counterpart is precision@3, pending annotation.
- **E5 is a weak measure here.** By schema, backfilled news carries `primary_agents = news_sentiment`, so it can only match news-led alerts. Panama's 0.00 is this artefact, not a disagreement.
- The without-duration variant changes these by ≤ 0.04.

### 7.7 Incidents

- **SerpApi failures:** a read timeout and a double HTTP 503 interrupted the backfill. Both were handled by pre-registered amendments (`d8cdc62`, `ea1e5bb`), and the cache meant no request was paid for twice.
- **Exposed API key:** the 503's `HTTPError` text contained the request URL, **including the SerpApi API key**, which was written to a local task log. That log has been redacted, and errors are now raised without the URL. **The key should be rotated.**
