# Explainability report (XAI-02, final)

Every number in this report is reproducible with:

```
python eval/explainability/make_figures.py
```

That command regenerates the AERs, scorecard, backfill (from cache), precedents, triage, METRIC 8, narratives and all figures. Add `--reuse` to redraw figures only. Branch `xai-02`; baseline `main` @ `b2306ae`.

## Summary (headline findings)

**1. Exact attribution is faithful; the deployed surrogate is not.**
- Exact agent Shapley values reproduce the composite to ≤ 1.1e-16 and remove the score departure fastest: deletion AUC 0.06–0.17 vs 0.50 random.
- The production SHAP surrogate has R² −0.32 to −4.39 on the real test windows and ranks agents at chance (0.49–0.59). It names the exact top agent on 1–42% of alert days.
- A surrogate refitted on real data narrows but does not close the gap (§4).

**2. Explanation was valuable as a diagnostic.** It exposed:
- the dashboard explaining a different (seed-42) row than it plotted (§9–10);
- a retriever returning wrong-region, duplicated and post-alert documents behind a non-selective 0.55 similarity floor (§7.3);
- a market agent saturated at its ceiling on ~30% of days (§4.3);
- a label that the composite does not track in Hormuz (F13).

**3. Explanation did not yield a trust signal.**
- Explanation features lose to the score alone in leave-one-region-out triage.
- The only Holm-surviving effect (Panama agreement, AUC 0.89) reverses in Hormuz (§8).
- The noise band does not separate TP from FP (E7).
- FP alerts show no more conflict evidence than calm days (lift 1.07 [0.97, 1.15]; §7.4, §11).

**4. Decision value (SYNTHETIC).**
- Canonical METRIC 8 (exact driver): hand-tuned 0.937 accuracy / 0.708 macro-F1.
- The optimized weights fall below the naive baseline (0.578 vs 0.797).
- The surrogate-driven badge never recommended rerouting (§9).

**5. Narrative.** It is faithful by construction (E9 = 1.00). The claim checker catches 96.9% of single-number corruptions (§10).

**6. Pending:** E8 precision@3 needs the author's annotation of `rag_annotation.csv` (0 of 60 pairs annotated on 2026-10-15).

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

**Primary outcome: null.** The pre-registered tag does not separate FP episodes from calm days.

#### 7.4b EXPLORATORY: stricter rule (post-hoc amendment `0c2cfff`, one run, `fp_strict_lift.csv`)

The rule was written after the null result and run once on the backfilled documents only:
- **ACLED:** only Battles, Explosions/Remote violence and Strategic developments count.
- **News:** a fixed conflict/shipping-attack inclusion list, with an exclusion list (shark, cyber, labour strike, …).

Lift is the FP-window rate ÷ calm-window rate (8-day windows), with a 1,000-resample bootstrap CI.

| Region | Counting docs | FP rate (n) | TP rate (n) | Calm rate (n) | Lift [95% CI] |
|---|---|---|---|---|---|
| Hormuz | 99 | 0.53 (34) | 0.40 (5) | 0.54 (439) | 0.99 [0.69, 1.32] |
| Bab el-Mandeb | 275 | 1.00 (10) | 0.86 (7) | 0.97 (155) | 1.03 [1.01, 1.05] |
| Panama | 15 | 0.07 (30) | 0.50 (8) | 0.07 (490) | 0.93 [0.00, 2.63] |
| Malacca | 53 | 0.45 (49) | — | 0.41 (543) | 1.09 [0.75, 1.47] |
| **Pooled** | 442 | 0.42 (123) | | 0.40 (1,627) | **1.07 [0.88, 1.27]** |

**Still null.**
- FP alerts are no more likely than ordinary days to coincide with conflict or shipping-attack reporting.
- Bab el-Mandeb's interval excludes 1 only because both rates sit at the ceiling (2023-25 is the Red Sea crisis throughout), so a lift of 1.03 has no practical meaning.
- The knowledge base therefore cannot be used to argue that false positives were "real risk without a traffic drop". The honest reading is that the FPs are not explained by contemporaneous conflict events in this corpus.

### 7.5 Annotation set (for the author)

- **Files:** `eval/explainability/rag_annotation.csv` (blind) and `rag_annotation_key.csv` (labels, similarities, query).
- **Content:** 60 (episode, document) pairs from 20 episodes (9 TP / 11 FP): Hormuz 6, Bab el-Mandeb 6, Panama 6, Malacca 2. Malacca has a single FP stratum; it was reduced from 3 to 2 episodes by the pre-registered rule.
- **Sources of the pairs:** SerpApi 52, PortWatch 5, static cases 3. ACLED events never reach the top 3: they are short, factual notes that embed far from a driver-phrased query.
- **To score:** fill `relevance` (0 / 1 / 2, scale in `PREREGISTERED_RULES.md` §B), then run `python eval/explainability/compute_precision_at_3.py`. It refuses to run while any `relevance` cell is blank.

### 7.6 Scorecard rows added (`scorecard.csv`, replacing the "pending" rows)

**E8 is precision@3 from the author's relevance annotation, and is pending** (`compute_precision_at_3.py`; lenient = relevance ≥ 1, strict = 2).

The similarity-based version is **dropped as a metric**. The 0.55 similarity floor is **non-selective**: every configuration, including the production retriever with its wrong-region, duplicated and future-dated hits, has a hit above it in 100% of episodes (§7.3). Passing it says nothing about relevance. The old rows remain in `scorecard.csv` only as `E8-diag` (similarity-floor grounding: 0.99–1.00 in every region), marked diagnostic.

| Metric | Hormuz | Bab el-Mandeb | Panama | Malacca |
|---|---|---|---|---|
| E8 precision@3 | pending | pending | pending | pending |
| E5 top exact agent ∈ precedents' `primary_agents` (episode) | 0.62 [0.46, 0.77] | 0.41 [0.28, 0.54] | **0.00** | 0.02 [0.00, 0.07] |

- **E5 is a weak measure here.** By schema, backfilled news carries `primary_agents = news_sentiment`, so it can only match news-led alerts. Panama's 0.00 is this artefact, not a disagreement.
- The without-duration variant changes these by ≤ 0.04.

### 7.7 Incidents

- **SerpApi failures:** a read timeout and a double HTTP 503 interrupted the backfill. Both were handled by pre-registered amendments (`d8cdc62`, `ea1e5bb`), and the cache meant no request was paid for twice.
- **Exposed API key:** the 503's `HTTPError` text contained the request URL, **including the SerpApi API key**, which was written to a local task log. That log has been redacted, and errors are now raised without the URL.
  - **Status (2026-10-15):** the key was rotated by the author.
  - The old (revoked) key remains only in the author's `.env` (to be replaced by the author) and in this session's Claude Code transcript.
  - No other project file, cache, eval DB or log holds it.

## 8. Phase 4: explanation-informed triage

Pre-registered in `PREREGISTERED_RULES.md` §C (commit `f89f445`). The data is real_eval; the unit is the Tier 5 alert episode (FP = 1).

- **TP episodes:** 5 / 7 / 8 (Hormuz / Bab el-Mandeb / Panama) with duration; 7 / 8 / 8 without. Malacca has none.
- **Hormuz small-n:** every Hormuz result carries the caveat "small-n (5 TP episodes with duration, 7 without; 22 TP days)".
- **Runtime:** the one-time C0 export took 43 s (bit-identity with the CSV: OK); the analysis took 10 s.

### 8.1 Primary indicators (first day of the episode), AUC for FP [95% CI], Holm-adjusted one-sided p

| Indicator (direction → FP) | Hormuz (5 TP / 34 FP) | Bab el-Mandeb (7 / 10) | Panama (8 / 30) |
|---|---|---|---|
| P1 top exact driver = geopolitical (1) | 0.41 [0.18, 0.66], p = 1.00 | 0.70 [0.55, 0.85], p = 0.11 | 0.50 (degenerate: no geopolitical-led episode) |
| P2 agreement (low) | 0.34 [0.18, 0.47], p = 1.00 | 0.42 [0.25, 0.61], p = 0.79 | **0.89 [0.74, 0.98], p = 0.0002** |
| P3 dispersion (high) | 0.59 [0.26, 0.87], p = 0.79 | 0.57 [0.29, 0.86], p = 0.67 | 0.49 [0.22, 0.78], p = 1.00 |
| *S0 score-only (low)* | *0.25 [0.05, 0.54]* | *0.37 [0.10, 0.66]* | *0.56 [0.29, 0.82]* |

Without duration the rows are near-identical; for example P2 is 0.43 / 0.41 / **0.89 (p = 0.0002)** and P1 is 0.32 / 0.64 / 0.50.

**Reading:**
- **One pre-registered result passes:** Panama P2 (agreement), in both variants.
- **It does not generalise.** The same indicator points the *wrong* way in Hormuz (0.34) and is null in Bab el-Mandeb.
- **The registered P1 hypothesis is null or reversed.** Geopolitical-led first days are *more* common among TP episodes in Hormuz (0.41; 0.32 without duration). Bab el-Mandeb's 0.70 does not survive Holm.

The agreement cut-off has two mechanical properties, reported and not altered:
- **Market never counts toward agreement.** Its train calm-day q0.90 is exactly 1.0, the agent's ceiling (§4.3), so its score can never exceed it.
- **Any non-zero natural-disaster score counts**, because that agent's calm q0.90 is 0.

### 8.2 Leave-one-region-out logistic model (primaries) vs score-only

| Held out | AUC primaries | AUC score-only | ΔAUC [paired 95% CI] |
|---|---|---|---|
| Hormuz | 0.35 [0.18, 0.53] | 0.75 [0.50, 0.95] | **−0.40 [−0.64, −0.16]** |
| Bab el-Mandeb | 0.53 [0.23, 0.81] | 0.63 [0.36, 0.91] | −0.10 [−0.59, 0.34] |
| Panama | **0.05 [0.00, 0.14]** | 0.56 [0.29, 0.83] | **−0.50 [−0.80, −0.21]** |
| *Without duration:* Hormuz / Bab el-Mandeb / Panama | 0.29 / 0.29 / 0.17 | 0.69 / 0.62 / 0.56 | −0.41 / −0.34 / −0.39 (Hormuz and Panama CIs exclude 0) |

**The explanation features transfer worse than the score alone, and in Panama they transfer inverted.**
- The model learns agreement's direction from regions where it is reversed and applies it to Panama, where it holds.
- The relationship between explanation features and alert correctness is **region-specific**; no cross-region trust rule is supported.

### 8.3 Malacca: share of 49 FP episodes flagged low-trust, vs the cost (share of the other regions' TP episodes also flagged)

| Rule | Malacca FP flagged (with / without dur.) | TP flagged elsewhere (cost) |
|---|---|---|
| P1 geopolitical-led | 0.59 [0.45, 0.73] / 0.59 | 0.15 [0.00, 0.30] / 0.26 |
| P2 agreement ≤ 1 | 0.84 [0.73, 0.94] / 0.84 | 0.60 [0.40, 0.80] / 0.61 |
| P3 dispersion > q0.75 | 0.10 [0.02, 0.20] / 0.10 | 0.20 [0.05, 0.40] / 0.22 |
| LORO primaries, P(FP) ≥ 0.5 | 0.82 [0.71, 0.92] / **0.31** | 0.90 [0.75, 1.00] / 0.91 |
| LORO score-only, P(FP) ≥ 0.5 | 0.71 [0.59, 0.84] / 0.41 | **1.00** / **1.00** |
| *X1 k80 (EXPLORATORY)* | *0.94 / 0.94* | *0.40 / 0.52* |
| *X2 surrogate disagrees (EXPLORATORY)* | *0.63 / 0.63* | *0.80 / 0.70* |
| *X4 noise robustness (EXPLORATORY)* | *0.88 / 0.88* | *0.85 / 0.83* |

- **P1 is the only rule that flags most Malacca FPs while sparing most TPs elsewhere** (59% vs 15%). It is a registered primary but did not reach significance in any labelled region, so the Malacca rate is descriptive, not evidence.
- **The two LORO models flag nearly every TP episode.** At the balanced-weight 0.5 cut-off they are not selective, and their Malacca rate collapses when the duration branch is removed (0.82 → 0.31).

### 8.4 Exploratory and retrospective (not interpreted)

**Secondary (`triage_results.csv`, tier "secondary"):**
- k80 0.70 / 0.58 / 0.67;
- surrogate disagreement 0.59 / 0.37 / 0.48;
- max precedent similarity 0.64 / 0.77 / 0.34;
- noise robustness 0.28 / 0.36 / 0.58 (Hormuz / Bab el-Mandeb / Panama, with duration).

**Retrospective (`triage_retrospective.csv`):**
- C1 (registered, episode-summed): 0.29 / 0.60 / 0.50. Hormuz is reversed: geopolitical leads its TP episodes.
- Episode length: 0.55 / 0.68 / 0.83.

### 8.5 Circularity control

Removing the shipping duration branch changes no primary conclusion:
- Panama P2 is identical (0.89, p = 0.0002);
- every other primary stays null;
- LORO stays worse than score-only.

The only material change is the Malacca LORO flag rate (0.82 → 0.31), which already failed on cost.

### 8.6 Answer to "does the explanation tell you when to trust the alert?"

**Not in a way that generalises.**
- One region-specific signal (Panama: FP alerts have fewer agents individually elevated) survives correction in both variants.
- Combining the primary indicators across regions is worse than the score alone.
- The registered geopolitical hypothesis does not hold in any labelled region.

With 5–8 TP episodes per region, the thesis may report the Panama result as a region-specific finding. It may not claim explanation-based triage.

## 9. Phase 5: decision value (**SYNTHETIC**, seed 44)

Pre-registered in `PREREGISTERED_RULES.md` §D (commit `f0f2253`). Source: `metric8_refresh.json`; runtime 24 s.

**These are synthetic results.** The "correct" actions are rule-derived from the injected scenario windows, not observed decisions.

### 9.1 METRIC 8 refreshed (current code, 5 agents) vs the stale 2026-07-30 values

| | Hand-tuned: refresh | Hand-tuned: stale | Optimized: refresh | Optimized: stale |
|---|---|---|---|---|
| Overall accuracy (365 days) | **0.948** | 0.866 | **0.578** | 0.490 |
| Macro-F1 (4 actions) | 0.855 | — | 0.726 | — |
| Recall: no_action / monitor / reroute / escalate | 0.96 / 0.77 / 0.83 / 1.00 | — | 0.52 / 1.00 / 1.00 / 1.00 | — |
| Scenario A / B / C accuracy | 0.73 / 0.95 / 0.82 | 0.07 / 1.00 / 0.09 | 1.00 / 1.00 / 1.00 | 0.13 / 1.00 / 0.18 |
| Per-case (10 historical cases) | 0.70 | 0.70 | 0.70 | 0.70 |
| Naive 2-σ baseline accuracy | 0.797 | 0.797 | 0.797 | 0.797 |

- **Hand-tuned:** accuracy is dominated by `no_action` (318 of 365 correct actions). Macro-F1 (0.855) is the fairer summary.
- **Optimized:** monitor recall 1.00 comes at the cost of `no_action` recall 0.52. It over-monitors 153 calm days, and its accuracy is below the naive baseline.
- **Why the stale numbers differ:** they predate the routing removal and the shipping level-shift change, and must not be cited.

### 9.2 Registered comparison: P0-surrogate vs P0-exact

Same day records; only `top_agent` differs.

Per method:

| Weight mode | Method | Accuracy | Macro-F1 | Recall: reroute | Recall: monitor |
|---|---|---|---|---|---|
| Hand-tuned | P0-surrogate | 0.940 | 0.660 | **0.00** (0/6) | 0.85 |
| Hand-tuned | P0-exact | 0.937 | 0.708 | 0.17 (1/6) | 0.77 |
| Optimized | P0-surrogate | 0.564 | 0.474 | **0.00** (0/5) | 1.00 |
| Optimized | P0-exact | 0.578 | **0.726** | **1.00** (5/5) | 1.00 |

Pairwise (P0-exact vs P0-surrogate):

| Weight mode | Days the top agent differs | Days the action differs | Discordant days (exact right / surrogate right) | McNemar exact p |
|---|---|---|---|---|
| Hand-tuned | 295 / 365 | 3 | 2 / 1 | **1.00** |
| Optimized | 294 / 365 | 5 | 5 / 0 | **0.063** |

**The two attributions disagree on the top agent on ~80% of days.**
- The surrogate names `news_sentiment` on 214 days and `shipping` on 90. Exact Shapley names `shipping` on 170–201 days.
- This is the seed-42 / seed-44 mismatch plus the surrogate's own error.

**But the decision rule consults `top_agent` only on HIGH, non-sustained days** (7 and 5 days), so the effect on actions is confined to 3–5 days, and neither McNemar test is significant.
- **The visible consequence:** the surrogate never recommends `reroute` (reroute recall 0.00 in both modes), because it never names shipping on those days.
- With exact attribution, reroute recall is 1.00 (optimized) or 0.17 (hand-tuned).

The stock METRIC 8 rule (argmax of weight × score, no baseline) gets 0.83 hand-tuned reroute recall. Exact Shapley measures each agent *against its calm level*, and on those days shipping's departure from calm is not the largest, so P0-exact names another agent.
- The registered comparison is surrogate vs exact. The stock rule is reported for context, not tested.

### 9.3 Not run

**P1** (downgrade on high uncertainty), **P2** (escalate needs a temporally valid precedent ≥ 0.6; reroute needs shipping as top exact driver) and the **simulated-reliance analysis** were **not run**. Their inputs tested null or non-selective in Phases 2–4:
- **Uncertainty:** the E7 AUCs span 0.5 (§4.1); noise robustness is exploratory and inconsistent (§8.4).
- **Precedent similarity:** the 0.55 floor is non-selective (§7.3), and precedent evidence does not separate FP alerts from calm days (§7.4, §7.4b).

Simulated reliance (RAIR/RSR under assumed operator accuracies) is listed under **Outlook**: it needs a human-subject estimate of operator accuracy that this thesis does not have.

## 10. Phase 6: narrative and dashboard

### 10.1 Narrative (`src/explainability/narrative.py`)

A deterministic template turns an AER into 6–8 sentences. Each sentence is one claim tied to named AER fields:
- status vs threshold;
- the two leading exact contributions;
- the within-agent terms;
- the agreement bonus and cap (Path C);
- the noise range;
- the counterfactual (or "already under the line");
- the closest pre-alert precedent;
- a SYNTHETIC label where applicable.

There are two styles: `decimal` (AERs, Analysis View) and `points` (×100 whole numbers, for the Decision View's no-raw-decimals rule).

`check_claims` verifies any text against the AER without knowing the template. It checks:
- numbers, bound to the claim kind recognised from the sentence's wording (±1% or half a displayed unit);
- agent names and their rank order;
- above/below vs the margin's sign;
- quoted titles and ISO dates.

**LLM paraphrase:** `narrate(…, paraphrase=…)` accepts a rewrite only if it scores 1.0, otherwise it keeps the template. It is not used: `anthropic` stays uninstalled and no API call is made.

**E9, narrative faithfulness** (`scorecard.csv`, real-data AERs):

| Region | AERs | Faithfulness (supported / claims) | Checker sensitivity (single-number corruptions rejected) |
|---|---|---|---|
| Hormuz | 181 | **1.000** | 1,405 / 1,448 (97.0%) |
| Bab el-Mandeb | 495 | **1.000** | 3,841 / 3,960 (97.0%) |
| Panama | 197 | **1.000** | 1,535 / 1,576 (97.4%) |
| Malacca | 188 | **1.000** | 1,445 / 1,504 (96.1%) |

- **Faithfulness of 1.000 holds by construction:** the template can only restate the record. It holds in both styles (2 × 1,061 narratives).
- **The informative number is the checker's sensitivity**, 96.9% overall. Every 2–3-decimal number was shifted by +0.07. The misses are shifts that land on another value the same sentence may cite (e.g. band low + 0.07 ≈ band high).
- **Other corruption checks:** on the full set, it rejected 1,052/1,052 agent-order swaps, 1,059/1,059 invented titles and 1,061/1,061 above/below flips. These ran during development and are repeated in `tests/test_xai_narrative.py`.

### 10.2 Decision View (synthetic seed 44, Path C)

**The seed-42 / seed-44 mismatch is fixed.** The page used to take its "top driver" from the SHAP surrogate on row *d* of the surrogate's own seed-42 training year. It now builds an exact AER for the **same seed-44 day it plots** (`core.dashboard_aer`):
- exact Shapley values of the route's composite (renormalised mean, agreement bonus, cap) over the route's active agents;
- seed-42 calm baselines and noise scales (`synthetic_calibration`);
- the alert line = `risk_medium`, where the status word turns High;
- one region-matched precedent dated before the day's source date, from the production KB (read only).

That record drives the map's top-driver label, the recommended action, the peak explanation and the new card.

**New card, "Why this status"** (right column, above "Recommended action"; whole numbers only, no scrolling). Rendered for the Hormuz Westbound TSS corridor on day 155 (source date 2025-06-04), inside Scenario B; the page's default peak day is 153:

> ⚠ **Critical**
> Risk 100 of 100 · alert line 51
> Main drivers: Geopolitical +40% · Shipping +33%
> Plausible range 86–100
> No single signal can bring it under the alert line on its own.
> Earlier precedent: "Iran Hormuz Mine Threat and Naval Exercises" (2010-07-01).
> *SYNTHETIC evaluation data. Driver shares are exact contributions relative to a calm day; the range reflects day-to-day input noise.*

On an ordinary day (day 70) the counterfactual line reads "Clears if shipping drops 70 points (below its usual calm level)". On a calm day (day 20) it reads "No signal is above its usual calm level · Already under the alert line".

**Peak explanation and spike panel:**
- **"Explain the marked peak"** now shows the `points` template (faithful by construction) instead of the surrogate's drivers.
- **The spike panel** uses the same template (`llm_explanations.explain_spike(…, aer=…)`, source `template`), and lists exact contribution shares instead of raw two-decimal agent scores. Those raw decimals violated the page's own rule, but rendered only after a click, so the test never saw them.
- **Legacy paths** (no AER) are unchanged and still tested.

### 10.3 Analysis View

- **New section 2b, "Exact alert explanation":**
  - a day slider (deep-linked from the Decision View);
  - a waterfall: calm-day composite → each agent's linear exact contribution → agreement bonus → cap → score, with the alert line;
  - the agreement-bonus share of the departure from calm, the efficiency residual, and the decimal narrative;
  - the full AER as JSON;
  - the region's real-data scorecard rows (E1/E2/E3/E5/E6/E9; exact vs surrogates).
- **The old per-day SHAP expander** is relabelled "surrogate (trained on seed 42), comparator only", with a caption stating it explains the seed-42 row (out-of-time R² 0.53; 18–42% top-agent agreement on real alerts). Its stale "20-feature, R² > 0.99" export caption is corrected.

### 10.4 Separate commit

`ee16c70` replaces "of six detection agents" in the composed narrative and the LLM prompt (`core.py`) with the actual agent count.

### 10.5 Tests

Full suite: **512 passed** (2026-10-15). That covers the Decision View AppTest's no-raw-decimals and forbidden-terms checks with the new card, the narrative and checker tests, and the frozen-score test.

### 10.6 Post-review fixes (2026-10-15)

**Card wording:**
- "Similar past event (historical analogue)" replaces "Earlier precedent".
- An infeasible counterfactual now reads "Would need {agent} to fall below its normal level, so not realistic on its own".

**Precedent date filter.** The dashboard's lookup uses the wrapper (region filter, date filter, de-duplication) with the cut-off on the series' **source timestamp** (2025-01-01 + day − 1), **not the display axis**.
- The display axis is a rolling window ending today.
- Filtering on it would admit 2026 articles (e.g. the real 2026 Hormuz closure coverage) as "analogues" for synthetic 2025 signals, which the page's own rules forbid.
- The card's caption states this.

## 11. Phase 7: figures (`thesis/figures/ch4/`, PNG 300 dpi + PDF + `.caption.txt`)

| Figure | Content | Data |
|---|---|---|
| F1 | Explanation pipeline schematic; surrogate as comparator | — |
| F2 | Alert anatomy waterfall: calm composite → exact agent φ → score vs threshold; highest-margin TP per region (Malacca FP) | real_eval, D |
| F3 | Attribution timeline: stacked exact φ over the test window, alert line, labelled days, TP/FP episodes | real_eval, D |
| F4 | Deletion and insertion curves (agent level): exact vs production surrogate vs random; episode AUCs in the titles | real_eval, D |
| F5 | Surrogate honesty: R² on its own synthetic year, out-of-time R² on real data (production vs refit), local error | real_eval / synthetic_seed42 |
| F6 | Compactness (k80 agents / features) and stability under noise | real_eval, D |
| F7 | Retrieval diagnostics before/after (temporal validity, region match, duplicates); precision@3 panel **pending annotation** | real_eval |
| F8 | Triage forest plot: primary indicators + score-only baseline, with/without duration, Holm and minimum-n flags | real_eval |
| F9 | Uncertainty (agent dispersion) TP vs FP, **null result** | real_eval |
| F10 | P0-surrogate vs P0-exact: macro-F1 and reroute recall | **SYNTHETIC** (seed 44), C |
| F12 | Scorecard heatmap, region × E1–E9 (E8 pending) | real_eval |
| F13 | Hormuz case: shock vs duration branch, max() winner, exact shipping φ | real_eval, D |
| F14 | Top exact driver of TP vs FP episodes | real_eval, D |
| F15 | Market saturation: score distribution per region, calm baseline | real_eval |
| F16 | FP characterisation: lift vs calm days, pre-registered rule (primary) and stricter rule (EXPLORATORY), line at 1.0 | real_eval |

F11 (simulated reliance) was dropped.

**Notes:**
- **F4 deviation:** curves are shown for exact, the production surrogate and random. The refitted surrogate's per-day order was not persisted, so its AUCs appear in the caption only.
- **F16 addition:** the pre-registered rule's lift with a bootstrap CI was computed for this figure (`fp_lift_primary.csv`): pooled 1.07 [0.97, 1.15]. It is consistent with the §7.4 shares.
- **F13 caption** is generated from the data. The duration branch wins on 82/133 labelled days, but shipping's exact φ is no larger on labelled days (0.044) than unlabelled (0.052), and labelled days are alerted less often (16.5% vs 26.6%).

Runtime: 45 s for all figures with `--reuse`.

## 12. Limitations

**Data and labels:**
- **Label circularity.** The label is the shipping level-shift statistic. Every result was repeated without the duration branch; nothing changed.
- **Small n.** 5 / 7 / 8 TP episodes (Hormuz has 5). Malacca has none, so it is an FP harness only.
- **Calibration** is not estimable (0 train positives in 3 regions). The score-noise band is not a probability.

**Scope:**
- **No human-subject study.** Every explainability metric is functionally grounded.
- **SYNTHETIC decision evaluation.** METRIC 8 correct actions are rule-derived from injected scenarios.

**Retrieval:**
- **Single annotator** (the author) for E8, blind to TP/FP. E8 is pending.
- **Knowledge-base coverage** before the backfill was zero for 2023-25 outside Bab el-Mandeb. The backfill is bounded (15 docs per region-month) and alert-blind, and its permissive pre-registered tag does not discriminate (§7.4).

**Pre-registered rule properties:**
- **Agreement cut-off quirks** (§8.1): market can never count (its calm q0.90 = 1.0); any non-zero disaster score counts.

**The registry/CSV discrepancy:**
- The method-comparison tiers (and therefore Path D and every explanation of it) include agents that `src/core/regions.py` marks **passive** for a region: **geopolitical in Panama, market in Malacca, and natural disaster in Bab el-Mandeb**. `scripts/run_method_comparison.py` builds tiers from the data columns present, not from `active_agents()`.
- Explanations deliberately use the CSV's agent sets so that their numbers match the reported detection results. A registry-faithful composite would drop those three terms from the respective regions' attributions.

**Comparators:**
- **Surrogate comparators.** The production surrogate is evaluated as deployed (synthetic-trained). The refit is one recipe (the same random forest); other surrogate families were not tried.

## 13. Final METRIC 9 scorecard (with duration; episode means, 95% CIs; TP episodes 5 / 7 / 8)

| ID | Metric | Method | Hormuz | Bab el-Mandeb | Panama | Malacca |
|---|---|---|---|---|---|---|
| E1 | Efficiency residual (max) | exact | 5.6e-17 | 5.6e-17 | 1.1e-16 | 8.3e-17 |
| E1 | Local error \|f−g\| | surrogate (prod) | 0.23 [0.22, 0.24] | 0.23 [0.22, 0.25] | 0.04 [0.04, 0.05] | 0.19 [0.18, 0.20] |
| E1 | R² on real test window | surrogate (prod) | −0.37 | −4.39 | −1.10 | −0.32 |
| E2 | Agent deletion AUC ↓ | exact | **0.15** [0.12, 0.18] | **0.12** [0.09, 0.16] | **0.06** [0.01, 0.09] | **0.17** [0.14, 0.19] |
| E2 | Agent deletion AUC ↓ | surrogate (prod) | 0.53 [0.49, 0.58] | 0.53 [0.48, 0.59] | 0.59 [0.56, 0.63] | 0.49 [0.44, 0.53] |
| E2 | Agent deletion AUC ↓ | surrogate (refit) | 0.20 [0.17, 0.23] | 0.34 [0.25, 0.41] | 0.13 [0.07, 0.20] | 0.23 [0.21, 0.24] |
| E3 | k80 agents | exact | 2.00 [1.90, 2.10] | 2.05 [1.93, 2.21] | 1.65 [1.50, 1.79] | 1.94 [1.86, 2.02] |
| E3 | k80 agents | surrogate (prod) | 2.35 [2.22, 2.50] | 2.26 [2.17, 2.36] | 2.59 [2.46, 2.72] | 2.42 [2.31, 2.53] |
| E4 | ρ under score noise σ = 0.05 | exact | 0.97 [0.96, 0.98] | 0.97 [0.96, 0.98] | 0.86 [0.84, 0.89] | 0.95 [0.93, 0.97] |
| E5 | Top agent = exact top | surrogate (prod) | 0.34 [0.23, 0.44] | 0.18 [0.07, 0.32] | 0.01 [0.00, 0.03] | 0.42 [0.31, 0.54] |
| E5 | Top agent = exact top | surrogate (refit) | 0.84 [0.76, 0.92] | 0.53 [0.38, 0.66] | 0.94 [0.86, 1.00] | 0.91 [0.86, 0.96] |
| E6 | Feasible single-agent counterfactual | exact | 0.97 [0.94, 0.99] | 0.97 [0.95, 0.99] | 1.00 [1.00, 1.00] | 0.97 [0.94, 0.99] |
| E7 | AUC of noise band (FP vs TP) | exact | 0.38 [0.12, 0.70] | 0.87 [0.66, 1.00] | 0.57 [0.38, 0.73] | n/a (no TP) |
| E8 | Precision@3 (annotation) | wrapper | pending | pending | pending | pending |
| E9 | Narrative faithfulness | template | 1.00 | 1.00 | 1.00 | 1.00 |

The without-duration variant changes no row by more than a few hundredths (`scorecard.csv`, `variant == without_duration`).

## 14. Commits (branch `xai-02`)

| Commit | Content |
|---|---|
| `f0191ac` | Context report |
| `4492b88` | Phase 1 |
| `fb7159b` / `9de7b4e` | Phase 2 |
| `ae8c7b7` | Pre-registration (backfill, retrieval, C1) |
| `d8cdc62` / `ea1e5bb` | Backfill amendments |
| `2b14011` | Phase 3 |
| `0c2cfff` / `41c9746` | Exploratory FP rule, E8 relabel |
| `f89f445` | §C pre-registration |
| `700d7c3` | Phase 4 |
| `f0f2253` / `1811dc9` | Phase 5 |
| `ee16c70` | "Six agents" text fix |
| `147cc1d` / `12aa383` | Phase 6 |
| `720ebba` | Post-review fixes |
| `e42bb14` | Phase 7 |
| (this commit) | Phase 8 |
