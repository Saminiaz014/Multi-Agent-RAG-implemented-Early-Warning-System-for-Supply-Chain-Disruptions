# Thesis Writing Brief — Complete Project State

**Date of record:** 2026-09-02
**Branch:** `phase-11-4region` (68 commits) · **HEAD:** `88c4274`
**Test suite:** 420 passed, 0 failed (5m22s)
**Scale:** 136 tracked files, 60 source modules, 25 test modules

This document is the single source of truth for writing up the thesis. Every
number in it was measured in this repository, not estimated. Where a number
should not be cited as evidence of skill, that is stated at the point of use.

Companion: `docs/SCORING_REFERENCE.md` — deeper per-agent formulas.

---

# PART I — WHAT THE SYSTEM IS

## 1.1 Title and claim

**Multi-Agent RAG-Implemented Early Warning System for Supply Chain Disruptions.**

A decision support system that detects, explains, and contextualises maritime
chokepoint disruptions. The claim is *not* "we detect disruptions better than
existing methods." The measured claim is narrower and defensible:

> A multi-agent architecture with explicit per-domain weighting produces
> **interpretable, auditable, decision-ready alerts** — attribution to a named
> domain, a SHAP feature breakdown, and a retrieved historical precedent —
> which single-model detectors cannot produce, at a detection cost that this
> evaluation quantifies honestly rather than hides.

That framing matters because the detection numbers (Part V) do not support a
stronger claim, and the evaluation was deliberately built to expose that.

## 1.2 Domain

Four maritime chokepoints, 2019–2026:

| Region | Display | Lat/Lon | Documented driver |
|---|---|---|---|
| `hormuz` | Strait of Hormuz | 26.50 / 56.50 | Geopolitical — sanctions, military, ~20% of global oil trade |
| `bab_el_mandeb` | Bab el-Mandeb | 12.58 / 43.33 | Security campaign — Houthi attacks, Cape of Good Hope diversion |
| `panama` | Panama Canal | 9.08 / −79.68 | Hydrological — Gatún Lake drought |
| `malacca` | Strait of Malacca | 2.50 / 101.80 | None documented in window (used as a control) |

## 1.3 Sub-research questions the code actually answers

| SRQ | Question | Answered by |
|---|---|---|
| SRQ1 | Can multi-domain signals detect chokepoint disruption? | `scripts/run_method_comparison.py` |
| SRQ2 | Does agent diversity add value over fewer agents? | Tier ablation (same script) + `notebooks/evaluation.py` METRIC 3 |
| SRQ3 | Can the score be explained faithfully? | `src/explainability/shap_explainer.py`, METRIC 2 |
| SRQ4 | Does weight optimization improve on hand-tuning? | `src/optimization/`, METRIC 5 |
| SRQ5 | Would a decision-maker be led to the correct action? | `src/evaluation/decision_effectiveness.py`, METRIC 8 |

---

# PART II — ARCHITECTURE

## 2.1 Pipeline

```
connector (ingestion)      raw domain frame, daily index
    ↓
agent.fit()                schema check and/or model calibration
agent.preprocess()         scaling / rolling baselines / derived columns
agent.detect()             per-row anomaly_score ∈ [0,1] + is_anomaly
agent.validate()           per-row `validated` flag (false-positive suppression)
agent.output()             contiguous anomaly windows (structured reports)
    ↓
RiskEngine.compute_risk()  composite risk + level + per-agent breakdown
    ↓
SHAP surrogate · RAG retrieval · Streamlit dashboard
    ↓
Optuna optimizer · evaluation harnesses
```

## 2.2 Three weight layers

| Layer | Weights | Location | Optimized? |
|---|---|---|---|
| **L1** intra-agent | features within one agent | `agent.set_weights()` | yes |
| **L2** inter-agent | agents against each other | `RiskEngine.weights` | yes |
| **L3** thresholds | detection cutoffs, risk bands, agreement bonuses | `set_threshold()` + `RiskEngine` | yes |

`weight_mode` in `config/settings.yaml` = `hand_tuned` (current) or `optimized`.

## 2.3 Models — there is no end-to-end learned model

| Agent | Model |
|---|---|
| shipping | Isolation Forest (200 trees, seed 42) + per-feature z-score + level-shift duration score |
| market | rolling 30-day trailing z-scores, weighted mean of \|z\| |
| geopolitical | weighted linear composite → sigmoid (gain 6, centre 0.5) |
| natural_disaster | weighted composite + single-event max override |
| routing | Isolation Forest (200 trees, seed 42) + transit-ratio z-score |
| news_sentiment | weighted composite of four normalised components |
| **aggregation** | renormalised weighted mean + non-linear agreement bonus |

The only fitted models are two Isolation Forests and their `StandardScaler`s.
Everything above is explicit arithmetic. The RandomForest in
`src/explainability/` is a **post-hoc surrogate** — it explains the score, it
never produces it. **This is a thesis strength, not a gap:** every number is
traceable to a formula a domain expert can audit.

---

# PART III — AGENT STATE (critical for accurate write-up)

Three distinct notions of "off", not interchangeable:

- **Active** — built, run, weighted.
- **Passive** — per-region evidence judgement. Domain is real, no documented
  driver at that chokepoint. Could activate if evidence appeared.
- **Dormant** — project-scope decision (`DORMANT_AGENTS`). No region may
  activate it; enforced by `_validate_registry`.

| Agent | Hormuz | Panama | Bab el-Mandeb | Malacca |
|---|---|---|---|---|
| shipping | active | active | active | active |
| market | active | active | active | **passive** |
| geopolitical | active | **passive** | active | active |
| natural_disaster | active | active | **passive** | active |
| routing | **dormant** | **dormant** | **dormant** | **dormant** |
| news_sentiment | active | active | active | active |

**Exclusion reasons (each is an affirmative finding, not missing data):**

- **market / Malacca** — all four documented Malacca events carry a null market
  field. Removed as a standards violation, not on plausibility grounds.
- **geopolitical / Panama** — documented disruption is purely hydrological.
- **natural_disaster / Bab el-Mandeb** — documented event is a security
  campaign; `disaster_relevance: none`.
- **routing / all four** — temporary uniform muting. `fetch_api()` is a
  `NotImplementedError` stub, so it only ever emitted synthetic values.
  **Note the asymmetry worth writing up:** Bab el-Mandeb's routing evidence is
  the strongest in the benchmark (85% of large containerships diverted via the
  Cape of Good Hope, +3,500–4,000 nm, +10–14 days per voyage — a documented
  percentage, not an extrapolation) and it is muted *in spite of* that.

**Consequence you must state:** with routing dormant everywhere, no evaluation
can measure routing's contribution in any region. Evaluation results alone
cannot settle whether to re-enable it.

---

# PART IV — DATA

## 4.1 Real evaluation dataset (`data/eval/`, built 2026-08-30)

| Region | Rows | Dropped | Date range | Positive days | Rate | Missing domain |
|---|---|---|---|---|---|---|
| hormuz | 2,434 | 358 | 2019-01-01 → 2025-08-30 | **159** | 6.53% | news_sentiment |
| bab_el_mandeb | 2,434 | 358 | 2019-01-01 → 2025-08-30 | **244** | 10.02% | news_sentiment |
| panama | 2,433 | 359 | 2019-01-01 → 2025-08-29 | **149** | 6.12% | — (all present) |
| malacca | 2,434 | 358 | 2019-01-01 → 2025-08-30 | **0** | 0.00% | news_sentiment |

**Live sources:** shipping = IMF PortWatch chokepoint transits; market = FRED
(Brent, freight PPI, freight services); natural_disaster = GDACS (Orange;Red
only) + USGS (M ≥ 4.0); geopolitical = ACLED; news_sentiment = GDELT when it
answered. Fetch cost is recorded per region (e.g. Hormuz geopolitical 663.6s).

**Two facts that must appear in the thesis:**

1. **The eval frame ends 2025-08-30.** The pinned April–May 2026 Hormuz
   shutdown is **not in it**. The 159 Hormuz positives are earlier events.
2. **news_sentiment is missing from three of four regions** — GDELT did not
   answer. Column sets are recorded per region in the manifest specifically so
   a five-feature region is never silently compared against a four-feature one.

## 4.2 Ground-truth label

```
y_true = 30-day mean of vessel_count is ≥20% below the trailing
         365-day median, sustained 14+ consecutive days
```

**Recorded weakness (in the manifest, travels with the data):** a rolling
baseline drifts down with a slow decline, so the label catches shocks and
misses slow-onset disruption. **Panama's 2023–24 drought cut transits 38% and
the label flags almost none of it.**

## 4.3 Synthetic splits (optimizer + `notebooks/evaluation.py`)

Rows cannot be shuffled — that leaks disruption shape across splits and
destroys the time ordering every rolling window depends on. Instead: three
**independent realisations** of the same world by re-seeding every connector.
Identical disruption structure (days 60–74, 150–170, 280–290), different noise
stream. Seed 42 train / 43 validation / 44 test, 365 days each.
`is_disruption` is an evaluation label only and is **never** an agent input.

## 4.4 RAG knowledge base — 10 historical cases

`cyclone_gonu_2007`, `hormuz_mine_threat_2010`, `somali_piracy_2011`,
`japan_earthquake_2011`, `iran_sanctions_2012`, `west_coast_port_strikes_2014`,
`hormuz_2019`, `ever_given_2021`, `covid_port_congestion_2021`,
`houthi_redsea_2024`.

---

# PART V — RESULTS

## 5.1 Method comparison — the headline evidence

**Design (three decisions that carry the weight):**

1. **One temporal split for everyone.** Last 30% of each region's series is the
   test window; nothing is fitted on it. A supervised model scored on its own
   training rows posts an inflated, non-comparable number. Temporal not random,
   because shuffling days leaks the future into the past.
2. **Tiers build real agents.** Each tier fits and runs actual agent classes,
   composited with the project's own weights. Approximating a tier by averaging
   "the first N feature columns" would measure nothing — the first two columns
   are both shipping features, so a "Tier 2" built that way contains no market
   signal at all.
3. **Malacca is a false-positive harness, not a detection test.** Zero labelled
   disruptions → AUC undefined. What it measures is alert rate on a quiet region.

**Circularity rating, recorded per method** — because the label is
"vessel_count vs a trailing long baseline," any method computing that same
statistic predicts it by construction:

| Rating | Meaning |
|---|---|
| `high` | reads vessel_count against a trailing long baseline — the label's own statistic |
| `medium` | short/rolling baseline, or fits on the label directly |
| `low` | reads other features, or shape rather than level |
| `n/a` | controls and oracles |

**All tiers are rated `high`**, because they include the shipping agent's
level-shift feature.

### HORMUZ (159 positives, 6.53%)

| Method | Kind | Circ. | AUC | F1 | FPR |
|---|---|---|---|---|---|
| ORACLE label t−1 | oracle | n/a | 0.986 | 0.977 | 0.005 |
| B7 logistic regression | supervised | medium | **0.968** | 0.682 | 0.020 |
| B6 CUSUM | unsupervised | high | 0.958 | 0.000 | 0.000 |
| M3 persistence | unsupervised | high | 0.851 | 0.503 | 0.140 |
| B8 random forest | supervised | medium | 0.747 | 0.378 | 0.100 |
| M8 matrix profile | unsupervised | low | 0.596 | 0.227 | 0.104 |
| M0 random | control | n/a | 0.526 | 0.165 | 0.094 |
| **Tier 1** | multi-agent | high | **0.502** | 0.034 | 0.164 |
| M1/M2 always/never | control | n/a | 0.500 | 0.000 | 0.000 |
| **Tier 2** | multi-agent | high | **0.465** | 0.071 | 0.139 |
| B3 isolation forest | unsupervised | low | 0.434 | 0.085 | 0.194 |
| **Tier 3/4/5** | multi-agent | high | **0.400** | 0.141 | 0.265 |

**Best baseline 0.986 vs best tier 0.502 → gap −0.484.** Tier 1 is at chance.

### BAB_EL_MANDEB (244 positives, 10.02%)

| Method | Circ. | AUC | F1 | FPR |
|---|---|---|---|---|
| ORACLE label t−1 | n/a | 0.997 | 0.996 | 0.002 |
| M3 persistence | high | 0.961 | 0.784 | 0.234 |
| **Tier 1** | high | **0.679** | 0.553 | 0.735 |
| **Tier 2** | high | **0.620** | 0.511 | 0.546 |
| B2 MA crossover | medium | 0.581 | 0.386 | 0.177 |
| B3 isolation forest | low | 0.500 | 0.410 | 0.628 |
| **Tier 3/4/5** | high | **0.449** | 0.441 | 0.682 |
| B6 CUSUM | high | 0.263 | 0.498 | 0.737 |

Gap −0.318. B7/B8 n/a — no positives in the training window.

### PANAMA (149 positives, 6.12%) — the best region

| Method | Circ. | AUC | F1 | FPR |
|---|---|---|---|---|
| ORACLE label t−1 | n/a | 0.996 | 0.993 | 0.002 |
| M3 persistence | high | 0.973 | 0.811 | 0.112 |
| **Tier 1** | high | **0.909** | 0.722 | 0.167 |
| **Tier 5** | high | **0.884** | 0.613 | 0.157 |
| **Tier 2** | high | **0.884** | 0.620 | 0.134 |
| **Tier 3/4** | high | **0.876** | 0.612 | 0.134 |
| B3 isolation forest | low | 0.814 | 0.435 | 0.170 |
| B6 CUSUM | high | 0.145 | 0.372 | 0.866 |

**Panama Tier 1 at 0.909 beats every non-oracle unsupervised baseline except
persistence.** This is the system's strongest real result.

### MALACCA (0 positives) — false-positive harness

AUC undefined. Alert rates on a region where nothing happened:

| Method | Alert rate |
|---|---|
| M2 never / B6 CUSUM / ORACLE | 0.000 |
| M3 persistence | 0.049 |
| B2 MA crossover | 0.059 |
| B4 EWMA | 0.068 |
| M0 random | 0.100 |
| B1 rolling z | 0.103 |
| M8 matrix profile | 0.107 |
| B5 AR residual | 0.111 |
| **Tier 1 / Tier 2** | **0.178 / 0.181** |
| B3 isolation forest | 0.208 |
| **Tier 3/4/5** | **0.257** |

**The tiers are the noisiest methods on the quiet region.** Adding agents makes
it worse (0.178 → 0.257). Write this up: it is the cost of the fusion design.

## 5.2 The central negative finding — more agents make it worse

Across every evaluable region, tier AUC **falls** as agents are added:

| Region | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 |
|---|---|---|---|---|---|
| hormuz | 0.502 | 0.465 | 0.400 | 0.401 | 0.401 |
| bab_el_mandeb | 0.679 | 0.620 | 0.449 | 0.449 | 0.449 |
| panama | 0.909 | 0.884 | 0.876 | 0.876 | 0.884 |
| malacca (alert rate ↓ better) | 0.178 | 0.181 | 0.257 | 0.257 | 0.257 |

**This directly contradicts the agent-diversity hypothesis (SRQ2) on real
data.** The mechanism is dilution: the label is shipping-derived, so every
non-shipping agent adds weighted score that is uncorrelated with it, pulling
the composite toward noise. Report this as found.

## 5.3 Root-cause analysis — why Hormuz sits at chance

Decomposing Hormuz's own anomaly score on the test window
(2023-08-31 → 2025-08-30, 133 positives / 731 days):

| Component | mean on normal | mean on disruption | **separation** |
|---|---|---|---|
| `duration_score` | 0.0497 | 0.4395 | **+0.3898** |
| `shock_score` | 0.5455 | 0.3982 | **−0.1473** |
| `anomaly_score` = `max(shock, duration)` | 0.5561 | 0.5315 | **−0.0246** |

**The duration signal separates the label cleanly (+0.39). The shock detectors
are anti-correlated (−0.15) — disruption days look *less* anomalous to them
than calm days. Because shock sits ~0.55 on quiet days and duration only
reaches ~0.44 on disrupted ones, `max()` returns shock nearly everywhere and
erases the working signal, leaving −0.02.**

The code comment defends `max()` over averaging to stop a calm shock detector
diluting an active duration signal. It prevents dilution and causes **masking**
instead. This is the actual root cause of Hormuz ≈ 0.50 — not forest training,
not the validation gate.

**Secondary:** `detect()` receives only the test slice, so the 365-day trailing
baseline restarts from scratch. The **first 120 test days have no baseline at
all**, and 17 of 133 positives fall in that blind window.

## 5.4 Two hypotheses tested and rejected (2026-09-01)

Both were implemented, measured, and reverted. Negative results with a
mechanism are thesis-grade material.

**Option 1 — replace the persistence gate with direct level-shift scoring.**
Gate: `level_shift_score > 0.50 AND duration_held >= 0.70`.

- **Structurally inert for the AUC.** `run_method_comparison.py:384` scores via
  `agent.detect(agent.preprocess(frame))` and never calls `validate()`.
- Full re-run: **0 of 76 rows changed**. Output byte-identical. Hormuz gap
  unchanged at −0.484.
- Where it *does* apply (synthetic path), it is destructive:

| Gate | TP | FP | FN | TN | TPR | FPR | F1 |
|---|---|---|---|---|---|---|---|
| persistence (current) | 44 | 10 | 3 | 308 | 0.936 | 0.031 | **0.871** |
| level_shift | 6 | 2 | 41 | 316 | 0.128 | 0.006 | **0.218** |

Cause: the trailing baseline needs 91 days shifted by 30, so **no baseline
exists until day 156** — the day 60–74 disruption is unscoreable by
construction; and only 23 of 365 days clear `score > 0.50`.

**Option 3 — retrain the Isolation Forest on mixed normal + disruption days.**

- **Premise false.** `_agent_frame` passes only `timestamp` + `shipping__*`;
  `y_true` never reaches the agent, so `fit()`'s leak-filter never fires. The
  forest producing Hormuz 0.502 was **already** trained on mixed data.
- Tested both regimes properly on real data with the harness's own split:

| Region | mixed | normal-only | Δ |
|---|---|---|---|
| hormuz | 0.5023 | 0.5240 | **+0.0216** |
| bab_el_mandeb | 0.6790 | 0.6790 | 0.0000 |
| panama | 0.9087 | 0.9087 | 0.0000 |

Bab el-Mandeb and Panama are identical because their train windows contain no
positives — filtering removes nothing. Option 3's direction is the *worse* of
the two on Hormuz, by a negligible margin.

## 5.5 Weight optimization

Optuna, TPE sampler (seed 42), median pruner, 100 trials, 1h cap. Dirichlet-style
renormalisation per weight group. Hard constraints reject
`risk_high ≤ risk_medium` and `agreement_bonus_5 ≤ agreement_bonus_3`.

**Objective:** `0.50·F1 + 0.30·lead_time_score − 0.20·FPR`
- F1 — HIGH-risk alert vs `is_disruption`
- lead_time_score — earliest MEDIUM alert within 5 days before onset, ÷ 5
- FPR — false-positive rate of HIGH alerts

**Test-split discipline:** touched exactly once, after the study.
`PipelineEvaluator.evaluated_splits` is an audit trail proving it.

**Best run:** trial 63 of 100, validation objective **0.7491** (2026-08-30).

| Agent | Hand-tuned L2 | Optimized L2 |
|---|---|---|
| shipping | 0.25 | **0.402** |
| market | 0.15 | 0.088 |
| geopolitical | 0.25 | 0.109 |
| natural_disaster | 0.10 | 0.095 |
| routing | 0.15 | 0.130 |
| news_sentiment | 0.10 | **0.177** |

| Band | Hand | Optimized |
|---|---|---|
| risk_critical | 0.90 | — (4-level path only) |
| risk_high | 0.69 | 0.766 |
| risk_medium | 0.51 | 0.276 |
| agreement_bonus_3 | 1.15 | 1.277 |
| agreement_bonus_5 | 1.25 | 1.443 |

**Hand-tuned bands are empirically calibrated, not chosen:** p60 / p85 / p97
quantiles of the composite on *calm* (label-negative) days, pooled across all
four regions 2019–2026, **9,183 days**. Recalibrated when the shipping agent
stopped batch-normalising its forest score — same quantiles of calm behaviour
on a new absolute scale, not a loosening.

## 5.6 Aggregation algorithm (`RiskEngine.compute_risk`)

```
1. keep agents present AND weighted AND producing scores
2. score_a   = mean(anomaly_scores_a)
3. w_norm_a  = w_a / Σw            renormalised over ACTIVE agents → sums to 1
4. base      = Σ w_norm_a · score_a
5. agreement = |{a : score_a > 0.5}|
6. amp       = 1.25 if agreement ≥ 5;  1.15 if ≥ 3;  else 1.00
7. risk      = min(base · amp, 1.0)
8. level     = high / medium / low by threshold
```

Step 3 makes a passive or dormant agent harmless — weight redistributes rather
than dragging the composite to zero. Step 6 is the only non-linearity.

---

# PART VI — GRAPHS (paths, provenance, staleness)

## 6.1 Method comparison — `eval/graphs_method_comparison/` (7)

Generated 2026-08-30 23:24 from `method_comparison_results.csv` (23:20).
**Current. Real data. Cite freely.**

| File | Content |
|---|---|
| `A1_auc_by_method.png` | AUC by method, all regions |
| `A2_false_positive_harness.png` | Malacca alert rates — the FP harness |
| `C_ranking_hormuz.png` | Hormuz ranking, bars coloured by circularity |
| `C_ranking_bab_el_mandeb.png` | as above |
| `C_ranking_panama.png` | as above |
| `D_auc_heatmap.png` | method × region AUC heatmap |
| `E_f1_vs_fpr.png` | F1 against FPR scatter |

Bar colour encodes circularity (`high` red, `medium` orange, `low` green,
`n/a` grey) — deliberate, so a ranking cannot be read as "tallest is best."

## 6.2 Tier ablation — `eval/graphs_ablation_tiers/` (3)

`B_tier_progression_{hormuz,bab_el_mandeb,panama}.png` — tier AUC against the
best baseline and a chance line. **Current.**

**Malacca has no chart by design:** `graph_tiers()` drops NaN-AUC rows, and
Malacca's are all NaN (0 positives). It is covered instead by
`A2_false_positive_harness.png`. All four regions' tier *data* is in the CSV.

## 6.3 Optimizer — `data/processed/` (6), gitignored

Generated 22:02–22:03 local = 20:02 UTC = exactly the 100-trial run recorded in
`optimized_weights.yaml`. **Current.**

`optimization_history.png`, `param_importances.png`, `parallel_coordinate.png`,
`contour_plot.png`, `weight_comparison.png`, `performance_comparison.png`

## 6.4 SHAP — `data/processed/` (4), gitignored

| File | Date | Status |
|---|---|---|
| `shap_comparison_waterfall.png` | 2026-08-30 23:28 | ⚠ suspect — see §7.1 |
| `shap_comparison_importance.png` | 2026-08-30 23:28 | ⚠ suspect |
| `shap_beeswarm_6agent.png` | **2026-06-20** | ✗ **STALE — do not cite** |
| `shap_waterfall_6agent.png` | **2026-06-20** | ✗ **STALE — do not cite** |

The two beeswarm/waterfall plots predate the current weights, the shipping
level-shift rework, the risk-band recalibration, and the explainer's own last
edit (2026-07-02). Regenerate before use.

## 6.5 EVAL01 — archived, previous iteration

Not in the working tree. On `main` / tag `eval01-archived` at
`docs/presentation/charts/`: `chart_A_baseline_auc_pr.png`,
`chart_B_baseline_fpr.png`, `chart_C_ablation_progression.png`,
`chart_D_iforest_decoy_false_alarm.png`, `chart_E_cusum_stickiness.png`.

Synthetic-benchmark charts from a harness that no longer runs. Retrieve with
`git checkout eval01-archived -- docs/presentation/charts/`. **Do not mix with
`eval/` graphs — both use A/B/C/D/E prefixes and are easily confused.**

---

# PART VII — DEFECTS AND OPEN ISSUES

## 7.1 BLOCKING — optimization records contradict each other

```
config/optimized_weights.yaml        → 100 trials, best trial 63, val 0.7491
data/processed/optimization_results.json → 5 trials, best trial 1, obj 0.6301
```

Entirely different weight vectors (news_sentiment 0.177 vs 0.046;
natural_disaster 0.095 vs 0.168). The JSON was overwritten at 23:28 by an
apparent smoke run; the YAML on disk is still the committed 100-trial result.

**Impact:** `notebooks/evaluation.py` **METRIC 5** reads that JSON. Run the
evaluation suite today and it reports the 5-trial numbers as your optimization
result. The two SHAP comparison plots were generated in that same session.

**Fix before writing Chapter "Results":** re-run the 100-trial optimization.

## 7.2 The 8-metric suite has never been run against the current system

`evaluation_results.json` and `thesis_comparison_table.json` are dated
**2026-07-30** — a month before the re-optimization and the shipping rework.
METRICS 1–8 as currently on disk describe a superseded system.

## 7.3 Code-level inconsistencies to disclose

1. **Routing normalises its Isolation Forest by batch min-max**; shipping was
   deliberately moved to fit-time percentiles for exactly the reason batch
   normalisation is wrong. Routing was never migrated — it went dormant first.
2. **`Orchestrator.run_timeseries_analysis` omits the agreement bonus** every
   other composite path applies.
3. **The agreement threshold (0.5) is a module constant**, not configurable or
   optimized, while the bonuses it gates are both.
4. **Geopolitical computes rolling baselines/deviations that `detect()` never
   reads**; news does the same with `sentiment_rolling_7d`. `sentiment_magnitude`
   is a schema feature that never enters any score.
5. **`PipelineEvaluator.build_agents` hardcodes agent config** (shipping
   `contamination=0.05`, `z_threshold=2.0`; market `baseline_years=5`,
   `threshold=0.50`) instead of reading `settings.yaml` — so the optimizer tunes
   a slightly different agent than the live pipeline runs.
6. **The SHAP surrogate sees 20 of the features, not all.** Absent from it:
   `tanker_count`, `vessel_count_trend`, `freight_services_pct_change`,
   `vessels_holding`, `alternative_route_traffic`, `sentiment_magnitude`,
   `recency_weighted_score`. Missing columns fill with 0.0 — indistinguishable
   from a true zero.
7. **The shipping duration score is near-circular** against the level-shift
   label. Cite operationally, never as detection skill.

## 7.4 Four composite paths that are not identical

| Path | Granularity | Agreement bonus? |
|---|---|---|
| `RiskEngine.compute_risk` | one scalar per run | yes |
| `RiskEngine.compute_risk_timeseries` | per day | yes |
| `Orchestrator.run_timeseries_analysis` | per day | **no** |
| `PipelineEvaluator._aggregate_daily` | per day, vectorised | yes |

Always say which produced a reported number. `_aggregate_daily` faithfully
mirrors `compute_risk`, so the optimizer maximises what the pipeline produces.

---

# PART VIII — WHAT YOU CANNOT CLAIM

State these explicitly; a defense will find them otherwise.

1. **Cannot claim the system beats baselines at detection.** On Hormuz it is at
   chance (0.502 vs 0.968 for logistic regression). Only Panama (0.909) is
   competitive.
2. **Cannot claim agent diversity improves detection.** Measured on real data,
   AUC *falls* monotonically as agents are added, in every evaluable region.
3. **Cannot cite tier AUC as clean detection skill.** Every tier is rated `high`
   circularity — tiers include the shipping level-shift feature, which is the
   label's own statistic.
4. **Cannot claim anything about routing.** Dormant in all four regions; never
   exercised.
5. **Cannot claim performance on the 2026 Hormuz shutdown.** Not in the eval
   frame (ends 2025-08-30).
6. **Cannot claim slow-onset detection.** The label itself misses it — Panama's
   38% drought decline is almost entirely unlabelled.
7. **Cannot cite METRIC 5 or any 8-metric number until §7.1 and §7.2 are fixed.**

---

# PART IX — SUGGESTED THESIS NARRATIVE

The honest story is stronger than the one originally hypothesised. Structure:

**Ch. 1 — Problem.** Chokepoint disruption, decision-maker needs, why raw
prediction is insufficient (attribution, precedent, auditability).

**Ch. 2 — Architecture.** Six agents, three weight layers, deliberate absence of
an end-to-end model. Emphasise auditability: every number traces to a formula.

**Ch. 3 — Evidence discipline.** Active/passive/dormant; per-region exclusions
as affirmative findings; the Bab el-Mandeb routing asymmetry (muted despite the
strongest evidence in the benchmark).

**Ch. 4 — Evaluation design.** One temporal split for all; tiers build real
agents; Malacca as an FP harness; **circularity ratings** — the design decision
that makes the whole evaluation credible.

**Ch. 5 — Results, reported as found.** Panama 0.909 is the win. Hormuz 0.502 is
the failure. Diversity *reduces* AUC. Tiers are noisiest on the quiet region.

**Ch. 6 — Root-cause analysis.** The `max(shock, duration)` masking finding
(§5.3) with its decomposition table — this is the chapter that turns a negative
result into a contribution. Include Options 1 and 3 as tested-and-rejected
hypotheses with mechanisms (§5.4).

**Ch. 7 — Explainability and decision support.** SHAP surrogate, RAG gating,
the transparent action rubric (deliberately not another model).

**Ch. 8 — Limitations.** Part VIII verbatim.

**The defensible contribution:** not a better detector, but *an evaluation
methodology that catches its own circularity* — plus a diagnosed mechanism for
why multi-domain fusion underperforms on a single-domain label. Most theses
would have reported the Panama number and stopped.

---

# APPENDIX — FILE MAP

| Purpose | Path |
|---|---|
| Agents | `src/agents/{base,shipping,market,geopolitical,disaster,routing,news}_agent.py` |
| Aggregation | `src/aggregation/risk_engine.py` |
| Orchestration | `src/orchestrator.py` |
| Region registry | `src/core/regions.py` · overlays `config/regions/*.yaml` |
| Config | `config/settings.yaml` · `config/optimized_weights.yaml` |
| Optimization | `src/optimization/{weight_optimizer,pipeline_evaluator,data_split,weight_config,optimization_analysis}.py` |
| Explainability | `src/explainability/shap_explainer.py` |
| RAG | `src/rag/context_retriever.py` · `data/knowledge_base/disruption_cases.json` |
| Decision eval | `src/evaluation/decision_effectiveness.py` |
| Real eval build | `scripts/build_eval_dataset.py` → `data/eval/` |
| Method comparison | `scripts/run_method_comparison.py` → `eval/method_comparison_results.csv` |
| Graph generation | `scripts/report_method_comparison.py` → `eval/graphs_*/` |
| 8-metric suite | `notebooks/evaluation.py` |
| Scoring detail | `docs/SCORING_REFERENCE.md` |
| Results narrative | `eval/COMPARISON_REPORT.md` |
| EVAL01 archive | `docs/eval01-archived/` · charts at tag `eval01-archived` |
