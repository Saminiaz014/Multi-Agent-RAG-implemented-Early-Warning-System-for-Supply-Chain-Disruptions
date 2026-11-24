# Thesis Writing Brief — Complete Project State

**State of record:** the git tag `v1.0.1-thesis` (after the freeze at `v1.0-thesis`).

This document is the guide for writing up the thesis. It states **no measured value of the real-data evaluation**: no result, dataset size, commit count or test count. Each is read from the place that produces it, so it cannot go stale here:

| What | Where |
|---|---|
| Detection results, ablations, split sensitivity, false-alarm rates, surrogate fidelity, lead time | [`eval/RESULTS_FROZEN.md`](../eval/RESULTS_FROZEN.md) and [`eval/results_main.md`](../eval/results_main.md), generated from the committed CSVs |
| Dataset size, date range, positives, provenance of the inputs | `data/eval/<region>.manifest.json` (git-ignored; written by `scripts/build_eval_dataset.py`) |
| Commits, test count, file count | `git log`, `pytest -q` |

The synthetic optimizer study (§5.5), the graph inventory (Part VI) and the dated defect records (Part VII) keep their own figures and dates as records of those studies. They are not results of the real-data evaluation and are not to be cited as such.

Where a number should not be cited as evidence of skill, that is stated at the point of use.

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

## 4.1 Real evaluation dataset (`data/eval/`)

Built by `scripts/build_eval_dataset.py` from the live connectors, merged on a daily index. Each region's row count, date range, positive days and rate, the domains present and missing, and the provenance of its news, market and ACLED inputs are in `data/eval/<region>.manifest.json` (git-ignored; regenerate with the script). None is restated here. The positives and scored days of the test window are at the head of each region's section in `eval/RESULTS_FROZEN.md`.

**Live sources:** shipping = IMF PortWatch chokepoint transits; market = FRED (Brent, freight PPI, freight services); natural_disaster = GDACS (Orange/Red only) + USGS (`USGSExtractor.min_magnitude`, a code default); geopolitical = ACLED; news_sentiment = GDELT 1.0 daily event export (`scripts/build_gdelt_news_csv.py`).

**Two facts that must appear in the thesis:**

1. **The test window ends at `evaluation.end_date` in `config/settings.yaml`**, a limit set by the ACLED data; the frame may run a few days past it, and those days are not scored. The pinned April–May 2026 Hormuz shutdown is **not in the window**; the positives are earlier events.
2. **Days GDELT never published are masked, not filled** (`rows_news_masked` and `news_source.missing_days` in the manifest): they stay NaN and are left out of every score. No domain is missing from any region's frame (`domains_missing` is empty in each manifest); which agents a region *uses* is a separate setting, the `enabled` flags in its region config.

## 4.2 Ground-truth label

```
y_true = 30-day mean of vessel_count is ≥20% below the trailing
         365-day median, sustained 14+ consecutive days
```

**Recorded weakness:** a trailing baseline follows a slow decline down, so the label is built to catch level shifts and can lose the start of a slow-onset disruption. The manifest records `label_definition`, `label_validation` and `label_secondary` with the frame. How much of a given episode is labelled is read from the test positives in `eval/RESULTS_FROZEN.md`, not from a sentence here.

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

**§5.1 to §5.4 state no number; §5.5 is the synthetic optimizer study and keeps its own figures.** The results are generated from the committed CSVs by `scripts/results_table.py` and `scripts/make_results_frozen.py`. Read them in [`eval/results_main.md`](../eval/results_main.md) (every method, one table per region: AUC with its bootstrap interval, and the paired ΔAUC against Tier 1) and in [`eval/RESULTS_FROZEN.md`](../eval/RESULTS_FROZEN.md) (main table, Eq. 1 ablation, agent ablations, split sensitivity, Malacca false-alarm rates, surrogate fidelity with the Bab el-Mandeb diagnostic, lead time, and the commit and manifest hashes). A sentence in the thesis about a result cites one of those two files.

## 5.1 Method comparison — the design

**Three decisions carry the weight:**

1. **One temporal split for everyone.** A fixed split date (`evaluation.split_date` in `config/settings.yaml`): every method is fitted on the days before it and scored from it to `evaluation.end_date`; nothing is fitted on the test window. A supervised model scored on its own training rows posts an inflated, non-comparable number. Temporal not random, because shuffling days leaks the future into the past.
2. **Tiers build real agents.** Each tier fits and runs actual agent classes, composited with the project's own weights. Approximating a tier by averaging "the first N feature columns" would measure nothing — the first two columns are both shipping features, so a "Tier 2" built that way contains no market signal at all.
3. **Malacca is a false-positive harness, not a detection test.** Zero labelled disruptions → AUC undefined. What it measures is the alert rate on a quiet region.

**Circularity rating, recorded per method** — because the label is "vessel_count vs a trailing long baseline," any method computing that same statistic predicts it by construction:

| Rating | Meaning |
|---|---|
| `high` | reads vessel_count against a trailing long baseline — the label's own statistic |
| `medium` | short/rolling baseline, or fits on the label directly |
| `low` | reads other features, or shape rather than level |
| `n/a` | controls and oracles |

**All tiers are rated `high`**, because they include the shipping agent's level-shift feature. Which method is best in a region is named in one place only, `eval/results_main.md`.

## 5.2 The central negative finding — what adding agents does

It is read from the tier rows of `eval/results_main.md` (AUC per tier with its interval, and the paired ΔAUC against Tier 1) and from the agent ablations in `eval/RESULTS_FROZEN.md`. Region by region, a ΔAUC interval that excludes zero is a change; one that includes zero is not; a tier that copies the tier before it (the region disables the agent it would add) is not a separate comparison. The agent-diversity hypothesis (SRQ2) is decided by those intervals, not by a sentence here.

**Proposed mechanism: dilution.** The label is shipping-derived, so every non-shipping agent adds weighted score that has no reason to correlate with it, pulling the composite toward noise. The agent ablations are where that reading can be checked. On the false-alarm side, Malacca's alert rates by tier are in the same page.

## 5.3 Root-cause analysis — why Hormuz Tier 1 scores low

A decomposition of the shipping agent's anomaly score into its two components, `duration` and `shock`, was run on an earlier evaluation window (a 70/30 split). It found the duration component separating the label, the shock component doing the opposite, and `max(shock, duration)` returning the shock component nearly everywhere: **masking**. No script was kept and it has not been rerun on the fixed split, so none of its numbers is restated: present it as a hypothesis for the current result, not a measurement of it ([README §4.7](../README.md#47-root-cause-analysis--why-hormuz-tier-1-scores-low)).

**Secondary (fixed, issue 11):** `detect()` received only the test slice, so the trailing baseline restarted from scratch and the first test days had no baseline. The comparison now fits each agent on the training days and scores it on training + test, keeping the test rows; `eval/diagnostics/split_invariance.py` checks that a day's score no longer depends on where the test window starts.

## 5.4 Two hypotheses tested and rejected

Both were implemented, measured and reverted (September 2026) on the earlier frame (70/30 split). They have not been rerun on the fixed split, so their numbers are not restated; the account matches README §4.8, and the earlier numbers are in the text of this file at the tag `v1.0-thesis`. Negative results with a mechanism are thesis-grade material.

**Option 1 — replace the persistence gate with direct level-shift scoring.**

- **Structurally inert for the harness's AUC.** The comparison harness scores through `agent.detect(agent.preprocess(frame))` and never calls `validate()`, so a change to the gate cannot change an AUC; a full rerun at the time changed no row.
- **Destructive where it does apply** (the synthetic path, which does call `validate()`): the trailing baseline needs a long history before any day is scoreable, so an early disruption is unscoreable by construction and few days clear the threshold. Detection fell sharply against the persistence gate.

**Option 3 — retrain the Isolation Forest on mixed normal + disruption days.**

- **Premise false.** `_agent_frame` passes only `timestamp` and the `shipping__*` columns, so `y_true` never reaches the agent and `fit()`'s leak filter never fires: the forest behind the Hormuz result had already been trained on mixed data.
- Both regimes were then compared properly on real data with the harness's own split. The mixed direction was the worse of the two on Hormuz, by a small margin; Bab el-Mandeb and Panama were identical because their training windows contain no positives, so filtering removes nothing.

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

**Hand-tuned bands are fixed display thresholds, not calibrated.** They cut the
composite into words for the dashboard and the narrative. They do not enter the real-data
evaluation (its alert rule is each method's training-window 90th percentile); the
synthetic weight optimizer does read `risk_high` and `risk_medium` (F1 and the lead-time
score).
Provenance: the values were set in commit `f7b87ff` (2026-08-30) by a quantile method
(p97 / p85 / p60 of the composite on label-negative days) run on the pipeline of that
time, test window included. No script was kept, so they cannot be reproduced and have
not been re-derived for the current five-agent composite. `scripts/calibrate_risk_bands.py`
reconstructs the method on training days only and gives different values; that is
documented, and not applied, in the comment above `thresholds` in `config/settings.yaml`.

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

Regenerated from `eval/method_comparison_results.csv` by `scripts/make_all_figures.py`; `eval/FIGURES.md` is the registry (each figure's source, command and whether it is regenerable) and wins over the inventory below.
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
best other method (no ceiling, no controls) and a chance line. **Current.**

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

1. **Cannot claim the system beats baselines at detection.** Compare each tier
   with the baselines region by region in `eval/results_main.md`, and say where a
   baseline is clearly better; do not generalise across regions.
2. **Cannot claim agent diversity improves detection.** In the frozen results no
   distinct tier is clearly above Tier 1 (paired ΔAUC, `eval/RESULTS_FROZEN.md`).
3. **Cannot cite tier AUC as clean detection skill.** Every tier is rated `high`
   circularity — tiers include the shipping level-shift feature, which is the
   label's own statistic.
4. **Cannot claim anything about routing.** Dormant in all four regions; never
   exercised.
5. **Cannot claim performance on the 2026 Hormuz shutdown.** Not in the
   evaluation window (it ends at `evaluation.end_date`).
6. **Cannot claim slow-onset detection.** A trailing-median label can lose the
   start of a slow decline; how much of an episode it labels is in the test
   positives (`eval/RESULTS_FROZEN.md`), not a sentence here.
7. **Cannot cite METRIC 5 or any 8-metric number until §7.1 and §7.2 are fixed.**
8. **Cannot cite lead time as early warning.** Against a lagged label every
   method leads every onset, the random control included (`eval/RESULTS_FROZEN.md`,
   Lead time).

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

**Ch. 5 — Results, reported as found.** Region by region, from
`eval/results_main.md`: where the tiers stand against the baselines, what the
paired ΔAUC says about adding agents, and the false-alarm rates on the quiet
region. No sentence without its table row.

**Ch. 6 — Root-cause analysis.** The `max(shock, duration)` masking hypothesis
(§5.3), stated as a hypothesis until it is rerun on the fixed split: a negative
result becomes a contribution only then. Include Options 1 and 3 as
tested-and-rejected hypotheses with mechanisms (§5.4).

**Ch. 7 — Explainability and decision support.** SHAP surrogate, RAG gating,
the transparent action rubric (deliberately not another model).

**Ch. 8 — Limitations.** Part VIII verbatim.

**The defensible contribution:** not a better detector, but *an evaluation
methodology that catches its own circularity* — plus a proposed mechanism for
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
