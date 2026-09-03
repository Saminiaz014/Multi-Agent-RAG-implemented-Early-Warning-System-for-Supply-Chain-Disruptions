# Risk Scoring Reference

How every agent scores, what it scores from, how the scores combine, and what
happens to those numbers between detection and the evaluation tables.

Source of truth for every figure below: `src/agents/*.py`,
`src/aggregation/risk_engine.py`, `config/settings.yaml`,
`config/optimized_weights.yaml`, `src/core/regions.py`,
`src/optimization/*`, `notebooks/evaluation.py`,
`scripts/run_method_comparison.py`.

---

## 0. Shape of the pipeline

```
connector (ingestion)      → raw domain frame, daily index
    ↓
agent.fit()                → schema check and/or model calibration
agent.preprocess()         → scaling / rolling baselines / derived columns
agent.detect()             → per-row anomaly_score ∈ [0,1] + is_anomaly
agent.validate()           → per-row `validated` flag (FP suppression)
agent.output()             → contiguous anomaly windows (reports)
    ↓
RiskEngine.compute_risk()  → composite risk score + level + breakdown
    ↓
SHAP surrogate / RAG / dashboard
    ↓
optimizer (Optuna) and evaluation harnesses
```

Three weight layers exist, and all three are tunable:

| Layer | What it weights | Lives in |
|---|---|---|
| **Layer 1** — intra-agent | features *inside* one agent | each agent's `set_weights()` |
| **Layer 2** — inter-agent | agents against each other | `RiskEngine.weights` |
| **Layer 3** — thresholds | detection/risk cutoffs, agreement bonuses | agent `set_threshold()` + `RiskEngine` |

`weight_mode` in `config/settings.yaml` selects `hand_tuned` (values in
`settings.yaml`) or `optimized` (values in `config/optimized_weights.yaml`,
written by Optuna). Currently set to `hand_tuned`.

---

## 1. Agent state

There are three distinct notions of "off", and they are not interchangeable.

- **Active** — built, run, weighted.
- **Passive** — a per-region evidence judgement. The domain is real but has no
  documented driver at that chokepoint. Could become active if evidence appears.
- **Dormant** — a project-scope decision (`DORMANT_AGENTS` in
  `src/core/regions.py`). No region may activate it; enforced by
  `_validate_registry`.

### Per-region roster (`src/core/regions.py`)

| Agent | Hormuz | Panama | Bab el-Mandeb | Malacca |
|---|---|---|---|---|
| shipping | active | active | active | active |
| market | active | active | active | **passive** |
| geopolitical | active | **passive** | active | active |
| natural_disaster | active | active | **passive** | active |
| routing | **dormant** | **dormant** | **dormant** | **dormant** |
| news_sentiment | active | active | active | active |

Reasons on record:

- **market / Malacca** — all four documented Malacca events carry a null market
  field. Removed as a standards violation, not on plausibility.
- **geopolitical / Panama** — the documented disruption is hydrological (Gatún
  Lake drought). No geopolitical driver applies.
- **natural_disaster / Bab el-Mandeb** — the documented event is a
  security/geopolitical campaign. `disaster_relevance: none` is an affirmative
  absence in the evidence, not missing data.
- **routing / everywhere** — temporary uniform muting, deferred to post-Phase-14
  evaluation. `routing_connector.fetch_api()` is still a `NotImplementedError`
  stub, so the agent only ever emitted synthetic or CSV values. Note the
  asymmetry: Bab el-Mandeb's routing evidence is the *strongest* in the
  benchmark (85% of large containerships diverted via the Cape of Good Hope) and
  it is muted in spite of that, not because of weakness. With routing off
  everywhere, evaluation cannot measure routing's contribution anywhere — so
  the evaluation results alone cannot settle whether to re-enable it.

### Data mode per agent (`config/settings.yaml`)

| Agent | Configured mode | Real source when live |
|---|---|---|
| shipping | `csv` | IMF PortWatch chokepoint transits (2019–2026) / Shuaiba arrivals |
| market | `csv` | FRED — Brent, freight PPI, freight services |
| geopolitical | `synthetic` | ACLED (used in the real eval dataset) |
| natural_disaster | `synthetic` | GDACS (Orange;Red) + USGS (M≥4.0) |
| routing | `synthetic` | none — API is a stub |
| news_sentiment | `synthetic` | NewsAPI / SerpAPI / GDELT |

A connector failure in CSV mode falls back to synthetic per-connector
(`Orchestrator._safe_fetch`), so a partial dataset never aborts a run.

---

## 2. Models used — one line each

| Agent | Model | Deterministic? |
|---|---|---|
| shipping | Isolation Forest (200 trees, `random_state=42`) + per-feature z-score + level-shift duration score | yes |
| market | rolling-window z-scores (30-day trailing), weighted mean of \|z\| | yes |
| geopolitical | weighted linear composite → sigmoid compression | yes |
| natural_disaster | weighted linear composite + single-event max override | yes |
| routing | Isolation Forest (200 trees, `random_state=42`) + transit-ratio z-score | yes |
| news_sentiment | weighted linear composite of four normalised components | yes |
| **aggregation** | **renormalised weighted mean + non-linear agreement bonus** | **yes** |

There is no learned end-to-end model producing the risk score. The only fitted
models are the two Isolation Forests (shipping, routing) and the
`StandardScaler`s in front of them. Everything above them is an explicit
arithmetic rule. The RandomForest in `src/explainability/` is a *post-hoc
surrogate* — it explains the score, it never produces it.

---

## 3. Agent-by-agent detail

### 3.1 Shipping — `src/agents/shipping_agent.py`

Primary signal source.

**Features**

| Feature | Origin | Notes |
|---|---|---|
| `vessel_count` | measured | daily transits |
| `avg_delay_hours` | derived | inverse-proportional to `vessel_count`, clipped [1, 72] |
| `congestion_index` | derived | rises when arrivals fall below a 30-day rolling baseline, ∈ [0,1] |
| `tanker_count` | measured, optional | most Hormuz-sensitive vessel class |
| `vessel_count_trend` | derived, optional | `vessel_count − vessel_count_7dma` |

Optional features are auto-discovered at `fit()` so the agent is a drop-in
across synthetic and CSV modes.

**Fit.** `StandardScaler` + `IsolationForest(contamination, n_estimators=200,
random_state=42)`. When `is_disruption` is present, **only non-disruption rows
are used to fit** — no leakage. The forest's score range is stored as the
fit-time 5th/95th percentiles (`_iforest_low` / `_iforest_high`), not min/max,
so one extreme training row cannot set the ceiling.

**Score.** Two signals that answer different questions, combined by `max()`:

```
iforest_norm = clip( (−decision_function − p5) / (p95 − p5), 0, 1 )
max_z_norm   = min( max_i |z_i| / z_threshold , 1 )

shock    = w_if · iforest_norm + w_z · max_z_norm          # "did something just change?"
duration = level_shift_score(vessel_count)                 # "are we still below normal?"

anomaly_score = max(shock, duration)
is_anomaly    = anomaly_score ≥ threshold
```

Normalisation is against the **fit-time** distribution, not the scored batch.
Batch min-max would map a uniformly disrupted month and a uniformly calm month
both onto [0,1] and destroy exactly the signal a sustained disruption carries.

`level_shift_score`:

```
rolling   = 30-day mean of vessel_count
baseline  = median of the 365 days ending 30 days ago   (trailing, not rolling)
shortfall = (baseline − rolling) / baseline
magnitude = clip(shortfall / 0.40, 0, 1)                 # a 40% sustained drop = 1.0
persist   = fraction of the trailing 14 days with shortfall > 0.10
score     = clip(magnitude · persist, 0, 1)
```

Two caveats carried in the code and repeated here because they matter for the
thesis: (a) the baseline is *trailing*, because a rolling one adapts to the new
level and reads a settled disruption as calm; (b) **this score is not evidence
of detection skill against a shipping-derived label** — it is computed from the
same series the level-shift label is built from, so it predicts that label
nearly by construction. It earns its place operationally (an alert should
persist while the disruption does), not evaluatively.

**Validation.** `validated = persistence ∧ breadth`
- persistence: part of a run of ≥ 2 consecutive `is_anomaly` days (no upper cap,
  so multi-month shutdowns stay flagged in full)
- breadth: ≥ 2 active features with `|z| > 1.5` on that row

**Confidence** (per window) = mean(`features_elevated` / `features_total`).

**Parameters** — `settings.yaml`: `contamination 0.1`, `threshold 0.65`,
`z_threshold 2.0`. Layer-1 blend default `if 0.70 / z 0.30`; optimized
`0.657 / 0.343`.

---

### 3.2 Market — `src/agents/market_agent.py`

**Features**

| Feature | Origin | Notes |
|---|---|---|
| `brent_crude_usd` | measured (FRED) | lead price-side indicator |
| `trade_volume_index` | derived | `1 − normalised 30-day Brent volatility`, ∈ [0,1] |
| `freight_rate_index` | measured (FRED PPI) | monthly, ffilled, rebased to ~100 |
| `freight_services_pct_change` | measured, optional | FRED-mode only |

**No Isolation Forest here, deliberately.** Market series are strongly
autocorrelated and drift; a global `StandardScaler` would introduce look-ahead
bias. Instead z-scores are recomputed against a 30-day *trailing* rolling window
on every call.

**Preprocess.** Clip history to the last `baseline_years` (default 5) so 1990s
$20/bbl regimes do not pollute the current baseline; then attach
`<feature>_rolling_mean` and `<feature>_rolling_std` (window 30, `min_periods=2`).

**Score.**

```
z_i           = (x_i − rolling_mean_i) / rolling_std_i        (0 where std < 1e-9)
anomaly_score = min( Σ_i w_i · |z_i| / z_threshold , 1.0 )
is_anomaly    = anomaly_score ≥ threshold
```

Layer-1 weights:

| | oil | trade | freight | freight-services |
|---|---|---|---|---|
| 3-feature (synthetic) | 0.40 | 0.35 | 0.25 | — |
| 4-feature (FRED) | 0.35 | 0.30 | 0.20 | 0.15 |
| optimized | 0.588 | 0.119 | 0.293 | (fixed 0.15 share reserved) |

**Validation** — an oil-led AND gate:
1. persistence ≥ 2 consecutive days, AND
2. `|oil_z| > z_threshold`, AND
3. `|trade_z| > z_threshold` OR `|freight_z| > z_threshold`

The asymmetry is intentional: oil is the most direct price-side signal of a
Hormuz disruption, and a freight or volume move with quiet oil is far more
likely ordinary market noise. `freight_services` contributes to the *score* but
is excluded from the corroboration OR, so the gate is identical in synthetic and
FRED modes.

**Parameters** — `settings.yaml`: `z_threshold 1.5`, `threshold 0.50`,
`baseline_years 5`. Optimized `market_z_threshold = 1.791`.

---

### 3.3 Geopolitical — `src/agents/geopolitical_agent.py`

Categorical/event-based, not a continuous series, so: weighted composite scoring
with persistence + breadth validation rather than statistical anomaly detection.

**Features** (all ∈ [0,1]): `sanctions_severity`, `military_activity_index`,
`diplomatic_incident_score`, `regime_stability_index`.

**Preprocess.** Attaches 14-day rolling baselines and per-feature deviations.
*These columns are computed but not consumed by `detect()`* — the composite reads
the raw levels. Worth knowing before you cite them.

**Score.**

```
raw = Σ_present w_k · v_k  /  Σ_present w_k          # stability inverted: v = 1 − x
anomaly_score = 1 / (1 + exp(−6 · (raw − 0.5)))      # sigmoid, centred 0.5, gain 6
is_anomaly    = anomaly_score ≥ threshold
```

Weight renormalisation is not cosmetic. ACLED — the only free source covering
this domain — carries no sanctions data, and no free source publishes a daily
sanctions severity series. Rather than invent the highest-weighted feature, the
remaining weights rescale proportionally: with sanctions (0.35) absent, military
and diplomatic go 0.25 → 0.385 and stability 0.15 → 0.231. The score stays a
weighted *mean* over [0,1] features, never a partial sum that would read as
artificially calm. A renormalisation is logged as a warning, not silently.

**Validation** — `persistence ∧ breadth`
- persistence: run of **≥ 3** consecutive days (longer than every other agent —
  geopolitical noise is spiky)
- breadth: ≥ 2 of the 4 features elevated, where "elevated" means
  `> 0.4` for the three risk features and `< 0.6` for stability

**Confidence** = mean(`features_elevated`) / 4.

**Weights** — hand: sanctions 0.35 / military 0.25 / diplomatic 0.25 /
stability 0.15. Optimized: 0.360 / 0.280 / 0.273 / 0.087.
Threshold 0.50 hand, 0.658 optimized.

---

### 3.4 Natural disaster — `src/agents/disaster_agent.py`

Same weighted-composite pattern as geopolitical, with **single-day validation**.
A magnitude-6.5 earthquake on day N *is* the signal; requiring multi-day
persistence would mask it.

**Features**: `earthquake_severity`, `tsunami_risk`, `cyclone_severity`,
`severe_weather_index`, all ∈ [0,1] after a proximity decay (full weight within
500 km of 26.5N/56.5E, decaying to 1500 km).

**Preprocess.** Pass-through — disasters are sparse, a rolling baseline is
meaningless.

**Score.**

```
composite     = Σ_k w_k · x_k
max_single    = max_k x_k
anomaly_score = clip( max(composite, max_single), 0, 1 )
is_anomaly    = (composite ≥ composite_threshold) ∨ (max_single ≥ single_event_threshold)
```

**Validation.** Only a floor: `validated = is_anomaly ∧ (max_single ≥ 0.10)`.
No persistence gate at all.

**Confidence** = `min(1, 0.5 + share of features > 0.30)`.

**Parameters** — hand: weights 0.35 / 0.30 / 0.20 / 0.15, composite threshold
0.30, single-event 0.40. Optimized: 0.229 / 0.379 / 0.215 / 0.178, thresholds
0.380 / 0.480.

Ambee categorical inputs are mapped to numeric severity via
`agents.natural_disaster.severity_mapping` (proximity 0.6 / alert 0.4 blend) —
a deliberate approximation, since those endpoints carry no underlying magnitude.

---

### 3.5 Routing — `src/agents/routing_agent.py` (**dormant**)

Retained, tested, and documented here for completeness. It contributes nothing
to any current run.

**Features**: `rerouting_percentage`, `avg_route_deviation_km`,
`transit_volume_ratio`, `vessels_holding`, `alternative_route_traffic`.

**Fit.** `StandardScaler` + `IsolationForest(contamination=0.08, n_estimators=200,
random_state=42)` on non-disruption rows only. Model tagged with a
`model_version` (default `hormuz_v1.0`) so it can be retargeted per corridor.

**Score.**

```
iforest_norm  = batch min-max of (−decision_function)
transit_znorm = min( |z(transit_volume_ratio)| / 3 , 1 )
anomaly_score = w_model · iforest_norm + w_z · transit_znorm
```

⚠ Note the inconsistency worth flagging in the write-up: routing normalises its
forest score by **batch** min-max, whereas shipping was deliberately changed to
**fit-time percentiles** for exactly the reason described in §3.1. Routing was
never migrated because it went dormant first.

**Validation**: persistence ≥ 2 days AND `rerouting_percentage ≥ 10`.

**Parameters** — hand: `model_score 0.6 / transit_zscore 0.4`, threshold 0.55.
Optimized: `0.425 / 0.575`, threshold 0.491.

---

### 3.6 News sentiment — `src/agents/news_agent.py`

**Features**

| Feature | Used in score? | Notes |
|---|---|---|
| `sentiment_score` | indirectly | required; drives velocity |
| `sentiment_magnitude` | **no** | schema/reporting only |
| `source_consensus` | yes | absent for GDELT timelines |
| `article_volume` | yes | via a spike ratio |
| `recency_weighted_score` | yes | the actual sentiment term |

Derived in `preprocess()`: `sentiment_rolling_7d` (computed, **unused in
scoring**), `sentiment_velocity` = `sentiment_score.diff(3)`,
`volume_rolling_30d`.

**Score.** Four components, each normalised into [0,1] before weighting:

```
neg_sent  = clip(−recency_weighted_score, 0, 1)
consensus = clip(source_consensus, 0, 1)                    # None when unavailable
velocity  = clip(−sentiment_velocity, 0, 1)                 # falling fast = risk
volume    = clip( article_volume / volume_rolling_30d / spike_multiplier , 0, 1 )

anomaly_score = Σ_available w_k · c_k  /  Σ_available w_k
```

Same renormalisation discipline as geopolitical: GDELT's timeline endpoints
return an aggregate tone per day with no per-source breakdown, so
`source_consensus` cannot be measured. Scoring it as zero would read as "outlets
disagree" rather than "not measured", and would silently damp every score by its
0.25 weight.

**Validation**: persistence ≥ 2 days AND `recency_weighted_score ≤ negative_threshold`
AND (`source_consensus ≥ consensus_threshold`, applied only when the column exists —
requiring it otherwise would suppress every flag).

**Confidence** = `clip(mean(consensus_norm) + 0.1, 0, 1)`.

**Parameters** — hand: sentiment 0.40 / consensus 0.25 / velocity 0.20 /
volume 0.15, composite threshold 0.40, negative threshold −0.30, consensus
threshold 0.40, volume spike multiplier 2.0. Optimized: 0.415 / 0.206 / 0.318 /
0.061, negative −0.412, consensus 0.394.

---

## 4. Aggregation — `src/aggregation/risk_engine.py`

Two entry points with different contracts. Both are live.

### 4.1 `aggregate()` — legacy, 4-level

Plain weighted mean of each agent's **mean** anomaly score over its whole score
array, renormalised across agents that have a configured weight. Classified
against four bands (`CRITICAL / HIGH / MEDIUM / LOW`). Kept for backwards
compatibility; `run_full_pipeline` returns it alongside the richer result.

### 4.2 `compute_risk()` — the one that matters

```
1.  keep agents that are present AND have a configured weight AND produced scores
2.  score_a   = mean(anomaly_scores_a)                    # per agent
3.  w_norm_a  = w_a / Σ w                                 # renormalised over ACTIVE agents → sums to 1
4.  base      = Σ_a w_norm_a · score_a
5.  agreement = |{ a : score_a > 0.5 }|
6.  amp       = 1.25  if agreement ≥ 5
                1.15  if agreement ≥ 3
                1.00  otherwise
7.  risk      = min(base · amp, 1.0)
8.  level     = high   if risk ≥ risk_high
                medium if risk ≥ risk_medium
                low    otherwise
```

Step 3 is what makes a passive or dormant agent harmless: its weight is simply
redistributed proportionally across the agents that did run, rather than the
composite being dragged toward zero by a missing contributor.

Step 6 is the only non-linearity in the whole aggregation. Rationale:
corroboration across *independent* domains raises confidence beyond what a
linear weighted sum expresses.

Note for accuracy in the write-up: the **agreement threshold (0.5) is a module
constant** and is not exposed to config or to the optimizer, while the two
**bonus multipliers are** both configurable and optimized. Only the multipliers
move.

The returned dict carries the full audit trail — `risk_score`, `risk_level`,
per-agent `{score, weight, contribution}`, `agent_agreement`, `timestamp`,
`weights_used`, and a generated one-sentence `reason` naming the lead driver,
its share of weighted risk, supporting agents above threshold, and any
amplification applied.

### 4.3 Layer-2 weights and risk bands

| Agent | Hand-tuned | Optimized |
|---|---|---|
| shipping | 0.25 | 0.402 |
| market | 0.15 | 0.088 |
| geopolitical | 0.25 | 0.109 |
| natural_disaster | 0.10 | 0.095 |
| routing | 0.15 | 0.130 |
| news_sentiment | 0.10 | 0.177 |

| Band | Hand-tuned | Optimized |
|---|---|---|
| `risk_critical` | 0.90 | — (4-level path only) |
| `risk_high` | 0.69 | 0.766 |
| `risk_medium` | 0.51 | 0.276 |
| `agreement_bonus_3` | 1.15 | 1.277 |
| `agreement_bonus_5` | 1.25 | 1.443 |

The hand-tuned bands are **empirically calibrated, not picked**: they are the
p60 / p85 / p97 quantiles of the composite score on *calm* (label-negative) days,
pooled across all four regions 2019–2026 (9,183 days). They were recalibrated
when the shipping agent stopped batch-normalising its forest score — the same
quantiles of calm behaviour, expressed on the new absolute scale, not a
loosening.

### 4.4 Per-day vs single-shot

Three code paths compute a composite, and they are not identical — be precise
about which one produced any number you report:

| Path | Granularity | Used by |
|---|---|---|
| `RiskEngine.compute_risk` | one scalar over the whole run (means the score arrays) | `run_full_pipeline`, API, dashboard |
| `RiskEngine.compute_risk_timeseries` | per day, calls `compute_risk` per day | dashboard time series |
| `Orchestrator.run_timeseries_analysis` | per day, weighted mean, **no agreement bonus** | dashboard analysis view |
| `PipelineEvaluator._aggregate_daily` | per day, vectorised, **with** agreement bonus | optimizer + all evaluation |

`_aggregate_daily` is a faithful vectorised mirror of `compute_risk`, so the
number the optimizer maximises is the number the live pipeline produces.
`run_timeseries_analysis` is the outlier — it omits amplification.

---

## 5. What happens to the features downstream

### 5.1 Explainability — `src/explainability/shap_explainer.py`

A `RandomForestRegressor(n_estimators=100, random_state=42)` is trained as a
**surrogate** to reproduce the pipeline's per-day composite risk from a flat
20-feature row, then explained with `shap.TreeExplainer`. Train-set R² is logged
and warned below 0.85.

The 20 canonical features (`ALL_FEATURE_NAMES`), each mapped to its agent by
`FEATURE_AGENT_MAP`:

- shipping (3): `vessel_count`, `avg_delay_hours`, `congestion_index`
- market (3): `brent_crude_usd`, `trade_volume_index`, `freight_rate_index`
- geopolitical (4): `sanctions_severity`, `military_activity_index`, `diplomatic_incident_score`, `regime_stability_index`
- natural_disaster (4): `earthquake_severity`, `tsunami_risk`, `cyclone_severity`, `severe_weather_index`
- routing (3): `rerouting_percentage`, `avg_route_deviation_km`, `transit_volume_ratio`
- news_sentiment (3): `sentiment_score`, `source_consensus`, `article_volume`

This is a **subset**, not the full agent input set. Dropped from the SHAP view:
`tanker_count`, `vessel_count_trend`, `freight_services_pct_change`,
`vessels_holding`, `alternative_route_traffic`, `sentiment_magnitude`,
`recency_weighted_score`. Missing columns are filled with 0.0 rather than
raising, so a disabled agent never crashes an explanation — which also means a
zeroed feature and a genuinely-zero feature look identical in the SHAP output.

Output: per-feature SHAP values, top-3 drivers with agent attribution, expected
value, and a generated natural-language explanation.

### 5.2 Retrieval — `src/rag/context_retriever.py`

Gated: historical precedents are retrieved only when the composite risk score
clears `rag.composite_threshold` (0.65), with `min_similarity 0.55` and
`all-MiniLM-L6-v2` embeddings, over two collections (10 static historical cases
+ live API-extracted context). A RAG failure is logged and never aborts a run.

### 5.3 Weight optimization — `src/optimization/`

**Splits (`data_split.py`).** You cannot shuffle rows here — that leaks the
shape of a disruption across splits and destroys the time ordering every
rolling window depends on. Instead, three **independent realisations** of the
same world are generated by re-seeding every connector: identical disruption
structure (same scenarios, same day positions 60–74 / 150–170 / 280–290), but
day-to-day noise drawn from a different random stream — seed 42 train, 43
validation, 44 test, 365 days each. A weight set that tunes on train and scores
on validation has demonstrably generalised across noise rather than memorised
one sample path. `is_disruption` is carried as an evaluation label only and is
**never** exposed to an agent as an input feature.

**Search.** Optuna, TPE sampler (`seed=42`), median pruner, 100 trials, 1-hour
cap. Raw values are suggested then renormalised so each weight group sums to 1
(Dirichlet-style — unconstrained search, valid injected weights). Hard
constraints reject trials where `risk_high ≤ risk_medium` or
`agreement_bonus_5 ≤ agreement_bonus_3`.

**Objective** (`pipeline_evaluator.py`) — fit on train, score on validation:

```
objective = 0.50 · F1  +  0.30 · lead_time_score  −  0.20 · FPR
```

- **F1** — HIGH-risk alert (`risk ≥ risk_high`) as the positive prediction vs
  `is_disruption`.
- **lead_time_score** — for each ground-truth window, the earliest MEDIUM-level
  alert within the 5 days before onset; mean lead days ÷ 5, clipped to [0,1].
- **FPR** — false-positive rate of HIGH-risk alerts, subtracted.

**Test split discipline.** Touched exactly once, in `optimize()`, after the
study finishes. `PipelineEvaluator.evaluated_splits` is an audit trail so the
leakage test can prove it stayed untouched during the search.

Best run on record: trial 63 of 100, validation objective 0.7491
(`config/optimized_weights.yaml`, 2026-08-30).

Caveat for the write-up: `PipelineEvaluator.build_agents` **hardcodes** some
agent config (shipping `contamination=0.05`, `z_threshold=2.0`; market
`baseline_years=5`, `threshold=0.50`) rather than reading `settings.yaml`. The
optimizer therefore tunes a slightly different agent than the live pipeline
runs unless those values match.

### 5.4 Evaluation

Two harnesses with different data and different purposes.

**A. `notebooks/evaluation.py` — synthetic, 8 metrics, both weight modes side by side.**

| Metric | What it measures |
|---|---|
| 1 Detection | per-agent (fixed 0.5 cutoff) + system precision/recall/F1/FPR/lead time, TEST split |
| 2 Faithfulness | SHAP top-3 vs *planted*, cleanly-disjoint anomalies (target > 0.80) |
| 3 Agent diversity | 6-agent vs 2-agent vs 1-agent ablation — agents disabled by **zeroing the aggregation weight and renormalising**, never by deleting code |
| 4 Baseline | naive 2-sigma (any feature, \|z\| > 2) vs hand-tuned vs optimized |
| 5 Optimization impact | deltas + top-5 shifted parameters from `optimization_results.json` |
| 6 RAG | retrieval relevance |
| 7 Generalization | validation vs test under optimized weights |
| 8 Decision effectiveness | risk + SHAP + precedent → correct action, over `{no_action, monitor, reroute, escalate}`, via a transparent rule set — not another model |

**B. `scripts/run_method_comparison.py` — real connector data, honest comparison.**

Reads the cached real per-region frames from `scripts/build_eval_dataset.py`
(PortWatch + FRED + GDACS/USGS + ACLED, GDELT when it answers). Three design
decisions carry the weight:

1. **One temporal split for everyone.** Last 30% of each region's series is the
   test window; nothing is fitted on it. A supervised model scored on its own
   training rows would post an inflated, non-comparable number. Temporal rather
   than random because shuffling days leaks the future into the past.
2. **Tiers build real agents.** A tier fits and runs actual agent classes and
   composites them with the project's own weights. Approximating a tier by
   averaging "the first N feature columns" would measure nothing — the first two
   columns of these frames are both shipping features, so a "Tier 2" built that
   way contains no market signal at all.
3. **Malacca is a false-positive harness, not a detection test.** Zero labelled
   disruptions 2019–2026, so AUC is undefined there; what it measures is how
   often each method fires when nothing is happening.

**Circularity is recorded per method** (`_CIRCULARITY`), because the label is
"30-day mean of `vessel_count` ≥ 20% below its trailing 365-day median, sustained
14+ days" — any method computing that same statistic predicts it by construction.
Ratings: `high` (reads `vessel_count` against a trailing long baseline — the
label's own statistic), `medium` (short/rolling baseline, or fits on the label),
`low` (other features, or shape rather than level), `n/a` (controls and oracles).
**Tiers 1+ are rated `high`**, because they include the shipping agent's
level-shift feature.

Label weakness travels with the data in the manifest: a 30-day rolling baseline
drifts down with a slow decline, so the label catches shocks and misses
slow-onset disruption. Panama's 2023–24 drought is the case in point — transits
fell 38% and the label flags almost none of it.

Results are in `eval/method_comparison_results.csv` and
`eval/COMPARISON_REPORT.md`.

---

## 6. Known inconsistencies worth naming before a reader finds them

1. **Routing normalises its Isolation Forest by batch min-max**; shipping was
   deliberately moved to fit-time percentiles. Routing was never migrated
   because it went dormant first (§3.1, §3.5).
2. **`run_timeseries_analysis` omits the agreement bonus** that every other
   composite path applies (§4.4).
3. **The agreement threshold (0.5) is not configurable** while the bonuses it
   gates are (§4.2).
4. **Geopolitical computes rolling baselines and deviations that `detect()`
   never reads** (§3.3); news does the same with `sentiment_rolling_7d` (§3.6).
5. **`PipelineEvaluator.build_agents` hardcodes agent config** rather than
   reading `settings.yaml` (§5.3).
6. **The SHAP surrogate sees 20 of the features**, not all of them, and fills
   absent columns with 0.0 — indistinguishable from a true zero (§5.1).
7. **The shipping duration score is near-circular against the level-shift
   label** and should be cited operationally, not as detection skill (§3.1).
8. **With routing dormant everywhere, no evaluation can measure its
   contribution** — so the evaluation results cannot, on their own, justify
   either keeping it off or turning it on (§1).
