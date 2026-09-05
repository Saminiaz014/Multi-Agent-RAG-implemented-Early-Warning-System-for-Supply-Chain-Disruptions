# Multi-Agent RAG-Implemented Early Warning System for Supply Chain Disruptions

A **Decision Support System (DSS)** that detects, explains, and contextualises disruptions at four maritime chokepoints. Six domain agents monitor heterogeneous external signals; a weighted aggregation layer fuses them into a composite risk score; a SHAP surrogate attributes that score to input features; and a retrieval layer grounds it in comparable historical events.

> ### The measured claim
>
> The claim is **not** "this detects disruptions better than existing methods." The evaluation in [§4](#4-evaluation) does not support that, and it was deliberately built to expose the fact. The defensible claim is narrower:
>
> **A multi-agent architecture with explicit per-domain weighting produces interpretable, auditable, decision-ready alerts — attribution to a named domain, a SHAP feature breakdown, and a retrieved historical precedent — which single-model detectors cannot produce, at a detection cost this evaluation quantifies honestly rather than hides.**
>
> The contribution is not a better detector. It is **an evaluation methodology that catches its own circularity**, plus a diagnosed mechanism for why multi-domain fusion underperforms on a single-domain label ([§4.7](#47-root-cause-analysis--why-hormuz-sits-at-chance)).

| | |
|---|---|
| **Regions** | Hormuz · Bab el-Mandeb · Panama · Malacca (2019–2026) |
| **Test suite** | 420 passed / 0 failed |
| **Headline result** | Panama Tier 1 **AUC 0.909** · Hormuz Tier 1 **AUC 0.502** (chance) |
| **Central negative finding** | Adding agents **reduces** AUC in every evaluable region |
| **Source of record** | [`docs/THESIS_BRIEF.md`](docs/THESIS_BRIEF.md) · [`docs/SCORING_REFERENCE.md`](docs/SCORING_REFERENCE.md) |
| **Build history** | [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) — the former phase-by-phase README body |

---

## Contents

- [1. Problem and Scope](#1-problem-and-scope)
- [2. Research Questions](#2-research-questions)
- [3. Methodology](#3-methodology)
  - [3.1 Research design](#31-research-design) · [3.2 Requirements](#32-design-requirements) · [3.3 Architecture](#33-system-architecture) · [3.4 Detection models](#34-detection-models) · [3.5 Evidence discipline](#35-evidence-discipline) · [3.6 Weighting](#36-weight-determination) · [3.7 Aggregation](#37-risk-aggregation) · [3.8 Explainability & RAG](#38-explainability-and-retrieval) · [3.9 Implementation](#39-implementation)
- [4. Evaluation](#4-evaluation)
  - [4.1 Design](#41-evaluation-design) · [4.2 Data](#42-dataset-and-ground-truth) · [4.3 Baselines](#43-baselines-tiers-and-circularity) · [4.4 Metrics](#44-metrics) · [4.5 Results](#45-results) · [4.6 Negative finding](#46-the-central-negative-finding) · [4.7 Root cause](#47-root-cause-analysis--why-hormuz-sits-at-chance) · [4.8 Rejected hypotheses](#48-hypotheses-tested-and-rejected) · [4.9 Optimization](#49-weight-optimization-results) · [4.10 Validity](#410-threats-to-validity--what-cannot-be-claimed) · [4.11 Defects](#411-known-defects-and-open-issues)
- [Project Structure](#project-structure) · [Installation](#installation) · [Running](#running)

> **On `Phase N` tags.** Sections 1–4 describe the system as it is. The reference sections that follow them (Project Structure, Installation, Configuration) still carry `Phase N` annotations recording *when* a file, dependency or config key was introduced. Those are pointers into [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md), not part of the current structure.

---

## 1. Problem and Scope

Maritime chokepoints concentrate global trade risk into a handful of narrow corridors. A disruption at one propagates across supply chains within days, and the decision-maker's problem is not obtaining data — vessel transits, commodity prices, conflict events and news are all available — but converting scattered, heterogeneous, largely unstructured signals into a judgement that can be acted on and defended.

Purely predictive systems are insufficient here. A risk score with no attribution cannot be audited, cannot be argued with, and gives a manager no basis for choosing between "do nothing" and "reroute a fleet." This system is therefore built around **attribution, precedent and auditability** as first-class outputs, not as post-hoc decoration on a classifier.

### Domain

| Region | Display | Lat / Lon | Documented driver in window |
|---|---|---|---|
| `hormuz` | Strait of Hormuz | 26.50 / 56.50 | Geopolitical — sanctions, military activity; ~20 % of global oil trade |
| `bab_el_mandeb` | Bab el-Mandeb | 12.58 / 43.33 | Security campaign — Houthi attacks, Cape of Good Hope diversion |
| `panama` | Panama Canal | 9.08 / −79.68 | Hydrological — Gatún Lake drought |
| `malacca` | Strait of Malacca | 2.50 / 101.80 | **None documented** — used as a control |

---

## 2. Research Questions

| SRQ | Question | Answered by | Verdict |
|---|---|---|---|
| **SRQ1** | Can multi-domain signals detect chokepoint disruption? | `scripts/run_method_comparison.py` | Partially — Panama yes (0.909), Hormuz no (0.502) |
| **SRQ2** | Does agent diversity add value over fewer agents? | Tier ablation · `notebooks/evaluation.py` METRIC 3 | **No — refuted on real data** ([§4.6](#46-the-central-negative-finding)) |
| **SRQ3** | Can the score be explained faithfully? | `src/explainability/shap_explainer.py` · METRIC 2 | Yes, with a stated surrogate caveat |
| **SRQ4** | Does weight optimization improve on hand-tuning? | `src/optimization/` · METRIC 5 | Unresolved — blocked by [§4.11](#411-known-defects-and-open-issues) |
| **SRQ5** | Would a decision-maker be led to the correct action? | `src/evaluation/decision_effectiveness.py` · METRIC 8 | Not currently citable — see [§4.11](#411-known-defects-and-open-issues) |

---

# 3. Methodology

## 3.1 Research design

The work follows **Design Science Research**: an artifact is built to address a practical problem, then evaluated against criteria fixed in advance. The artifact is the DSS; the evaluation is the method comparison in [§4](#4-evaluation).

The methodological commitment that shapes everything below is that **an evaluation must be able to fail**. Three design decisions enforce this — a temporal split applied identically to every method, tiers that instantiate real agent classes rather than column averages, and an explicit circularity rating attached to every method. Each is described in [§4.1](#41-evaluation-design). They are what allow the negative results in [§4.6](#46-the-central-negative-finding) to be trusted rather than explained away.

## 3.2 Design requirements

Derived from the problem statement in [§1](#1-problem-and-scope):

| # | Requirement | Realised by |
|---|---|---|
| R1 | Separate ingestion, detection, and decision support so each can change independently | Layered architecture, [§3.3](#33-system-architecture) |
| R2 | Handle heterogeneous structured and unstructured sources | Six connectors behind one ABC |
| R3 | Attribute every score to a named domain | Per-agent breakdown in `RiskEngine` |
| R4 | Attribute every score to named input features | SHAP surrogate, [§3.8](#38-explainability-and-retrieval) |
| R5 | Ground alerts in comparable past events | RAG retrieval, [§3.8](#38-explainability-and-retrieval) |
| R6 | Keep every number auditable by a domain expert | No end-to-end learned model, [§3.4](#34-detection-models) |
| R7 | Degrade gracefully when a source or agent is unavailable | Weight renormalisation, [§3.7](#37-risk-aggregation) |

## 3.3 System architecture

### Analysis

The pipeline is a strict one-way flow. Each stage has a single responsibility and a defined output contract, so a stage can be replaced without touching its neighbours — the property that made it possible to swap five connectors from synthetic to live data without changing a single agent.

```
connector (ingestion)      raw domain frame, daily index
    ↓
agent.fit()                schema check and/or model calibration
agent.preprocess()         scaling · rolling baselines · derived columns
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

### Implementation

| Layer | Responsibility | Code |
|---|---|---|
| **1 — Ingestion** | Fetch, validate, and normalise each domain to a daily index | `src/ingestion/*_connector.py`, all extending `BaseConnector` |
| **2 — Detection** | Per-domain anomaly scoring | `src/agents/*_agent.py`, all extending `BaseAgent` |
| **3 — Aggregation** | Fuse agent scores into a composite risk + level | `src/aggregation/risk_engine.py` |
| **4 — Explainability** | Attribute the composite to input features | `src/explainability/shap_explainer.py` |
| **5 — Retrieval** | Ground the alert in historical precedent | `src/rag/context_retriever.py` |
| **6 — Presentation** | Decision view and analysis view | `src/dashboard/`, `src/api/endpoints.py` |

Each connector retains `csv` and `synthetic` modes alongside `api`. `synthetic` is the only mode carrying ground-truth labels, which is why the optimizer and the 8-metric suite still use it.

### Layer 1 in detail — live data sources

Five of the six agents read a live, free public API; the sixth is dormant by decision ([§3.5](#35-evidence-discipline)).

| Agent | Live source | Access | Notes |
|---|---|---|---|
| **Shipping** | [IMF PortWatch](https://portwatch.imf.org) Daily Chokepoints (ArcGIS FeatureServer) | **No credentials** | Daily transits per chokepoint, 2019–2026, all four regions. The evidence-grade backbone of the evaluation set |
| **Market** | [FRED API](https://fred.stlouisfed.org/docs/api/fred/) | Free (key) | Brent Crude `DCOILBRENTEU`, Freight PPI `PCU4831114831111`, Freight Services Index `PCUATFREIATFREI` |
| **Geopolitical** | [ACLED](https://acleddata.com) | Free (OAuth) | Battles/Explosions → `military_activity_index`; Strategic developments → `diplomatic_incident_score`; Protests/Riots → inverse `regime_stability_index` |
| **Natural Disaster** | [GDACS](https://www.gdacs.org) + [USGS](https://earthquake.usgs.gov/fdsnws/event/1/) | **No credentials** | GDACS Orange/Red alerts only — green-inclusive queries exceed GDACS's silent 100-result cap every month. USGS supplies magnitude and the tsunami flag |
| **News Sentiment** | [GDELT DOC API v2](https://api.gdeltproject.org/api/v2/doc/doc) | **No credentials** | Tone scores. Answered for Panama only in the evaluation window |
| **Routing** | — | — | **Dormant** — no free source supplies evidence-grade rerouting data |

**Two deliberate absences, each an affirmative finding rather than a gap:**

- **`sanctions_severity` is not produced.** ACLED carries no sanctions data; OpenSanctions is paywalled and publishes current designations rather than a time series; and sanctions are discrete events, so any daily "severity" curve would be a modelling artefact. `GeopoliticalAgent` renormalises its weights over the three features ACLED does supply. Scoring the missing column as zero would read as *"no sanctions risk"* rather than *"not measured."*
- **`NewsAgent` renormalises the same way** when GDELT does not answer for a region.

> **Naming trap.** `ShippingConnector` and `MarketConnector` each carry two similarly named methods. `fetch_from_api()` is the real live path and is what `fetch()` dispatches to in `api` mode. `fetch_api()` is a leftover convenience hook that logs a warning and **silently returns synthetic data**. Call `fetch()` or `fetch_and_validate()`, never `fetch_api()` directly. Several docstrings in these two modules still describe `api` mode as an unimplemented aisstream.io stub and are themselves out of date.

## 3.4 Detection models

### Analysis

**There is no end-to-end learned model, and this is a deliberate design position rather than an unfinished component.** The only fitted models in the entire scoring path are two Isolation Forests and their scalers. Everything else is explicit arithmetic — weighted composites, rolling z-scores, a sigmoid.

The consequence is that every number the system produces traces to a formula a domain expert can read and dispute. That is the property the decision-support claim rests on. A gradient-boosted model over the same features would very likely score better on the metrics in [§4.5](#45-results) and would forfeit exactly the thing this system exists to provide.

The RandomForest in `src/explainability/` is a **post-hoc surrogate**. It explains the score; it never produces it.

### Implementation

| Agent | Model |
|---|---|
| `shipping` | Isolation Forest (200 trees, seed 42) + per-feature z-score + level-shift duration score |
| `market` | Rolling 30-day trailing z-scores, weighted mean of \|z\| |
| `geopolitical` | Weighted linear composite → sigmoid (gain 6, centre 0.5) |
| `natural_disaster` | Weighted composite + single-event max override |
| `routing` | Isolation Forest (200 trees, seed 42) + transit-ratio z-score — **dormant** |
| `news_sentiment` | Weighted composite of four normalised components |
| **aggregation** | Renormalised weighted mean + non-linear agreement bonus |

Per-agent formulas: [`docs/SCORING_REFERENCE.md`](docs/SCORING_REFERENCE.md).

## 3.5 Evidence discipline

### Analysis

An agent can be off for three distinct reasons, and conflating them would misrepresent the system's coverage. Each exclusion below is an **affirmative finding about the evidence**, not missing data:

- **Active** — built, run, and weighted.
- **Passive** — a per-region evidence judgement. The domain is real, but no documented driver exists at that chokepoint in the observation window. It would activate if evidence appeared.
- **Dormant** — a project-scope decision (`DORMANT_AGENTS`). No region may activate it; enforced at registry validation.

| Agent | Hormuz | Panama | Bab el-Mandeb | Malacca |
|---|---|---|---|---|
| shipping | active | active | active | active |
| market | active | active | active | **passive** |
| geopolitical | active | **passive** | active | active |
| natural_disaster | active | active | **passive** | active |
| routing | **dormant** | **dormant** | **dormant** | **dormant** |
| news_sentiment | active | active | active | active |

**Exclusion reasons:**

- **market / Malacca** — all four documented Malacca events carry a null market field. Removed as a data-standards violation, not on plausibility grounds.
- **geopolitical / Panama** — the documented disruption is purely hydrological.
- **natural_disaster / Bab el-Mandeb** — the documented event is a security campaign; `disaster_relevance: none`.
- **routing / all four** — uniform muting. `fetch_api()` is a `NotImplementedError` stub, so the agent only ever emitted synthetic values.

**The asymmetry worth writing up:** Bab el-Mandeb's routing evidence is the *strongest in the benchmark* — 85 % of large containerships diverted via the Cape of Good Hope, +3,500–4,000 nm and +10–14 days per voyage, a documented percentage rather than an extrapolation — and routing is muted there *in spite of* that, because no free source supplies it as a time series.

**Consequence that must be stated:** with routing dormant everywhere, no evaluation can measure routing's contribution in any region. Evaluation results alone cannot settle whether to re-enable it.

### Implementation

`src/core/regions.py` holds the registry. `DORMANT_AGENTS` is a module-level `frozenset`; per-region activation lives in each `RegionConfig` and is overlaid from `config/regions/*.yaml`. Registry validation raises if a region config activates a dormant agent, so reviving routing is a deliberate edit to `DORMANT_AGENTS` rather than an accidental YAML toggle. Passive agents are simply absent from a region's active set and are handled by the renormalisation in [§3.7](#37-risk-aggregation).

## 3.6 Weight determination

### Analysis

Weights exist at three layers, all searchable by the optimizer:

| Layer | What it weights | Location | Optimized? |
|---|---|---|---|
| **L1** intra-agent | features within one agent | `agent.set_weights()` | yes |
| **L2** inter-agent | agents against each other | `RiskEngine.weights` | yes |
| **L3** thresholds | detection cutoffs, risk bands, agreement bonuses | `set_threshold()` + `RiskEngine` | yes |

`weight_mode` in `config/settings.yaml` selects `hand_tuned` (current default) or `optimized`.

**The hand-tuned risk bands are empirically calibrated, not chosen by intuition.** They are the p60 / p85 / p97 quantiles of the composite score on *calm* (label-negative) days, pooled across all four regions 2019–2026 — **9,183 days**:

```yaml
thresholds:
  risk_critical: 0.90   # p97 of calm days
  risk_high:     0.69   # p85
  risk_medium:   0.51   # p60  -> ~60% of calm days read LOW
  risk_low:      0.30
```

They were recalibrated when the shipping agent stopped batch-normalising its forest score. Batch min-max forced every scored window onto `[0,1]` by its own extremes, so the old `0.8 / 0.6 / 0.4` boundaries were tuned to a compressed, window-relative distribution. On absolute scores the calm median rose to ≈0.47. The new bands are the **same quantiles of calm behaviour expressed on the new scale — a re-scaling, not a loosening**.

### Implementation

| Concern | Code |
|---|---|
| Weight config and `weight_mode` switching | `src/optimization/weight_config.py` · `config/optimized_weights.yaml` |
| Optuna study, parameter space, constraints | `src/optimization/weight_optimizer.py` |
| Objective evaluation and split discipline | `src/optimization/pipeline_evaluator.py` |
| Split construction | `src/optimization/data_split.py` |

Optimization procedure and results: [§4.9](#49-weight-optimization-results).

## 3.7 Risk aggregation

### Analysis

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

**Step 3 is what satisfies R7.** Renormalising over *active* agents makes a passive or dormant agent harmless: its weight redistributes across the agents that did report, rather than contributing a zero that drags the composite down. Without it, muting routing in all four regions would have silently depressed every score by its weight.

**Step 6 is the only non-linearity in the entire scoring path.** It encodes the intuition that three domains agreeing is worth more than the sum of three domains individually.

### Implementation

`src/aggregation/risk_engine.py`. Note that **four composite paths exist and are not identical** — see [§4.11](#411-known-defects-and-open-issues) before citing any number.

## 3.8 Explainability and retrieval

### Analysis

Two mechanisms answer two different user questions, and they are deliberately complementary:

- **"Which features drove this score?"** — a SHAP surrogate over the feature space produces per-feature attributions and a top-drivers list.
- **"Has this happened before?"** — a retrieval layer matches the current multi-domain signal profile against a curated knowledge base of historical disruptions.

Feature attribution alone tells a manager *what moved*; precedent tells them *what it meant last time*. The decision-support argument in [§1](#1-problem-and-scope) needs both.

### Implementation

- **SHAP** — `src/explainability/shap_explainer.py`. A RandomForest surrogate is fitted to reproduce the composite, then explained. **The surrogate sees 20 of the features, not all** — see [§4.11](#411-known-defects-and-open-issues) item 6.
- **RAG** — `src/rag/context_retriever.py`, ChromaDB with a local ONNX embedding model (no API key). Knowledge base: 10 curated historical cases in `data/knowledge_base/disruption_cases.json` — `cyclone_gonu_2007`, `hormuz_mine_threat_2010`, `somali_piracy_2011`, `japan_earthquake_2011`, `iran_sanctions_2012`, `west_coast_port_strikes_2014`, `hormuz_2019`, `ever_given_2021`, `covid_port_congestion_2021`, `houthi_redsea_2024`. Retrieval is threshold-gated so a quiet day does not attach spurious precedent.
- **Action rubric** — `src/evaluation/decision_effectiveness.py` maps a risk level and its drivers to a recommended action through a transparent rubric, **deliberately not another model**.

## 3.9 Implementation

| | |
|---|---|
| **Scale** | 60 source modules · 25 test modules · 420 tests |
| **Stack** | Python 3.10+, pandas, scikit-learn, SHAP, ChromaDB, Optuna, FastAPI, Streamlit |
| **Config** | `config/settings.yaml` base + `config/regions/*.yaml` overlays, merged by `src/core/config_manager.py` |
| **Region registry** | `src/core/regions.py` — `RegionConfig`, `DORMANT_AGENTS`, `RETIRED_REGION_ALIASES`, validation |
| **Orchestration** | `src/orchestrator.py`; CLI entry `main.py` |

Full layout in [Project Structure](#project-structure); construction history in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md).

---

# 4. Evaluation

## 4.1 Evaluation design

### Analysis

Three decisions carry the weight of the entire evaluation. Each exists to prevent a specific way of accidentally reporting a flattering number.

**1. One temporal split, applied to everyone.** The last 30 % of each region's series is the test window and nothing is fitted on it. A supervised model scored on its own training rows posts an inflated number that is not comparable with an unsupervised detector. The split is **temporal rather than random** because shuffling days of a time series leaks the future into the past and destroys the ordering every rolling window depends on.

**2. Tiers instantiate real agents.** A tier is a set of agent classes; each is fitted and run, and the composite is the weighted mean of their scores under the project's own weights. Approximating a tier by averaging "the first N feature columns" would measure nothing — the first two columns of these frames are both shipping features, so a "Tier 2" built that way would contain no market signal at all despite its label.

**3. Malacca is a false-positive harness, not a detection test.** With zero labelled disruptions its AUC is undefined. What it measures is the **alert rate on a region where nothing happened** — the number that decides whether an early warning system is usable in practice.

### Circularity rating — the decision that makes the evaluation credible

The ground-truth label is "vessel_count against a trailing long baseline." **Any method computing that same statistic predicts the label by construction.** Rather than leave this implicit, every method carries a recorded rating:

| Rating | Meaning |
|---|---|
| `high` | reads vessel_count against a trailing long baseline — the label's own statistic |
| `medium` | short/rolling baseline, or fits on the label directly |
| `low` | reads other features, or shape rather than level |
| `n/a` | controls and oracles |

**Every tier is rated `high`**, because all tiers include the shipping agent's level-shift feature. This is why tier AUC cannot be cited as clean detection skill — and why the result graphs colour bars by circularity, so a ranking cannot be misread as "tallest is best."

## 4.2 Dataset and ground truth

### The evaluation dataset

Built by `scripts/build_eval_dataset.py` from the live connectors, merged on a daily index:

| Region | Rows | Dropped | Date range | Positive days | Rate | Missing domain |
|---|---|---|---|---|---|---|
| hormuz | 2,434 | 358 | 2019-01-01 → 2025-08-30 | **159** | 6.53 % | news_sentiment |
| bab_el_mandeb | 2,434 | 358 | 2019-01-01 → 2025-08-30 | **244** | 10.02 % | news_sentiment |
| panama | 2,433 | 359 | 2019-01-01 → 2025-08-29 | **149** | 6.12 % | — (all present) |
| malacca | 2,434 | 358 | 2019-01-01 → 2025-08-30 | **0** | 0.00 % | news_sentiment |

**Live sources:** shipping = IMF PortWatch chokepoint transits · market = FRED (Brent, freight PPI, freight services) · natural_disaster = GDACS (Orange/Red only) + USGS (M ≥ 4.0) · geopolitical = ACLED · news_sentiment = GDELT where it answered.

**Two facts that must be stated whenever this dataset is cited:**

1. **The frame ends 2025-08-30.** The pinned April–May 2026 Hormuz shutdown is **not in it**. The 159 Hormuz positives are earlier events.
2. **news_sentiment is missing from three of four regions** — GDELT did not answer. Column sets are recorded per region in the manifest specifically so a five-feature region is never silently compared against a four-feature one.

### The label, and its recorded weakness

```
y_true = 30-day mean of vessel_count is ≥20% below the trailing
         365-day median, sustained 14+ consecutive days
```

**The weakness travels with the data in the manifest, not in a comment:** a rolling baseline drifts downward during a slow decline, so the label catches shocks and misses slow-onset disruption. **Panama's 2023–24 drought cut transits by 38 % and the label flags almost none of it.**

### Synthetic splits

The optimizer and `notebooks/evaluation.py` use synthetic data because it is the only source carrying labels. Rows cannot be shuffled, so instead three **independent realisations** of the same world are generated by re-seeding every connector: identical disruption structure (days 60–74, 150–170, 280–290), different noise stream. Seeds 42 / 43 / 44 for train / validation / test, 365 days each. `is_disruption` is an evaluation label only and is **never** an agent input.

## 4.3 Baselines, tiers, and circularity

Methods span controls, classical detectors, supervised models, and an oracle:

- **Controls** — M0 random, M1 always-alert, M2 never-alert. A method that cannot beat these has no signal.
- **Unsupervised** — B1 rolling z, B2 MA crossover, B3 isolation forest, B4 EWMA deviation, B5 AR residual, B6 CUSUM, M3 persistence, M8 matrix profile.
- **Supervised** — B7 logistic regression, B8 random forest (n/a where the training window has no positives).
- **Oracle** — label at t−1. An upper bound, not a competitor.
- **Tiers 1–5** — the multi-agent system, progressively adding agents.

## 4.4 Metrics

| Metric | What it captures | Note |
|---|---|---|
| **AUC** | Ranking quality, threshold-free | Undefined on Malacca (0 positives) |
| **F1** | Balance of precision and recall at the operating threshold | |
| **FPR** | False-positive rate — the number that decides operational usability | |
| **Alert rate** | Fraction of days alerted | The only meaningful metric on Malacca |

An 8-metric suite (`notebooks/evaluation.py`) additionally covers detection, explainability faithfulness, agent diversity, lead time, optimization gain, RAG quality, and decision effectiveness. **Those numbers are not currently citable** — see [§4.11](#411-known-defects-and-open-issues).

## 4.5 Results

Reported as found. Source: `eval/method_comparison_results.csv` (76 rows), graphs in `eval/graphs_method_comparison/` and `eval/graphs_ablation_tiers/`.

### Hormuz — 159 positives (6.53 %)

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

**Best baseline 0.986 vs best tier 0.502 → gap −0.484. Tier 1 is at chance**, and Tiers 2–5 fall below it.

### Bab el-Mandeb — 244 positives (10.02 %)

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

Gap −0.318. B7/B8 not applicable — no positives in the training window.

### Panama — 149 positives (6.12 %) — the strongest region

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

**Panama Tier 1 at 0.909 beats every non-oracle unsupervised baseline except persistence.** This is the system's strongest real result.

### Malacca — 0 positives — false-positive harness

AUC undefined. What is measured is the alert rate on a region where nothing happened:

| Method | Alert rate |
|---|---|
| M2 never · B6 CUSUM · ORACLE | 0.000 |
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

**The tiers are the noisiest methods on the quiet region**, and adding agents makes it worse (0.178 → 0.257). This is the cost of the fusion design, and it is reported rather than buried.

## 4.6 The central negative finding

**Across every evaluable region, tier AUC falls as agents are added.**

| Region | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 |
|---|---|---|---|---|---|
| hormuz | 0.502 | 0.465 | 0.400 | 0.401 | 0.401 |
| bab_el_mandeb | 0.679 | 0.620 | 0.449 | 0.449 | 0.449 |
| panama | 0.909 | 0.884 | 0.876 | 0.876 | 0.884 |
| malacca *(alert rate — lower is better)* | 0.178 | 0.181 | 0.257 | 0.257 | 0.257 |

**This directly refutes the agent-diversity hypothesis (SRQ2) on real data.**

**Mechanism — dilution.** The label is shipping-derived. Every non-shipping agent therefore contributes weighted score that is uncorrelated with the label, pulling the composite toward noise. The finding is not that multi-domain fusion is worthless; it is that **fusion cannot help against a single-domain label**, and the evaluation was constructed well enough to show it.

## 4.7 Root-cause analysis — why Hormuz sits at chance

Decomposing Hormuz's own anomaly score on the test window (2023-08-31 → 2025-08-30, 133 positives / 731 days):

| Component | Mean on normal | Mean on disruption | **Separation** |
|---|---|---|---|
| `duration_score` | 0.0497 | 0.4395 | **+0.3898** |
| `shock_score` | 0.5455 | 0.3982 | **−0.1473** |
| `anomaly_score` = `max(shock, duration)` | 0.5561 | 0.5315 | **−0.0246** |

**The duration signal separates the label cleanly (+0.39). The shock detectors are anti-correlated (−0.15) — disruption days look *less* anomalous to them than calm days.** Because shock sits ≈0.55 on quiet days while duration only reaches ≈0.44 on disrupted ones, `max()` returns shock nearly everywhere and **erases the working signal**, leaving −0.02.

The code comment defends `max()` over averaging on the grounds that a calm shock detector would dilute an active duration signal. It does prevent dilution — and causes **masking** instead. **This is the actual root cause of Hormuz ≈ 0.50**: not forest training, not the validation gate.

**Secondary cause.** `detect()` receives only the test slice, so the 365-day trailing baseline restarts from scratch. The **first 120 test days have no baseline at all**, and 17 of 133 positives fall inside that blind window.

This decomposition is the finding that turns a negative result into a contribution: the architecture is not merely underperforming, it is underperforming for a located, fixable reason.

## 4.8 Hypotheses tested and rejected

Both were implemented, measured, and reverted. Negative results with a mechanism are evidence.

### Option 1 — replace the persistence gate with direct level-shift scoring

Gate: `level_shift_score > 0.50 AND duration_held >= 0.70`.

- **Structurally inert for the AUC.** `run_method_comparison.py:384` scores via `agent.detect(agent.preprocess(frame))` and never calls `validate()`.
- Full re-run: **0 of 76 rows changed**, output byte-identical, Hormuz gap unchanged at −0.484.
- Where it *does* apply (the synthetic path) it is destructive:

| Gate | TP | FP | FN | TN | TPR | FPR | F1 |
|---|---|---|---|---|---|---|---|
| persistence (current) | 44 | 10 | 3 | 308 | 0.936 | 0.031 | **0.871** |
| level_shift | 6 | 2 | 41 | 316 | 0.128 | 0.006 | **0.218** |

Cause: the trailing baseline needs 91 days shifted by 30, so **no baseline exists until day 156** — the day 60–74 disruption is unscoreable by construction — and only 23 of 365 days clear `score > 0.50`.

### Option 3 — retrain the Isolation Forest on mixed normal + disruption days

- **Premise false.** `_agent_frame` passes only `timestamp` + `shipping__*`, so `y_true` never reaches the agent and `fit()`'s leak-filter never fires. The forest producing Hormuz 0.502 was **already** trained on mixed data.
- Tested both regimes properly, on real data, with the harness's own split:

| Region | mixed | normal-only | Δ |
|---|---|---|---|
| hormuz | 0.5023 | 0.5240 | **+0.0216** |
| bab_el_mandeb | 0.6790 | 0.6790 | 0.0000 |
| panama | 0.9087 | 0.9087 | 0.0000 |

Bab el-Mandeb and Panama are identical because their training windows contain no positives, so filtering removes nothing. Option 3's direction is the *worse* of the two on Hormuz, by a negligible margin.

## 4.9 Weight optimization results

Optuna, TPE sampler (seed 42), median pruner, 100 trials, 1 h cap, Dirichlet-style renormalisation per weight group. Hard constraints reject `risk_high ≤ risk_medium` and `agreement_bonus_5 ≤ agreement_bonus_3`.

**Objective:** `0.50·F1 + 0.30·lead_time_score − 0.20·FPR`, where lead_time_score is the earliest MEDIUM alert within 5 days before onset ÷ 5.

**Test-split discipline:** touched exactly once, after the study. `PipelineEvaluator.evaluated_splits` is an audit trail proving it.

**Best run:** trial 63 of 100, validation objective **0.7491**.

| Agent | Hand-tuned L2 | Optimized L2 |
|---|---|---|
| shipping | 0.25 | **0.402** |
| market | 0.15 | 0.088 |
| geopolitical | 0.25 | 0.109 |
| natural_disaster | 0.10 | 0.095 |
| routing | 0.15 | 0.130 |
| news_sentiment | 0.10 | **0.177** |

The optimizer independently concentrates weight on shipping — consistent with the dilution mechanism in [§4.6](#46-the-central-negative-finding), since shipping is the domain the label is derived from.

> ⚠️ **These weights cannot currently be cited as a result.** See [§4.11](#411-known-defects-and-open-issues) item 1.

## 4.10 Threats to validity — what cannot be claimed

Stated explicitly, because a defence will find them otherwise.

1. **Cannot claim the system beats baselines at detection.** On Hormuz it is at chance (0.502 vs 0.968 for logistic regression). Only Panama (0.909) is competitive.
2. **Cannot claim agent diversity improves detection.** Measured on real data, AUC *falls* monotonically as agents are added, in every evaluable region.
3. **Cannot cite tier AUC as clean detection skill.** Every tier is rated `high` circularity — tiers include the shipping level-shift feature, which is the label's own statistic.
4. **Cannot claim anything about routing.** Dormant in all four regions; never exercised.
5. **Cannot claim performance on the 2026 Hormuz shutdown.** It is not in the evaluation frame, which ends 2025-08-30.
6. **Cannot claim slow-onset detection.** The label itself misses it — Panama's 38 % drought decline is almost entirely unlabelled.
7. **Cannot cite METRIC 5 or any 8-metric number** until the defects in [§4.11](#411-known-defects-and-open-issues) are resolved.

## 4.11 Known defects and open issues

### Blocking

1. **The optimization records contradict each other.**

   ```
   config/optimized_weights.yaml            → 100 trials, best trial 63, val 0.7491
   data/processed/optimization_results.json →   5 trials, best trial  1, obj 0.6301
   ```

   Entirely different weight vectors (news_sentiment 0.177 vs 0.046; natural_disaster 0.095 vs 0.168). The JSON was overwritten by an apparent smoke run; the committed YAML is still the 100-trial result. **`notebooks/evaluation.py` METRIC 5 reads that JSON**, so running the suite today reports the 5-trial numbers as the optimization result. The two SHAP comparison plots were generated in that same session and are suspect. **Fix: re-run the 100-trial optimization before writing up results.**

2. **The 8-metric suite has never been run against the current system.** `evaluation_results.json` and `thesis_comparison_table.json` are dated 2026-07-30 — a month before the re-optimization and the shipping rework. METRICS 1–8 as they sit on disk describe a superseded system.

### Code-level inconsistencies to disclose

1. **Routing normalises its Isolation Forest by batch min-max**, the exact practice shipping was deliberately moved away from. Routing went dormant before it was migrated.
2. **`Orchestrator.run_timeseries_analysis` omits the agreement bonus** that every other composite path applies.
3. **The agreement threshold (0.5) is a module constant**, not configurable or optimized — while the bonuses it gates are both.
4. **Geopolitical computes rolling baselines and deviations that `detect()` never reads**; news does the same with `sentiment_rolling_7d`. `sentiment_magnitude` is a schema feature that never enters any score.
5. **`PipelineEvaluator.build_agents` hardcodes agent config** instead of reading `settings.yaml`, so the optimizer tunes a slightly different agent than the live pipeline runs.
6. **The SHAP surrogate sees 20 of the features, not all.** Absent: `tanker_count`, `vessel_count_trend`, `freight_services_pct_change`, `vessels_holding`, `alternative_route_traffic`, `sentiment_magnitude`, `recency_weighted_score`. Missing columns fill with 0.0 — indistinguishable from a true zero.
7. **The shipping duration score is near-circular** against the level-shift label. Cite operationally, never as detection skill.

### Four composite paths that are not identical

Always state which produced a reported number.

| Path | Granularity | Agreement bonus? |
|---|---|---|
| `RiskEngine.compute_risk` | one scalar per run | yes |
| `RiskEngine.compute_risk_timeseries` | per day | yes |
| `Orchestrator.run_timeseries_analysis` | per day | **no** |
| `PipelineEvaluator._aggregate_daily` | per day, vectorised | yes |

`_aggregate_daily` faithfully mirrors `compute_risk`, so the optimizer maximises what the pipeline actually produces.

### Figure provenance

| Set | Path | Status |
|---|---|---|
| Method comparison (7) | `eval/graphs_method_comparison/` | **Current, real data — cite freely** |
| Tier ablation (3) | `eval/graphs_ablation_tiers/` | **Current** |
| Optimizer (6) | `data/processed/` (gitignored) | Current |
| SHAP comparison (2) | `data/processed/` (gitignored) | ⚠ Suspect — generated in the 5-trial session |
| SHAP beeswarm/waterfall (2) | `data/processed/` (gitignored) | ✗ **Stale (2026-06-20) — do not cite** |

Malacca deliberately has no tier-progression chart: `graph_tiers()` drops NaN-AUC rows and all of Malacca's are NaN. It is covered instead by `A2_false_positive_harness.png`. All four regions' tier data is in the CSV.

---

## Reference Documents

| Document | Contents |
|---|---|
| [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) | **The phase-by-phase build history** — the former body of this README, preserved verbatim. Read it for *how* the system was built and why decisions were later revised |
| `docs/THESIS_BRIEF.md` | The source of record for every measured number in [§4](#4-evaluation). Where this README and the brief disagree, the brief wins |
| `docs/SCORING_REFERENCE.md` | End-to-end walkthrough of how a raw feature becomes a composite risk score and a band; per-agent formulas |
| `docs/DASHBOARD_USAGE.md` | Running and reading the two-page dashboard |
| `docs/REGION_USAGE_GUIDE.md` | Adding and operating a region |
| `eval/COMPARISON_REPORT.md` | The results narrative generated from `method_comparison_results.csv` |

The `docs/` entries are subject to the tracking caveat under *Repository hygiene* below.

### Repository hygiene

Two directories are deliberately kept out of the repository, and one of them does not currently work as intended:

- **`thesis/`** — the FAPS LaTeX template and thesis drafts. Gitignored with an explicit *"never commit/push"* note. The compiled `FAPS-Thesis.pdf` and all LaTeX build artefacts live here and are not versioned.
- **`docs/`** — listed in `.gitignore`, **but all 16 files under it are already tracked**, so the ignore rule has no effect. `.gitignore` only applies to untracked files. To actually stop versioning them, the files must first be removed from the index:

  ```bash
  git rm -r --cached docs/
  ```

  Until that is run, the entries above remain in the repository and edits to them will still be committed. Left as-is deliberately for now — the reference documents are useful to have versioned.

---

## Project Structure

```
supply-chain-dss/
├── config/
│   ├── settings.yaml           # base config: agent toggles, weights, thresholds, RAG, API, logging
│   ├── optimized_weights.yaml  # Phase 4 — Optuna-tuned weights (weight_mode: optimized)
│   └── regions/                # Phase 11 — per-region overrides, merged onto settings.yaml
│       ├── hormuz.yaml
│       ├── bab_el_mandeb.yaml  # absorbs the retired red_sea / suez keys
│       ├── panama.yaml
│       └── malacca.yaml
├── data/
│   ├── raw/                    # raw CSV ingestion data (populate per connector)
│   │   ├── shipping_hormuz.csv # synthetic Hormuz dataset (Phase 1 artefact)
│   │   └── market_data.csv     # synthetic Brent / trade volume / freight data (Phase 1 artefact)
│   ├── processed/              # cleaned DataFrames, SHAP PNGs (Phase 4 depth), evaluation_results.json + thesis_comparison_table.json (Phase 9a, gitignored)
│   └── knowledge_base/         # historical disruption cases as JSON
│       ├── disruption_cases.json   # the 10 historical RAG cases
│       └── decision_labels.json    # Phase 9a — auditable ground-truth action labels for the 10 cases
├── src/
│   ├── core/                    # Phase 11 — region abstraction
│   │   ├── regions.py          # RegionConfig registry, DORMANT_AGENTS, RETIRED_REGION_ALIASES, validate_region()
│   │   └── config_manager.py   # base + per-region config merging
│   ├── ingestion/
│   │   ├── base_connector.py   # ABC for all data source connectors
│   │   ├── shipping_connector.py    # IMF PortWatch chokepoint transits (live) / CSV / synthetic
│   │   ├── market_connector.py      # FRED Brent + freight (live) / CSV / synthetic
│   │   ├── geopolitical_connector.py # ACLED conflict events (live); no sanctions_severity by design
│   │   ├── disaster_connector.py    # GDACS + USGS (live); replaced the dead Ambee path
│   │   ├── news_connector.py        # GDELT DOC API v2 (live)
│   │   └── routing_connector.py     # dormant — fetch_api() raises NotImplementedError by design
│   ├── agents/
│   │   ├── base_agent.py       # ABC + DetectionResult dataclass
│   │   ├── shipping_agent.py   # IsolationForest + Z-score + level_shift_score() duration signal (Phase 14)
│   │   ├── market_agent.py     # Rolling Z-score detector for Brent / trade volume / freight (Phase 2)
│   │   ├── geopolitical_agent.py # renormalises weights over features ACLED actually supplies
│   │   ├── disaster_agent.py   # sparse hazard severity
│   │   ├── news_agent.py       # renormalises over present features when GDELT is silent
│   │   └── routing_agent.py    # present but dormant
│   ├── aggregation/
│   │   └── risk_engine.py      # weighted composite risk scoring
│   ├── explainability/
│   │   └── shap_explainer.py   # SurrogateShapExplainer + compare_explanations / compute_faithfulness / generate_comparison_plot (Phase 4 depth)
│   ├── rag/
│   │   └── context_retriever.py # ChromaDB similarity search; query_gated() (Phase 7), evaluate_retrieval_quality() (Phase 4 depth)
│   ├── evaluation/              # Phase 9a — thesis evidence bundle
│   │   └── decision_effectiveness.py # ACTIONS, predict_action(), evaluate_decision_effectiveness() — SRQ5
│   ├── extractors/              # Phase 7 — live API extraction layer for RAG knowledge base
│   │   ├── base_extractor.py        # ABC: rate limiting, ${VAR} env-var resolution, doc normalization
│   │   ├── newsapi_extractor.py     # current news (news_sentiment), ~30-day lookback cap
│   │   ├── serpapi_extractor.py     # date-unbounded Google News — historical RAG backfill (10 cases x 2007-2024)
│   │   ├── gdacs_extractor.py       # Phase 13 — natural_disaster primary (Orange/Red alerts only)
│   │   ├── usgs_extractor.py        # Phase 13 — seismic detail below GDACS's alert bar
│   │   ├── disaster_combined_extractor.py # Phase 13 — GDACS+USGS behind one monthly cap
│   │   ├── portwatch_extractor.py   # Phase 13 — monthly chokepoint traffic summaries
│   │   ├── ambee_extractor.py       # retained but DISABLED — valid key, zero documents
│   │   ├── reliefweb_extractor.py   # natural_disaster fallback (needs an approved appname)
│   │   ├── fred_extractor.py        # market signals around known disruption windows
│   │   ├── acled_extractor.py       # geopolitical conflict events, OAuth via the `acled` client
│   │   ├── aisstream_monitor.py     # live-only AIS WebSocket monitor (unused while routing is dormant)
│   │   └── knowledge_base_builder.py # orchestrates all extractors -> dedupe -> ChromaDB upsert
│   ├── api/
│   │   └── endpoints.py        # FastAPI — 10 endpoints: /health /predict /explain /agents /agents/toggle /weights /weights/switch /optimization/results /populate /status (Phase 8)
│   ├── dashboard/               # Phase 9b — two-page Streamlit dashboard
│   │   ├── app.py                   # entry point / st.navigation router
│   │   ├── core.py                  # shared cached data layer, globe builder, narrative + JPEG helpers
│   │   ├── decision_view.py         # Page 1 — manager view (no scrolling, no raw scores)
│   │   ├── analysis_view.py         # Page 2 — thesis evaluation view (all 8 metrics, JPEG export)
│   │   └── pages/                   # thin multipage shims (1_Decision_View.py, 2_Analysis_View.py)
│   └── orchestrator.py         # main pipeline runner; RAG block now calls query_gated() (Phase 7)
├── scripts/
│   ├── populate_knowledge_base.py  # CLI: python scripts/populate_knowledge_base.py [--extractors a,b,c] [--region r]
│   ├── build_eval_dataset.py       # Phase 15 — assemble the real per-region evaluation set + manifest
│   ├── run_method_comparison.py    # Phase 15 — temporal-split method comparison and agent ablation
│   ├── report_method_comparison.py # Phase 15 — graphs + findings from method_comparison_results.csv
│   ├── migrate_retired_region_keys.py   # Phase 13 — migrate red_sea/suez KB docs to bab_el_mandeb
│   ├── run_disaster_extraction_2018_2026.py # Phase 13 — GDACS+USGS historical backfill
│   └── run_abc_extraction_2018_2026.py      # Phase 13 — ACLED+FRED historical backfill
├── tests/
│   ├── test_agents.py
│   ├── test_ingestion.py       # shipping + market connector schema, ranges, separation, cross-source correlation
│   ├── test_risk_engine.py
│   ├── test_scenarios.py
│   ├── test_new_agents.py      # geopolitical, natural-disaster, routing, news-sentiment agents + 6-agent integration
│   ├── test_optimization.py    # Optuna weight optimizer, parameter space, objective function, no-leakage guard
│   ├── test_shap_6agent.py     # 20-feature SHAP surrogate, explain output, text generation, disabled-agent path
│   ├── test_rag_6domain.py     # 10-case knowledge base, multi-domain query, similarity thresholds, format_context
│   ├── test_extractors.py      # Phase 7 — 26 tests, all extractors + KnowledgeBaseBuilder + query_gated(), HTTP fully mocked
│   ├── test_api_6agent.py      # Phase 8 — 12 tests for all 10 API endpoints with mocked orchestrator
│   ├── test_phase4_depth.py    # Phase 4 depth — 4 tests: SHAP comparison, faithfulness > 0.8, RAG quality > 0.7, plots
│   ├── test_evaluation.py      # Phase 9a — 10 tests: scenario risk levels, agent diversity, decision effectiveness (SRQ5)
│   ├── test_dashboard.py       # Phase 9b — 14 tests: no-raw-scores scan, route sync, JPEG export, region parameterization
│   └── test_fred_api.py        # standalone (non-pytest) FRED connectivity diagnostic, run by hand
├── logs/                       # pipeline execution logs (gitignored)
├── notebooks/
│   └── evaluation.py           # Phase 9a — executable 8-metric thesis evaluation suite
├── .streamlit/
│   └── config.toml             # Phase 9b — dark analytical dashboard theme
├── .env                        # API keys/credentials (gitignored) — see Phase 7 Configuration Additions
├── requirements.txt
├── main.py                     # entrypoint
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `shap`, `chromadb`, `fastapi`, `uvicorn`, `pyyaml`, `plotly`, `optuna`, `kaleido`, `pytest`, `httpx`. (`optuna` + `kaleido` back the weight optimizer and its figure export. `chromadb` bundles its own ONNX embedding model — `sentence-transformers` is no longer required.)

**Phase 7 additions:** `acled` (OAuth-authenticated ACLED client), `requests` (HTTP transport for every extractor + `DisasterConnector.fetch_api()`), `python-dotenv` (loads `.env` for API-key resolution), `websockets` (only needed if `aisstream.enabled: true`).

**Phase 9b additions:** `streamlit` (dashboard). Optional: `anthropic` — only needed if you set `ANTHROPIC_API_KEY` to enable LLM-generated risk explanations on the Decision view (a compositional fallback is used otherwise).

Copy your own keys into `.env` at the project root (gitignored — never commit real values):
```
FRED_API_KEY=
NEWSAPI_KEY=
AISSTREAM_API_KEY=
ACLED_USERNAME=
ACLED_PASSWORD=
AMBEE_API_KEY=
SERPAPI_API_KEY=
MAPTILER_API_KEY=          # Phase 9b, optional — upgrades the dashboard map style/terrain (keyless OpenFreeMap + AWS DEM otherwise)
ANTHROPIC_API_KEY=         # Phase 9b, optional — LLM risk explanations
```
Every key is optional — each extractor and `DisasterConnector.fetch_api()` log a warning and degrade gracefully (empty results / fallback to synthetic) when its key is missing, so a partial `.env` never breaks the pipeline. The dashboard map is fully keyless by default (MapLibre GL JS + OpenFreeMap + AWS Open Data terrain); without `ANTHROPIC_API_KEY` risk explanations use the deterministic compositional fallback.

---

## Running

### Pipeline entrypoint

```bash
python main.py                        # default region: hormuz
python main.py --region panama        # four regions, see §1
```

Expected output:
```
2026-04-25 17:17:33 | INFO | __main__ | Pipeline initialized | region=hormuz
Pipeline initialized (region: hormuz)
```

`--region` accepts `hormuz`, `panama`, `bab_el_mandeb`, `malacca`. With it
absent, `SUPPLY_CHAIN_REGION` is consulted, then the `hormuz` default.

### Generate the synthetic datasets

```bash
python -c "from src.ingestion import ShippingConnector; ShippingConnector(config={}).save_raw()"
python -c "from src.ingestion import MarketConnector; MarketConnector(config={}).save_raw()"
```

The first command writes `data/raw/shipping_hormuz.csv` (365 rows) and prints the Welch t-statistic separating normal vs. disruption vessel counts. The second writes `data/raw/market_data.csv` (365 rows) with Brent crude, trade volume, and freight rate signals lag-aligned to the shipping disruption windows.

### Reproducing the evaluation

The three scripts behind [§4](#4-evaluation), in order. Each writes artefacts the next one reads.

```bash
python scripts/build_eval_dataset.py        # live connectors -> data/eval/ + manifest
python scripts/run_method_comparison.py     # -> eval/method_comparison_results.csv
python scripts/report_method_comparison.py  # -> eval/graphs_*/ + COMPARISON_REPORT.md
```

`build_eval_dataset.py` hits the live APIs and is the slow step (Hormuz geopolitical alone takes ~11 minutes). The manifest it writes records the column set per region, so a later comparison cannot silently score a five-feature region against a four-feature one.

The 8-metric suite runs separately on the synthetic splits:

```bash
python notebooks/evaluation.py
```

> ⚠️ Its output is **not currently citable** — see [§4.11](#411-known-defects-and-open-issues). Re-run the 100-trial optimization first, or METRIC 5 will report a 5-trial smoke run as the optimization result.

### Populate the RAG knowledge base from live APIs

```bash
python scripts/populate_knowledge_base.py                       # all extractors in extraction.enabled_extractors
python scripts/populate_knowledge_base.py --extractors serpapi   # one-time historical backfill only
```

Extracts → deduplicates by document id → backs up to `data/knowledge_base/live_extracted_backup.json` (gitignored) → upserts into the `live_extracted_context` ChromaDB collection. Safe to re-run; missing API keys degrade individual extractors to zero documents rather than failing the run.

Document ids are region-scoped (`acled_{region}_{country}_{year}`). They were not originally: countries shared between chokepoints produced identical ids, so the second region's rows were silently dropped by the deduplicator. Any extractor spanning overlapping country sets needs the same treatment.

> **Historical note.** Earlier runs of this script are reported in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) with document counts that include the `ambee` extractor and the `red_sea` / `suez` region keys. Both are retired — Ambee returns zero documents with a valid key, and the two region keys were folded into `bab_el_mandeb`. Current sources are in [§3.3](#layer-1-in-detail--live-data-sources).

### API server

```bash
uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`.

### Dashboard

```bash
streamlit run src/dashboard/app.py
```

Opens at `http://localhost:8501`. The first load takes ~1 minute (split generation, agent fitting, SHAP surrogate training, RAG index) — everything is cached afterwards. The **Decision View** is the default page; switch to the **Analysis View** from the sidebar for the full thesis evidence and per-chart JPEG export.

### Tests

```bash
pytest tests/ -v
```

The full suite is **420 tests / 420 passing** across 23 collected test files, in ~3m35s (`test_fred_api.py` is a standalone diagnostic script and contributes no test items). Verified on the project venv; run it with `.venv/Scripts/python -m pytest -q`. Run the agent evaluations with output:

```bash
pytest tests/test_agents.py::test_shipping_agent_evaluation -v -s
pytest tests/test_agents.py::test_market_agent_evaluation -v -s
```

---

## Configuration Reference (`config/settings.yaml`)

| Key | Default | Description |
|---|---|---|
| `agents.shipping.enabled` | `true` | Toggle shipping agent on/off |
| `agents.shipping.detection_method` | `isolation_forest` | Algorithm for shipping anomaly detection |
| `agents.shipping.contamination` | `0.1` | Expected anomaly fraction for Isolation Forest |
| `agents.shipping.threshold` | `0.65` | Minimum combined score to raise a shipping flag (eval harness uses 0.55) |
| `agents.shipping.z_threshold` | `3.0` | Z-score normalisation cap for the secondary fallback channel |
| `agents.market.enabled` | `true` | Toggle market agent (enabled in Phase 3 so the default run exercises all six agents) |
| `agents.market.detection_method` | `zscore` | Algorithm for market anomaly detection |
| `agents.market.z_threshold` | `2.5` | Per-feature absolute z-score elevation cutoff (eval harness uses 1.2) |
| `agents.market.threshold` | `0.55` | Minimum combined score to raise a market flag (eval harness uses 0.40) |
| `agents.market.window` | `30` | Trailing rolling-window length, in days |
| `weights.shipping` | `0.4` | Contribution weight in composite score |
| `weights.market` | `0.3` | Contribution weight in composite score |
| `weights.geopolitical` | `0.3` | Contribution weight in composite score |
| `thresholds.risk_high` | `0.7` | Composite score cutoff for HIGH risk |
| `thresholds.risk_medium` | `0.4` | Composite score cutoff for MEDIUM risk |
| `weight_mode` | `hand_tuned` | `hand_tuned` (settings.yaml) or `optimized` (`config/optimized_weights.yaml`) |
| `optimization.n_trials` | `100` | Optuna trial budget for `python main.py --optimize` |
| `optimization.objective_weights` | `{f1: 0.5, lead_time: 0.3, fpr_penalty: 0.2}` | Blend of F1 / lead-time / FPR the optimizer maximises |
| `optimization.seeds` | `{train: 42, validation: 43, test: 44}` | Per-split RNG seeds for the train/val/test realisations |
| `rag.collection_name` | `disruption_cases` | ChromaDB collection name |
| `rag.top_k` | `3` | Number of historical precedents to retrieve |
| `rag.composite_threshold` | `0.65` | *(Phase 7)* Minimum composite risk score for `query_gated()` to fire at all |
| `rag.min_similarity` | `0.55` | *(Phase 7)* Minimum cosine similarity for a match to be included |
| `rag.collections.static_cases` / `.live_context` | `disruption_cases` / `live_extracted_context` | *(Phase 7)* The two ChromaDB collections `query_gated()` merges results from |
| `api_keys.*` | `"${VAR_NAME}"` | *(Phase 7)* `fred`, `newsapi`, `acled_username`, `acled_password`, `aisstream`, `serpapi` — resolved from `.env` at runtime |
| `extraction.enabled_extractors` | `[newsapi, serpapi, ambee, fred, acled]` | *(Phase 7)* Extractors run by `KnowledgeBaseBuilder.build()` / `scripts/populate_knowledge_base.py` |
| `extraction.chokepoints` | `{hormuz, red_sea, malacca, suez}` | *(Phase 7)* Per-region countries + bounding boxes used by every extractor |
| `extraction.rate_limits` | per-source `requests/min` caps | *(Phase 7)* Enforced by `BaseExtractor._rate_limit_wait()` |
| `agents.natural_disaster.location` | `"hormuz"` | *(Phase 7)* Which `monitoring_points` region `DisasterConnector.fetch_api()` queries against Ambee |
| `agents.natural_disaster.severity_mapping` | proximity/alert → `[0,1]` tables | *(Phase 7)* Categorical→numerical mapping for Ambee's two severity fields |
| `aisstream.enabled` | `false` | *(Phase 7)* Toggle for the live AIS WebSocket monitor (no historical backfill exists for this source) |

---

