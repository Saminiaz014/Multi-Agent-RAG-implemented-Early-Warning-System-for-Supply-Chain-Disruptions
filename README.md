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
  - [3.1 Research design](#31-research-design) · [3.2 Requirements](#32-design-requirements) · [3.3 Architecture](#33-system-architecture)
  - [3.4 Ingestion](#34-layer-1--ingestion) · [3.5 Detection agents](#35-layer-2--detection-agents) — [shipping](#351-shippingagent) · [market](#352-marketagent) · [geopolitical](#353-geopoliticalagent) · [disaster](#354-disasteragent) · [news](#355-newsagent) · [routing](#356-routingagent--dormant)
  - [3.6 Aggregation](#36-layer-3--risk-aggregation) · [3.7 Explainability](#37-layer-4--explainability) · [3.8 Retrieval](#38-layer-5--retrieval) · [3.9 Presentation](#39-layer-6--presentation)
  - [3.10 Evidence discipline](#310-evidence-discipline) · [3.11 Weighting](#311-weight-determination-and-optimization) · [3.12 Configuration](#312-configuration-and-region-overlays) · [3.13 Verification](#313-verification-methodology) · [3.14 Summary](#314-implementation-summary)
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
| **SRQ4** | Does weight optimization improve on hand-tuning? | `src/optimization/` · METRIC 5 | **Yes, on the synthetic objective** — lead time 1.67 → 5.00 d, F1 +0.039, at +0.0031 FPR ([§4.9](#49-weight-optimization-results)). Says nothing about real-data detection |
| **SRQ5** | Would a decision-maker be led to the correct action? | `src/evaluation/decision_effectiveness.py` · METRIC 8 | Not currently citable — see [§4.11](#411-known-defects-and-open-issues) |

---

# 3. Methodology

This chapter describes the research approach, the requirements the artifact was built to satisfy, and the design and implementation of every layer of the system. Each section states the **analysis** — the reasoning behind a decision and the alternative it rejects — followed by the **implementation** that realises it in code.

The organising principle throughout is that a design decision is only defensible if the alternative was considered and the reason for rejecting it is recorded. Where a decision later proved wrong, that is stated at the point of use rather than in a separate errata.

---

## 3.1 Research design

### Analysis

The work follows **Design Science Research**: an artifact is constructed to address a practical problem, then evaluated against criteria fixed before the results are known. The artifact is the decision support system described in §3.3–§3.9; the evaluation is the method comparison in [§4](#4-evaluation).

The methodological commitment that shapes every subsequent decision is that **an evaluation must be able to fail**. It is trivially easy to build a detection benchmark that flatters the system under test: score the model on rows it was fitted to, compare against baselines that were not tuned, choose a label the model's own inputs construct, and report the best of several runs. Each of those produces a number that is real, reproducible, and worthless.

Three design decisions guard against this, and they are described in full in [§4.1](#41-evaluation-design):

1. **One temporal split, applied identically to every method** — nothing is scored on rows it was fitted to, and the split is temporal rather than random so the future cannot leak into the past.
2. **Tiers instantiate real agent classes** — an ablation that approximates a tier by averaging feature columns measures nothing about the agents.
3. **Every method carries a circularity rating** — because the ground-truth label is derived from vessel counts, any method reading vessel counts against a long baseline predicts it partly by construction, and that must be visible on the face of every results table.

These are what allow the negative findings in [§4.6](#46-the-central-negative-finding) and [§4.7](#47-root-cause-analysis--why-hormuz-sits-at-chance) to be trusted rather than explained away. They are stated here, in the methodology, because they are design commitments made before the results existed — not defences constructed afterwards.

### The consequence for what this system claims

A second commitment follows from the first. Because the evaluation was built to be able to fail, and did fail on two of four regions, the claim the artifact makes is narrower than the one originally hypothesised. The system is not presented as a better detector. It is presented as a system that produces **auditable, attributable, decision-ready alerts**, evaluated by a method that quantifies what that costs in detection performance. The full statement is at the top of this document; the evidence for the cost is [§4.5](#45-results).

---

## 3.2 Design requirements

### Analysis

Requirements are derived from the problem statement in [§1](#1-problem-and-scope), specifically from the observation that a risk score without attribution cannot support a decision. R3–R6 exist because of that; R1, R2 and R7 exist because the system must survive contact with real, intermittent, heterogeneous data sources.

| # | Requirement | Rationale | Realised by |
|---|---|---|---|
| **R1** | Separate ingestion, detection, and decision support so each can change independently | Sources change far more often than detection logic | Layered architecture with two ABCs, [§3.3](#33-system-architecture) |
| **R2** | Handle heterogeneous structured and unstructured sources | Evidence arrives as transit counts, prices, event records, and news text | Six connectors behind one contract, [§3.4](#34-layer-1--ingestion) |
| **R3** | Attribute every score to a named domain | A manager must know *which* domain raised the alarm | Per-agent breakdown in `RiskEngine.compute_risk`, [§3.6](#36-layer-3--risk-aggregation) |
| **R4** | Attribute every score to named input features | Domain attribution alone does not say *what moved* | SHAP surrogate, [§3.7](#37-layer-4--explainability) |
| **R5** | Ground alerts in comparable past events | Precedent is what converts a number into a judgement | Threshold-gated RAG, [§3.8](#38-layer-5--retrieval) |
| **R6** | Keep every number auditable by a domain expert | An unauditable score cannot be defended in a review | No end-to-end learned model, [§3.5](#35-layer-2--detection-agents) |
| **R7** | Degrade gracefully when a source or agent is unavailable | Free-tier APIs fail, and some domains have no evidence in some regions | Weight renormalisation at two levels, [§3.6](#36-layer-3--risk-aggregation) |

**R7 is load-bearing and easy to underestimate.** Five of the six live sources are free-tier services that fail intermittently, one domain has no usable free source at all, and three of the four regions have at least one domain with no documented driver. A system that required all six agents to report would produce nothing for most region-days. Renormalisation appears twice — inside an agent over its own features ([§3.5](#35-layer-2--detection-agents)) and across agents in the aggregator ([§3.6](#36-layer-3--risk-aggregation)) — because absence occurs at both levels.

---

## 3.3 System architecture

### Analysis

The pipeline is a strict one-way flow with no feedback path. Each stage has one responsibility and a defined output contract, so a stage can be replaced without touching its neighbours. This is not architectural decoration: it is the property that made it possible to migrate five connectors from synthetic generation to live APIs without modifying a single agent, and to mute an entire agent domain across all four regions without touching the aggregator.

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

### The two contracts

Everything in the system is built against one of two abstract base classes. Both are deliberately small — a contract that specifies too much prevents exactly the substitution it exists to enable.

**`BaseConnector`** (`src/ingestion/base_connector.py`) declares two abstract methods:

| Method | Contract |
|---|---|
| `fetch()` | Return a raw domain frame on a daily index, from whichever mode is configured |
| `validate(df)` | Schema, domain and gap checks; return the cleaned frame |
| `fetch_and_validate()` | Concrete convenience wrapper; agents call this |

**`BaseAgent`** (`src/agents/base_agent.py`) declares four abstract methods plus a `DetectionResult` dataclass:

| Method | Contract |
|---|---|
| `fit(df)` | Calibrate any model or baseline the agent needs |
| `detect(df)` | Produce `anomaly_score ∈ [0,1]` and `is_anomaly` per row |
| `set_weights(...)` | Override intra-agent feature weights — the L1 layer the optimizer searches |
| `set_threshold(...)` | Override the detection cutoff — part of the L3 layer |
| `fit_detect(df)` | Concrete convenience wrapper |

`set_weights` and `set_threshold` are abstract rather than optional because **every agent must be tunable by the optimizer**. An agent that hardcoded its weights would silently drop out of the L1 search space in [§3.11](#311-weight-determination-and-optimization).

### The six-stage lifecycle

Beyond the ABC, all six agents implement the same six-stage lifecycle by convention. The ABC does not enforce stages 2, 4 and 5 because not every agent needs a meaningful implementation of each — the disaster agent's `preprocess` is nearly a pass-through — but every agent provides them, so the orchestrator and the evaluation harness can treat agents uniformly.

| Stage | Purpose | Consumed by |
|---|---|---|
| `fit` | Calibrate scaler, forest, or baseline window | Once per pipeline run |
| `preprocess` | Derive features, fill gaps, compute rolling statistics, scale | `detect` |
| `detect` | Score every row into `[0,1]`, flag against threshold | `validate`, and the evaluation harness |
| `validate` | Suppress false positives via persistence and corroboration gates | `output` |
| `output` | Collapse contiguous flagged rows into structured window reports | Dashboard, API |
| `run` / `run_dataframe` | Orchestrate the above and return reports or a frame | `Orchestrator` |

**A consequence that matters for reading [§4](#4-evaluation):** the method comparison harness scores agents via `detect(preprocess(frame))` and **never calls `validate()`**. Everything in the validation stage — every persistence gate and corroboration rule described below — is therefore invisible to the AUC numbers in [§4.5](#45-results). This is the reason one of the two rejected hypotheses in [§4.8](#48-hypotheses-tested-and-rejected) turned out to be structurally inert.

### Implementation

| Layer | Responsibility | Code |
|---|---|---|
| **1 — Ingestion** | Fetch, validate, normalise each domain to a daily index | `src/ingestion/*_connector.py` |
| **2 — Detection** | Per-domain anomaly scoring | `src/agents/*_agent.py` |
| **3 — Aggregation** | Fuse agent scores into a composite risk and level | `src/aggregation/risk_engine.py` |
| **4 — Explainability** | Attribute the composite to input features | `src/explainability/shap_explainer.py` |
| **5 — Retrieval** | Ground the alert in historical precedent | `src/rag/context_retriever.py` |
| **6 — Presentation** | Decision view, analysis view, REST API | `src/dashboard/`, `src/api/endpoints.py` |
| **Orchestration** | Wire the above, per region | `src/orchestrator.py`, `main.py` |

---

## 3.4 Layer 1 — Ingestion

### Analysis

Every connector supports three modes, selected per connector by `source_mode` or `data_mode`. The three exist for genuinely different purposes, and conflating them would undermine the evaluation.

| Mode | Purpose | Carries labels? |
|---|---|---|
| `api` | Live data from the domain's real source. The evidence basis for [§4](#4-evaluation) | No |
| `csv` | A downloaded static extract. Reproducible, offline, no rate limits | Sometimes |
| `synthetic` | Generated data with injected disruption scenarios | **Yes** |

**`synthetic` is not a placeholder mode.** It is the only mode that carries ground-truth `is_disruption` labels, which is why the Optuna optimizer and the 8-metric suite still run on it: they need a label the data generator placed deliberately, not one inferred from the data. The real evaluation set built in [§4.2](#42-dataset-and-ground-truth) uses `api` mode and derives its label statistically — with the circularity consequences recorded in [§4.1](#41-evaluation-design).

### Live data sources

Five of the six agents read a live, free public API. The sixth is dormant by decision ([§3.10](#310-evidence-discipline)).

| Agent | Live source | Access | Notes |
|---|---|---|---|
| **Shipping** | [IMF PortWatch](https://portwatch.imf.org) Daily Chokepoints (ArcGIS FeatureServer) | **No credentials** | Daily transits per chokepoint, 2019–2026, all four regions. The evidence-grade backbone of the evaluation set |
| **Market** | [FRED API](https://fred.stlouisfed.org/docs/api/fred/) | Free (key) | Brent Crude `DCOILBRENTEU`, Freight PPI `PCU4831114831111`, Freight Services `PCUATFREIATFREI` |
| **Geopolitical** | [ACLED](https://acleddata.com) | Free (OAuth) | Battles/Explosions → `military_activity_index`; Strategic developments → `diplomatic_incident_score`; Protests/Riots → inverse `regime_stability_index` |
| **Natural Disaster** | [GDACS](https://www.gdacs.org) + [USGS](https://earthquake.usgs.gov/fdsnws/event/1/) | **No credentials** | GDACS Orange/Red alerts only; USGS supplies magnitude and the tsunami flag |
| **News Sentiment** | [GDELT DOC API v2](https://api.gdeltproject.org/api/v2/doc/doc) | **No credentials** | Tone scores. Answered for Panama only in the evaluation window |
| **Routing** | — | — | **Dormant** — no free source supplies evidence-grade rerouting data |

### Connector-by-connector

**`ShippingConnector`** — the most consequential connector, because the ground-truth label is derived from its output.

- `api` mode queries PortWatch's `Daily_Chokepoints_Data` FeatureServer layer for daily transit counts at the chokepoint itself, for every region, 2019-01-01 to present, refreshed daily, with no credentials. `n_total` becomes `vessel_count` and `n_tanker` becomes `tanker_count`; the per-vessel-type counts travel with the frame for provenance and future features.
- `csv` mode reads a static PortWatch export for a single Persian Gulf port (Shuaiba, Kuwait) — the same provider, a narrower scope.
- `synthetic` mode generates a 365-day series with three injected disruption scenarios and a ground-truth label.
- The connector pins the April–May 2026 Hormuz shutdown as known ground truth (`_KNOWN_SHUTDOWN_START` / `_END`). **That window is outside the evaluation frame**, which ends 2025-08-30 — see [§4.2](#42-dataset-and-ground-truth).
- Raises `ValueError` rather than returning empty when a region has no `portwatch_chokepoint` configured or the API returns no rows: an empty series here means a name mismatch, not a quiet chokepoint.

**`MarketConnector`** — three FRED series on mixed frequencies. Brent is daily; freight PPI and freight services are monthly and are forward-filled to daily. Trade volume is derived from inverse Brent volatility. In `csv` mode the ground-truth label is computed from co-occurring Brent and freight spikes.

**`GeopoliticalConnector`** — delegates ACLED's OAuth lifecycle to `ACLEDExtractor` rather than re-implementing token exchange. Fetches per country, per year, and **warns when a year hits the per-year event cap**, because a capped page is indistinguishable from a complete one. Raises `ValueError` when a region has no `acled_countries` configured — a live query with no country filter would silently return the wrong thing.

**`DisasterConnector`** — queries GDACS `geteventlist` restricted to `Orange;Red` alert levels, and USGS `fdsnws` for seismicity. The alert-level restriction is not conservatism: green-inclusive GDACS queries exceed its **silent 100-result cap** every month, so including them would truncate the response without any error. GDACS hazard codes `FL`, `WF`, `DR` fold into `severe_weather_index`; tropical cyclones drive `cyclone_severity`; USGS supplies `earthquake_severity` on a documented magnitude scale and a real `tsunami_risk` flag.

**`NewsConnector`** — GDELT DOC API v2 tone scores. GDELT answered for only one of four regions across the evaluation window, which is why `NewsAgent` renormalises over present components ([§3.5](#35-layer-2--detection-agents)).

**`RoutingConnector`** — `fetch_api()` raises `NotImplementedError`. This is a recorded decision, not an unfinished task; see [§3.10](#310-evidence-discipline).

### Two deliberate absences

Both are affirmative findings about the evidence rather than gaps in the implementation, and both are handled by renormalisation rather than by imputation:

- **`sanctions_severity` is not produced.** ACLED carries no sanctions data. OpenSanctions is paywalled and publishes current designations rather than a time series. Sanctions are discrete events, so any daily "severity" curve would be a modelling artefact rather than a measurement. Scoring the missing column as zero would read as *"no sanctions risk"* rather than *"not measured"*.
- **`source_consensus` is absent for GDELT-sourced news.** Same reasoning: zero would read as *"outlets disagree"*.

> **Naming trap.** `ShippingConnector` and `MarketConnector` each carry two similarly named methods. `fetch_from_api()` is the real live path and is what `fetch()` dispatches to in `api` mode. `fetch_api()` is a leftover convenience hook that logs a warning and **silently returns synthetic data**. Call `fetch()` or `fetch_and_validate()`, never `fetch_api()` directly. Several docstrings in these two modules still describe `api` mode as an unimplemented aisstream.io stub and are themselves out of date.

### The unified daily schema

Whichever mode produced it, every connector emits the same contract: a **daily-frequency frame on a monotonically increasing timestamp index**, with the domain's feature columns and nothing else. Agents therefore never know which mode produced their input, which is what allowed the migration from synthetic to live data to leave the agent layer untouched.

`validate()` enforces the contract before an agent ever sees the frame:

Taking `ShippingConnector.validate()` as the reference implementation:

| Check | Behaviour |
|---|---|
| `timestamp` and `vessel_count` present | **Assert** — a missing feature column is a contract violation |
| No NaN in `timestamp` or `vessel_count` | **Assert** |
| `vessel_count >= 0` | **Assert** — a negative transit count is not recoverable |
| `congestion_index` within `[0, 1]` | **Assert**, when the column is present |
| Timestamps monotonically increasing | **Assert** — out-of-order rows silently corrupt every rolling window |
| Gaps > 2 days | **Warn** and retain, reporting the count and the largest |
| Gaps ≤ 2 days | **Forward-fill** (`ffill(limit=2)`) |

The asymmetry in the last two rows is deliberate. A weekend gap in a daily series is a reporting artefact and should be filled. A longer gap is a real absence of evidence, and the `limit=2` on the forward-fill is what stops it being papered over — filling a three-week gap would manufacture exactly the calm baseline the level-shift detector in [§3.5.1](#351-shippingagent) measures against, turning missing data into a spurious "traffic is normal" signal.

Assertions rather than exceptions are a deliberate choice at this boundary: these are conditions that should be impossible if the connector above is correct, so they document invariants rather than handle expected failures. Expected failures — a region with no configured chokepoint, an API returning nothing — raise `ValueError` further up, where the orchestrator's `_safe_fetch` can catch them and degrade gracefully.

---

## 3.5 Layer 2 — Detection agents

### Analysis — there is no end-to-end learned model

The only fitted models anywhere in the scoring path are **two Isolation Forests and their `StandardScaler`s**. Every other agent is explicit arithmetic: weighted composites, rolling z-scores, one sigmoid.

This is a deliberate design position and the foundation of R6. The consequence is that every number the system produces traces to a formula a domain expert can read, check, and dispute. A gradient-boosted model over the same features would very likely score better on the metrics in [§4.5](#45-results) — and would forfeit exactly the property the decision-support claim rests on.

The RandomForest in `src/explainability/` is a **post-hoc surrogate**. It explains the score; it never produces it.

| Agent | Model |
|---|---|
| `shipping` | Isolation Forest (200 trees, seed 42) + per-feature z-score + level-shift duration score |
| `market` | Rolling 30-day trailing z-scores, weighted mean of \|z\| |
| `geopolitical` | Weighted linear composite → sigmoid (gain 6, centre 0.5) |
| `natural_disaster` | Weighted composite + single-event max override |
| `routing` | Isolation Forest (200 trees, seed 42) + transit-ratio z-score — **dormant** |
| `news_sentiment` | Weighted composite of four normalised components |

---

### 3.5.1 ShippingAgent

The most detailed agent, because it carries the most weight and because its behaviour is the root cause of the central negative result in [§4.7](#47-root-cause-analysis--why-hormuz-sits-at-chance).

**Features.** Three base columns — `vessel_count`, `avg_delay_hours`, `congestion_index` — plus two optional PortWatch columns auto-discovered when present: `tanker_count`, and `vessel_count_7dma` from which `vessel_count_trend` is derived.

#### `fit(df)` — calibrate without leakage

```python
if "is_disruption" in train.columns:
    train = train.loc[~train["is_disruption"].astype(bool)]
```

The scaler and forest are fitted **only on label-negative rows** when a ground-truth column is present, so the notion of "normal" the forest learns is not contaminated by the disruptions it must later detect.

> **A caveat that matters for [§4.8](#48-hypotheses-tested-and-rejected).** On the real evaluation path this filter never fires, because the harness passes only `timestamp` and `shipping__*` columns to the agent — `y_true` never reaches it. The forest producing the Hormuz result was therefore trained on mixed normal and disruption days. This was discovered while testing a hypothesis that assumed the opposite.

The forest is `IsolationForest(contamination=0.1, random_state=42, n_estimators=200)`. Its raw scores are then anchored:

```python
train_scores = -self._iforest.decision_function(scaled)
self._iforest_low  = np.percentile(train_scores, 5)
self._iforest_high = np.percentile(train_scores, 95)
```

**Percentiles, not min/max.** A single extreme training row would otherwise set the ceiling and compress everything beneath it into a narrow band.

#### `preprocess(data)` — project into the trained space

Selects active features, derives optional ones, forward-fills gaps, and applies the **already-fitted** scaler. New data is projected into the training space rather than re-standardised against itself.

#### `level_shift_score(vessel_count)` — the duration signal

The shock detectors cannot hold a flag across a sustained disruption. Two of the three base features — `congestion_index` and `avg_delay_hours` — are themselves derived from `vessel_count` against a 30-day rolling baseline, so once traffic settles at a lower level that baseline follows it down and both return to calm values. **Measured on the evaluation set their correlation with a sustained-disruption label is −0.06 and −0.05: no signal at all**, leaving one useful feature of three.

The duration score measures how far traffic sits below its own *trailing annual* baseline:

```python
rolling  = vessel_count.rolling(30, min_periods=10).mean()
baseline = vessel_count.shift(30).rolling(365, min_periods=91).median()
shortfall = (baseline - rolling) / baseline
magnitude = (shortfall / 0.40).clip(0, 1)          # 40% drop = full strength

below       = (shortfall > 0.10).astype(float)      # floor, or noise accumulates
persistence = below.rolling(14, min_periods=14).mean()
score = (magnitude * persistence).clip(0, 1)
```

The baseline is **trailing and shifted**, not rolling, precisely because a rolling baseline adapts to the new level and reads a settled disruption as calm. The persistence factor means a one-off dip scores near zero however deep, while a shift that holds for a fortnight reaches full strength.

> **This score is not evidence of detection skill against a shipping-derived label.** It is computed from the very series such a label is built from and will predict one nearly by construction. It earns its place **operationally** — an alert should persist while the disruption does — not evaluatively. This is why every tier in [§4.5](#45-results) is rated `high` circularity.

#### `detect(data)` — and the masking bug

```python
iforest_norm = clip((−decision_function(scaled) − low) / (high − low), 0, 1)
max_z_norm   = min(max(|scaled|, axis=1) / z_threshold, 1)
shock        = 0.70 · iforest_norm + 0.30 · max_z_norm

duration = level_shift_score(raw_vessel_count)
combined = np.maximum(shock, duration)
```

Two decisions here, one right and one wrong.

**Right — normalising against the fit-time range, not the scored batch.** Batch min-max destroys the signal a sustained disruption carries: when every day in a window is anomalous, rescaling by that window's own extremes maps its least-anomalous day to 0 and its most to 1, so a uniformly disrupted month and a uniformly calm one both span `[0,1]`. Anchoring to the fit-time distribution makes scores absolute and comparable across windows. This change is what forced the risk-band recalibration in [§3.11](#311-weight-determination-and-optimization).

**Wrong — `max()` rather than a blend.** The code comment defends `max()` on the grounds that the two components answer different questions ("did something just change?" versus "are we still below normal?") and are meant to be true at different times, so averaging would let a calm shock detector drag down an active duration signal.

The reasoning is sound and the outcome is not. Measured on the Hormuz test window, `shock` sits at **0.55 on calm days** while `duration` only reaches **0.44 on disrupted ones**, so `max()` returns `shock` nearly everywhere and **erases the working signal**. It prevents dilution and causes **masking** instead. Full decomposition in [§4.7](#47-root-cause-analysis--why-hormuz-sits-at-chance).

#### `validate(signals)` — two-stage false-positive suppression

A row is `validated=True` only when **both** hold:

1. **Persistence** — the flag is part of a run of at least **2** consecutive flagged rows. There is no upper cap, so a multi-month shutdown is flagged in full.
2. **Multi-feature** — at least **2** active features show `|z| > 1.5` on that row, so lone-feature outliers are dropped.

Recall that the evaluation harness never calls this method.

#### `output(validated_signals)`

Collapses contiguous validated rows into window reports carrying start/end dates, duration, peak score, mean score, and the features that were elevated.

---

### 3.5.2 MarketAgent

**Features.** `brent_crude_usd`, `trade_volume_index`, `freight_rate_index`, plus optional `freight_services_pct_change`.

**A different detection strategy, deliberately.** Market series are strongly autocorrelated and trend-bearing; an isolation forest over raw levels would flag every sustained price regime as anomalous. Instead each feature is scored against its own **trailing rolling window** (30 days, with a 5-year baseline configuration), so the question asked is "is today unusual relative to the recent past?" rather than "is today unusual relative to all history?".

```python
z = (value − rolling_mean) / rolling_std        # sd ≤ 1e-9 → z = 0
anomaly_score = min( Σ wᵢ·|zᵢ| / z_threshold , 1.0 )
```

Degenerate flat windows are handled explicitly: when the rolling standard deviation collapses below `1e-9` the feature contributes `z = 0` rather than an infinity.

**`validate` — oil-led corroboration.** A row survives only when all three hold:

1. the flag is part of a run of at least **2** days;
2. `|oil_zscore| > z_threshold` — oil is treated as the lead indicator;
3. **at least one** of `|trade_volume_zscore|` or `|freight_zscore|` also exceeds the threshold.

The optional freight-services feature contributes to the anomaly score but is **deliberately excluded from the corroboration test**, so the gate behaves identically in synthetic and FRED modes. This is a small decision with a real consequence: it keeps the validation logic comparable across data modes rather than silently stricter on live data.

---

### 3.5.3 GeopoliticalAgent

**Features and weights.** Four features with hand-set intra-agent weights:

| Feature | Weight | Direction |
|---|---|---|
| `sanctions_severity` | 0.35 | higher = worse |
| `military_activity_index` | 0.25 | higher = worse |
| `diplomatic_incident_score` | 0.25 | higher = worse |
| `regime_stability_index` | 0.15 | **inverted** — a stable regime lowers risk |

**Renormalisation over present features.** ACLED supplies no sanctions data, so the highest-weighted feature is absent on every live run. Rather than invent it, the agent scores over what is present and rescales the remaining weights to sum to 1:

```python
raw = Σ  wₖ · valueₖ        for features actually present
used = Σ wₖ
composite = raw / used
```

Rescaling is **proportional, so relative importance is preserved**: with sanctions (0.35) absent, military and diplomatic rise 0.25 → 0.385 and stability 0.15 → 0.231. The score stays comparable across sources because it is always a weighted *mean* over `[0,1]` features, never a partial sum that would read as artificially calm.

The absence is logged as a **warning, not silently**: a genuinely missing column and a source that never supplies one look identical at this point in the code, and only one of those is acceptable. If no scoring feature is present at all, the agent raises rather than returning zero.

**Sigmoid compression.** The composite is passed through `1 / (1 + exp(−6·(raw − 0.5)))`, centred so a raw score of 0.5 maps to 0.5 and extremes saturate gracefully rather than clipping.

**Validation.** Persistence over **3** days (longer than shipping's 2, because geopolitical signals are noisier day to day) plus at least **2** features above an elevation threshold of 0.4.

---

### 3.5.4 DisasterAgent

**Features and weights.** `earthquake_severity` (0.35), `tsunami_risk` (0.30), `cyclone_severity` (0.20), `severe_weather_index` (0.15).

**A single-event override, and why.** The disaster signal is **sparse by construction**: most days sit at near-zero baseline noise. A purely weighted composite would dilute a single catastrophic event across four features and score a major earthquake below threshold. So the agent takes the larger of the two:

```python
composite  = Σ wᵢ · featureᵢ
max_single = max(all four features)
anomaly_score = clip( max(composite, max_single), 0, 1 )
is_anomaly    = (composite >= 0.30) or (max_single >= 0.40)
```

This is the same `max()` construction that causes masking in the shipping agent — but here it is correct, because the components are genuinely alternative evidence of the same thing rather than two detectors with different calm-day baselines. The distinction is worth stating explicitly: `max()` is safe when the inputs share a scale and a meaning, and dangerous when they do not.

**Validation.** Single-day events are acceptable — an earthquake does not need to persist — so the only filter is a minimum severity of 0.10, suppressing sub-threshold noise.

---

### 3.5.5 NewsAgent

**Components and weights.** Four normalised components: `sentiment` (0.40), `consensus` (0.25), `velocity` (0.20), `volume` (0.15).

```python
neg_sent      = clip(−recency_weighted_score, 0, 1)
consensus     = clip(source_consensus, 0, 1)          # None for GDELT
velocity      = clip(−sentiment_velocity, 0, 1)       # sentiment dropping fast
volume_factor = clip(article_volume / volume_rolling_30d / 2.0, 0, 1)
```

Negative velocity is the interesting component: it captures sentiment *dropping fast*, which leads a disruption more reliably than sentiment being low, since a persistently negative corridor reads as low without anything new happening.

**The same renormalisation as geopolitical**, for the same reason and with the same warning-not-silence discipline: `source_consensus` is absent for GDELT-sourced data, and scoring it as zero would read as "outlets disagree" rather than "not measured", silently damping every score by its 0.25 weight.

**A feature that never enters any score.** `sentiment_magnitude` is part of the declared schema but is not read by `detect()`. It is listed among the disclosed inconsistencies in [§4.11](#411-known-defects-and-open-issues).

---

### 3.5.6 RoutingAgent — dormant

Five features (`rerouting_percentage`, `avg_route_deviation_km`, `transit_volume_ratio`, `vessels_holding`, `alternative_route_traffic`) and a working Isolation Forest implementation, muted in all four regions. The reasoning is evidential rather than technical and is set out in [§3.10](#310-evidence-discipline).

One code-level consequence is worth recording here: **routing still normalises its Isolation Forest by batch min-max**, the exact practice the shipping agent was deliberately moved away from. It went dormant before it was migrated, so the defect is latent rather than active — but it would bite immediately if the agent were revived.

---

## 3.6 Layer 3 — Risk aggregation

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

**Step 3 is what satisfies R7 at the agent level.** Renormalising over *active* agents makes a passive or dormant agent harmless: its weight redistributes across the agents that did report, rather than contributing a zero that drags the composite down. Without this, muting routing across all four regions would have silently depressed every score in the system by routing's 0.15 weight — a 15 % reduction that would have been invisible in the output and would have quietly invalidated the hand-calibrated risk bands.

**Step 6 is the only non-linearity in the entire scoring path.** It encodes the judgement that three independent domains agreeing is worth more than the arithmetic sum of three domains individually — corroboration across domains is qualitatively different from a single loud domain.

Agents are excluded with a logged warning in two cases: no configured weight, and no scores produced. Both are conditions that should be visible rather than absorbed.

### Implementation

`src/aggregation/risk_engine.py`. Beyond `compute_risk` it returns a full breakdown for downstream consumption — `contributing_agents` with per-agent `{score, weight, contribution}`, `agent_agreement`, a human-readable `reason` string, and metadata recording the active agent count and the redistributed weights. **That breakdown is what satisfies R3**; it is the difference between an alert that says "risk 0.74" and one that says "risk 0.74, driven by geopolitical (0.31) and shipping (0.24), three domains in agreement."

The agreement threshold is the module constant `_AGREEMENT_THRESHOLD = 0.5`, and the bonuses default to `1.15` and `1.25` but are read from config so the optimizer can tune them. **The threshold is not tunable while the bonuses it gates are** — a disclosed inconsistency, [§4.11](#411-known-defects-and-open-issues).

> **Four composite paths exist and are not identical.** `RiskEngine.compute_risk`, `RiskEngine.compute_risk_timeseries`, `Orchestrator.run_timeseries_analysis`, and `PipelineEvaluator._aggregate_daily` differ in granularity, and one of them omits the agreement bonus. Always state which produced a reported number — see [§4.11](#411-known-defects-and-open-issues).

---

## 3.7 Layer 4 — Explainability

### Analysis

R3 is satisfied by the aggregator's per-agent breakdown, but domain attribution answers only half the question. "Geopolitical drove this" does not tell a manager *what moved*; "military activity is 2.4σ above its baseline" does.

The system therefore fits a **RandomForest surrogate** to reproduce the pipeline's composite score from the canonical input features, then explains the surrogate with a SHAP `TreeExplainer`. TreeExplainer gives fast, exact Shapley values on a tree model, which a KernelExplainer over the real pipeline could not.

**The honest statement of what this buys and what it costs.** The surrogate is a model of the pipeline, not the pipeline. Its attributions are exact *for the surrogate*, and are only as faithful to the real system as the surrogate's fit. The fit quality is reported alongside every explanation as `surrogate_r2` rather than being assumed, so a reader can judge whether the attribution is trustworthy on a given run.

An alternative was available: because the scoring path is explicit arithmetic ([§3.5](#35-layer-2--detection-agents)), exact analytic attributions could in principle be derived without any surrogate. That would be strictly better and is listed as future work.

### Implementation

`src/explainability/shap_explainer.py`. `RandomForestRegressor(n_estimators=100, random_state=42)` over **20 canonical features** in `ALL_FEATURE_NAMES`, mapped to their producing domain by `FEATURE_AGENT_MAP`:

| Domain | Features in the surrogate |
|---|---|
| Shipping (3) | `vessel_count`, `avg_delay_hours`, `congestion_index` |
| Market (3) | `brent_crude_usd`, `trade_volume_index`, `freight_rate_index` |
| Geopolitical (4) | `sanctions_severity`, `military_activity_index`, `diplomatic_incident_score`, `regime_stability_index` |
| Natural Disaster (4) | `earthquake_severity`, `tsunami_risk`, `cyclone_severity`, `severe_weather_index` |
| Routing (3) | `rerouting_percentage`, `avg_route_deviation_km`, `transit_volume_ratio` |
| News (3) | `sentiment_score`, `source_consensus`, `article_volume` |

> **The surrogate sees 20 features, not all of them.** Seven inputs that reach the agents never reach the explainer: `tanker_count`, `vessel_count_trend`, `freight_services_pct_change`, `vessels_holding`, `alternative_route_traffic`, `sentiment_magnitude`, `recency_weighted_score`. Missing columns are filled with `0.0`, which is **indistinguishable from a true zero**. An explanation can therefore omit a driver that genuinely moved the score. Disclosed in [§4.11](#411-known-defects-and-open-issues).

Output per explanation: `top_drivers` (feature, producing agent, SHAP value), a generated natural-language `text`, `expected_value`, and `surrogate_r2`.

---

## 3.8 Layer 5 — Retrieval

### Analysis

Feature attribution says what moved. Precedent says what it meant last time. The decision-support argument in [§1](#1-problem-and-scope) needs both, and the retrieval layer supplies the second.

**Retrieval is threshold-gated, which is the design decision worth defending.** `query_gated()` returns `None` outright when the composite risk score is below `rag.composite_threshold` (0.65). A retrieval system that always returns its top-k will always return something — on a quiet day it returns the least-irrelevant historical case, which a reader will reasonably interpret as "the system thinks this resembles the 2021 Suez blockage." Gating on composite risk means **the absence of precedent is itself informative**.

A second filter applies within a triggered query: matches below `min_similarity` (0.55 cosine) are dropped even when the gate has opened, so a triggered alert with weak precedent returns fewer matches rather than padding to top-k.

### Implementation

`src/rag/context_retriever.py`, ChromaDB with the `all-MiniLM-L6-v2` embedding model running locally via ONNX — **no API key and no network dependency at query time**.

Two collections are queried and merged by similarity:

| Collection | Contents |
|---|---|
| `disruption_cases` | 10 curated historical cases, hand-written |
| `live_extracted_context` | Documents harvested from live APIs by `KnowledgeBaseBuilder` |

The 10 curated cases: `cyclone_gonu_2007`, `hormuz_mine_threat_2010`, `somali_piracy_2011`, `japan_earthquake_2011`, `iran_sanctions_2012`, `west_coast_port_strikes_2014`, `hormuz_2019`, `ever_given_2021`, `covid_port_congestion_2021`, `houthi_redsea_2024`.

A triggered query returns the composite score and the threshold it cleared, the merged matches with source, similarity and metadata, and a formatted summary for display.

---

## 3.9 Layer 6 — Presentation

### Analysis

Two audiences need incompatible things from the same run. A supply-chain manager needs a status word, a recommended action, and a reason, with no scrollbar and no raw numbers to misread. A thesis author needs every metric, every raw signal, and per-day SHAP values.

Rather than compromise on one page, the dashboard is split by audience:

- **Decision View** — single viewport, plain language, status word, recommended action, a MapLibre GL JS pitched 3-D chokepoint map, and a natural-language risk narrative. **No raw scores anywhere**, and that constraint is machine-verified by a test that scans the rendered output for decimal patterns.
- **Analysis View** — scrollable, all metrics, raw signals, per-day SHAP values, per-chart JPEG export for direct use in the thesis.

The action rubric behind the Decision View maps a risk level and its drivers to a recommended action through a **transparent rule table, deliberately not another model**. Putting a classifier at the last step would reintroduce exactly the unauditable judgement the whole system exists to avoid.

### Implementation

`src/dashboard/` (`app.py` router, `core.py` cached data layer, `decision_view.py`, `analysis_view.py`) and `src/api/endpoints.py`.

The REST API exposes ten endpoints:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/predict` | Run the pipeline, return composite risk |
| `POST` | `/explain` | Return the SHAP attribution for a prediction |
| `GET` | `/agents` | List agents and their state |
| `POST` | `/agents/toggle` | Enable/disable an agent at runtime |
| `GET` | `/weights` | Current weight vector and mode |
| `POST` | `/weights/switch` | Switch `hand_tuned` ↔ `optimized` |
| `GET` | `/optimization/results` | Optuna study results |
| `POST` | `/populate` | Trigger knowledge-base population |
| `GET` | `/status` | System and agent health |

---

## 3.10 Evidence discipline

### Analysis

An agent can be off for three distinct reasons, and conflating them would misrepresent the system's coverage. **Each exclusion below is an affirmative finding about the evidence, not missing data.**

- **Active** — built, run, and weighted.
- **Passive** — a per-region evidence judgement. The domain is real, but no documented driver exists at that chokepoint in the observation window. It would activate if evidence appeared.
- **Dormant** — a project-scope decision. No region may activate it; enforced at registry validation.

| Agent | Hormuz | Panama | Bab el-Mandeb | Malacca |
|---|---|---|---|---|
| shipping | active | active | active | active |
| market | active | active | active | **passive** |
| geopolitical | active | **passive** | active | active |
| natural_disaster | active | active | **passive** | active |
| routing | **dormant** | **dormant** | **dormant** | **dormant** |
| news_sentiment | active | active | active | active |

**Exclusion reasons, each traceable to a documented source:**

- **market / Malacca** — all four documented Malacca events carry a null market field. Removed as a **data-standards violation**, not on plausibility grounds. The distinction matters: the claim is not "markets don't matter in Malacca" but "the benchmark specification does not record a market field for these events."
- **geopolitical / Panama** — the documented disruption is purely hydrological. The region spec states outright that no geopolitical driver applies. This is the one region where the domain that dominates elsewhere contributes nothing, **which is what makes the multi-agent-value claim falsifiable rather than foreordained**.
- **natural_disaster / Bab el-Mandeb** — the documented event is a security campaign; `disaster_relevance: none`.
- **routing / all four** — uniform muting, because `fetch_api()` is a stub and the agent would only ever emit synthetic values.

### The asymmetry worth writing up

Bab el-Mandeb's routing evidence is the **strongest in the entire benchmark**: 85 % of large containerships diverted via the Cape of Good Hope, adding 3,500–4,000 nautical miles and 10–14 days per voyage — a documented percentage, not an extrapolation. Routing is muted there *in spite of* that, because no free source supplies it as a daily time series.

This is the clearest illustration of the difference between "we know this happened" and "we can measure this daily." The system's coverage is bounded by the second, not the first.

### The consequence that must be stated

With routing dormant everywhere, **no evaluation can measure routing's contribution in any region**. The tier ablation in [§4.6](#46-the-central-negative-finding) is silent on it. Evaluation results alone therefore cannot settle whether to re-enable it; that decision needs a data source, not a metric.

### Implementation

`src/core/regions.py` holds the registry. `DORMANT_AGENTS` is a module-level `frozenset`; per-region activation lives in each `RegionConfig` and is overlaid from `config/regions/*.yaml`. Registry validation **raises** if a region config activates a dormant agent, so reviving routing is a deliberate edit to `DORMANT_AGENTS` rather than an accidental YAML toggle.

`RETIRED_REGION_ALIASES` maps the retired `red_sea` and `suez` keys onto `bab_el_mandeb` so historical knowledge-base documents still resolve; migrated documents keep their original text, which still names Suez, so the distinction stays visible to anyone reading them.

---

## 3.11 Weight determination and optimization

### Analysis

Weights exist at three layers, all searchable:

| Layer | What it weights | Location | Optimized? |
|---|---|---|---|
| **L1** intra-agent | features within one agent | `agent.set_weights()` | yes |
| **L2** inter-agent | agents against each other | `RiskEngine.weights` | yes |
| **L3** thresholds | detection cutoffs, risk bands, agreement bonuses | `set_threshold()` + `RiskEngine` | yes |

`weight_mode` in `config/settings.yaml` selects `hand_tuned` (current default) or `optimized`. Hand-tuned L2 weights: shipping 0.25, geopolitical 0.25, market 0.15, routing 0.15, natural_disaster 0.10, news_sentiment 0.10.

### The risk bands are calibrated, not chosen

This is the part most likely to be assumed arbitrary and is not. The bands are the **p60 / p85 / p97 quantiles of the composite score on calm (label-negative) days**, pooled across all four regions 2019–2026 — **9,183 days**:

```yaml
thresholds:
  risk_critical: 0.90   # p97 of calm days
  risk_high:     0.69   # p85
  risk_medium:   0.51   # p60  -> ~60% of calm days read LOW
  risk_low:      0.30
```

They were recalibrated when the shipping agent stopped batch-normalising its forest score ([§3.5.1](#351-shippingagent)). Batch min-max had forced every scored window onto `[0,1]` by its own extremes, so the old `0.8 / 0.6 / 0.4` boundaries were tuned to a compressed, window-relative distribution. On absolute scores the calm median rose to ≈0.47, which put most ordinary days at MEDIUM or worse under the old bands.

**The new bands are the same quantiles of calm behaviour expressed on the new scale — a re-scaling, not a loosening.** Stating it that way is the difference between a defensible recalibration and a suspicious one.

### Optimization procedure

Optuna with a TPE sampler (seed 42), median pruner, 100 trials, 1-hour cap, and Dirichlet-style renormalisation per weight group so a sampled weight vector always sums to 1. Hard constraints reject incoherent configurations outright — `risk_high ≤ risk_medium`, and `agreement_bonus_5 ≤ agreement_bonus_3`.

**Objective:**

```
0.50 · F1  +  0.30 · lead_time_score  −  0.20 · FPR
```

where `lead_time_score` is the earliest MEDIUM alert within 5 days before onset, divided by 5. The three terms encode the operational trade-off directly: correctness, earliness, and the cost of crying wolf.

**Test-split discipline.** The test split is touched exactly once, after the study concludes. `PipelineEvaluator.evaluated_splits` is an audit trail that records every split evaluated during the study, so the claim can be checked rather than trusted.

Results, and a blocking issue that currently prevents them from being cited: [§4.9](#49-weight-optimization-results) and [§4.11](#411-known-defects-and-open-issues).

---

## 3.12 Configuration and region overlays

### Analysis

Four regions share one pipeline. The alternative — a region parameter threaded through every constructor — was rejected because it puts region logic in code that has no business knowing about regions, and because it makes "what is different about Panama?" a question answered by grepping rather than by reading one file.

Instead, configuration is **layered**: a base `settings.yaml` carries everything region-independent, and each `config/regions/<key>.yaml` overlays only what genuinely differs. The overlay mirrors the base file's key paths exactly, so a reader can diff them mentally.

The overlays are also where **evidence judgements are recorded in prose**. Panama's `geopolitical: enabled: false` sits under a comment explaining that the documented disruption is hydrological and that this is the region which makes the multi-agent claim falsifiable. That reasoning belongs next to the switch it justifies, not in a document that can drift away from it.

### Implementation

`src/core/config_manager.py`:

| Function | Purpose |
|---|---|
| `load_base_config()` | Read `config/settings.yaml` |
| `load_region_overlay(region)` | Read `config/regions/<key>.yaml`, resolving retired aliases |
| `_deep_merge(base, overlay)` | Recursive merge — overlay wins at the leaf |
| `load_config_for_region(region)` | The composed configuration the pipeline actually runs on |
| `resolve_active_region()` | CLI `--region` → `SUPPLY_CHAIN_REGION` → `hormuz` default |
| `available_regions()` | Enumerate configured regions |

Region resolution order is deliberate: an explicit flag beats an environment variable beats a default, so a scripted run cannot be silently redirected by a stale shell export.

---

## 3.13 Verification methodology

### Analysis

The test suite is not only a correctness net; several of its modules exist to **hold design claims to account** that would otherwise drift into aspiration. Three are worth describing because they test properties this chapter asserts.

**Region isolation is verified end-to-end, not by configuration.** `test_config_manager.py` checks that the composed configuration agrees with the region registry. That is necessary and insufficient: a passive agent could still be constructed, scored, and weighted despite its `enabled: false` flag, and only a full pipeline run reveals it. `test_region_isolation.py` therefore runs `run_full_pipeline()` for all four regions through a module-scoped fixture and asks a different question of each result — that no disabled agent appears in the output, that no dormant agent is built anywhere, and that weights redistribute as [§3.6](#36-layer-3--risk-aggregation) claims.

**The no-raw-scores constraint is machine-checked.** The Decision View's promise in [§3.9](#39-layer-6--presentation) is that a manager never sees a raw score. That is a claim about rendered output, so `test_dashboard.py` scans the rendered page for decimal patterns and fails if one appears. A design commitment enforced only by reviewer diligence would not survive a year of edits.

**Region overlays are checked for copy-paste.** `test_region_config_completeness.py` asserts that every region supplies each of the three region-specific connector settings **and that the values are distinct per region rather than copied from Hormuz**. The second half is the useful half: a config that is present but inherited verbatim from the first region looks complete and behaves as if the region abstraction were never applied.

### The suite

| Area | Modules | Tests |
|---|---|---|
| Ingestion and connectors | `test_ingestion.py`, `test_region_specific_connectors.py` | 93 |
| Extractors and knowledge base | `test_extractors.py`, `test_disaster_combined_extractor.py` | 60 |
| Agents | `test_agents.py`, `test_new_agents.py`, `test_scenarios.py` | 73 |
| Regions and config | `test_region_configs.py`, `test_config_manager.py`, `test_regions.py`, `test_region_isolation.py`, `test_region_config_completeness.py` | 54 |
| Dashboard | `test_dashboard_ux.py`, `test_dashboard_regions.py`, `test_dashboard.py` | 66 |
| API | `test_api_6agent.py`, `test_api_regions.py` | 24 |
| Aggregation | `test_risk_engine.py` | 17 |
| Explainability, RAG, evaluation, optimization | `test_evaluation.py`, `test_optimization.py`, `test_shap_6agent.py`, `test_rag_6domain.py`, `test_phase4_depth.py` | 33 |
| **Total** | **23 modules** | **420** |

All HTTP is mocked in the extractor and connector tests, so the suite runs offline and deterministically. `test_fred_api.py` is a standalone live diagnostic script, not collected by pytest and not counted above.

**What the suite does not verify.** It confirms the system behaves as designed; it says nothing about whether the design detects disruptions. That question belongs entirely to [§4](#4-evaluation), and the answer there is substantially less favourable than a green test suite might suggest. The two are independent, and conflating them would be the most misleading thing this document could do.

---

## 3.14 Implementation summary

| | |
|---|---|
| **Scale** | 60 source modules · 25 test modules · **420 tests, all passing** (~3m35s) |
| **Stack** | Python 3.10+, pandas, scikit-learn, SHAP, ChromaDB, Optuna, FastAPI, Streamlit |
| **Fitted models** | Two Isolation Forests + scalers (shipping, routing) and one post-hoc RandomForest surrogate. Nothing else is learned |
| **Config** | `config/settings.yaml` base + `config/regions/*.yaml` overlays |
| **Region registry** | `src/core/regions.py` — `RegionConfig`, `DORMANT_AGENTS`, `RETIRED_REGION_ALIASES`, validation |
| **Orchestration** | `src/orchestrator.py`; CLI entry `main.py --region <key>` |

`Orchestrator` builds only the enabled agents for the active region (`_build_enabled_agents`), routes each agent to its own connector frame (`_frame_for_agent`), and **degrades gracefully**: a failing connector or agent is logged and skipped rather than aborting the run, which is R7 at the orchestration level.

Full file layout in [Project Structure](#project-structure). Construction history — including the reasoning behind decisions later revised — in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md).

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

**Best run:** trial 63, validation objective **0.7491**. Reproduced 2026-09-05: **100 trials requested, 57 completed, 43 pruned** by the median pruner, of which 8 were rejected by the hard constraints. Quote all three numbers — "57 trials" alone reads as a truncated run, and "100 trials" alone hides the pruning.

> ### These numbers are not comparable to [§4.5](#45-results)
>
> Everything in this section is measured on **synthetic splits**: three independent 365-day realisations of the same generated world (seeds 42 / 43 / 44), carrying a ground-truth `is_disruption` label the data generator placed deliberately.
>
> [§4.5](#45-results) is measured on the **real evaluation set**: live connector data 2019–2025, with a label derived statistically from vessel counts.
>
> Different data, different label, different harness. **An F1 of 0.933 here does not contradict a Tier 1 AUC of 0.502 there, and neither number transfers to the other setting.** The optimizer runs on synthetic data because it is the only source carrying a label that was placed rather than inferred — see [§3.4](#34-layer-1--ingestion). What this section establishes is that *tuning improves the pipeline against its own objective*, not that the pipeline detects real disruptions.

### Weights

| Agent | Hand-tuned L2 | Optimized L2 |
|---|---|---|
| shipping | 0.25 | **0.402** |
| market | 0.15 | 0.088 |
| geopolitical | 0.25 | 0.109 |
| natural_disaster | 0.10 | 0.095 |
| routing | 0.15 | 0.130 |
| news_sentiment | 0.10 | **0.177** |

The optimizer independently concentrates weight on shipping — consistent with the dilution mechanism in [§4.6](#46-the-central-negative-finding), since shipping is the domain the label is derived from. It reaches that conclusion from the synthetic objective alone, without access to the real-data ablation.

### Measured performance (synthetic splits, 365 days each)

| Split | Config | F1 | Precision | Recall | FPR | Lead time | Objective |
|---|---|---|---|---|---|---|---|
| Validation | optimized | 0.978 | 1.000 | 0.957 | 0.000 | 4.33 d | **0.7491** |
| Validation | hand-tuned | 0.932 | 1.000 | 0.872 | 0.000 | 1.67 d | 0.5659 |
| Test | optimized | 0.933 | 0.977 | 0.894 | 0.0031 | 5.00 d | **0.7660** |
| Test | hand-tuned | 0.894 | 1.000 | 0.809 | 0.000 | 1.67 d | 0.5471 |

**The gain is lead time, not accuracy.** F1 improves by 0.039 on the test split; detection lead time improves from 1.67 to 5.00 days — the full width of the 5-day window the objective scores. That is what the `0.30·lead_time_score` term was written to buy.

**And it is paid for in false positives.** Hand-tuning holds FPR at exactly 0.000 on both splits. Optimization gives up 0.0031 on test. The `−0.20·FPR` term permits that trade deliberately; whether 3 false positives per 1,000 days is worth 3.3 days of warning is an operational judgement the objective encodes but does not settle.

**`weight_mode` remains `hand_tuned` by default.** Switching to `optimized` is a deliberate config change, because the trade above should be made knowingly.

### Reproducibility

The 2026-09-05 re-run reproduced `config/optimized_weights.yaml` **byte-identically apart from the date header** — every weight to six decimal places, same best trial. The optimizer is fully deterministic under its seed, so this section can be regenerated with:

```bash
python main.py --optimize --trials 100
```

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

### Resolved

1. ~~**The optimization records contradict each other.**~~ **Fixed 2026-09-05.**

   `data/processed/optimization_results.json` had been overwritten by a 5-trial smoke run (best trial 1, objective 0.6301) carrying a different weight vector from the committed YAML (news_sentiment 0.046 vs 0.177; natural_disaster 0.168 vs 0.095). Because **`notebooks/evaluation.py` METRIC 5 reads that JSON**, the suite would have reported the smoke run as the optimization result.

   Re-running the full study (`python main.py --optimize --trials 100`) regenerated both artifacts consistently. The YAML came back **byte-identical apart from its date header**, confirming that the committed weights were always the genuine 100-trial result and that the JSON was the only corrupted artifact. Results are now in [§4.9](#49-weight-optimization-results).

### Blocking

1. **The 8-metric suite has never been run against the current system.** `evaluation_results.json` and `thesis_comparison_table.json` are dated **2026-07-30** — before the re-optimization and before the shipping level-shift rework. METRICS 1–8 as they sit on disk describe a superseded system.

   Note that resolving the optimization contradiction above does **not** clear this. METRIC 5 will now read a correct JSON, but the other seven metrics have not been recomputed. **No 8-metric number should be cited until `notebooks/evaluation.py` is re-run end to end.**

### Code-level inconsistencies to disclose

1. **Routing normalises its Isolation Forest by batch min-max**, the exact practice shipping was deliberately moved away from. Routing went dormant before it was migrated.
2. **`Orchestrator.run_timeseries_analysis` omits the agreement bonus** that every other composite path applies.
3. **The agreement threshold (0.5) is a module constant**, not configurable or optimized — while the bonuses it gates are both.
4. **Geopolitical computes rolling baselines and deviations that `detect()` never reads**; news does the same with `sentiment_rolling_7d`. `sentiment_magnitude` is a schema feature that never enters any score.
5. **`PipelineEvaluator.build_agents` hardcodes agent config** instead of reading `settings.yaml`, so the optimizer tunes a slightly different agent than the live pipeline runs.
6. **The SHAP surrogate sees 20 of the features, not all.** Absent: `tanker_count`, `vessel_count_trend`, `freight_services_pct_change`, `vessels_holding`, `alternative_route_traffic`, `sentiment_magnitude`, `recency_weighted_score`. Missing columns fill with 0.0 — indistinguishable from a true zero.
7. **The shipping duration score is near-circular** against the level-shift label. Cite operationally, never as detection skill.
8. **The test suite writes into the figure directory.** `tests/test_phase4_depth.py::test_comparison_plots_saved` calls `generate_comparison_plot(..., save_dir="data/processed/")`, so **running `pytest` silently overwrites `shap_comparison_waterfall.png` and `shap_comparison_importance.png` with plots built from test fixtures**. Any figure taken from that directory must be regenerated from a real pipeline run *after* the last test invocation. The test should write to a `tmp_path` fixture instead.

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
| Optimizer (6) | `data/processed/` (gitignored) | **Current** — regenerated 2026-09-05 by the 100-trial re-run |
| SHAP comparison (2) | `data/processed/` (gitignored) | ⚠ **Do not cite.** Overwritten by the *test suite* on every run — see the disclosure below. They show fixture data, not a real pipeline run |
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

> ⚠️ Its output is **not currently citable** — see [§4.11](#411-known-defects-and-open-issues). The optimization records it reads were repaired on 2026-09-05, so METRIC 5 is now sound, but METRICS 1–8 on disk still date from 2026-07-30 and describe a superseded system. Re-run this script end to end before citing any of them.

### Populate the RAG knowledge base from live APIs

```bash
python scripts/populate_knowledge_base.py                       # all extractors in extraction.enabled_extractors
python scripts/populate_knowledge_base.py --extractors serpapi   # one-time historical backfill only
```

Extracts → deduplicates by document id → backs up to `data/knowledge_base/live_extracted_backup.json` (gitignored) → upserts into the `live_extracted_context` ChromaDB collection. Safe to re-run; missing API keys degrade individual extractors to zero documents rather than failing the run.

Document ids are region-scoped (`acled_{region}_{country}_{year}`). They were not originally: countries shared between chokepoints produced identical ids, so the second region's rows were silently dropped by the deduplicator. Any extractor spanning overlapping country sets needs the same treatment.

> **Historical note.** Earlier runs of this script are reported in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) with document counts that include the `ambee` extractor and the `red_sea` / `suez` region keys. Both are retired — Ambee returns zero documents with a valid key, and the two region keys were folded into `bab_el_mandeb`. Current sources are in [§3.4](#live-data-sources).

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
| `agents.shipping.z_threshold` | `2.0` | Z-score normalisation cap for the secondary fallback channel (code default is 3.0; settings.yaml sets 2.0) |
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

