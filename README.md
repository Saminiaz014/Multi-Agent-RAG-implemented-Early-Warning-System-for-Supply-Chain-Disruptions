# Multi-Agent RAG-Implemented Early Warning System for Supply Chain Disruptions

A thesis-grade **Decision Support System (DSS)** for detecting, explaining, and contextualising supply chain disruptions in the **Strait of Hormuz** maritime corridor. The system combines multi-agent anomaly detection, SHAP-based explainability, and retrieval-augmented generation (RAG) for historical precedent retrieval — producing structured, interpretable alerts rather than raw predictions.

---

## Research Context

The Strait of Hormuz is one of the world's most critical maritime chokepoints, carrying approximately 20% of global oil trade. Disruptions — caused by geopolitical tensions, sanctions, vessel incidents, or market shocks — propagate rapidly across global supply chains. This system provides decision-makers with:

- **Early warning** from multi-source signal monitoring
- **Explainability** — which features drove the anomaly score
- **Historical grounding** — which past disruption events are most analogous
- **Risk classification** — composite HIGH / MEDIUM / LOW risk levels with configurable thresholds

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Raw Signals                        │
│     (AIS vessel data · oil futures · incident feeds)    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Ingestion Layer                        │
│   BaseConnector ABC → domain-specific connectors        │
│   fetch() · validate() · fetch_and_validate()           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Multi-Agent Detection Layer                │
│   BaseAgent ABC → ShippingAgent · MarketAgent · ...     │
│   Isolation Forest (shipping) · Z-score (market)        │
│   Each agent produces: anomaly_scores + anomaly_flags   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Risk Aggregation Engine                    │
│   Weighted average of per-agent anomaly scores          │
│   Config-driven weights · RiskLevel enum (HIGH/MED/LOW) │
└────────────┬──────────────────────┬─────────────────────┘
             │                      │
             ▼                      ▼
┌────────────────────┐   ┌──────────────────────────────┐
│  SHAP Explainer    │   │    RAG Context Retriever     │
│  TreeExplainer /   │   │  ChromaDB + sentence-         │
│  KernelExplainer   │   │  transformers (local, no API) │
│  top-N feature     │   │  top-k similar historical     │
│  contributions     │   │  disruption cases             │
└────────┬───────────┘   └──────────────┬───────────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  Decision Output                        │
│  { composite_score, risk_level, agent_scores,           │
│    shap_contributions, historical_precedents }          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Layer  (Phase 8)              │
│  GET  /health · /agents · /weights · /status            │
│  GET  /optimization/results                             │
│  POST /predict · /explain · /populate                   │
│  POST /agents/toggle · /weights/switch                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            Streamlit Dashboard  (Phase 9b)              │
│  Page 1 · Decision View — status words, action badge,   │
│           MapLibre 3-D map, LLM risk narrative          │
│  Page 2 · Analysis View — all 8 evaluation metrics,     │
│           range explorer, per-chart JPEG export         │
└─────────────────────────────────────────────────────────┘
```

Since Phase 10 the whole chain runs unattended: a **local Apache Airflow** DAG (Docker Compose, `airflow/`) executes pipeline → RAG populate → evaluation daily at 08:00 with hard-stop failure semantics, and the dashboard surfaces the refresh through a "Data refreshed: …" badge.

---

## Data Sources

Each of the six agents is backed by a free, public data source. Every source has a CSV and a `synthetic` mode, so the full pipeline runs locally with **no API keys required**. Live integrations exist at two levels: the Phase 7 extractors in `src/extractors/` (ACLED, FRED, NewsAPI, ReliefWeb, SerpAPI, Ambee, aisstream) feed the RAG corpus from real APIs, and `DisasterConnector.fetch_api()` queries Ambee directly; the remaining connectors' `api` source mode is stubbed (`NotImplementedError` with planned-integration docstrings). All keys are optional and blank by default.

| Agent | Source(s) | Access | Notes |
|---|---|---|---|
| **Shipping** | [aisstream.io](https://aisstream.io) | Free WebSocket AIS | Live vessel positions/arrivals, bounding-box filtered to the Shuaiba / Hormuz corridor |
| **Market** | [FRED API](https://fred.stlouisfed.org/docs/api/fred/) | Free (key) | Brent Crude `DCOILBRENTEU`, Freight PPI `PCU4831114831111`, Freight Services Index `PCUATFREIATFREI` |
| **Geopolitical** | [ACLED](https://acleddata.com), [OpenSanctions](https://www.opensanctions.org), [GDELT](https://www.gdeltproject.org) | ACLED/OpenSanctions (key), GDELT (no key) | Conflict events · sanctions data · global event database |
| **Natural Disaster** | [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/), [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive), [Ambee](https://www.getambee.com) | USGS/NOAA (no key), Ambee (key) | Earthquakes (GeoJSON) · cyclones (public-domain CSV, North-Indian basin) · Ambee live disaster feed (the wired `fetch_api()` path) |
| **Routing** | [aisstream.io](https://aisstream.io) | Free WebSocket AIS | Vessel positions with Hormuz / Cape-of-Good-Hope bounding-box filters for rerouting detection |
| **News Sentiment** | [GDELT DOC API](https://www.gdeltproject.org/), [NewsAPI.org](https://newsapi.org) | GDELT (no key), NewsAPI (free dev tier) | GDELT tone scores (no key) · NewsAPI free tier (1,000 req/day) |

> Shipping and market run on **real CSV extracts by default** (IMF PortWatch Shuaiba daily calls, FRED Brent + freight indices); the other four agents default to `synthetic` mode for reproducible, label-bearing evaluation. Flip `data_mode` / `source_mode` per source to switch between `synthetic`, `csv`, and `api`.

---

## Phase 0 — Project Scaffolding

> **Summary.** Project scaffold + ABCs + skeleton modules.
> Adds: `config/settings.yaml`, `main.py` entrypoint, `src/orchestrator.py`, `BaseConnector` and `BaseAgent` ABCs (with `DetectionResult` dataclass), `RiskEngine` with weighted aggregation and `RiskLevel` enum, SHAP explainer wrapper, ChromaDB RAG retriever, FastAPI `/predict` `/explain` `/health` endpoints, and the initial 16-test suite (`tests/test_agents.py`, `test_risk_engine.py`, `test_scenarios.py`).

This session delivered the **complete project scaffold** — every module is independently importable and tested, forming the foundation for domain-specific agent and connector implementations.

### Core Infrastructure

| Module | File | Description |
|--------|------|-------------|
| Config system | `config/settings.yaml` | All tunable parameters — agent weights, detection thresholds, RAG settings, API host/port, logging config |
| Entrypoint | `main.py` | Loads YAML config, wires dual-sink logging (console + file), runs pipeline |
| Orchestrator | `src/orchestrator.py` | Registers agents, sequences detection → aggregation stages, returns structured output dict |

### Abstract Base Classes

| Class | File | Enforced Contract |
|---|---|---|
| `BaseConnector` | `src/ingestion/base_connector.py` | `fetch()`, `validate()`, `fetch_and_validate()` — schema validation built in |
| `BaseAgent` | `src/agents/base_agent.py` | `fit()`, `detect()`, `fit_detect()` — plus `DetectionResult` dataclass |

Both use Python's `ABC` / `@abstractmethod` pattern, ensuring domain implementations cannot be instantiated without satisfying the full interface.

### Risk Aggregation

`src/aggregation/risk_engine.py` — `RiskEngine` takes a list of `DetectionResult` objects, applies configurable per-agent weights, computes a weighted mean composite score, and maps it to a `RiskLevel` enum (`HIGH ≥ 0.7`, `MEDIUM ≥ 0.4`, `LOW` otherwise). Unknown agents are skipped with a warning; non-unit weights are auto-renormalised.

### SHAP Explainability

`src/explainability/shap_explainer.py` — `ShapExplainer` wraps any fitted sklearn estimator. It auto-selects `TreeExplainer` for tree-based models (e.g., Isolation Forest) and falls back to `KernelExplainer` with a 50-sample background for other models. Exposes `explain()` returning raw SHAP values and sorted mean-absolute contributions, and `top_features(n)` for quick reporting.

### RAG Context Retrieval

`src/rag/context_retriever.py` — `ContextRetriever` uses a local persistent **ChromaDB** collection and **sentence-transformers** (`all-MiniLM-L6-v2`) for embedding. `load_knowledge_base()` indexes JSON case files incrementally (skips already-indexed IDs). `retrieve(query_text)` returns the top-k most semantically similar historical disruption cases with cosine distances. Fully local — no external API keys.

### FastAPI Endpoints

`src/api/endpoints.py` — Three routes:
- `POST /predict` — accepts feature vector + agent name, returns composite score and risk level
- `POST /explain` — returns SHAP top features and RAG context (wired to pipeline in next phase)
- `GET /health` — liveness probe

### Test Suite

| Test file | Coverage |
|---|---|
| `tests/test_agents.py` | ABC enforcement, `DetectionResult` shape/type, correct flag logic |
| `tests/test_risk_engine.py` | HIGH/MEDIUM/LOW boundary cases, unknown-agent skipping, weight renormalisation |
| `tests/test_scenarios.py` | End-to-end orchestrator runs with synthetic normal vs. disrupted Hormuz signal data |

---

## Phase 1 — Data Ingestion

> **Summary.** Two synthetic data connectors with shared ground truth.
> Adds: `src/ingestion/shipping_connector.py` (365-day Hormuz dataset with three injected disruption scenarios — Moderate Tension days 60-74, Major Blockage days 150-170, Brief Incident days 280-290) and `src/ingestion/market_connector.py` (Brent / trade volume / freight rate signals lag-aligned to shipping with 2-day propagation and mean-reverting tails). Welch t = 10.04 on vessel_count separation, Pearson r = 0.746 between shipping vessel_count and market trade_volume_index in disruption windows. 30 new ingestion tests; 46/46 total passing.

This session delivered the **first two concrete data connectors** — a synthetic Strait of Hormuz shipping signal generator and a temporally aligned market signal generator. Together they give detection agents a reproducible, multi-source dataset with shared ground-truth disruption labels, enabling cross-agent validation without depending on live AIS or market feeds.

### Shipping Connector

`src/ingestion/shipping_connector.py` — `ShippingConnector` extends `BaseConnector` and produces a 365-day daily-frequency DataFrame with:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Daily index starting 2025-01-01 |
| `vessel_count` | int | Tankers transiting the strait (baseline 60-80) |
| `avg_delay_hours` | float | Mean transit delay in hours (baseline 2-8) |
| `congestion_index` | float | Corridor saturation, 0=free → 1=gridlock (baseline 0.1-0.4) |
| `oil_price_usd` | float | Brent reference price (baseline $70-85/bbl) |
| `is_disruption` | bool | Ground-truth label — **NOT** an input feature |

Baseline normal-period values track published Strait of Hormuz traffic statistics; daily Gaussian noise prevents flat-line series.

### Injected Disruption Scenarios

Three disruptions are seeded into every generated dataset, each with a gradual ramp-up and decay so the resulting series resembles real incident progression rather than step functions:

| Scenario | Window | Vessel Drop | Delay Multiplier | Congestion | Oil Premium |
|---|---|---|---|---|---|
| Moderate Tension | days 60-74 | 20-35% | 2-3× | 0.50-0.70 | +10-15% |
| Major Blockage | days 150-170 | 50-70% | 4-6× | 0.70-0.95 | +25-40% |
| Brief Incident | days 280-290 | 10-20% | 1.5-2× | 0.40-0.60 | +5-10% |

Per-scenario magnitudes are sampled once via the seeded RNG so each disruption has a coherent character; intensity is then scaled per-day to produce the ramp/decay shape. Total labelled disruption days ≈ 47 of 365.

### Public Methods (Shipping)

| Method | Purpose |
|---|---|
| `generate_dataset(days, seed)` | Produce the full DataFrame; reproducible by seed |
| `fetch()` | Base-class hook — reads `days`/`seed` from `self.config` |
| `validate(df)` | Schema + range checks (no NaN, congestion in [0,1], vessel_count ≥ 0, ~46 disruption days) |
| `save_raw(path)` | Generate + validate + write CSV (default `data/raw/shipping_hormuz.csv`) |
| `to_signal_records(df)` | Convert to unified signal schema: `{timestamp, source, feature, value, location}` |

### Statistical Validation

The connector prints a Welch t-statistic separating normal vs. disrupted `vessel_count` distributions on every generation. With seed 42, the 365-day dataset produces:

```
[ShippingConnector] vessel_count separation — normal mean=70.05 (n=318), disruption mean=46.32 (n=47), Welch t=10.04
```

A t-statistic above 5 confirms the injected disruptions are clearly separable from normal traffic — a precondition for downstream Isolation Forest training.

### Market Connector

`src/ingestion/market_connector.py` — `MarketConnector` extends `BaseConnector` and produces a 365-day daily-frequency DataFrame with three market signals temporally aligned to the shipping connector's disruption windows:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Daily index aligned with shipping data |
| `brent_crude_usd` | float | Brent crude price (baseline $75-82/bbl), reacts to disruptions with a 1-2 day lag |
| `trade_volume_index` | float | Normalised trade throughput in `[0, 1]` (baseline ~0.85), drops during disruptions |
| `freight_rate_index` | float | Composite freight rate (baseline 100-120), spikes during disruptions |
| `is_disruption` | bool | Ground-truth label — **identical** to the shipping connector's flag for cross-agent validation |

**Information-propagation lag.** Market features lag the underlying shipping disruption by `lag_days` (default 2, configurable per call). The shipping window starts at day `s`; the market envelope starts at day `s + lag_days`, so traders observe price/volume anomalies after the physical event begins — mirroring real-world signal propagation.

**Mean-reverting tails.** After the window closes, the disruption envelope decays exponentially with persistence factor 0.7 (~30% decay per day) instead of snapping back to baseline. This produces realistic post-disruption dynamics where prices and freight rates remain elevated for several days before fully settling.

**Aligned scenarios.** The default `disruption_periods` argument seeds severity-scaled responses for the same three windows used by the shipping connector: Moderate Tension (severity 0.45, days 60-74), Major Blockage (severity 1.00, days 150-170), and Brief Incident (severity 0.25, days 280-290). Severity controls the magnitude of the Brent uplift (up to +20%), the trade-volume drop (up to −0.45), and the freight-rate spike (up to +50%).

### Public Methods (Market)

| Method | Purpose |
|---|---|
| `generate_dataset(days, seed, disruption_periods, lag_days)` | Produce the full DataFrame; reproducible by seed; disruption windows and lag both configurable |
| `fetch()` | Base-class hook — reads `days`, `seed`, and `lag_days` from `self.config` |
| `validate(df)` | Schema + range checks (no NaN, `trade_volume_index` in [0, 1], positive prices and freight) |
| `save_raw(path)` | Generate + validate + write CSV (default `data/raw/market_data.csv`) |
| `to_signal_records(df)` | Convert to unified signal schema: `{timestamp, source, feature, value, location}` |

### Cross-Source Correlation

Because both connectors share the same disruption windows, market signals confirm shipping anomalies. With seed 42, the Pearson correlation between shipping `vessel_count` and market `trade_volume_index` over the 47 ground-truth disruption days is:

```
[test] vessel_count <-> trade_volume_index Pearson r (disruption days, n=47): 0.746
```

A correlation above 0.5 confirms the two synthetic sources are mutually consistent during disruption windows — the precondition for meaningful multi-agent validation.

### Test Coverage (Phase 1)

| Test file | Coverage |
|---|---|
| `tests/test_ingestion.py` | 30 tests: 14 for `ShippingConnector` (schema, NaN/range checks, ~46 disruption days, scenario window placement, Welch t > 5, seed reproducibility, `validate()` happy/sad path, unified-signal JSON schema, CSV persistence) + 16 for `MarketConnector` (schema, baseline ranges, ground-truth alignment with shipping, lagged response peak, mean-reverting decay, seed reproducibility, validate happy/sad path, signal-record schema, CSV persistence, and the cross-source Pearson r > 0.5 correlation check) |

All 30 ingestion tests pass alongside the 16 prior-week tests — **46/46 total**.

---

## Phase 2 — Detection Agents

> **Summary.** First two concrete detection agents wired to the Week-2 datasets.
> Adds: `src/agents/shipping_agent.py` (Isolation Forest + Z-score fallback with leak-free fit on non-disruption rows, persistence + multi-feature validation, **TPR = 0.936 / FPR = 0.003**) and `src/agents/market_agent.py` (trailing 30-day rolling Z-scores, oil-led validation gate, **TPR = 0.809 / FPR = 0.022**). Both expose `run_dataframe` + `to_detection_result` for `RiskEngine` integration. 8 new agent tests; 54/54 total passing.

This session delivered the **first two concrete detection agents** of the multi-agent pipeline — `ShippingAgent` (physical-flow side: vessel counts, transit delays, corridor congestion) and `MarketAgent` (price-side: Brent crude, trade volume, freight rates) — both wired to the synthetic Strait of Hormuz datasets produced in Phase 1. Together they form a two-channel early-warning system: the shipping agent fires first on the physical event, and the market agent corroborates it 1-2 days later as the price reaction propagates. Both agents conform to the same `BaseAgent` ABC, emit the same dict schema, and slot into the existing `RiskEngine.aggregate()` call site without any orchestrator changes.

| Agent | File | Detection Strategy | TPR | FPR |
|---|---|---|---|---|
| `ShippingAgent` | `src/agents/shipping_agent.py` | Isolation Forest (multivariate) + Z-score fallback | **0.936** | **0.003** |
| `MarketAgent` | `src/agents/market_agent.py` | Rolling 30-day Z-scores (per-feature) | **0.809** | **0.022** |

The two strategies are deliberately different — see [Why a Different Detection Strategy](#why-a-different-detection-strategy) below for the design rationale. The rest of this section documents the shipping agent first (since it is the lead indicator), then the market agent.

### Why the Shipping Agent Matters

Anomaly detection in a single dimension is brittle: a one-day vessel-count dip is more likely a port maintenance event than a strait closure, and a single elevated delay reading can be weather noise. The shipping agent's design directly attacks both failure modes:

- **Multivariate primary detection** — Isolation Forest sees `vessel_count`, `avg_delay_hours`, and `congestion_index` jointly. A coordinated drop across all three is a far stronger disruption signal than any one feature alone.
- **Univariate fallback** — A Z-score channel captures extreme single-feature spikes the forest may smooth over (rare, sharp events that don't match any historical isolation pattern).
- **Two-stage validation** — Even a strong score is suppressed unless (a) it persists ≥ 2 days and (b) ≥ 2 of 3 features are elevated. This is the part that turns a noisy classifier into a usable early-warning signal.

The result is a deterministic, config-driven, leakage-free detector that the rest of the pipeline (SHAP, RAG, risk engine) can build on without re-tuning.

### Architectural Placement

`src/agents/shipping_agent.py` — `ShippingAgent` extends `BaseAgent`. It satisfies the abstract contract (`fit()`, `detect()`) and adds four pipeline-specific methods (`preprocess`, `validate`, `output`, `run`) plus two adapters (`run_dataframe`, `to_detection_result`) that bridge to the existing `RiskEngine` aggregation path. The agent is **stateless across `run()` invocations** — only the fitted `StandardScaler` and `IsolationForest` are retained, both produced by `fit()`.

```
              ┌────────────────────┐
              │  Raw shipping CSV  │
              └─────────┬──────────┘
                        ▼
            ┌─────────────────────────┐
            │   ShippingAgent.fit()   │   ← fit on is_disruption == False rows
            │  (StandardScaler + IF)  │     to prevent leakage
            └─────────────┬───────────┘
                          ▼
            ┌──────────────────────────┐
            │  preprocess(data)        │   select features · ffill ·
            │  → scaled DataFrame      │     scaler.transform → z-space
            └──────────────┬───────────┘
                           ▼
            ┌──────────────────────────┐
            │  detect(scaled)          │   IsolationForest.decision_function
            │  → anomaly_score [0,1]   │   + max|z| / z_threshold
            │  → is_anomaly bool       │   combined: 0.7·IF + 0.3·z_norm
            └──────────────┬───────────┘
                           ▼
            ┌──────────────────────────┐
            │  validate(signals)       │   ≥ 2-day run AND ≥ 2 of 3
            │  → validated bool        │   features with |z| > 1.5
            └──────────────┬───────────┘
                           ▼
            ┌──────────────────────────┐
            │  output(validated)       │   group consecutive validated days
            │  → List[dict] per window │   → unified anomaly-report schema
            └──────────────────────────┘
```

### Configuration (`config/settings.yaml`)

Read from `agents.shipping` — every numeric is overridable per call without touching code:

| Key | Default | Effect |
|---|---|---|
| `contamination` | `0.10` | Expected anomaly fraction the Isolation Forest budgets for. Higher → more sensitive, more false positives. The evaluation harness uses `0.13` (≈ 47/365 ground-truth rate) for a tight match to the synthetic prior. |
| `threshold` | `0.65` | Minimum **combined** score to set `is_anomaly=True`. The evaluation harness uses `0.55` to give the validation stage room to filter rather than relying on a hard pre-validation cutoff. |
| `z_threshold` | `3.0` | Used twice — (1) as the Z-score normalisation cap so a single feature ≥ 3σ saturates the secondary score, and (2) reserved for downstream univariate-fallback decisions. |

Internal-only constants (top of `shipping_agent.py`):

| Constant | Value | Purpose |
|---|---|---|
| `_FEATURE_COLUMNS` | `(vessel_count, avg_delay_hours, congestion_index)` | The three scored features. `oil_price_usd` is intentionally excluded — it lives in the market agent. |
| `_LOCATION` | `"Strait of Hormuz"` | Stamped on every output dict for downstream geographic indexing. |
| `_FEATURE_ELEVATION_Z` | `1.5` | Per-feature elevation cutoff for the multi-feature validation check. |
| `_PERSISTENCE_DAYS` | `2` | Minimum run length for the persistence check. |
| `_MIN_FEATURES_ELEVATED` | `2` | Minimum number of features that must be elevated on a row for it to clear validation. |

### Method-by-Method Walkthrough

#### `fit(df)` — Train scaler + forest with no leakage

```python
if "is_disruption" in df.columns:
    train = df.loc[~df["is_disruption"].astype(bool)]   # 318 normal rows
features = train[FEATURES].ffill().dropna()
self._scaler  = StandardScaler().fit(features)
self._iforest = IsolationForest(
    contamination=0.13, random_state=42, n_estimators=200
).fit(scaler.transform(features))
```

The crucial detail: when ground truth is available, **disruption rows are excluded from the fit** so neither the scaler's mean/variance nor the forest's isolation paths can be polluted by the very anomalies they're meant to detect. In production (no `is_disruption` column), the agent fits on the entire training window and trusts the contamination prior to absorb baseline outliers. `n_estimators=200` (up from sklearn's 100) trades a fraction of a second for noticeably more stable scores on the 365-row dataset.

#### `preprocess(data)` — Project new data into the trained z-space

Selects the three features, forward-fills short gaps (sensor dropouts), drops residual NaNs, and applies the *fitted* scaler. The `timestamp` column is preserved for downstream window labelling and `is_disruption` is carried through unchanged so the same frame can flow into evaluation. This method **never refits** — calling it before `fit()` raises `RuntimeError`.

#### `detect(data)` — Hybrid score, [0, 1] normalised

```python
iforest_norm = (-decision_function - min) / (max - min)         # IF in [0,1]
max_z_norm   = min(max(|z|) / z_threshold, 1.0)                 # Z in [0,1]
anomaly_score = 0.7 * iforest_norm + 0.3 * max_z_norm
is_anomaly    = anomaly_score >= threshold
```

`decision_function` returns higher values for "more normal", so we negate before normalising. Min-max normalisation is computed *over the detection window* — this is intentional: it means the highest-scoring row in a given run always saturates at 1.0, giving downstream consumers a stable upper bound. The 70/30 weight favours the multivariate signal but lets a single extreme feature drag a row over the line. The output frame retains the per-feature z-scores as `vessel_count_zscore`, `delay_zscore`, `congestion_zscore` so SHAP and the validation stage can reason about them directly.

#### `validate(signals)` — Two-stage false-positive suppression

```python
# (1) Persistence: each True must sit in a run of length >= 2
runs = identify_consecutive_anomalies(is_anomaly)
persistent = mask where run_length >= 2

# (2) Multi-feature: at least 2 of 3 features must show |z| > 1.5
elevated   = sum(|z_i| > 1.5 for i in features)
feature_ok = elevated >= 2

validated = persistent AND feature_ok
```

The persistence check is implemented as an in-place run-length scan rather than a `rolling().sum()` because we need the **forward** condition too (a row at the start of a 5-day disruption must still be marked persistent on day 1, not day 2). The multi-feature check is what kills "vessel count dipped because Eid al-Fitr"-style false positives — a single-feature dip alone, however statistically extreme, will not survive validation. Both gates are AND'd, so a row clears validation only if it is both persistent and broad-based.

#### `output(validated_signals)` — Window-level structured reports

Consecutive validated days are collapsed into anomaly windows. For each window the agent emits:

```python
{
  "agent": "shipping",
  "anomaly_score": float,           # max combined score across the window
  "confidence": float,              # mean (features_elevated / 3) over window
  "signals": {
    "vessel_count_zscore": float,   # max |z| in the window
    "delay_zscore":        float,
    "congestion_zscore":   float,
  },
  "start_timestamp": "YYYY-MM-DD",
  "end_timestamp":   "YYYY-MM-DD",
  "location": "Strait of Hormuz",
}
```

`confidence` is deliberately distinct from `anomaly_score`: a window can score 1.0 (saturated isolation forest) but only have 2 of 3 features firing on most days (`confidence ≈ 0.67`), or score 0.7 with all three features elevated every day (`confidence = 1.0`). Downstream consumers (the LLM-facing summariser, the SHAP report) can use this split to reason about *certainty* independently of *severity*.

#### `run(data)` — Pipeline orchestration

Auto-fits if needed (one-shot evaluation mode), then chains
`preprocess → detect → validate → output`, logging each stage's row counts. Returns the list of structured window dicts. For per-row evaluation, `run_dataframe(data)` exposes the intermediate validated DataFrame, and `to_detection_result(validated)` adapts that frame to the existing `DetectionResult` dataclass so the agent slots into `RiskEngine.aggregate()` unchanged.

### Evaluation Results (synthetic Hormuz dataset, seed 42)

Run `pytest tests/test_agents.py::test_shipping_agent_evaluation -v -s` to reproduce:

```
======================================================================
ShippingAgent — End-to-End Evaluation on Synthetic Hormuz Dataset
======================================================================
Total rows               : 365
Ground-truth disruption  : 47 days
Predicted (validated)    : 45 days
Anomaly windows reported : 5

Confusion Matrix
                 pred=Normal   pred=Disruption
true=Normal              317                 1
true=Disruption            3                44

TPR (recall)     : 0.936
FPR              : 0.003
Precision        : 0.978

Classification Report
              precision    recall  f1-score   support
      Normal      0.991     0.997     0.994       318
  Disruption      0.978     0.936     0.957        47
    accuracy                          0.989       365
    macro avg     0.984     0.967     0.975       365
    weighted avg  0.989     0.989     0.989       365
======================================================================
```

| Metric | Target | Achieved |
|---|---|---|
| **True Positive Rate** | ≥ 0.80 | **0.936** |
| **False Positive Rate** | ≤ 0.15 | **0.003** |
| Precision | — | 0.978 |
| F1 (Disruption) | — | 0.957 |

The agent recovers 44 of 47 ground-truth disruption days with a **single false positive** across 318 normal days. The three missed days fall at the extreme edges of the ramp-up / decay tails of the Major Blockage scenario, where the disruption envelope is small enough that the multi-feature elevation gate (≥ 2 of 3 features with |z| > 1.5) correctly judges them indistinguishable from baseline noise — a false negative we accept in exchange for the near-zero false-positive rate.

The five reported windows correspond cleanly to the three injected scenarios (with the Major Blockage window split into a main body and two short trailing fragments), confirming the pipeline preserves event-level structure rather than just per-row classification.

### Test Coverage (Shipping Agent)

`tests/test_agents.py` — 4 new shipping-agent tests added on top of the 5 prior ABC contract tests, all passing:

| Test | Asserts |
|---|---|
| `test_shipping_agent_evaluation` | TPR ≥ 0.80, FPR ≤ 0.15 against the synthetic dataset; prints confusion matrix, classification report, and a sample window dict |
| `test_shipping_agent_run_output_schema` | Every dict from `run()` contains all required keys, scores are in `[0, 1]`, and the `signals` block has exactly the three z-score features |
| `test_shipping_agent_determinism` | Two fresh agents with identical config produce bit-identical anomaly scores and validated flags (deterministic — `random_state=42`) |
| `test_shipping_agent_no_leakage` | `fit()` discards disruption rows when `is_disruption` is present — verified by checking the fitted scaler's mean matches the non-disruption mean and **not** the full-dataset mean |

### Non-Functional Guarantees

- **Determinism** — `IsolationForest(random_state=42)`; no other randomness. Same input + config → identical output, byte-for-byte.
- **No data leakage** — `fit()` filters on `is_disruption == False` whenever the column is present; the scaler is never refit during inference; `preprocess()` raises if called before `fit()`.
- **Type hints** — Every public signature is fully annotated (`pd.DataFrame`, `list[dict[str, Any]]`, etc.).
- **Google-style docstrings** — Every public method has Args / Returns sections; module-level docstring explains the agent's role in the broader pipeline.
- **Logging** — Every pipeline stage emits an `INFO`-level line with row counts and key parameters, routed through the project's standard logger.

### Market Agent

`src/agents/market_agent.py` — `MarketAgent` is the **second concrete detection agent** and the cross-confirmation layer for the shipping signal. Where the shipping agent watches the physical-flow side of a Strait of Hormuz disruption (vessel counts, transit delays, corridor congestion), the market agent watches the **price-side and trade-flow** reaction (Brent crude price, normalised trade volume, composite freight rates). Because the market envelope lags the underlying shipping disruption by ~1-2 days and decays slowly afterwards, it functions as an **independent corroborator**: a shipping flag confirmed by an aligned market response is far less likely to be a false alarm than a shipping-only flag.

#### Why a Different Detection Strategy

Market features are inherently noisier than shipping features and exhibit strong temporal autocorrelation — yesterday's Brent price is the single best predictor of today's. An Isolation Forest on the raw values would either (a) require a meticulously curated training window to avoid memorising baseline drift, or (b) collapse on a stationary subset that's not representative of live market conditions. Instead, the market agent uses **rolling-window Z-scores** that adapt to slow drift naturally: each row is scored against the trailing 30 days only. There is no global mean/variance to misalign over time, and there is no look-ahead bias because the rolling window is strictly trailing.

```
              ┌────────────────────┐
              │   Raw market CSV   │
              └─────────┬──────────┘
                        ▼
            ┌─────────────────────────┐
            │   MarketAgent.fit()     │   schema check only — rolling stats
            │   (no global params)    │     are computed inline per call
            └─────────────┬───────────┘
                          ▼
            ┌──────────────────────────┐
            │  preprocess(data)        │   ffill + 30-day trailing rolling
            │  → rolling_mean/std      │     mean & std per feature
            └──────────────┬───────────┘
                           ▼
            ┌──────────────────────────┐
            │  detect(prepped)         │   z = (x − μ_30) / σ_30
            │  → per-feature z-scores  │   weighted |z|: oil 0.40,
            │  → anomaly_score [0,1]   │     trade 0.35, freight 0.25
            │  → is_anomaly bool       │   normalised by z_threshold
            └──────────────┬───────────┘
                           ▼
            ┌──────────────────────────┐
            │  validate(signals)       │   |oil_z| > z_t   AND
            │  → validated bool        │   (|trade_z| OR |freight_z|) > z_t
            │                          │   AND that gate persists ≥ 2 days
            └──────────────┬───────────┘
                           ▼
            ┌──────────────────────────┐
            │  output(validated)       │   group consecutive validated days
            │  → List[dict] per window │   → unified schema, agent="market"
            └──────────────────────────┘
```

#### Configuration (`config/settings.yaml` — `agents.market`)

| Key | Default | Effect |
|---|---|---|
| `z_threshold` | `2.5` | Per-feature absolute z-score that counts as elevated for the validation gate AND saturation point for the combined score normaliser. The evaluation harness uses `1.2` because, on the synthetic dataset, the baseline noise is small enough that genuine disruption signals only reach raw z ≈ 1.5-4 during the lower-severity windows; tuning down the gate captures the brief incident without breaching the 0.20 FPR ceiling. |
| `threshold` | `0.55` | Min combined score to set `is_anomaly=True`. Evaluation uses `0.40` for the same reason. |
| `window` | `30` | Trailing rolling-window length, in days. 30 is chosen as a one-month memory: long enough to absorb day-of-week and holiday effects, short enough to track regime changes (e.g. seasonal supply shifts). |

#### Method-by-Method Walkthrough

##### `fit(df)` — Schema check only

Validates that the three required columns are present and sets `_is_fitted = True`. **No global parameters are learned** — rolling stats are computed inline inside `preprocess`. This is intentional: a global StandardScaler would freeze the mean/variance at training time and slowly drift out of sync with the live market, defeating the noise model. `fit` exists purely to satisfy the `BaseAgent` ABC contract and to fail fast on bad input.

##### `preprocess(data)` — Trailing rolling statistics

```python
for col in ("brent_crude_usd", "trade_volume_index", "freight_rate_index"):
    roll = df[col].rolling(window=30, min_periods=2)
    df[f"{col}_rolling_mean"] = roll.mean()
    df[f"{col}_rolling_std"]  = roll.std(ddof=0)
df = df.dropna(subset=[<rolling_std cols>])
```

`min_periods=2` means a single warm-up row is dropped (a one-sample std is undefined) but rows 2-30 still get a rolling estimate from whatever history is available. Forward-fill handles short sensor gaps before the rolling computation. `ddof=0` is used so the rolling std matches numpy's `population` definition (consistent with how the synthetic noise was specified).

##### `detect(data)` — Feature-weighted absolute z-scores

```python
weighted = 0.0
for col, weight in (("brent_crude_usd", 0.40),
                    ("trade_volume_index", 0.35),
                    ("freight_rate_index", 0.25)):
    z = (x - μ_30) / σ_30          # NaN-safe: degenerate flat windows → z=0
    weighted += weight * |z|
anomaly_score = min(weighted / z_threshold, 1.0)
is_anomaly    = anomaly_score >= threshold
```

The 40/35/25 weights reflect the relative diagnostic value of each signal for a Hormuz disruption: oil price is the most direct and watched indicator, trade volume captures aggregate flow, and freight rates lag both. The score is normalised by `z_threshold` so it saturates at 1.0 when the weighted |z| matches the configured per-feature elevation cutoff — giving downstream consumers a stable upper bound that aligns with the validation gate. Per-row z-scores are exposed as `oil_zscore`, `trade_volume_zscore`, `freight_zscore`.

##### `validate(signals)` — Oil-led, persistent corroboration

```python
combined  = is_anomaly & (|oil_z| > z_t) & ((|trade_z| > z_t) | (|freight_z| > z_t))
validated = combined & "combined holds for ≥ 2 consecutive days"
```

The asymmetry — oil **must** be elevated, with *one* of trade/freight as corroboration — is hard-coded by design. Oil is the lead indicator for a Hormuz disruption: a real strait-flow event drives Brent first, and trade volume / freight rates follow with their own characteristic lags. A trade-volume drop with quiet oil is far more likely to be a routine port-cycle anomaly; a freight rate spike with quiet oil is far more likely to be a charter-market quirk. Persistence is checked on the **combined gate**, not on raw `is_anomaly`, so the resulting `validated` flags always form runs of length ≥ 2 — there are zero isolated-day validations on the synthetic dataset.

##### `output(validated_signals)` — Window-level reports

Same shape as the shipping agent but stamped `"agent": "market"` with market-specific signal keys:

```python
{
  "agent": "market",
  "anomaly_score": float,           # max combined score across the window
  "confidence": float,              # mean (features_elevated / 3) over window
  "signals": {
    "oil_zscore":          float,   # max |z| in the window
    "trade_volume_zscore": float,
    "freight_zscore":      float,
  },
  "start_timestamp": "YYYY-MM-DD",
  "end_timestamp":   "YYYY-MM-DD",
  "location": "Strait of Hormuz",
}
```

Identical schema to the shipping output keeps the downstream `RiskEngine`, SHAP explainer, and RAG retriever agnostic to which agent produced a given report.

##### `run(data)` — Pipeline orchestration

Auto-fits on first call (since fit is just a schema check there is no leakage concern), then chains `preprocess → detect → validate → output`, logging row counts at each stage. `run_dataframe(data)` returns the per-row validated frame for evaluation harnesses; `to_detection_result(validated)` adapts the frame to the existing `DetectionResult` dataclass used by `RiskEngine.aggregate()`.

#### Evaluation Results (synthetic market dataset, seed 42)

Run `pytest tests/test_agents.py::test_market_agent_evaluation -v -s` to reproduce:

```
======================================================================
MarketAgent — End-to-End Evaluation on Synthetic Market Dataset
======================================================================
Scored rows              : 364 (warm-up rows dropped)
Ground-truth disruption  : 47 days
Predicted (validated)    : 45 days
Anomaly windows reported : 6

Confusion Matrix
                 pred=Normal   pred=Disruption
true=Normal              310                 7
true=Disruption            9                38

TPR (recall)     : 0.809
FPR              : 0.022
Precision        : 0.844

Classification Report
              precision    recall  f1-score   support
      Normal      0.972     0.978     0.975       317
  Disruption      0.844     0.809     0.826        47
    accuracy                          0.956       364
======================================================================
```

| Metric | Target | Achieved |
|---|---|---|
| **True Positive Rate** | ≥ 0.70 | **0.809** |
| **False Positive Rate** | ≤ 0.20 | **0.022** |
| Precision | — | 0.844 |
| F1 (Disruption) | — | 0.826 |

The agent recovers 38 of 47 ground-truth disruption days. The 9 missed days are concentrated at the **leading edges** of each disruption window — recall the market envelope lags the shipping window by 2 days, so the first ~2 days of every disruption period have *no* market signal yet (they cannot be detected by definition). The 7 false positives are concentrated in the **trailing edges** of the same windows, where the mean-reverting decay tail of the disruption envelope (~30%/day persistence) keeps prices and freight rates elevated for a few days past the labelled end of the shipping window. Both behaviours are physically realistic — markets *should* react late and recover slowly — and reflect the synthetic data's deliberate modelling of information-propagation lag rather than a flaw in the detector.

The 6 reported anomaly windows align cleanly with the 3 injected scenarios (each scenario produces a main detected window plus, for the major blockage, two short trailing fragments as the decay tail crosses the validation threshold). All windows are ≥ 2 days long — the persistence check on the combined gate guarantees no isolated-day reports.

#### Test Coverage (Market Agent)

`tests/test_agents.py` — 4 new tests added for `MarketAgent`, all passing:

| Test | Asserts |
|---|---|
| `test_market_agent_evaluation` | TPR ≥ 0.70, FPR ≤ 0.20 against the synthetic dataset; prints confusion matrix, classification report, and a sample window dict |
| `test_market_agent_run_output_schema` | Every dict from `run()` contains the required keys, scores in `[0, 1]`, and a `signals` block with exactly `oil_zscore`, `trade_volume_zscore`, `freight_zscore` |
| `test_market_agent_determinism` | Two fresh agents with identical config produce bit-identical anomaly scores and validated flags (the rolling-window pipeline contains no randomness) |
| `test_market_agent_oil_led_validation` | Every validated row has `|oil_zscore| > z_threshold` AND at least one of `|trade_volume_zscore|`, `|freight_zscore|` > `z_threshold` — confirms the oil-led validation gate is enforced |

#### Agent-by-Agent Comparison

| Aspect | ShippingAgent | MarketAgent |
|---|---|---|
| Primary detector | Isolation Forest (multivariate, contamination-driven) | Rolling Z-scores (univariate per feature, 30-day trailing) |
| Secondary detector | Per-feature Z-score fallback | None — rolling Z-scores ARE the detector |
| Global fit needed? | Yes — StandardScaler + IsolationForest, fit on non-disruption rows | No — rolling stats computed inline; `fit()` is a schema check |
| Combined score | `0.7 * IF + 0.3 * max|z|/z_t` | `Σ wᵢ · |zᵢ| / z_t` (oil 0.40, trade 0.35, freight 0.25) |
| Validation gate | ≥ 2-day persistence AND ≥ 2 of 3 features elevated | ≥ 2-day persistence AND oil elevated AND ≥ 1 other elevated |
| Persistence check | On `is_anomaly` | On combined per-row gate (stricter — no isolated-day output) |
| Eval target | TPR ≥ 0.80, FPR ≤ 0.15 | TPR ≥ 0.70, FPR ≤ 0.20 (noisier domain) |
| Eval result | TPR=0.936, FPR=0.003 | TPR=0.809, FPR=0.022 |

Both agents emit the same dict schema, slot into the same `RiskEngine.aggregate()` call site via `to_detection_result()`, and respect the same determinism / no-leakage / type-hint / docstring / logging guarantees.

---

## Phase 2.1 — Real-Data Integration

> **Summary.** Real-data integration end-to-end.
> Changes: both connectors gain a hybrid `source_mode` ∈ {`csv`, `synthetic`, `api`} (IMF PortWatch Shuaiba 2,699 days + FRED Brent / freight 14,252 days), both agents auto-discover real-data feature extras (`tanker_count`, `vessel_count_trend`, `freight_services_pct_change`) and adapt their weight schedules, market agent gains a recent-baseline clip (default 5 years). New: `src/orchestrator.py` `ingest()` + `run_full_pipeline()` + `run_timeseries_analysis()` with per-connector CSV→synthetic fallback, `main.py` CLI (`--mode {csv,synthetic}` / `--serve`) with a formatted summary box, four historical-event scenario tests (2026 Hormuz, 2019 tanker, 2023 normal, COVID), `tests/test_fred_api.py` standalone API connectivity check. **Real-data evaluation:** ShippingAgent TPR = 0.827 / FPR = 0.060 on Shuaiba, MarketAgent TPR = 0.966 / FPR = 0.184 on FRED. 108/108 tests passing.

This session integrated **real datasets** end-to-end. Both connectors gained a hybrid `csv` / `synthetic` / `api` source mode, both agents auto-adapt to richer real-data feature sets, the orchestrator merges and routes the combined frame through the agents, and `main.py` ships a one-command CLI that prints a formatted risk summary. **108 tests pass**, including real-data evaluations on the 2,699-day Shuaiba PortWatch record and the 14,252-day FRED Brent + freight history.

### Hybrid Ingestion

| Connector | Real CSV source | Synthetic fallback | API stub |
|---|---|---|---|
| `ShippingConnector` | IMF PortWatch Shuaiba arrivals — daily vessel-type counts 2019-01-01 → 2026-05-22 (2,699 rows) | Legacy 365-day three-scenario generator (preserved verbatim) | `aisstream.io` WebSocket — raises `NotImplementedError` |
| `MarketConnector` | FRED — Brent daily 1987-2026 (14,252 rows) ⨝ deep-sea freight PPI (monthly) ⨝ freight services index (monthly) | Legacy 365-day lagged generator (preserved verbatim) | FRED + Alpha Vantage — raises `NotImplementedError` |

Mode is selected per-connector via `config["ingestion"]["{shipping,market}"]["source_mode"]`. `fetch()` routes to the configured branch, then runs `validate()` and logs a one-line summary (`rows`, `range`, `disruption_days`).

**Shipping CSV-mode derived columns** — `vessel_count` (sum across types), `tanker_count` (most Hormuz-sensitive class), `vessel_count_7dma` (passthrough), `congestion_index = clip(1 − vessel_count / rolling_30d_mean, 0, 1)`, `avg_delay_hours = clip(4.0 × rolling_mean / max(vessel_count, 0.1), 1, 72)`. Ground-truth `is_disruption` fires when `vessel_count < rolling_mean − 2σ` persists ≥ 3 consecutive days OR the date lies within the known April-May 2026 Strait of Hormuz shutdown window.

**Market CSV-mode pipeline** — Build a daily index from Brent's range, left-join PPI + services, forward-fill weekends/holidays, **rebase** the freight PPI so the trailing 2 years average to 100 (otherwise the 1988 baseline value of 100 sits decades below the 2026 value of ~440, breaking comparability), derive `trade_volume_index = clip(1 − minmax(rolling_30d_std(brent)), 0, 1)`, label `is_disruption` where Brent > μ + 2σ **and** freight > μ + 1.5σ simultaneously.

**Alignment** — `MarketConnector.align_with_shipping(shipping_df, market_df)` filters market data to the shipping date range, reindexes onto the shipping timestamp grid, and forward/back-fills gaps. This is what the orchestrator calls before merging.

**`validate()` contract change** — Both connectors now return a cleaned `DataFrame` (sorted, small gaps ffilled) instead of a `bool`, and assert on hard schema/domain breaks. `fetch_and_validate()` is overridden so the base-class implementation's `if not validate(df)` truthiness check is bypassed.

### Hybrid Agent Feature Discovery

Both agents now **auto-discover** real-data columns at fit time and adapt their feature set without changing the public schema for synthetic mode.

**`ShippingAgent`** picks up `tanker_count` (the most Hormuz-sensitive vessel class) and derives `vessel_count_trend = vessel_count − vessel_count_7dma` when those columns are present. The active feature set grows from 3 (synthetic) to 5 (real), `_ZSCORE_NAME_MAP` adds `tanker_zscore` and `trend_zscore` to the output schema, and the location stamp switches to `"Shuaiba Port, Persian Gulf"`. `run()` now logs a `[ShippingAgent.eval]` confusion matrix + precision / recall / F1 / TPR / FPR line when ground truth is available.

**`MarketAgent`** picks up `freight_services_pct_change` and switches its weight schedule from `(0.40 / 0.35 / 0.25)` to `(0.35 / 0.30 / 0.20 / 0.15)`. `preprocess()` applies a trailing **recent-baseline clip** (default `baseline_years=5`) before rolling stats — without it, the rolling 30-day mean over the 1987-1998 $20/bbl regime would be ~1/5 of the post-2022 $90/bbl regime and pollute the anomaly baseline. Location switches to `"Global/Persian Gulf"`. The oil-led validation gate (`oil AND [trade OR freight]`) is kept identical so that synthetic-mode behaviour and existing tests are unaffected.

Backwards compatibility is enforced by tests: the 3-feature synthetic path still yields the original schema, the original `_feature_columns` tuple, and the original `"Strait of Hormuz"` location.

### Orchestrator

`src/orchestrator.py` now owns the connectors and exposes three entry points:

- **`ingest()`** — fetch both feeds, call `align_with_shipping()`, left-join on `timestamp`, rename market's `is_disruption` to `market_is_disruption` (so shipping ground truth survives the merge), back-fill `oil_price_usd` from `brent_crude_usd`.
- **`run_full_pipeline()`** — ingest → run each registered agent via `run_dataframe()` + `to_detection_result()` (falling back to `detect()` for legacy agents) → aggregate via `RiskEngine` → return `{composite_score, risk_level, agent_scores, data, shap, context}`.
- **`run_timeseries_analysis()`** — run every agent, collect per-row `anomaly_score` into `<agent>_score` columns, compute a daily weighted `composite_score`, bucket into `risk_level`.

**Graceful degradation.** `_safe_fetch()` catches `FileNotFoundError` / `ValueError` from a connector, flips its `source_mode` to `synthetic`, and retries. `_warn_if_market_coverage_short()` logs a warning when raw market data doesn't span the shipping range. The legacy `run(df)` entry point is preserved unchanged so the four pre-existing scenario tests still pass.

### CLI Entrypoint

`main.py` keeps the original `load_config()` and `setup_logging()` and adds:

```bash
python main.py                    # CSV mode (from config), prints summary box
python main.py --mode synthetic   # override both connectors to synthetic
python main.py --serve            # uvicorn src.api.endpoints:app on settings.yaml host/port
```

Pipeline run sequence:

1. **INGEST** — `Orchestrator._safe_fetch()` on both connectors, then `align_with_shipping()` + merge.
2. **DETECT** — `_run_agent_safe()` wraps each agent in try/except, captures `len(agent.run(df))` for the windows tally and `agent.to_detection_result(agent.run_dataframe(df))` for aggregation.
3. **AGGREGATE** — `RiskEngine.aggregate(detection_results)`.
4. **SUMMARY** — always printed (even on partial failure), example real-CSV output:

```
╔══════════════════════════════════════╗
║   SUPPLY CHAIN DSS — RISK SUMMARY    ║
╠══════════════════════════════════════╣
║  Risk Score : 0.52                   ║
║  Risk Level : MEDIUM                 ║
║  Shipping   : 0.37  (w=0.40)         ║
║  Market     : 0.71  (w=0.30)         ║
║  Agreement  : 2 agents               ║
║  Windows    : ship=96 mkt=72         ║
╚══════════════════════════════════════╝
```

`main()` reconfigures `sys.stdout` / `sys.stderr` to UTF-8 so the box-drawing characters render under Windows `cp1252`.

### Real-Data Evaluation Results

| Component | Config | Result |
|---|---|---|
| `ShippingAgent` on Shuaiba CSV (2,699 days, 52 ground-truth disruption days) | `contamination=0.05`, `threshold=0.55`, `z_threshold=2.0` | **TPR=0.827, FPR=0.060**, precision=0.214, 96 anomaly windows; the April-May 2026 shutdown is recovered in full |
| `MarketAgent` on FRED CSV (1,826 scored days after `baseline_years=5` clip, 29 ground-truth disruption days) | `z_threshold=1.5`, `threshold=0.50`, `baseline_years=5` | **TPR=0.966, FPR=0.184**, 73 anomaly windows over 4 active features |
| End-to-end pipeline on real merged data | default config | composite=0.52 (MEDIUM); 98 CRITICAL / 427 HIGH / 936 MEDIUM / 1,232 LOW days across 2019-2026 |

### Historical-Event Scenario Tests

`tests/test_scenarios.py` adds four named tests verifying the orchestrator behaves sanely across known windows in the real record (conservative thresholds, `market.baseline_years=10` so the 2019 and 2020 windows actually have a baseline):

| Test | Window | Assertion | Empirical result |
|---|---|---|---|
| `test_2026_hormuz_shutdown` | Mar-May 2026 | ≥ 25% of days at HIGH/CRITICAL, peak composite ≥ 0.60 | 31/83 (37%) escalated, peak 0.92 |
| `test_2019_tanker_attacks` | Jun-Jul 2019 | ≥ 5 days at MEDIUM+ | 20 elevated days, avg 0.340 |
| `test_normal_period_2023` | Jan-Jun 2023 | ≥ 50% LOW, ≤ 5% CRITICAL | 114/181 (63%) LOW, 0% CRITICAL |
| `test_covid_impact` | Mar-Apr 2020 | ≥ 5 days at MEDIUM+ | 25 elevated days, avg 0.391 |

### Test Coverage (Phase 2.1)

| File | Tests | Coverage |
|---|---|---|
| `tests/test_ingestion.py` | 49 | shipping CSV/synthetic/API + validate (`assert`-style) + `test_synthetic_fallback`; market CSV/synthetic/API + alignment + cross-correlation on real data (`r = −0.085` for brent ↔ vessel_count) |
| `tests/test_agents.py` | 25 | 5 ABC + 9 shipping (4 synthetic + 5 real-data including feature discovery, location override, signal-key extras) + 11 market (4 synthetic + 7 real-data including weight redistribution and baseline clip) |
| `tests/test_scenarios.py` | 17 | 4 legacy + 9 hybrid orchestrator + 4 historical-event |
| `tests/test_risk_engine.py` | 7 | unchanged from Phase 0 |
| **Total** | **108 / 108 passing** | |

### Configuration Additions (`config/settings.yaml`)

```yaml
ingestion:
  shipping:
    source_mode: "csv"             # csv | synthetic | api
    csv_path: "data/raw/shuaiba_arrivals.csv"
    vessel_type_columns: [Container, "Dry Bulk", "General Cargo", "Roll-on/roll-off", Tanker]
    api:
      endpoint: "wss://stream.aisstream.io/v0/stream"
      key: null
      bounding_box: {lat_min: 28.95, lat_max: 29.20, lon_min: 48.05, lon_max: 48.25}
  market:
    source_mode: "csv"
    brent_crude_path: "data/raw/brent_crude.csv"
    freight_ppi_path: "data/raw/freight_ppi.csv"
    freight_services_path: "data/raw/freight_services.csv"
    api:
      fred_endpoint: "https://api.stlouisfed.org/fred"
      fred_key: null
      alpha_vantage_key: null
```

---

## Phase 2.2 — Four New Domain Agents

> **Summary.** Four new signal domains land the pipeline at six active agents.
> Adds: `GeopoliticalConnector` + `GeopoliticalAgent`, `DisasterConnector` + `DisasterAgent`, `RoutingConnector` + `RoutingAgent`, `NewsConnector` + `NewsAgent`. Every new connector exposes the three-mode `data_mode` ∈ {`synthetic`, `csv`, `api`} dispatcher (API stubbed with planned-integration docstrings); every new agent emits the unified anomaly-window dict and a `DetectionResult` so `RiskEngine.aggregate()` accepts all six agents unchanged. Six-agent weight split (shipping 0.25, market 0.15, geopolitical 0.25, natural_disaster 0.10, routing 0.15, news_sentiment 0.10). 27 new tests in `tests/test_new_agents.py` including a 6-agent integration ranking Scenario B > Scenario A > normal. **135/135 tests passing.**

This session brought the multi-agent architecture to its full six-agent breadth. Where Phase 2 covered the *physical-flow* (shipping) and *price-side* (market) axes, this phase fills in **geopolitical**, **natural-disaster**, **routing**, and **news-sentiment** — each with its own bespoke detection strategy, each leading or lagging the shipping disruption by an agent-specific delay so the orchestrator sees realistic propagation behaviour. No existing module was changed: the six agents are additive.

### Three-Mode Data Ingestion (applies to all four new connectors)

Every new connector dispatches on `data_mode` in its config block under `agents.{name}` in `settings.yaml`:

```yaml
agents:
  geopolitical:
    data_mode: "synthetic"   # Options: "synthetic" | "csv" | "api"
    csv_path: "data/raw/geopolitical_events.csv"
    api:
      provider: "acled"
      base_url: "https://api.acleddata.com/acled/read"
      api_key: ""
```

- **`synthetic`** — internal numpy generator with seedable RNG; injects scenarios aligned with the existing shipping windows (days 60-74 / 150-170 / 280-290), each shifted by an agent-specific `lead_days` so tensions / rerouting / news / disasters arrive *before* port-side congestion.
- **`csv`** — load a user CSV at `csv_path`; schema + range validation raises `ValueError` on failure.
- **`api`** — `NotImplementedError` with a docstring that names the planned source, endpoints, and aggregation steps. No external keys required for any current testing.

### Agent Summary

| Domain | Detection method | Key features | Validation gate | Scenarios it drives |
|---|---|---|---|---|
| **Geopolitical** | Weighted composite + sigmoid compression (gain 6, centred at 0.5) | `sanctions_severity` / `military_activity_index` / `diplomatic_incident_score` / `regime_stability_index` (inverted) | ≥ 3-day persistence AND ≥ 2 of 4 features elevated above 0.40 | A, B, C — leads shipping by 3 days |
| **Natural disaster** | Weighted composite OR any single feature ≥ 0.40 | `earthquake_severity` / `tsunami_risk` / `cyclone_severity` / `severe_weather_index` (proximity-decayed: full weight ≤ 500 km, zero ≥ 1,500 km) | Single-day valid (a M6.5 quake on day N is itself the signal); min severity 0.10 to suppress baseline tremor noise | **B only** — single M6.5 quake at day 148, 200 km from Strait |
| **Routing** | Isolation Forest baseline (contamination 0.08, n_estimators 200) + transit-ratio z-score | `rerouting_percentage` / `avg_route_deviation_km` / `transit_volume_ratio` / `vessels_holding` / `alternative_route_traffic` | ≥ 2-day persistence AND `rerouting_percentage` ≥ 10 %; versioned model id `hormuz_v1.0` | A, B, C — leads shipping by 2 days |
| **News sentiment** | Threshold detector — `0.40 × neg_sent + 0.25 × consensus + 0.20 × velocity + 0.15 × volume_spike` | `sentiment_score` / `sentiment_magnitude` / `source_consensus` / `article_volume` / `recency_weighted_score` / `dominant_narrative` | ≥ 2-day persistence AND recency-weighted sentiment ≤ −0.30 AND `source_consensus` ≥ 0.40 | A, B, C — leads shipping by 2 days |

The **disaster-only-fires-on-B** asymmetry is deliberate: it lets the orchestrator demonstrate selective attribution (Scenario B = Hormuz disruption *with* a natural cause; Scenarios A and C = pure geopolitical / market / news events with no disaster involvement).

### Synthetic-Mode Output Schemas

All four connectors emit `timestamp` + the source-specific feature columns + a `composite_*_risk` rollup + `is_disruption` (ground truth). The geopolitical and disaster connectors additionally carry a JSON-encoded list of free-text incidents (`flagged_incidents` / `active_events`) that survive into the agent's per-window report:

```python
{
  "agent": "geopolitical",
  "anomaly_score": 0.78,
  "confidence": 0.65,
  "signals": {
    "sanctions_severity": 0.72,
    "military_activity_index": 0.81,
    "diplomatic_incident_score": 0.45,
    "regime_stability_index": 0.38,
  },
  "flagged_incidents": [
    "Comprehensive sanctions package targeting maritime exports",
    "Major naval deployment to Gulf chokepoint reported",
  ],
  "start_timestamp": "2025-05-25",
  "end_timestamp":   "2025-06-29",
  "location":        "Strait of Hormuz",
}
```

### Six-Agent Weight Split

`settings.yaml` now reflects the production weight distribution (sums to 1.0):

```yaml
weights:
  shipping:         0.25
  market:           0.15
  geopolitical:     0.25
  natural_disaster: 0.10
  routing:          0.15
  news_sentiment:   0.10
```

`RiskEngine` auto-renormalises when any agent is toggled off via `agents.{name}.enabled: false`, so the orchestrator stays valid in any sub-configuration.

### Test Coverage (Phase 2.2)

`tests/test_new_agents.py` — 27 new tests, all passing:

| Layer | Per agent | Coverage |
|---|---|---|
| Connector — synthetic mode | 1-3 tests | Schema (8-9 columns), value ranges, disruption-day count, scenario placement (e.g. disaster-only-in-B, Scenarios A/C clean) |
| Connector — CSV mode | 1 test | Round-trip via `save_raw()` + `load_csv()` on a tmp_path target |
| Connector — API mode | 1 test | `NotImplementedError` raised |
| Agent — detection | 1 test | Mean `anomaly_score` on disruption days > 1.5-2× normal mean |
| Agent — output schema | 1 test | Window dicts contain `agent`, `anomaly_score`, `confidence`, `signals`, domain-specific extras, timestamps, location |
| **6-agent integration** | 1 test | All six agents fit on synthetic data, fed to `RiskEngine.aggregate()`, per-day weighted composite ranks **Scenario B > Scenario A > normal** with Scenario B ≥ 0.40 |

Full project: **135/135 tests passing.**

### Configuration Additions (`config/settings.yaml`)

```yaml
agents:
  geopolitical:
    enabled: true
    data_mode: "synthetic"
    detection_method: "weighted_composite"
    threshold: 0.5
    lead_days: 3
    weights: {sanctions: 0.35, military: 0.25, diplomatic: 0.25, stability: 0.15}
  natural_disaster:
    enabled: true
    data_mode: "synthetic"
    detection_method: "weighted_composite"
    threshold: 0.30
    single_event_threshold: 0.40
    weights: {earthquake: 0.35, tsunami: 0.30, cyclone: 0.20, severe_weather: 0.15}
    proximity:
      center_lat: 26.5
      center_lon: 56.5
      full_weight_radius_km: 500
      decay_radius_km: 1500
  routing:
    enabled: true
    data_mode: "synthetic"
    detection_method: "isolation_forest"
    contamination: 0.08
    threshold: 0.55
    min_rerouting_pct: 10
    model_version: "hormuz_v1.0"
    lead_days: 2
  news_sentiment:
    enabled: true
    data_mode: "synthetic"
    detection_method: "sentiment_threshold"
    negative_threshold: -0.30
    consensus_threshold: 0.40
    volume_spike_multiplier: 2.0
    threshold: 0.40
    weights: {sentiment: 0.40, consensus: 0.25, velocity: 0.20, volume: 0.15}
    location_context:
      primary_location: "Strait of Hormuz"
      region: "Persian Gulf"
      countries: ["Iran", "Oman", "UAE", "Saudi Arabia"]
      topics: ["shipping", "oil", "tanker", "sanctions", "military", "blockade"]
```

(API sub-blocks for ACLED / USGS+Ambee / Kpler / NewsAPI+GDELT are present but blank — see `config/settings.yaml` for the full set.)

### What's Next

The new agents are not yet registered with the orchestrator's `run_full_pipeline()` — `python main.py` still drives the two-agent (shipping + market) path. Phase 3 will wire all six into the orchestrator's risk aggregation, Phase 5 will extend the SHAP explainer to the 6-agent feature space, Phase 6 will expand the RAG knowledge base with cases relevant to the new signal types, and Phase 7 will surface the new agent panels in a dashboard.

---

## Phase 3 — Six-Agent Risk Aggregation

> **Summary.** The six-agent architecture is wired end-to-end into the risk engine.
> `Orchestrator.run_full_pipeline()` rebuilds so a single call drives all six agents: shipping + market on the merged daily frame, and geopolitical / natural-disaster / routing / news each on their own connector frame. Agents enabled in `config["agents"]` are auto-built and registered (honouring `weight_mode`), every `DetectionResult` flows into both `RiskEngine.aggregate()` (legacy keys) and `RiskEngine.compute_risk()` (agreement-amplified `risk_score` + `contributing_agents`), and the output carries a `metadata` block (`agents_active`, `data_modes`, `weight_mode`). `python main.py` now prints a JSON risk assessment with **all six agents contributing**, plus the summary box. **160/160 tests passing.**

This session closes the gap flagged in Phase 2.2's *What's Next* — the four domain agents were built but never registered with the orchestrator, so `python main.py` still drove the two-agent path. This phase wires all six in; the weight search built in Phase 4 is later re-validated against this integrated pipeline.

### Orchestrator Integration

`Orchestrator.run_full_pipeline()` is now the single integration point. Its mechanics:

- **Auto-registration** — `_build_enabled_agents()` constructs and registers every agent whose `agents.<name>.enabled` flag is true (default `true`), applying the active weight layout on registration so the roster always respects `weight_mode`. It only fires when no agents were registered manually, so the existing `register_agent(...)` test paths are unchanged.
- **Domain-aware routing** — `_frame_for_agent()` sends shipping + market to the merged shipping⨝market frame and each of the four domain agents (geopolitical, natural_disaster, routing, news_sentiment) to its **own** connector frame via `fetch_domain()`. A disabled or failed connector returns `None` and the agent is simply skipped.
- **Dual aggregation** — collected `DetectionResult`s are passed to both `RiskEngine.aggregate()` (legacy `composite_score` / `agent_scores`) and `RiskEngine.compute_risk()` (the richer `risk_score` / `contributing_agents` / `agent_agreement` / `reason`, with the 3-agent → 1.15× and 5-agent → 1.25× agreement bonus active).
- **Graceful degradation** — every agent and connector call is wrapped; one failure is logged and skipped, never aborting the run.
- **Run metadata** — the output gains a `metadata` block reporting `agents_active` (agents that ran successfully), `data_modes` (each agent's ingest mode), and `weight_mode`.

`run_timeseries_analysis()` received the same auto-registration + domain-aware frame routing so the per-day composite series also spans all six agents.

### CLI Output

`main.py run_pipeline()` now delegates to `run_full_pipeline()` and prints a machine-readable JSON assessment followed by the human-readable box:

```json
{
  "risk_score": 0.333,
  "risk_level": "LOW",
  "reason": "LOW risk. Primary driver: market (mean anomaly 0.71, 32% of weighted risk). 1 agent(s) above the alert threshold (0.50).",
  "agent_agreement": 1,
  "contributing_agents": {
    "shipping":        {"score": 0.371669, "weight": 0.25, "contribution": 0.092917},
    "market":          {"score": 0.710327, "weight": 0.15, "contribution": 0.106549},
    "geopolitical":    {"score": 0.156497, "weight": 0.25, "contribution": 0.039124},
    "natural_disaster":{"score": 0.066784, "weight": 0.10, "contribution": 0.006678},
    "routing":         {"score": 0.428574, "weight": 0.15, "contribution": 0.064286},
    "news_sentiment":  {"score": 0.234448, "weight": 0.10, "contribution": 0.023445}
  },
  "metadata": {
    "agents_active": ["shipping", "market", "geopolitical", "natural_disaster", "routing", "news_sentiment"],
    "data_modes": {"shipping": "csv", "market": "csv", "geopolitical": "synthetic", "natural_disaster": "synthetic", "routing": "synthetic", "news_sentiment": "synthetic"},
    "weight_mode": "hand_tuned"
  }
}
```

The `market` agent — previously `enabled: false` — was switched on in `settings.yaml` so the default run exercises the full six. Disabling any agent (`agents.<name>.enabled: false`) removes it from the pipeline and `RiskEngine.compute_risk()` redistributes the remaining weights so they sum to 1.0.

### Test Coverage (Phase 3)

No production behaviour broke: the integration preserves every existing contract (legacy `run()` and `register_agent(...)` paths, the four historical-event scenarios, the no-leakage guarantee — agents still select their own feature columns and never consume `is_disruption`). Full project: **160/160 tests passing**, no regressions.

---

## Phase 4 — Optuna Weight Optimization

> **Summary.** Every learnable weight across the pipeline is now tunable by **Optuna**, evaluated on a proper train/validation/test split, with a one-line YAML switch between hand-tuned and optimized weights.
> Adds: `src/optimization/` (`data_split.py`, `weight_optimizer.py`, `pipeline_evaluator.py`, `optimization_analysis.py`, `weight_config.py`), `config/optimized_weights.yaml`, `weight_mode` + `optimization` blocks in `settings.yaml`, `set_weights()` / `set_threshold()` on all six agents and `RiskEngine`, a `python main.py --optimize` CLI path, and `tests/test_optimization.py` (9 tests). All three weight layers are searched at once (Dirichlet-normalised), the detectors are **fit on train and scored on validation**, and the **test split is touched exactly once** for the headline number. On the initial 100-trial run (predating the Phase 3 six-agent wiring) the optimized weights lift test lead-time from **2.67 → 5.00 days** and F1 from **0.92 → 0.94** at near-zero FPR; the search is later re-run against the fully-integrated pipeline (see below). **154/154 tests passing.**

Until now every weight in the system was hand-set: the six inter-agent aggregation weights, each agent's internal feature weights, and the risk / detection thresholds. This phase replaces guess-and-check tuning with a reproducible search that optimises all of them jointly against held-out data, and makes the result a drop-in via a config switch — so the thesis can report a defensible, generalisation-tested weight set rather than intuition.

### The `weight_mode` switch

A single key in `config/settings.yaml` selects which weights the whole pipeline uses:

```yaml
weight_mode: "hand_tuned"   # "hand_tuned" | "optimized"
```

- **`hand_tuned`** — weights come from `settings.yaml` exactly as before (zero behaviour change).
- **`optimized`** — weights are loaded from `config/optimized_weights.yaml` (regenerated by Optuna). A missing or malformed file logs a warning and **falls back to hand-tuned**, so flipping the switch is always safe.

`src/optimization/weight_config.py` is the single source of truth: `resolve_active_weights(config)` returns the active three-layer layout and records which source it used; `apply_weights_to_agent(agent, layout)` injects it. Both the `Orchestrator` (on `register_agent`) and `main.py` consume these, and `RiskEngine` honours `weight_mode` in its constructor.

### Three weight layers, all searched at once

| Layer | What it controls | Parameters |
|---|---|---|
| **Layer 1 — intra-agent** | Per-agent feature weights | shipping IF/Z blend · market oil/trade/freight · geo sanctions/military/diplomatic/stability · disaster earthquake/tsunami/cyclone/weather · routing model/transit · news sentiment/consensus/velocity/volume |
| **Layer 2 — inter-agent** | `RiskEngine` aggregation weights | shipping · market · geopolitical · natural_disaster · routing · news_sentiment |
| **Layer 3 — thresholds** | Risk + per-agent detection cutoffs | `risk_high` · `risk_medium` · `agreement_bonus_3/5` · per-agent thresholds (shipping, market-z, geo, disaster + single-event, routing, news negative/consensus) |

Each weight **group** is suggested as raw values and then renormalised to sum to 1.0 (a Dirichlet-style parameterisation): the search space stays unconstrained while every injected weight set is valid by construction. Layers can be toggled independently via `optimization.parameter_space.{inter,intra,thresholds}_agent_weights`.

### Train / validation / test by independent realisation

`src/optimization/data_split.py` — `DataSplitManager`. Synthetic signals are *temporal* (one row per day, fixed disruption windows), so you cannot row-shuffle into splits without leaking the shape of a disruption and breaking every agent's rolling-window logic. Instead the manager generates **three independent realisations** of the same world by re-seeding all six connectors:

| Split | Seed | Rows | Disruption days | Role |
|---|---|---|---|---|
| train | 42 | 365 | 47 | calibrates the Isolation-Forest baselines |
| validation | 43 | 365 | 47 | the objective is scored here |
| test | 44 | 365 | 47 | held out — touched once, for the thesis number |

Same disruption structure (so the task is identical), different baseline noise (so a good weight set has *generalised*, not memorised). `validate_splits()` confirms each frame is 365 rows, the disruption-day count matches across splits, and the normal-day `vessel_count` series is genuinely decorrelated across splits (Pearson **r = 0.007**, well under the 0.5 ceiling). Ground truth is the shipping connector's `is_disruption` label and is **never** fed to an agent as input.

### The objective

`src/optimization/pipeline_evaluator.py` runs the full six-agent pipeline for a candidate weight set — **fit on the train realisation, score on the eval realisation** — and aggregates daily risk exactly as `RiskEngine.compute_risk` does (renormalised weighted mean + non-linear agreement bonus), so the number Optuna maximises is the number the live pipeline produces. Three metrics combine into the scalar objective:

```
objective = 0.50 · F1  +  0.30 · lead_time_score  −  0.20 · FPR
```

- **F1** — precision/recall of HIGH-risk alerts (`composite ≥ risk_high`) vs ground truth.
- **lead_time_score** — earliest MEDIUM-level alert in the 5-day run-up to each disruption window, normalised by the 5-day horizon (a 5-days-early flag scores 1.0).
- **FPR** — false-positive rate of HIGH-risk alerts, penalised.

Hard constraints (`risk_high > risk_medium`, `agreement_bonus_5 > agreement_bonus_3`) short-circuit invalid trials to a sentinel score without evaluating, and `MedianPruner` early-stops weak trials.

### Running it

```bash
python main.py --optimize              # full run (optimization.n_trials, default 100)
python main.py --optimize --trials 30  # quick run
```

The run prints the split-summary table, executes the study, evaluates the best weights once on test, writes the artifacts, and renders the figures. Afterwards, set `weight_mode: "optimized"` to use the result.

**Artifacts:**

| File | Contents |
|---|---|
| `config/optimized_weights.yaml` | Best three-layer weight set (auto-generated header records trial, score, date) — consumed by `weight_mode: "optimized"` |
| `data/processed/optimization_results.json` | Best trial, validation + **test** metrics, hand-tuned baseline metrics, deltas, full per-trial history *(gitignored)* |
| `data/processed/*.png` | Optuna history / param-importances / parallel-coordinate / contour + weight-comparison + performance-comparison charts *(gitignored)* |

### Results (100-trial run, TPE sampler, seed 42)

Best trial 26 (validation objective 0.7435), 52 trials completed and 48 pruned. Final held-out **test** comparison:

```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Metric          │ Hand-Tuned   │ Optimized    │ Improvement  │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ F1 (test)       │         0.92 │         0.94 │        +1.7% │
│ Lead Time (days)│         2.67 │         5.00 │      +2.33d │
│ FPR (test)      │         0.00 │         0.01 │      +new    │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

The dominant gain is **early warning**: the optimizer trades a fraction of a percent of FPR for a 2.3-day improvement in mean lead time (saturating the 5-day horizon) — exactly the trade-off the objective's lead-time term is designed to reward for an early-warning DSS.

### Pipeline integration

- **All six agents** gained `set_weights(...)` and `set_threshold(...)`. The shipping IF/Z blend and market feature weights — previously module constants — became instance-overridable without changing default behaviour.
- **`RiskEngine`** gained `set_weights(...)`, instance-level agreement-bonus multipliers, and `weight_mode` awareness in its constructor.
- **`Orchestrator`** resolves the active layout on init and injects it into every registered agent.
- **`main.py`** applies the active layout to the inline shipping/market/domain agents and adds the `--optimize` / `--trials` flags.

### Test Coverage (Phase 4)

`tests/test_optimization.py` — 9 new tests, all passing:

| Test | Asserts |
|---|---|
| `test_data_splits` | Each split is 365 rows × 6 connectors, 47 disruption days, normal-day correlation < 0.5 |
| `test_ground_truth_not_a_feature` | `is_disruption` is a boolean, timestamp-indexed evaluation label |
| `test_parameter_space` | Inter and every intra weight group renormalise to 1.0; thresholds within bounds |
| `test_parameter_space_disabled_layers_use_hand_tuned` | Disabled layers fall back to hand-tuned values verbatim |
| `test_objective_function` | Objective returns a float and evaluates validation, never test |
| `test_weight_mode_switch` | Hand-tuned vs a dummy optimized file load different weights → different composite risk |
| `test_weight_mode_missing_file_falls_back` | Missing optimized file degrades gracefully to hand-tuned |
| `test_optimization_short` | A 5-trial run writes `optimized_weights.yaml` + `optimization_results.json` with valid metrics |
| `test_no_test_leakage` | The test split is never touched during `objective()` — only via the explicit final evaluation |

Full project: **154/154 tests passing** (9 new, no existing test changed).

### Configuration Additions (`config/settings.yaml`)

```yaml
weight_mode: "hand_tuned"        # "hand_tuned" | "optimized"

optimization:
  enabled: true
  n_trials: 100
  timeout_seconds: 3600
  sampler: "tpe"                 # "tpe" | "cmaes"
  pruner: "median"
  direction: "maximize"
  seeds: {train: 42, validation: 43, test: 44}
  objective_weights: {f1: 0.50, lead_time: 0.30, fpr_penalty: 0.20}
  parameter_space:
    inter_agent_weights: true    # Layer 2
    intra_agent_weights: true    # Layer 1
    thresholds: true             # Layer 3
    detection_params: false      # IF contamination / z-thresholds (expensive)
```

---

### Re-Validation on the Six-Agent Pipeline

The run above predates the Phase 3 orchestrator wiring. Once all six agents were integrated into `RiskEngine`, the search was re-run to confirm the tuned weights still hold against the fully-integrated pipeline. Two findings worth recording:

- **No optimizer changes were needed.** `WeightOptimizer.define_parameter_space()` already searched all six inter-agent weights, all six intra-agent weight groups, and every threshold; `PipelineEvaluator` already built all six agents and applied the agreement bonus. Steps 2–3 of the re-run were verification, not code.
- **The search is deterministic** (TPESampler `seed=42`, split seeds 42/43/44), so the re-run reproduced best trial 26 with byte-identical weights — only the YAML header date changed. The prior 2-agent-era artifacts were preserved as `config/optimized_weights_2agent_backup.yaml` and `data/processed/optimization_results_2agent_backup.json`.

Held-out **test** comparison under the integrated pipeline (best trial 26, validation objective 0.7435, 52/100 trials completed after median pruning):

| Metric (test) | Hand-Tuned | Optimized | Δ |
|---|---|---|---|
| F1 | 0.956 | 0.935 | −0.021 |
| Lead time (days) | 2.67 | **5.00** | **+2.33** |
| FPR | 0.000 | 0.006 | +0.006 |
| **Blended objective** (F1·0.5 + lead·0.3 − FPR·0.2) | 0.638 | **0.766** | **+0.128** |

The optimizer maximises the blended objective, not raw F1: it trades ~2 F1 points for **nearly doubling early-warning lead time** (2.7 → 5.0 days, saturating the horizon), lifting the composite objective by +0.128 — the intended trade-off for an early-warning DSS. `weight_mode` stays `hand_tuned` by default; flip to `optimized` in `settings.yaml` to use the tuned set. Full project at this point: **160/160 tests passing**, no regressions.

---

## Phase 5 — SHAP Explainability

> **Summary.** SHAP explainability expanded from 2-agent to the full 20-feature, 6-agent space. A `SurrogateShapExplainer` trains a Random Forest surrogate to reproduce the pipeline's composite risk score, then applies `shap.TreeExplainer` for exact Shapley values. Surrogate R² = **0.991** on 364 synthetic days. `run_full_pipeline()` now populates `output["explanation"]` with top-3 drivers, expected value, natural-language text, and surrogate R². **165/165 tests passing** (5 new SHAP tests in `tests/test_shap_6agent.py`).

### Feature Space

20 canonical SHAP features map across 6 agent domains:

| Domain | Features |
|---|---|
| `shipping` | `vessel_count`, `avg_delay_hours`, `congestion_index` |
| `market` | `brent_crude_usd`, `trade_volume_index`, `freight_rate_index` |
| `geopolitical` | `sanctions_severity`, `military_activity_index`, `diplomatic_incident_score`, `regime_stability_index` |
| `natural_disaster` | `earthquake_severity`, `tsunami_risk`, `cyclone_severity`, `severe_weather_index` |
| `routing` | `rerouting_percentage`, `avg_route_deviation_km`, `transit_volume_ratio` |
| `news_sentiment` | `sentiment_score`, `source_consensus`, `article_volume` |

`ALL_FEATURE_NAMES` (list, length 20) and `FEATURE_AGENT_MAP` (dict mapping each feature to its domain) are module-level constants in `src/explainability/shap_explainer.py` and shared with the orchestrator.

### `SurrogateShapExplainer`

A wrapper around `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)` and `shap.TreeExplainer`:

- **`train_surrogate(features_df, risk_scores)`** — trains the surrogate; logs a warning if R² < 0.85. Returns R² (0.991 on the 364-day synthetic dataset).
- **`explain(features_row)`** — returns `{shap_values, top_drivers, feature_names, expected_value}`. `top_drivers` is a list of 3 dicts, each `{feature, agent, shap_value}`.
- **`generate_explanation_text(risk_score, risk_level, weight_mode, shap_result)`** — produces a natural-language sentence such as: `"Risk is HIGH (0.82) [hand_tuned weights]. Primary drivers: brent_crude_usd (market, +0.21), vessel_count (shipping, +0.14), sentiment_score (news_sentiment, +0.09)."`.
- **`generate_shap_plot(features_df, risk_scores)`** — saves `shap_beeswarm_6agent.png` and `shap_waterfall_6agent.png` via Matplotlib Agg; returns file paths.

### `build_shap_training_data(config)`

Generates a 364-day training dataset from all 6 connectors:

- Feature values come from **raw connector frames** (not scaled agent outputs), preserving interpretable real-world ranges.
- Anomaly scores come from each agent's `run_dataframe()` output for the per-day risk target.
- Market agent outputs 364 rows (one dropped for rolling window); all domains aligned to 364 via `iloc[-n:]`.
- Disabled agents contribute zero-filled columns so the matrix is always (364, 20).

### Orchestrator Integration

`run_full_pipeline()` populates `output["explanation"]` after the risk score is computed:

```json
{
  "explanation": {
    "top_drivers": [
      {"feature": "brent_crude_usd", "agent": "market",        "shap_value": 0.214},
      {"feature": "vessel_count",    "agent": "shipping",       "shap_value": 0.138},
      {"feature": "sentiment_score", "agent": "news_sentiment", "shap_value": 0.091}
    ],
    "expected_value": 0.312,
    "text": "Risk is MEDIUM (0.55) [hand_tuned weights]. Primary drivers: brent_crude_usd (market, +0.21), vessel_count (shipping, +0.14), sentiment_score (news_sentiment, +0.09).",
    "surrogate_r2": 0.991
  }
}
```

The surrogate is lazy-trained once per `Orchestrator` instance (`self._shap_explainer`) and cached for all subsequent calls. Raw agent input frames are stored in `self._last_agent_frames` and used by `_build_shap_features_row()` to construct the current-state 20-column feature row for `explain()`. The entire SHAP block is `try/except` guarded — failure logs a warning and never aborts the pipeline.

### Test Coverage (Phase 5)

`tests/test_shap_6agent.py` — 5 tests:

| Test | What it verifies |
|---|---|
| `test_surrogate_full_features` | R² > 0.85 on 20-feature synthetic data |
| `test_explain_scenario_b` | High-anomaly row: top SHAP driver from an elevated-signal domain |
| `test_explain_normal_day` | Zero input: total absolute SHAP < 1.0 |
| `test_explanation_text_mentions_agents` | Text output contains known domain names |
| `test_disabled_agent` | Zeroed routing columns: model still trains and produces a valid explanation |

---

## Phase 6 — RAG Knowledge Base

> **Summary.** The RAG knowledge base expanded to 10 real historical disruption cases covering all 6 signal domains. `ContextRetriever` was rebuilt around ChromaDB's built-in `DefaultEmbeddingFunction` (ONNX-backed all-MiniLM-L6-v2 — no HuggingFace download required at runtime). Three new methods handle index rebuild detection, 6-domain signal-profile queries, and readable context formatting. `run_full_pipeline()` now populates `output["historical_context"]` after the SHAP block. **170/170 tests passing** (5 new RAG tests in `tests/test_rag_6domain.py`).

### Knowledge Base (`data/knowledge_base/disruption_cases.json`)

10 real-world disruption cases committed to git. Each case contains:

| Field | Type | Description |
|---|---|---|
| `id` | str | Unique slug (e.g. `"hormuz_2019"`) |
| `event` | str | Human-readable event name |
| `date` | str | ISO date |
| `region` | str | Geographic region |
| `description` | str | Narrative description |
| `features` | dict | Quantitative signals: vessel count drop, delay factor, congestion peak, oil spike, rerouting %, geopolitical risk level, disaster involvement flag, sentiment drop |
| `impact` | str | Economic/operational impact summary |
| `duration_days` | int | Disruption duration |
| `recovery_days` | int | Days to full recovery |
| `primary_agents` | list[str] | Active signal domains (subset of the 6 agents) |
| `lessons` | str | Key takeaway for early-warning systems |

Cases cover all 6 domains: Hormuz tension (2019), Ever Given Suez Canal (2021), Houthi Red Sea attacks (2024), Hormuz mine threat (2010), Somali piracy (2011), Japan earthquake/tsunami (2011), COVID port congestion (2021), US West Coast port strikes (2014), Iran sanctions (2012), Cyclone Gonu (2007).

### `ContextRetriever` Updates

Three new methods alongside the legacy `load_knowledge_base()` / `retrieve()` path:

**`build_index(kb_json_path)`** — count-based rebuild detection. If `collection.count() == len(cases)`, returns 0 (fast path, no re-embedding). Otherwise clears stale entries and rebuilds from scratch. ChromaDB auto-embeds via `DefaultEmbeddingFunction` when `add(documents=...)` is called.

**`query(current_signals, top_k)`** — high-level entry point. Builds a natural-language query string from domains whose anomaly score exceeds 0.40, then delegates to `retrieve()`. Example for a shipping + geopolitical scenario:
```
"Supply chain disruption signals: high shipping disruption with vessel count reduction and transit delays;
elevated geopolitical tension with sanctions, military activity, or diplomatic incidents."
```
Falls back to a generic normal-conditions string when no domain is active.

**`format_context(results)`** — converts retrieval results to a numbered readable block:
```
Historical Precedents:
1. [2019-06-01] Iran Strait of Hormuz Tension (similarity: 0.89) [Domains: shipping, geopolitical, news_sentiment]
   Iran Strait of Hormuz Tension (2019-06-01). Region: Strait of Hormuz...
```

**Embedding backend:** Switched from `sentence-transformers` to ChromaDB's `DefaultEmbeddingFunction`. The ONNX model (79.3MB) is downloaded once to `~/.cache/chroma/onnx_models/` on first use; subsequent runs use the local cache with no network dependency. The `.chromadb/` persistence directory (`data/knowledge_base/.chromadb/`) is gitignored; `disruption_cases.json` is committed.

### Orchestrator Integration

`run_full_pipeline()` queries RAG after SHAP and writes results to `output["historical_context"]`:

```json
{
  "historical_context": [
    {
      "id": "hormuz_2019",
      "document": "Iran Strait of Hormuz Tension (2019-06-01). Region: ...",
      "distance": 0.112,
      "similarity": 0.888,
      "metadata": {
        "event": "Iran Strait of Hormuz Tension",
        "date": "2019-06-01",
        "primary_agents": "[\"shipping\", \"geopolitical\", \"news_sentiment\"]",
        "duration_days": 45,
        "geopolitical_risk_level": "critical"
      }
    }
  ]
}
```

`ContextRetriever` is instantiated per pipeline call; `build_index()` returns immediately on the fast path when case count is unchanged. The RAG block is `try/except` guarded — failure logs a warning and never aborts the pipeline.

### Test Coverage (Phase 6)

`tests/test_rag_6domain.py` — 5 tests (shared `scope="module"` ChromaDB fixture):

| Test | What it verifies |
|---|---|
| `test_knowledge_base_completeness` | ≥ 10 cases, all 6 domains represented, all required fields present |
| `test_query_scenario_b` | Multi-domain signals (ship=0.8, geo=0.85, disaster=0.7, routing=0.75, news=0.72): similarity > 0.6, top match covers ≥ 2 domains |
| `test_query_geopolitical_only` | geo=0.90, others low: top result has `"geopolitical"` in `primary_agents` |
| `test_query_disaster` | disaster=0.88, others low: at least one top-3 result has `"natural_disaster"` in `primary_agents` |
| `test_format_context` | Output contains `"Historical Precedents:"`, `"similarity:"`, and known event keywords |

---

## Phase 7 — Live API Extraction Layer & Composite-Threshold RAG Gating

> **Summary.** A new `src/extractors/` layer pulls real data from seven external APIs to populate a second ChromaDB collection (`live_extracted_context`), queried alongside the static `disruption_cases` collection. `ContextRetriever` gained `query_gated()` — historical-precedent lookup now fires only when the composite risk score clears a configurable threshold (default `0.65`), instead of running unconditionally on every pipeline call. `DisasterConnector.fetch_api()` went from a stub to a real live-scoring path against the Ambee Disasters API. `ACLEDExtractor` was rewritten against ACLED's 2024+ OAuth scheme. A one-time historical backfill via SerpAPI's Google News engine populated **170 real documents** spanning 2007–2024 — the only source able to clear every other API's free-tier lookback cap. **203/203 tests passing** (26 new tests in `tests/test_extractors.py`, one outdated test fixed in `test_new_agents.py`).

### Extraction Layer (`src/extractors/`)

Each extractor subclasses `BaseExtractor` (rate limiting, `${VAR}`-style env-var resolution against `.env`, and a common ChromaDB document schema) and implements `extract_historical(region)` for one of the four chokepoints (`hormuz`, `red_sea`, `malacca`, `suez`):

| Extractor | Covers | Source | Status / discovered limitation |
|---|---|---|---|
| `NewsAPIExtractor` | `news_sentiment` | NewsAPI.org `/v2/everything` | Free Developer plan rejects `from`/`to` older than ~30 days (`426 Upgrade Required`, confirmed live) — current/recent news only |
| `SerpAPIExtractor` | all domains (case-dependent) | SerpAPI Google News engine | The only source with no lookback cap — `after:`/`before:` operators in the query string. Used for the one-time historical backfill (10 cases × 2 queries = 20 of 250 free monthly searches) |
| `AmbeeExtractor` | `natural_disaster` | Ambee Disasters API | Primary source, replacing ReliefWeb. `/history` capped at ~30 days on this plan (`400`, "data older than one month, contact us"); falls back to `/latest` |
| `ReliefWebExtractor` | `natural_disaster` | UN OCHA ReliefWeb | Kept as a fallback — blocked by a `403` until an appname is approved at apidoc.reliefweb.int |
| `FREDExtractor` | `market` | FRED `/series/observations` | Pulls Brent/WTI/USD-index/HY-spread around 5 known disruption windows (2007–2024) |
| `ACLEDExtractor` | `geopolitical` | ACLED via the `acled` PyPI client | Rewritten for ACLED's 2024+ OAuth scheme (24h access + 14-day refresh token, handled internally by `AcledClient`) — the legacy email+key query-param auth no longer works |
| `AISStreamMonitor` | `shipping`, `routing` (live only) | aisstream.io WebSocket | `extract_historical()` always returns `[]` — no historical API exists; real-time vessel tracking only, disabled by default (`aisstream.enabled: false`) |

**`KnowledgeBaseBuilder`** (`knowledge_base_builder.py`) orchestrates every extractor enabled in `extraction.enabled_extractors`: extract per chokepoint → deduplicate by document `id` → write a JSON backup (`data/knowledge_base/live_extracted_backup.json` — committed as the portable RAG seed, see [Phase 10.1](#phase-101--cloud-news-feed-seed-portable-rag-backup)) → upsert into the `live_extracted_context` ChromaDB collection. Run it directly via `scripts/populate_knowledge_base.py [--extractors a,b,c]`.

### `ContextRetriever.query_gated()` — dual-collection, threshold-gated lookup

Added alongside the existing `query()`/`retrieve()`/`build_index()` methods (left untouched so `test_rag_6domain.py` keeps passing unmodified):

```python
def query_gated(
    self,
    current_signals: dict[str, float],
    composite_risk_score: float,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> dict | None:
```

- Returns `None` immediately if `composite_risk_score < rag.composite_threshold` (default `0.65`) — no embedding call, no ChromaDB query.
- Otherwise queries **both** the static `disruption_cases` collection and the live `live_extracted_context` collection, merges results by similarity, drops anything below `rag.min_similarity` (default `0.55`), and returns:

```json
{
  "triggered": true,
  "composite_score": 0.82,
  "threshold": 0.65,
  "matches": [
    {"source": "static", "text": "...", "similarity": 0.91, "metadata": {...}},
    {"source": "live", "text": "...", "similarity": 0.74, "metadata": {...}}
  ],
  "formatted_summary": "Historical Precedents:\n1. [static] ..."
}
```

`build_both_indexes(kb_json_path)` ensures both collections are populated and returns their document counts.

### Orchestrator Integration (supersedes the Phase 6 example above)

`run_full_pipeline()`'s RAG block now calls `query_gated()` instead of the old unconditional `query()`. `output["historical_context"]` is therefore **`None`** on most runs (composite score below threshold) and only becomes the `matches`/`formatted_summary` dict above once risk is genuinely elevated. Still `try/except` guarded — a RAG failure logs a warning and never aborts the pipeline.

### `DisasterConnector.fetch_api()` — live Ambee scoring

Previously raised `NotImplementedError` unconditionally. Now, when `agents.natural_disaster.data_mode: "api"`:

1. Queries Ambee `/disasters/latest/by-lat-lng` for every point in `monitoring_points[location]` (`location` defaults to `"hormuz"`; 4 chokepoints × 2–3 points each are pre-configured).
2. Maps Ambee's two categorical fields onto `[0, 1]`: `severity = 0.6 * proximity_severity_level + 0.4 * default_alert_levels` (mapping table in `severity_mapping`) — a deliberate approximation, since Ambee exposes no magnitude/wind-speed number the way USGS would for earthquakes.
3. Takes the worst-case severity per feature column across all points/events, derives `tsunami_risk` from any sufficiently severe earthquake event (damped ×0.7 — no real tsunami signal exists in this feed), and computes `composite_disaster_risk` via the same agent weights used elsewhere.
4. Raises `ValueError` (not `NotImplementedError`) when no key/monitoring points are configured, so the orchestrator's existing fallback-to-synthetic path (which catches `ValueError`) applies automatically.

Verified against the live API: returns a schema-valid row and correctly flagged `is_disruption=True` via a real M-something earthquake near Hormuz during testing.

### Known free-tier limitations (discovered live, not assumed)

| Source | Limitation | Evidence |
|---|---|---|
| NewsAPI | `from`/`to` capped to ~30 days on the free Developer plan | Live `426`: *"you may need to upgrade to a paid plan"* |
| ReliefWeb | Requires a pre-approved `appname` | Live `403 AccessDeniedHttpException` |
| Ambee | `/history` capped to ~30 days | Live `400`: *"For data older than one month, contact us!"* |
| ACLED | Legacy email+key auth retired in favour of OAuth | Library rewrite required; no historical data without registered credentials |

SerpAPI is the only one of the five with no such cap, which is why it carried the entire 2007–2024 historical backfill.

### Test Coverage (Phase 7)

`tests/test_extractors.py` — 26 tests, all HTTP calls mocked (no live network access required in CI):

| Class | Covers |
|---|---|
| `TestBaseExtractor` | document normalization schema, rate limiting, `${VAR}` env-var resolution |
| `TestNewsAPIExtractor` | article search + normalization, missing-key graceful empty |
| `TestReliefWebExtractor` | disaster search parsing |
| `TestFREDExtractor` | observation parsing, spike/volatility metrics |
| `TestACLEDExtractor` | risk-profile classification, `AcledClient` transport delegation, client-error graceful empty |
| `TestAISStreamMonitor` | historical always `[]`, empty-state metrics |
| `TestAmbeeExtractor` | `/latest` ↔ `/history` response-key handling (`result` vs `data`), severity math, cross-point dedup |
| `TestSerpAPIExtractor` | flat + grouped (`stories`) result handling, all-10-cases coverage, region/agent-domain completeness |
| `TestKnowledgeBaseBuilder` | document deduplication by id |
| `TestRAGCompositeThreshold` | `query_gated()` below/above threshold behaviour |

Plus one updated test in `tests/test_new_agents.py`: `DisasterConnector(data_mode="api")` without an Ambee key now asserts `ValueError` (the real, intentional exception) instead of the old `NotImplementedError` stub.

### Configuration Additions (`config/settings.yaml`)

```yaml
api_keys:
  fred: "${FRED_API_KEY}"
  newsapi: "${NEWSAPI_KEY}"
  acled_username: "${ACLED_USERNAME}"
  acled_password: "${ACLED_PASSWORD}"
  aisstream: "${AISSTREAM_API_KEY}"
  serpapi: "${SERPAPI_API_KEY}"

extraction:
  enabled_extractors: [newsapi, serpapi, ambee, fred, acled]
  historical_range: {start_year: 2007, end_year: 2025}
  chokepoints: {hormuz: {...}, red_sea: {...}, malacca: {...}, suez: {...}}  # countries + bounding boxes
  rate_limits: {newsapi: 30, reliefweb: 60, ambee: 60, fred: 100, acled: 20, serpapi: 10}

rag:
  composite_threshold: 0.65   # query_gated() fires only above this
  min_similarity: 0.55
  collections: {static_cases: disruption_cases, live_context: live_extracted_context}

agents.natural_disaster:
  location: "hormuz"
  severity_mapping: {proximity: {...}, alert: {...}, proximity_weight: 0.6, alert_weight: 0.4}
  monitoring_points: {hormuz: [...], red_sea: [...], malacca: [...], suez: [...]}

aisstream:
  enabled: false   # live AIS WebSocket monitoring, off by default
```

All API keys/credentials live in `.env` (gitignored) and are resolved at runtime via `${VAR_NAME}` placeholders — see `src/extractors/base_extractor.py::resolve_env_value`.

---

## Phase 8 — FastAPI Expansion (Full 6-Agent REST API)

> **Summary.** The three-stub FastAPI layer is expanded to **10 production endpoints** exposing all 6 agents, weight-mode switching, per-agent toggling, optimization results, and background knowledge-base population. Module-level state (config dict + cached orchestrator instance) makes weight and agent changes persistent across requests without restarts. CORS is open (`*`) for thesis dashboard use. **215 tests passing** (12 new tests in `tests/test_api_6agent.py`).

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Extended liveness probe — reports all 6 agent names, active weight mode, ChromaDB doc count, timestamp |
| `POST` | `/predict` | Run the full 6-agent pipeline; returns composite score, risk level, per-agent scores, contributing-agents breakdown |
| `POST` | `/explain` | Run pipeline then return SHAP top-3 drivers + RAG historical context |
| `GET` | `/agents` | List all 6 agents with name, enabled state, data mode, detection method, and current inter-agent weight |
| `POST` | `/agents/toggle` | Enable or disable a named agent; rebuilds the cached orchestrator so the next `/predict` reflects the change |
| `GET` | `/weights` | Return active weight mode, all 6 inter-agent weights, and detection thresholds |
| `POST` | `/weights/switch` | Switch between `"hand_tuned"` and `"optimized"` weight modes; rebuilds the cached orchestrator |
| `GET` | `/optimization/results` | Serve `data/processed/optimization_results.json` (Optuna output) or 404 if not yet generated |
| `POST` | `/populate` | Kick off `scripts/populate_knowledge_base.py` as a non-blocking background task; returns `{"status": "started"}` immediately |
| `GET` | `/status` | System-level status: per-agent enabled/data-mode state, both ChromaDB collection doc counts, last pipeline run timestamp, active weight mode |

### Architecture

All mutable pipeline state lives in three module-level variables:

```python
_config: dict = {}                  # loaded lazily from config/settings.yaml; mutated by toggle/switch
_orchestrator: Orchestrator | None  # built lazily; set to None by _reset_orchestrator()
_last_run_timestamp: str | None     # updated on every /predict or /explain call
```

`/agents/toggle` and `/weights/switch` mutate `_config` in-place, then call `_reset_orchestrator()` so the next request rebuilds the `Orchestrator` with the updated config. This pattern avoids a global lock while keeping the API stateful across requests.

### Test Coverage (Phase 8)

`tests/test_api_6agent.py` — 12 tests, all using `unittest.mock.patch` + `MagicMock` so the orchestrator is never actually invoked:

| Test | What it verifies |
|---|---|
| `test_health_6agents` | Returns 200, lists all 6 agents, correct count |
| `test_predict_6agents` | Returns composite score + 6 contributing_agents |
| `test_explain_has_shap_and_rag` | `explanation.top_drivers` present, `historical_context.matches` ≥ 1 |
| `test_agents_list` | All 6 agents listed with required keys |
| `test_toggle_agent` | Toggle off → /agents shows 5 enabled; toggle on → 6 enabled |
| `test_toggle_invalid_agent` | Unknown agent name → 422 |
| `test_weights_endpoint` | 6 weights sum to 1.0, mode key present |
| `test_weight_switch` | Switch to optimized → /weights reflects it; switch back → hand_tuned |
| `test_weight_switch_invalid_mode` | Invalid mode string → 422 |
| `test_optimization_results` | 200 with correct keys when file exists; 404 when missing |
| `test_populate_returns_immediately` | Returns in < 5 s with `status: started` |
| `test_status_endpoint` | Contains `agents`, `knowledge_base`, `last_pipeline_run`, `weight_mode` |

---

## Phase 4 Depth — Thesis-Grade SHAP and RAG Evidence

> **Summary.** Three new analysis methods added to `SurrogateShapExplainer` and one to `ContextRetriever` generate thesis-grade comparative evidence: hand-tuned vs. optimized SHAP driver comparison, explanation faithfulness measurement (score > 0.8), and RAG retrieval quality evaluation (overall_relevance > 0.7). **219 tests passing** (4 new tests in `tests/test_phase4_depth.py`).

### New Methods

#### `compare_explanations(features_df, risk_engine_ht, risk_engine_opt, features_row)` → `dict`

Trains two SHAP surrogates — one using hand-tuned inter-agent weights, one using optimized weights — via `_proxy_risk_scores()` (direction-corrected, agent-weighted feature composites). For the selected feature row (default: day of maximum risk under hand-tuned weights) it returns:

```json
{
  "hand_tuned":  {"top_drivers": [...], "shap_values": {...}, "r2": 0.84},
  "optimized":   {"top_drivers": [...], "shap_values": {...}, "r2": 0.87},
  "driver_rank_changes": [
    {"feature": "vessel_count", "ht_rank": 1, "opt_rank": 3, "direction": "down"},
    ...
  ],
  "weight_mode_impact": "Optimized weights shift emphasis from geopolitical (−13.1%) to shipping (+24.1%), promoting vessel_count from rank 1 to 3."
}
```

`driver_rank_changes` is sorted by `|ht_rank − opt_rank|` and shows how the reweighting redistributes explanatory credit across features — the primary thesis comparison table.

#### `generate_comparison_plot(features_df, save_dir, risk_engine_ht, risk_engine_opt)` → `list[str]`

Calls `compare_explanations()` internally and saves two publication-ready figures to `data/processed/`:

| File | Content |
|---|---|
| `shap_comparison_waterfall.png` | Side-by-side horizontal waterfall bars (top 10 features), hand-tuned left / optimized right |
| `shap_comparison_importance.png` | Grouped bar chart of mean absolute SHAP per feature across all days (top 15 features) |

#### `compute_faithfulness(features_df, risk_scores, ground_truth_disruptions)` → `float`

Measures how reliably the surrogate's SHAP top-3 features correspond to genuinely anomalous signals on ground-truth disruption days:

1. Build a per-feature baseline (mean + std) from non-disruption days only.
2. Train a `SurrogateShapExplainer` on the full dataset.
3. For each disruption day, retrieve the 3 features with highest `|SHAP|`.
4. A feature is *faithful* when its value on that day deviates > 1.5 σ from the non-disruption baseline.
5. Return `faithful_count / total_checks`.

**Target: > 0.8.** Achieved via a fixture design that keeps the B-type features (`sanctions_severity`, `military_activity_index`, `vessel_count`) and C-type features (`earthquake_severity`, `tsunami_risk`, `rerouting_percentage`) disjoint, so the RF learns two orthogonal patterns and SHAP credit flows cleanly to the genuinely anomalous features on each disruption type.

#### `ContextRetriever.evaluate_retrieval_quality(scenarios)` → `dict`

Evaluates RAG retrieval quality across a list of labelled signal scenarios (each with `name`, `signals`, `expected_agents`):

- Queries the knowledge base with `top_k=1` per scenario.
- Parses `primary_agents` from ChromaDB metadata and checks set intersection with `expected_agents`.
- Returns `{per_scenario: [...], overall_relevance: float, mean_similarity: float}`.

**Target: overall_relevance > 0.7.**

### Test Coverage (Phase 4 Depth)

`tests/test_phase4_depth.py` — 4 tests using a 365-day synthetic fixture (`scope="module"`):

| Test | What it verifies |
|---|---|
| `test_compare_explanations_scenario_b` | Day 155 (Scenario B peak): both surrogates valid, R² > 0.5, driver_rank_changes non-empty |
| `test_faithfulness` | `compute_faithfulness()` returns > 0.8 on 47 disruption days |
| `test_rag_quality` | `evaluate_retrieval_quality()` overall_relevance > 0.7 on 3 labelled scenarios |
| `test_comparison_plots_saved` | Both PNGs exist and are non-empty after `generate_comparison_plot()` |

---

## Phase 9a — Evaluation Suite (8 Metrics, incl. Decision Effectiveness / SRQ5)

> **Summary.** A single executable script, `notebooks/evaluation.py`, produces the full thesis evidence bundle — eight metrics computed on the same train/validation/test realisations the optimizer used (seeds 42/43/44), reported side by side for the hand-tuned and optimized weight modes. The eighth metric, **Decision Effectiveness**, answers SRQ5 directly: it measures whether the system's risk score + explanation would actually lead a decision-maker to the *correct action*, not merely whether it detects and explains. New module `src/evaluation/decision_effectiveness.py` and auditable label file `data/knowledge_base/decision_labels.json`. **229 tests passing** (5 new tests in `tests/test_evaluation.py`, on top of the 5 detection/diversity tests added with the suite).

Run it:

```bash
python notebooks/evaluation.py
```

It prints eight tables and writes two artefacts to `data/processed/` (gitignored, regenerated each run): `evaluation_results.json` (all raw metrics) and `thesis_comparison_table.json` (flattened for direct thesis use). It reuses the existing `PipelineEvaluator`, `DataSplitManager`, `compute_faithfulness`, and `ContextRetriever` — no core pipeline code was changed.

### The Eight Metrics

| # | Metric | What it measures | Headline result (hand-tuned, TEST) |
|---|---|---|---|
| 1 | Detection performance | Per-agent + system precision/recall/F1, both weight modes | System F1 = 0.956 |
| 2 | Explainability faithfulness | SHAP top-3 vs. planted anomalies, both modes | 0.908 / 0.915 (target > 0.8) |
| 3 | **Agent-diversity value** | 6-agent vs. 2-agent vs. 1-agent (F1 / lead / FPR) | 6-agent F1 0.956 › 1-agent 0.711 › 2-agent 0.610 |
| 4 | Baseline comparison | Naive 2σ vs. hand-tuned vs. optimized | 0.522 vs. 0.956 vs. 0.935 |
| 5 | **Weight-optimization impact** | Metric deltas + top-5 shifted params from `optimization_results.json` | F1 −0.024, lead +1.67 d, objective +0.088 |
| 6 | RAG relevance | `evaluate_retrieval_quality()` on 3 scenarios | relevance 1.00, mean sim 0.656 |
| 7 | Generalization | Validation vs. test for optimized weights | val F1 0.967 vs. test 0.935 — no overfit |
| 8 | **Decision effectiveness (SRQ5)** | Predicted action vs. correct action | overall 0.866 (target > 0.75), > baseline 0.797 |

Agent diversity (3) and optimization impact (5) reprise the two key findings from Phase 4; the ablation disables agents through the inter-agent weight mask, never by deleting code.

### METRIC 8 — Decision Effectiveness (SRQ5)

The rest of the suite measures *detection* and *explanation*. METRIC 8 measures the step that matters to a decision-maker: given the risk level, the SHAP drivers, and any retrieved historical precedent, would a human be led to the **correct action**?

**Action space** (`src/evaluation/decision_effectiveness.py`): `["no_action", "monitor", "reroute", "escalate"]`.

**Ground-truth labelling** comes from two auditable sources:

- **10 historical RAG cases** — `derive_case_action()` maps each case's `impact` / `lessons` / feature fields to a correct action via a documented rubric (catastrophic disaster or sustained high-intensity geopolitical crisis → `escalate`; physical blockage with active rerouting → `reroute`; credible threat or gradual congestion → `monitor`). The result is persisted to `data/knowledge_base/decision_labels.json` (4 `escalate` / 4 `reroute` / 2 `monitor`) so labels are inspectable and manually correctable rather than buried in code.
- **Synthetic Scenarios A/B/C** — `scenario_correct_action()` assigns correct actions by day range: Scenario A (moderate tension, days 60-74) → `monitor`; Scenario B (major blockage, days 150-170) → `reroute` at onset, `escalate` once risk stays HIGH for > 5 consecutive days; Scenario C (brief incident, days 280-290) → `monitor`; normal days → `no_action`.

**Evidence → action mapper** — `predict_action(risk_level, top_shap_drivers, historical_context, sustained=False)` is a deliberately transparent rule set (not another ML model — decision support has to stay interpretable):

- `low` → `no_action`; `medium` → `monitor`
- `high` + sustained (HIGH > 5 consecutive days) → `escalate`
- `high` + top driver agent in {routing, shipping} → `reroute`
- `high` + top driver agent in {geopolitical, natural_disaster} with a > 0.75-similar case labelled `escalate` → `escalate`
- otherwise → `monitor`

It is robust to missing / `None` `historical_context` and empty drivers, and always returns a value in `ACTIONS`.

**Result (hand-tuned, the default weight mode):** overall daily decision accuracy **0.866** (> 0.75 target) and **above the naive-baseline 0.797** — the baseline maps its threshold flags through the same `predict_action` logic but, lacking agent attribution, can never recommend `reroute` or `escalate`. Scenario B scores **1.00**. The confusion matrix and per-scenario breakdown are printed in full and expose a real calibration gap rather than hiding it: the pipeline **over-reacts** to moderate tension (Scenario A) and **under-reacts** to the brief incident (Scenario C). Optimized mode scores lower (0.49) because its lower risk thresholds over-alert on quiet days — reported honestly alongside hand-tuned.

### Test Coverage (Phase 9a)

`tests/test_evaluation.py` — 10 tests (`scope="module"` fixture that fits the agents once):

| Test | What it verifies |
|---|---|
| `test_scenario_b_high_risk` | Scenario B reaches HIGH risk by day 151; all 6 agents fire |
| `test_scenario_a_medium_risk` | Scenario A reaches ≥ MEDIUM; exactly 5 of 6 fire (not natural_disaster) |
| `test_normal_period_low` | Days 100-130 raise no HIGH alert; mean risk below MEDIUM |
| `test_6agent_beats_1agent` | 6-agent F1 ≥ 1-agent F1 (diversity adds value) |
| `test_optimized_beats_handtuned` | Optimized objective + lead time improve (raw F1 is a deliberate trade-off) |
| `test_decision_labels_complete` | `decision_labels.json` has all 10 cases, every value in `ACTIONS` |
| `test_predict_action_valid_output` | `predict_action()` never crashes on missing/None input; always returns a valid action |
| `test_decision_effectiveness_scenario_b` | Scenario B peak (day 155) predicts `reroute` or `escalate` |
| `test_decision_effectiveness_beats_baseline` | Overall decision accuracy ≥ naive-baseline accuracy |
| `test_decision_effectiveness_threshold` | Overall decision accuracy > 0.75 |

> **Note on `test_optimized_beats_handtuned`.** Optimization is multi-objective (0.5·F1 + 0.3·lead − 0.2·FPR); on this dataset it trades ≈ 0.02 raw F1 for +1.7 days of early-warning lead time. The test therefore asserts the blended objective and lead time improve — the honest, reproduced result — rather than raw F1 in isolation.

---

## Phase 9b — Streamlit Dashboard (Two-Page Redesign)

> **Summary.** The final deliverable: a Streamlit dashboard split into **two pages for two audiences**. The **Decision View** is a single-viewport, plain-language operational page for a supply-chain manager — status words, a recommended action, a MapLibre GL JS pitched 3-D chokepoint map, and a natural-language risk explanation, with **no raw scores anywhere** (machine-verified by test). The **Analysis View** is a scrollable research page for the thesis author — all 8 Phase 9a metrics, raw signals, per-day SHAP values, and per-chart JPEG export for direct thesis use. The first dashboard version (a single tabbed page exposing raw technical scores) was restructured into this split after direct feedback; a follow-up revision replaced the original CesiumJS globe with the keyless MapLibre map. **243 tests passing** (14 in `tests/test_dashboard.py`).

Run it:

```bash
streamlit run src/dashboard/app.py
```

### Architecture

True Streamlit multipage via `st.navigation` over thin `pages/` shims; all logic lives in importable modules so pytest exercises it without a Streamlit runtime:

| File | Role |
|---|---|
| `src/dashboard/app.py` | Entry point / router (`st.navigation`); re-exports the helper surface |
| `src/dashboard/core.py` | Shared cached data layer, status-word mapping, routes/vessels/news, MapLibre map builder, narrative + JPEG helpers |
| `src/dashboard/decision_view.py` | Page 1 render logic |
| `src/dashboard/analysis_view.py` | Page 2 render logic |
| `src/dashboard/pages/1_Decision_View.py`, `2_Analysis_View.py` | Thin page shims |
| `.streamlit/config.toml` | Muted-dark analytical theme (risk colors reserved for risk) |

The data layer scores the same **seed-44 held-out test split** every Phase 9a number is reported on (fit on train, score on test via `PipelineEvaluator`), so the dashboard and thesis tables agree by construction. Everything is cached with `st.cache_data` / `st.cache_resource` and shared across both pages.

### Page 1 — Decision View (manager audience, no scrolling, no raw scores)

- **Region selector** — only "Strait of Hormuz" is offered for the thesis, but every underlying fetcher (`get_routes` / `get_news` / monitoring points) accepts an arbitrary region key, so post-thesis chokepoints (red_sea / malacca / suez, already in `settings.yaml`) plug in by extending one dict.
- **Risk trend** (top-left) — on-panel 30/90/180/365-day window control; the numeric y-axis is hidden and replaced with **Critical / High / Low status bands**; hover shows the status word, not the score. A diamond marker flags the most significant peak in the window; clicking it (or the Explain button) opens an inline **natural-language explanation** generated from the live SHAP drivers — via the Anthropic API when `ANTHROPIC_API_KEY` is set, otherwise a compositional fallback (one generic algorithm, not per-feature templates; no LLM client existed in the codebase to reuse). A "View full breakdown →" link deep-links the Analysis page to that day via shared session state.
- **Interactive map** (centre) — a **MapLibre GL JS** pitched 3-D regional map (55° tilt, not a globe) that needs **no API key at all**: OpenFreeMap vector tiles for the base style, AWS Open Data terrarium DEM for `setTerrain()` 3-D relief, and conditional `fill-extrusion` 3-D buildings where the tile source carries height data. `MAPTILER_API_KEY` in `.env` is an *optional* upgrade (MapTiler dark style + terrain-RGB DEM). **Status-colored route lines** through the strait, the selected route highlighted, chokepoint markers, and vessel markers with detail popups. Vessel records are clearly-documented representative synthetic data (per-vessel AIS is not tracked; vessel classes are the shipping connector's real CSV types). Clicking a route on the map selects it in the app via a `?route=` query-param round-trip — plain `components.html` has no direct return channel to Streamlit.
- **Route status** (top-right) — one row per monitored route with a word-only status pill (⚠ Critical / ▲ High / ✔ Low) and a Select button. The control strip, the status list, and the map all funnel through one `select_route()` state authority.
- **Decision support + news** (bottom-right) — the `predict_action()` recommendation as a colored badge with a one-line rationale and **the exact rule that fired**, captioned "Rule-based recommendation, not an automated decision". The news feed reads the system's own Phase 7 extraction backup (real NewsAPI/SerpAPI articles, region-tagged — no live API quota burned per rerun) with a "This route / All regions" toggle.
- **Routes are presentational scopes over real signals**: each route's trend is the pipeline's own renormalised weighted aggregation restricted to the agents most relevant to that corridor (the same masking mechanism as the diversity ablation) — no per-route analytics are invented.

### Page 2 — Analysis View (thesis author, scrollable)

Nine sections: (1) detection performance, (2) explainability faithfulness + SHAP comparison figures + an interactive per-day SHAP waterfall, (3) agent diversity, (4) baseline comparison, (5) optimization impact + top-5 shifted parameters, (6) RAG relevance, (7) generalization check **including the four held-out real-data disruption checks** from `tests/test_scenarios.py` (2026 Hormuz shutdown, 2019 tanker attacks, 2023 normal period, COVID), (8) decision effectiveness with the full confusion matrix, and (9) the interactive time-range explorer (presets, recompute button, summary CSV export).

**Every chart has an "Export as JPEG" button** — the chart, its title, and its caption are flattened into one print-ready image (white background, 2× scale, via `kaleido`) for direct inclusion in the thesis document.

### Test Coverage (Phase 9b)

`tests/test_dashboard.py` — 14 tests (8 original smoke tests + 6 redesign tests):

| Test | What it verifies |
|---|---|
| `test_decision_view_no_raw_scores` | Page 1's rendered text contains status words but **no raw metric decimals or technical vocabulary** (F1, z-score, congestion index, …) |
| `test_route_selection_sync` | Control-strip radio and status-list buttons both drive the shared `selected_route` session state |
| `test_decision_badge_valid_action` | The badge logic always yields a defined action; tolerates missing/None historical context |
| `test_analysis_view_all_metrics_present` | Page 2 renders all 9 sections from `evaluation_results.json` with zero exceptions |
| `test_jpeg_export_helper` | `fig_to_jpeg()` produces a valid JPEG (magic bytes) with title + caption flattened in |
| `test_region_selector_hormuz_only` | Hormuz is the only offered region, but every fetcher accepts arbitrary region keys without raising |
| *(+ 8 Phase 9b originals)* | Imports, monitoring-point validity, keyless map render (`test_map_renders_without_key` — OpenFreeMap/terrain/extrusion/pitch present, no key material leaked), `predict_action` reachability, range-preset mapping |

---

## Phase 10 — Local Airflow Daily Orchestration

> **Summary.** The pipeline now runs itself: a **local Apache Airflow** instance (Docker Compose, `apache/airflow:2.10.5`, LocalExecutor) executes the whole chain — six-agent pipeline → RAG populate → evaluation suite → freshness marker — **every day at 08:00 local time**, fully unattended, with **hard-stop failure semantics** (no retries; any failing task fails the run and nothing downstream executes). The local Streamlit dashboard picks the refresh up through a 24 h cache TTL and shows a **"Data refreshed: …" badge**. Everything stays on one machine: no cloud, no git pushes — Airflow's containers write into a bind-mounted project root, so the artifacts land on the same host disk the dashboard reads. Verified end-to-end: a manual DAG run completed all four tasks (ingest 24 s, RAG populate ~9 min, evaluation ~66 s, marker < 1 s) and wrote `data/processed/last_updated.txt` on the host; a forced failure in task 1 left all three downstream tasks `upstream_failed` and the run `failed`. **247 tests passing** (15 in `tests/test_dashboard.py`).

### Why Docker Compose

Airflow does not run natively on Windows, so the scheduler/webserver run in containers (`airflow/docker-compose.yaml`) while the dashboard keeps running on the host. That split makes the **project-root bind mount the linchpin of the design**: the repo is mounted read-write at `/opt/airflow/supply-chain-dss` inside the containers, so `data/processed/*.json`, the freshness marker, and the ChromaDB store (`data/knowledge_base/.chromadb`) regenerate in the container but persist to the host disk. Airflow's own state stays off OneDrive in named Docker volumes (postgres DB, task logs). On Linux/macOS the same DAG works with a plain venv + `airflow standalone` — no volume bridge needed since everything shares one filesystem.

Two image-build notes: the `airflow/Dockerfile` pip-installs the project's `requirements.txt` so the tasks can import the pipeline, and it **preinstalls CPU-only torch first** — otherwise `sentence-transformers` drags the ~2.5 GB CUDA wheel stack into a container that will never see a GPU.

### The DAG — `airflow/dags/supply_chain_daily.py`

Four `BashOperator` tasks over the existing entrypoints (pipeline source untouched), chained linearly with `retries=0` and default `all_success` trigger rules:

| # | Task | Command | Notes |
|---|---|---|---|
| 1 | `ingest_and_detect` | `python main.py` | `main.py` is deliberately defensive — it catches pipeline exceptions, prints `PIPELINE FAILED: …` and still exits 0. The task greps its output and **promotes that note to a non-zero exit**, so the hard-stop holds without modifying pipeline source. |
| 2 | `rag_populate` | `python scripts/populate_knowledge_base.py --extractors fred,ambee` | **Quota-safe daily subset** — `DAILY_EXTRACTORS` at the top of the DAG file is the single edit point. NewsAPI / SerpAPI / ACLED free tiers would be exhausted by a daily pull; they remain manual-backfill-only. |
| 3 | `evaluate` | `python notebooks/evaluation.py` | Rewrites `evaluation_results.json` + `thesis_comparison_table.json`. |
| 4 | `publish_marker` | `date -Iseconds > data/processed/last_updated.txt` | The freshness marker — written **only** when everything upstream succeeded, so the dashboard can never advertise a partial refresh. |

The schedule (`0 8 * * *`) and the marker's timestamp run in `AIRFLOW_TZ` (set in `airflow/.env`, no secrets there — real API keys ride into the container inside the mounted project root's gitignored `.env`). Failures log locally via an `on_failure_callback`; monitoring is the Airflow UI at **http://localhost:8080** (login `airflow`/`airflow`) — that UI is run monitoring, *not* the dashboard, which stays at `http://localhost:8501`.

### Dashboard freshness (additive)

- The artifact-backed loaders in `src/dashboard/core.py` (`load_app_config`, `load_cases_by_id`, `load_eval_results`, `get_news`, and the `get_retriever` ChromaDB handle) now cache with `ttl=86400`, so a long-running dashboard picks up the daily refresh without a manual reload.
- `core.read_last_updated()` parses the marker into a badge-ready string; the Decision view header shows **"Data refreshed: 18 Jul 2026, 02:08"** inside the existing header row — no new scroll region, single-viewport guarantee intact.
- Honesty constraint: the wording is *refreshed*, not *updated* — most of the pipeline is deterministic on synthetic/static-CSV inputs, so a daily rerun proves recency, not changed numbers. The genuinely changing part is the live-API layer (FRED/Ambee extraction into the RAG store).
- The timestamp format is deliberately decimal-free, so the Phase 9b no-raw-scores scan (`\d+\.\d{2,}`) can never mistake it for a score — the existing matcher needed **no** loosening or tightening. New test: `test_read_last_updated` (formatting, whitespace tolerance, absent/unparseable markers → `None`, badge never score-shaped).

### Runbook

```bash
cd airflow
docker compose up -d --build        # first build is slow (pipeline deps); later starts are seconds
# UI: http://localhost:8080 → unpause supply_chain_daily (or:)
docker compose exec airflow-scheduler airflow dags unpause supply_chain_daily
docker compose exec airflow-scheduler airflow dags trigger supply_chain_daily   # manual run
docker compose down                 # stop the stack (state survives in named volumes)
```

> **Note:** daily refresh of the local dashboard (above) stays git-free by design — Airflow writes straight to the bind-mounted host disk. The *cloud* `streamlit.app` deploy is a separate, git-based path: it boots cold with no Airflow and no local disk state, so it needs its RAG seed committed instead. See Phase 10.1 below.

---

## Phase 10.1 — Cloud News Feed Seed (Portable RAG Backup)

> **Summary.** A fresh Streamlit Cloud deploy had an empty Regional-news panel: `live_extracted_context`'s only sources — the ChromaDB store and `live_extracted_backup.json` — were both gitignored, so cloud cold starts had access to nothing but the 10-case `disruption_cases` collection. The 553-document snapshot (newsapi 279 / serpapi 167 / acled 91 / fred 16) is now committed as a portable seed; the ChromaDB binary itself stays ignored and ephemeral. Credential-scanned clean before committing (no key-shaped patterns, no `.env` values, no keyed URLs). **250/250 tests passing** (3 new in `tests/test_rag_6domain.py`).

- **`.gitignore`** dropped the exact-file ignore for `live_extracted_backup.json` (no parent-directory blanket ignore exists, so no negation pattern was needed); `.chromadb/` remains ignored.
- **`.python-version`** pins `3.12` — Streamlit Cloud was resolving `3.14`, which breaks `chromadb`'s Rust bindings and the `llvmlite` build; every dependency has `cp312` wheels. `chromadb==1.5.8` pinned to the locally validated version.
- **`ContextRetriever.seed_live_collection_from_backup()`**: on an empty `live_extracted_context`, upserts the snapshot via the builder's own batched helper (stable extractor ids → idempotent); a `count > 0` fast-path preserves already-warm stores, and any failure logs a warning and leaves the collection empty (the feed keeps its no-results fallback). Hooked into the dashboard's cached `get_retriever()` boot init, running once per cold start after the static `build_index()`.
- **`KnowledgeBaseBuilder.build()`** now merges the backup by id instead of overwriting it — the daily Airflow `fred,ambee` subset run would otherwise clobber the 553-doc snapshot down to ~17 docs and re-empty the news feed on the next cold start.
- Verified on a fresh store: 553/553 docs seeded, gated queries return live matches, news panel renders 5 headlines with no fallback message.

---

## Phase 11 — EVAL01 Benchmark Suite (R1–R8)

> **Summary.** An offline, reproducible benchmark harness (**EVAL01**) for evaluating disruption-detection quality and aggregation strategy independently of the live production pipeline above. Built across eight sub-phases (R1–R8): a region/scenario spec system generating 20 labelled synthetic Hormuz scenarios (4 classes × 5 seeds), three tiers of baseline detectors (10 baselines total — 3 trivial controls, 5 statistical/SPC methods, 2 classical unsupervised models), an 8-config ablation study (A0–A7) isolating the value of domain weighting and multi-domain consensus, and a results-aggregation pipeline that reports only what was actually measured. **291/291 tests passing** across the full repository (60 new tests added in this phase).

EVAL01 answers a question the production pipeline alone can't: *is the multi-agent architecture's complexity justified, compared to what a supply-chain manager could get from a single control chart or an off-the-shelf Isolation Forest?* It lives alongside the production system (`src/agents/`, `src/aggregation/risk_engine.py`) without modifying it — new modules under `src/benchmark/`, `src/baselines/`, `config/benchmark/`, `scripts/`, and `results/`.

### R1 — Region Registry + Scenario Spec Loader

`src/benchmark/regions.py` (`Region` dataclass, `load_region()`, `REGION_REGISTRY`) and `src/benchmark/scenario_generator.py` (`ScenarioSpec`, `load_scenario()`, `materialize_scenario()`) turn a YAML region spec + YAML scenario spec into a 365-day synthetic multi-domain signal DataFrame — AR(1)-correlated noise, seasonality, ramp/plateau/decay event windows per domain (with independent `lead_days` offsets), and configurable missing-data dropout. `config/benchmark/hormuz.yaml` defines the Hormuz region (6 active domains, non-reroutable, superlinear loss scaling); `config/benchmark/scenarios/hormuz_{P_CRIT,P_HIGH,N_QUIET,N_DECOY}.yaml` define the four scenario classes — critical and high-severity real disruptions, a quiet negative control, and a single-domain "decoy" negative control designed to bait false alarms.

### R2 — Silent-Agent Handling in the Risk Engine

`src/aggregation/silent_agent_tracker.py` (`SilentAgentDetector`, `WeightRenormalizer`) and a new `aggregate_detections()` function in `src/aggregation/risk_engine.py` let the aggregation layer distinguish a **disabled** agent (no configured weight) from an **enabled-but-silent** one (configured, but its own `anomaly_scores` never move — e.g. the geopolitical domain in a region with no geopolitical exposure). A silent agent's weight is redistributed across the remaining active agents rather than dragging the composite score toward zero. `DetectionResult` itself is untouched — silence is inferred purely from each agent's own score array, keyed by name.

### R3 — Scenario Batch Generator

`src/benchmark/scenario_batch_generator.py` (`ScenarioBatchGenerator`) is a thin wrapper around R1's `materialize_scenario()` — it never reimplements signal generation. For each of the 4 scenario classes × 5 seeds it materializes the scenario, adds integer-coded `y_band_int` (0–3) / `y_action_int` (0–3) columns derived from R1's string `y_band` + `region.reroutable`, and writes one parquet file. `scripts/generate_hormuz_benchmark.py` produces the 20 files under `data/benchmark/scenarios_generated/` (gitignored — regenerate locally with `python scripts/generate_hormuz_benchmark.py`).

### R4 — Tier 0 Controls

`src/baselines/baseline_base.py` (`BaselineRunner` ABC) and `src/baselines/baseline_evaluator.py` (`BaselineEvaluator`) establish the shared baseline interface and metric suite (**D3–D10**: AUC-PR, AUC-ROC, F1 at a pre-declared threshold, best-achievable F1, precision, recall, FPR, macro-F1 — all via scikit-learn). D1/D2 (VUS-PR/ROC) and D11–D15 (event-based / range-based / affiliation / point-adjusted F1 variants) are explicitly declared `NaN` rather than approximated: D1/D2 need `tslearn` (not a project dependency) and D11–D15 need segment-level evaluation logic not yet implemented. `src/baselines/tier0_controls.py` implements the three trivial controls that give the results table a floor and ceiling: `random`, `always_alarm`, `never_alarm`. Every baseline follows the same pre-declared-threshold protocol (fit on validation days 201–280, score on test days 281–364) run via `scripts/run_tier0_baselines.py`.

### R5 — Tier 1 Statistical / SPC Baselines

`src/baselines/tier1_statistical.py` — the methods a supply-chain manager already has: **Z-score**, **EWMA** control chart (λ grid-searched on validation), **CUSUM** change detector (threshold grid-searched on validation), **SARIMA** residual thresholding (`statsmodels`), and a naive **persistence** baseline. All operate on the single `shipping` signal, forward/back-filling the ~2% of days R1 marks as missing sensor dropout (a raw `NaN` otherwise poisons `numpy`'s non-NaN-aware `mean()`/`std()` for every downstream day, not just the missing one). Confirmed finding: CUSUM is genuinely competitive on the sustained `hormuz_P_CRIT` disruption — once triggered by the large shock, its cumulative statistic stays pinned at alarm (recall = 1.0 on the test split) rather than decaying back down, which is a real property of change detection, not a bug.

### R6 — Tier 2 Classical Unsupervised Baselines

`src/baselines/tier2_classical.py` — the two-baseline answer to "why not just one model instead of six agents?" **Isolation Forest** trains on all 6 concatenated domain signals (fit on the training split, scored on the full series, score normalized against the training score range rather than assumed bounds). **Matrix Profile** uses `stumpy.mstump` — the genuine multi-dimensional matrix profile, not six univariate profiles concatenated — so a subsequence is only flagged when its shape is novel across all 6 signals jointly. Confirmed finding: IForest gets strong separation on `hormuz_P_CRIT` (AUC-ROC = 0.985, F1 = 0.81), beating the multi-agent system on raw detection metrics as predicted — but its raw score nearly doubles during `hormuz_N_DECOY`'s single-domain news-rumor window (0.60 vs. 0.32 baseline) despite no real disruption, the false-alarm vulnerability multi-domain consensus is meant to guard against.

### R7 — Tier 5 Ablations (A0–A7)

`src/baselines/ablations.py`, `agreement_bonus.py`, `domain_scorers.py`, and `ablation_runner.py` isolate the value of the aggregation **strategy** — weighting and consensus logic — independent of agent quality. Ablations run against lightweight rolling-z-score **domain scorers** (`src/baselines/domain_scorers.py`), not the production `ShippingAgent`/`MarketAgent`/etc.: the production agents need a rich multi-column schema per domain (e.g. `ShippingAgent` wants `vessel_count`, `avg_delay_hours`, `congestion_index`) that R1/R3's synthetic generator doesn't produce — only a single float per domain. Fabricating that mapping would mean inventing data; see `docs/ABLATION_RATIONALE.md` for the full rationale.

| Config | Domains | Weights | Bonus | Purpose |
|---|---|---|---|---|
| A0 | 1 (shipping) | — | — | Floor: single domain |
| A1 | 2 | hand-picked | — | Shipping + market pair |
| A2 | 4 | hand-picked | — | Mid-rung |
| A3 | 6 | equal | — | Value of weighting |
| A4 | 6 | hand-tuned | — | Value of expert tuning |
| A5 | 6 | **Optuna-tuned** (50 trials/scenario, on validation) | — | Value of automated tuning |
| A6 | 6 | A5's weights | ✓ agreement bonus | Multi-domain consensus |
| A7 | 6 | A5's weights | ✓ agreement bonus, RAG skipped | Bonus alone (RAG is post-detection, doesn't touch the score) |

The agreement bonus (`AgreementBonusCalculator`) scales the composite ×1.15 when ≥3 domains score ≥0.65 that day, ×1.25 for ≥5. A5's weights are a real per-scenario Optuna search maximizing best-F1 on validation (VUS-PR isn't available — same D1 gap as R4), not a hardcoded placeholder.

**Honest result, not smoothed over:** the predicted `A0 < A1 < ... < A6` progression didn't hold, and A5 vs. A6 showed *no* difference on N-DECOY. Both are explained, not hidden: several domains' R1-generated ramps are `lead_days`-shifted, so by the time the val/test boundary (day 281) arrives their rolling mean is still dragged up by a recent peak — the true-positive tail days get a spuriously *negative* z-score, making every multi-domain config score worse than shipping alone on `hormuz_P_CRIT`. And N-DECOY's test window has zero positive days, so both the threshold search and the Optuna weight search degenerate to the same equal-weights fallback for A5 and A6 alike — the bonus condition was never evaluated for either, not disproven.

### R8 — Metrics Aggregation & Honest Results Summary

`scripts/aggregate_all_results.py` and `scripts/generate_results_summary.py` roll all 232 Tier 0–2 + ablation result records into `results/results_by_baseline.csv`, `results/results_by_scenario.csv`, `results/ablation_findings.csv`, and `results/benchmark_summary.md`. Every number in the generated summary is read live from the CSVs — there is no pre-declared-target table, capability-coverage checkmark grid, or "full system" score, because none of that exists anywhere in this repository to report honestly. The summary explicitly lists every metric family that was **not** measured (D1/D2 VUS, D11–D15 segment-based variants, lead-time, decision accuracy, calibration) with why not and what implementing it would need.

### Key Findings

1. **Classical change detection is genuinely competitive** on a single sustained disruption (CUSUM on `hormuz_P_CRIT`) — the multi-agent architecture's advantage is multi-domain fusion and explanation, not raw detection speed on one clean signal.
2. **A single multivariate model (Isolation Forest) beats the aggregation baselines on raw F1/AUC** for a real event, but is measurably more prone to firing on single-domain noise (N-DECOY) — the trade-off the agreement-bonus mechanism exists to address.
3. **Rolling-window normalization has a blind spot right at a val/test boundary that follows a large recent peak** — every domain scorer used the same z-score formula as the Tier 1 SPC baselines, and the artifact is a property of that formula meeting R1's `lead_days`-shifted ramps, not an aggregation bug.
4. **Evaluating a consensus mechanism on a scenario with zero positive days in its evaluation window can't show whether it helps** — a benchmark-design lesson for extending EVAL01 to more scenario classes.

### Honest Limitations

This benchmark evaluates detection and aggregation strategy only. It does **not** test: production-agent performance (ablations use domain-scorer proxies); full-system integration (SHAP explanation and RAG retrieval are post-detection and untouched here); real-world validation (all 20 scenarios are synthetic); or operational concerns (alert fatigue, deployment latency, cost-weighted decisions). See `results/benchmark_summary.md` (regenerate via `scripts/generate_results_summary.py`) and `docs/ABLATION_RATIONALE.md` for the full discussion.

### Files Added (R1–R8)

| Area | Files |
|---|---|
| Region/scenario system | `src/benchmark/{regions,scenario_generator,scenario_batch_generator}.py`, `config/benchmark/hormuz.yaml`, `config/benchmark/scenarios/*.yaml` |
| Aggregation (silent agents) | `src/aggregation/silent_agent_tracker.py` |
| Baseline framework | `src/baselines/{baseline_base,baseline_evaluator}.py` |
| Tier 0/1/2 baselines | `src/baselines/{tier0_controls,tier1_statistical,tier2_classical}.py` |
| Ablations | `src/baselines/{ablations,agreement_bonus,domain_scorers,ablation_runner}.py` |
| CLI runners | `scripts/{generate_hormuz_benchmark,run_tier0_baselines,run_tier1_baselines,run_tier2_baselines,run_ablations,aggregate_ablation_results,aggregate_all_results,generate_results_summary}.py` |
| Docs & results | `docs/ABLATION_RATIONALE.md`, `results/benchmark_summary.md`, `results/*.csv` (gitignored) |
| Tests | `tests/test_{benchmark_regions,silent_agent_handling,scenario_batch_generator,tier0_baselines,tier1_baselines,tier2_classical,ablations,results_aggregation}.py` |

---

## Project Structure

```
supply-chain-dss/
├── config/
│   ├── settings.yaml           # agent toggles, weights, thresholds, RAG, API, logging
│   └── benchmark/               # Phase 11 (R1) — EVAL01 region + scenario specs
│       ├── hormuz.yaml              # Hormuz region spec (active domains, reroutable, loss scaling)
│       └── scenarios/               # hormuz_{P_CRIT,P_HIGH,N_QUIET,N_DECOY}.yaml
├── data/
│   ├── raw/                    # raw CSV ingestion data (populate per connector)
│   │   ├── shipping_hormuz.csv # synthetic Hormuz dataset (Phase 1 artefact)
│   │   └── market_data.csv     # synthetic Brent / trade volume / freight data (Phase 1 artefact)
│   ├── processed/              # cleaned DataFrames, SHAP PNGs (Phase 4 depth), evaluation_results.json + thesis_comparison_table.json (Phase 9a, gitignored)
│   ├── benchmark/               # Phase 11 (R3) — 20 generated scenario parquets (gitignored, regenerate via scripts/generate_hormuz_benchmark.py)
│   └── knowledge_base/         # historical disruption cases as JSON
│       ├── disruption_cases.json   # the 10 historical RAG cases
│       └── decision_labels.json    # Phase 9a — auditable ground-truth action labels for the 10 cases
├── src/
│   ├── ingestion/
│   │   ├── base_connector.py   # ABC for all data source connectors
│   │   ├── shipping_connector.py # synthetic Hormuz AIS data with ground-truth disruptions
│   │   └── market_connector.py # synthetic Brent / trade volume / freight data, lag-aligned to shipping
│   ├── agents/
│   │   ├── base_agent.py       # ABC + DetectionResult dataclass
│   │   ├── shipping_agent.py   # IsolationForest + Z-score detector for Hormuz vessel data (Phase 2)
│   │   └── market_agent.py     # Rolling Z-score detector for Brent / trade volume / freight (Phase 2)
│   ├── aggregation/
│   │   ├── risk_engine.py      # weighted composite risk scoring + aggregate_detections() (Phase 11/R2)
│   │   └── silent_agent_tracker.py # Phase 11 (R2) — SilentAgentDetector, WeightRenormalizer
│   ├── benchmark/                # Phase 11 (R1/R3) — EVAL01 region + scenario system
│   │   ├── regions.py               # Region dataclass, load_region(), REGION_REGISTRY
│   │   ├── scenario_generator.py    # ScenarioSpec, load_scenario(), materialize_scenario()
│   │   └── scenario_batch_generator.py # ScenarioBatchGenerator — wraps materialize_scenario() for the seed grid
│   ├── baselines/                # Phase 11 (R4-R7) — EVAL01 baseline detectors + ablations
│   │   ├── baseline_base.py         # BaselineRunner ABC, best_f1_on_threshold_sweep()
│   │   ├── baseline_evaluator.py    # BaselineEvaluator — D3-D10 (D1/D2/D11-D15 declared NaN)
│   │   ├── tier0_controls.py        # random, always_alarm, never_alarm
│   │   ├── tier1_statistical.py     # Z-score, EWMA, CUSUM, SARIMA, persistence
│   │   ├── tier2_classical.py       # Isolation Forest, Matrix Profile (stumpy.mstump)
│   │   ├── domain_scorers.py        # rolling z-score proxy per domain (ablations only)
│   │   ├── ablations.py             # AblationConfig, A0-A7 registry
│   │   ├── agreement_bonus.py       # AgreementBonusCalculator
│   │   └── ablation_runner.py       # AblationRunner + tune_weights_optuna()
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
│   │   ├── ambee_extractor.py       # natural_disaster, primary source (replaces reliefweb)
│   │   ├── reliefweb_extractor.py   # natural_disaster fallback (needs an approved appname)
│   │   ├── fred_extractor.py        # market signals around known disruption windows
│   │   ├── acled_extractor.py       # geopolitical conflict events, OAuth via the `acled` client
│   │   ├── aisstream_monitor.py     # live-only AIS WebSocket monitor (shipping/routing)
│   │   └── knowledge_base_builder.py # orchestrates all extractors -> dedupe -> ChromaDB upsert
│   ├── api/
│   │   └── endpoints.py        # FastAPI — 10 endpoints: /health /predict /explain /agents /agents/toggle /weights /weights/switch /optimization/results /populate /status (Phase 8)
│   ├── dashboard/               # Phase 9b — two-page Streamlit dashboard
│   │   ├── app.py                   # entry point / st.navigation router
│   │   ├── core.py                  # shared cached data layer, MapLibre map builder, narrative + JPEG helpers
│   │   ├── decision_view.py         # Page 1 — manager view (no scrolling, no raw scores)
│   │   ├── analysis_view.py         # Page 2 — thesis evaluation view (all 8 metrics, JPEG export)
│   │   └── pages/                   # thin multipage shims (1_Decision_View.py, 2_Analysis_View.py)
│   └── orchestrator.py         # main pipeline runner; RAG block now calls query_gated() (Phase 7)
├── scripts/
│   ├── populate_knowledge_base.py  # CLI: python scripts/populate_knowledge_base.py [--extractors a,b,c]
│   ├── generate_hormuz_benchmark.py    # Phase 11 (R3) — writes the 20 scenario parquets
│   ├── run_tier0_baselines.py          # Phase 11 (R4) — random/always/never-alarm
│   ├── run_tier1_baselines.py          # Phase 11 (R5) — Z-score/EWMA/CUSUM/SARIMA/persistence
│   ├── run_tier2_baselines.py          # Phase 11 (R6) — Isolation Forest, Matrix Profile
│   ├── run_ablations.py                # Phase 11 (R7) — A0-A7, seed=42 only
│   ├── aggregate_ablation_results.py   # Phase 11 (R7) — ablation summary table
│   ├── aggregate_all_results.py        # Phase 11 (R8) — Tier 0-2 + ablations -> CSVs
│   └── generate_results_summary.py     # Phase 11 (R8) — results/benchmark_summary.md
├── docs/
│   └── ABLATION_RATIONALE.md    # Phase 11 (R7) — why domain scorers, not production agents
├── results/                     # Phase 11 (R4-R8) — benchmark outputs (gitignored except tier0)
│   └── baselines/
│       └── tier0/                   # committed exception — random/always/never-alarm result JSONs
├── airflow/                    # Phase 10 — local daily orchestration (Docker Compose)
│   ├── Dockerfile                  # apache/airflow:2.10.5 + pipeline deps (CPU-only torch)
│   ├── docker-compose.yaml         # LocalExecutor stack; project root bind-mounted rw into the containers
│   ├── .env                        # AIRFLOW_UID / AIRFLOW_TZ / UI login — deliberately secret-free
│   └── dags/
│       └── supply_chain_daily.py   # 08:00 daily: ingest → rag_populate(fred,ambee) → evaluate → publish marker; retries=0 hard-stop
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
│   ├── test_dashboard.py       # Phase 9b/10 — 15 tests: no-raw-scores scan, route sync, JPEG export, region parameterization, freshness marker
│   ├── test_fred_api.py        # standalone (non-pytest) FRED connectivity diagnostic, run by hand
│   ├── test_benchmark_regions.py       # Phase 11 (R1) — region + scenario spec loading, silent-agent signal
│   ├── test_silent_agent_handling.py   # Phase 11 (R2) — SilentAgentDetector, WeightRenormalizer, aggregate_detections()
│   ├── test_scenario_batch_generator.py # Phase 11 (R3) — wraps R1, integer labels, unclipped signals
│   ├── test_tier0_baselines.py         # Phase 11 (R4) — control baselines + BaselineEvaluator
│   ├── test_tier1_baselines.py         # Phase 11 (R5) — SPC baselines, validation-tuned hyperparameters
│   ├── test_tier2_classical.py         # Phase 11 (R6) — IForest/Matrix Profile shape, determinism, NaN handling
│   ├── test_ablations.py               # Phase 11 (R7) — A0-A7 config, agreement bonus, domain scorers
│   └── test_results_aggregation.py     # Phase 11 (R8) — regex/name-extraction fixes, mean/std aggregation
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

**Phase 11 (EVAL01) additions:** `statsmodels` (SARIMA baseline, R5), `stumpy` (Matrix Profile baseline, R6), `tabulate` (`DataFrame.to_markdown()` for `results/benchmark_summary.md`, R8). `optuna` (already listed above) is reused for the R7 ablation weight search.

**Phase 9b additions:** `streamlit` (dashboard). Optional: `anthropic` — only needed if you set `ANTHROPIC_API_KEY` to enable LLM-generated risk explanations on the Decision view (a compositional fallback is used otherwise).

**Phase 10 additions:** none on the host — Airflow and the pipeline deps it needs live entirely inside the Docker image (`airflow/Dockerfile`). The only host prerequisite is **Docker Desktop** (WSL2 backend on Windows).

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
python main.py
```

Expected output:
```
2026-04-25 17:17:33 | INFO | __main__ | Pipeline initialized
Pipeline initialized
```

### Generate the synthetic datasets

```bash
python -c "from src.ingestion import ShippingConnector; ShippingConnector(config={}).save_raw()"
python -c "from src.ingestion import MarketConnector; MarketConnector(config={}).save_raw()"
```

The first command writes `data/raw/shipping_hormuz.csv` (365 rows) and prints the Welch t-statistic separating normal vs. disruption vessel counts. The second writes `data/raw/market_data.csv` (365 rows) with Brent crude, trade volume, and freight rate signals lag-aligned to the shipping disruption windows.

### Populate the RAG knowledge base from live APIs (Phase 7)

```bash
python scripts/populate_knowledge_base.py                       # all extractors in extraction.enabled_extractors
python scripts/populate_knowledge_base.py --extractors serpapi   # one-time historical backfill only
```

Extracts → deduplicates by document id → backs up to `data/knowledge_base/live_extracted_backup.json` (gitignored) → upserts into the `live_extracted_context` ChromaDB collection. Safe to re-run; missing API keys degrade individual extractors to zero documents rather than failing the run.

A full live run populated **531 documents** across all five extractors (newsapi 265, serpapi 169, fred 16, acled 80, ambee 1) over the four chokepoints, with `documents_stored == documents_deduplicated` and no errors. That run also surfaced and fixed a cross-region de-duplication bug in `acled_extractor.py`: countries shared between chokepoints (Saudi Arabia in `hormuz` + `red_sea`, Egypt in `red_sea` + `suez`) previously produced identical `acled_{country}_{year}` document ids, so the second region's rows were silently dropped by the deduplicator — the id is now region-scoped (`acled_{region}_{country}_{year}`).

### API server

```bash
uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`.

### Dashboard (Phase 9b)

```bash
streamlit run src/dashboard/app.py
```

Opens at `http://localhost:8501`. The first load takes ~1 minute (split generation, agent fitting, SHAP surrogate training, RAG index) — everything is cached afterwards. The **Decision View** is the default page; switch to the **Analysis View** from the sidebar for the full thesis evidence and per-chart JPEG export.

### Daily orchestration (Phase 10)

```bash
cd airflow
docker compose up -d --build
```

Airflow UI at `http://localhost:8080` (login `airflow`/`airflow`) — unpause `supply_chain_daily` once and it runs the full chain daily at 08:00 local time (`AIRFLOW_TZ` in `airflow/.env`); trigger manual runs from the UI or with `docker compose exec airflow-scheduler airflow dags trigger supply_chain_daily`. The dashboard reflects a completed run via its "Data refreshed: …" badge within its 24 h cache TTL (or immediately on a fresh `streamlit run`). See the Phase 10 section for the hard-stop semantics and the container→host volume bridge.

### Tests

```bash
pytest tests/ -v
```

The full suite is **250 tests / 250 passing** across 13 collected test files (`test_fred_api.py` is a standalone diagnostic, not collected by pytest). Run the agent evaluations with output:

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

