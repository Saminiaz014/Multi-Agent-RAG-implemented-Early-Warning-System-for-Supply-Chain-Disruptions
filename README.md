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
│                   FastAPI Layer                         │
│   POST /predict  ·  POST /explain  ·  GET /health       │
└─────────────────────────────────────────────────────────┘
```

---

## What Was Built (Week 1)

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

## What Was Built (Week 2)

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

### Test Coverage (Week 2)

| Test file | Coverage |
|---|---|
| `tests/test_ingestion.py` | 30 tests: 14 for `ShippingConnector` (schema, NaN/range checks, ~46 disruption days, scenario window placement, Welch t > 5, seed reproducibility, `validate()` happy/sad path, unified-signal JSON schema, CSV persistence) + 16 for `MarketConnector` (schema, baseline ranges, ground-truth alignment with shipping, lagged response peak, mean-reverting decay, seed reproducibility, validate happy/sad path, signal-record schema, CSV persistence, and the cross-source Pearson r > 0.5 correlation check) |

All 30 ingestion tests pass alongside the 16 prior-week tests — **46/46 total**.

---

## Project Structure

```
supply-chain-dss/
├── config/
│   └── settings.yaml           # agent toggles, weights, thresholds, RAG, API, logging
├── data/
│   ├── raw/                    # raw CSV ingestion data (populate per connector)
│   │   ├── shipping_hormuz.csv # synthetic Hormuz dataset (Week 2 artefact)
│   │   └── market_data.csv     # synthetic Brent / trade volume / freight data (Week 2 artefact)
│   ├── processed/              # cleaned, feature-ready DataFrames
│   └── knowledge_base/         # historical disruption cases as JSON
├── src/
│   ├── ingestion/
│   │   ├── base_connector.py   # ABC for all data source connectors
│   │   ├── shipping_connector.py # synthetic Hormuz AIS data with ground-truth disruptions
│   │   └── market_connector.py # synthetic Brent / trade volume / freight data, lag-aligned to shipping
│   ├── agents/
│   │   └── base_agent.py       # ABC + DetectionResult dataclass
│   ├── aggregation/
│   │   └── risk_engine.py      # weighted composite risk scoring
│   ├── explainability/
│   │   └── shap_explainer.py   # SHAP Tree/Kernel explainer wrapper
│   ├── rag/
│   │   └── context_retriever.py # ChromaDB similarity search
│   ├── api/
│   │   └── endpoints.py        # FastAPI /predict, /explain, /health
│   └── orchestrator.py         # main pipeline runner
├── tests/
│   ├── test_agents.py
│   ├── test_ingestion.py       # shipping + market connector schema, ranges, separation, cross-source correlation
│   ├── test_risk_engine.py
│   └── test_scenarios.py
├── logs/                       # pipeline execution logs (gitignored)
├── notebooks/                  # exploration and evaluation notebooks
├── requirements.txt
├── main.py                     # entrypoint
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `shap`, `chromadb`, `sentence-transformers`, `fastapi`, `uvicorn`, `pyyaml`, `plotly`, `pytest`, `httpx`.

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

### API server

```bash
uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`.

### Tests

```bash
pytest tests/ -v
```

---

## Configuration Reference (`config/settings.yaml`)

| Key | Default | Description |
|---|---|---|
| `agents.shipping.enabled` | `true` | Toggle shipping agent on/off |
| `agents.shipping.detection_method` | `isolation_forest` | Algorithm for shipping anomaly detection |
| `agents.shipping.contamination` | `0.1` | Expected anomaly fraction for Isolation Forest |
| `agents.shipping.threshold` | `0.65` | Minimum score to raise a shipping flag |
| `agents.market.enabled` | `false` | Toggle market agent |
| `agents.market.z_threshold` | `2.5` | Z-score cutoff for market anomalies |
| `weights.shipping` | `0.4` | Contribution weight in composite score |
| `weights.market` | `0.3` | Contribution weight in composite score |
| `weights.geopolitical` | `0.3` | Contribution weight in composite score |
| `thresholds.risk_high` | `0.7` | Composite score cutoff for HIGH risk |
| `thresholds.risk_medium` | `0.4` | Composite score cutoff for MEDIUM risk |
| `rag.collection_name` | `disruption_cases` | ChromaDB collection name |
| `rag.top_k` | `3` | Number of historical precedents to retrieve |

---

