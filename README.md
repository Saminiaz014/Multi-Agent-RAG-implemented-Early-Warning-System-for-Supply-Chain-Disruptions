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
│   Config-driven weights · RiskLevel enum (CRIT/HIGH/MED/LOW) │
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

## What Was Built

This session delivered the **complete project scaffold** — every module is independently importable and tested, forming the foundation for domain-specific agent and connector implementations.

### Core Infrastructure

| Module | File | Description |
|---|---|---|
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
| `tests/test_risk_engine.py` | CRITICAL/HIGH/MEDIUM/LOW boundary cases, unknown-agent skipping, weight renormalisation |
| `tests/test_scenarios.py` | End-to-end orchestrator runs with synthetic normal vs. disrupted Hormuz signal data |

---

## Project Structure

```
supply-chain-dss/
├── config/
│   └── settings.yaml           # agent toggles, weights, thresholds, RAG, API, logging
├── data/
│   ├── raw/                    # raw CSV ingestion data (populate per connector)
│   ├── processed/              # cleaned, feature-ready DataFrames
│   └── knowledge_base/         # historical disruption cases as JSON
├── src/
│   ├── ingestion/
│   │   └── base_connector.py   # ABC for all data source connectors
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
| `thresholds.risk_critical` | `0.8` | Composite score cutoff for CRITICAL risk |
| `thresholds.risk_high` | `0.7` | Composite score cutoff for HIGH risk |
| `thresholds.risk_medium` | `0.4` | Composite score cutoff for MEDIUM risk |
| `rag.collection_name` | `disruption_cases` | ChromaDB collection name |
| `rag.top_k` | `3` | Number of historical precedents to retrieve |

---
