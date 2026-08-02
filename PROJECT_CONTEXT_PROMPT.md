# PROJECT CONTEXT PROMPT — supply-chain-dss

> Paste this entire document as the opening/system context in any workspace (Claude, ChatGPT, Cursor, a claude.ai Project, etc.) to give the assistant full working knowledge of the project. It describes the architecture, every major module, configuration, conventions, and known gotchas.

---

## 1. What this project is

**`supply-chain-dss`** is a Master's-thesis-grade Python project: a **multi-agent Decision Support System (DSS) for detecting and explaining supply chain disruptions in the Strait of Hormuz maritime corridor** (with secondary chokepoints: Red Sea / Bab el-Mandeb, Malacca, Suez).

- **Academic goal:** explainable AI for geopolitical supply chain risk. Rigour and explainability matter more than production speed. Treat every change as an academic deliverable.
- **Core pipeline:** `Signals → Detection → Validation → Context → Risk → Explanation → Decision`
- **Stack:** Python 3.10+ (venv at `.venv/`, currently Python 3.13), pandas, scikit-learn, SHAP, Optuna, ChromaDB, FastAPI, Streamlit, Apache Airflow 2.10.5 (Docker Compose), pytest (**247 tests, all passing**).
- **Conventions:** Google-style docstrings, type hints on all function signatures, config-driven behavior via `config/settings.yaml`. All agents extend the `BaseAgent` ABC; all data connectors extend the `BaseConnector` ABC; all RAG extractors extend the `BaseExtractor` ABC.

---

## 2. Repository layout

```
supply-chain-dss/
├── main.py                      # CLI entry: runs full pipeline, prints JSON + summary box
├── config/
│   ├── settings.yaml            # THE central config (agents, weights, thresholds, APIs, RAG)
│   ├── optimized_weights.yaml   # Optuna-tuned weights (weight_mode: "optimized")
│   └── optimized_weights_2agent_backup.yaml
├── src/
│   ├── orchestrator.py          # Orchestrator — wires everything together
│   ├── agents/                  # 6 detection agents + BaseAgent ABC
│   ├── ingestion/               # 6 data connectors + BaseConnector ABC
│   ├── aggregation/risk_engine.py       # weighted risk aggregation
│   ├── explainability/shap_explainer.py # SHAP surrogate explainer
│   ├── rag/context_retriever.py         # ChromaDB historical-context RAG
│   ├── extractors/              # live-API extractors that populate the RAG KB
│   ├── optimization/            # Optuna weight optimization
│   ├── evaluation/decision_effectiveness.py
│   ├── api/endpoints.py         # FastAPI service (10 endpoints)
│   └── dashboard/               # Streamlit two-view dashboard
├── airflow/                     # Docker Compose local Airflow (Phase 10)
│   ├── dags/supply_chain_daily.py
│   └── .env                     # AIRFLOW_TZ etc. (secret-free, force-added past .gitignore)
├── scripts/populate_knowledge_base.py   # one-shot live RAG population
├── data/
│   ├── raw/                     # CSVs: shuaiba_arrivals, brent_crude, freight_ppi, ...
│   ├── processed/               # gitignored — pipeline outputs, optimization results, last_updated.txt
│   └── knowledge_base/          # disruption_cases.json (10 curated) + live_extracted_backup.json (553 docs)
└── tests/                       # 14 test modules, 247 tests
```

---

## 3. The six agents (`src/agents/`)

All extend **`BaseAgent`** (ABC in `base_agent.py`). Each agent consumes a pandas frame from its connector, runs anomaly detection, and emits a normalized [0,1] risk score plus per-feature signals.

| Agent | File | Detection method | Key config (settings.yaml `agents.*`) |
|---|---|---|---|
| **Shipping** | `shipping_agent.py` | Isolation Forest (contamination 0.1) | threshold 0.65; vessel-arrival counts at Shuaiba port, Kuwait, by vessel type |
| **Market** | `market_agent.py` | Z-score (z_threshold 1.5, 5-yr baseline) | threshold 0.50; Brent crude + freight PPI + freight services |
| **Geopolitical** | `geopolitical_agent.py` | Weighted composite | threshold 0.5; weights: sanctions .35, military .25, diplomatic .25, stability .15; lead_days 3 |
| **Natural disaster** | `disaster_agent.py` | Weighted composite + proximity decay | threshold 0.30; weights: earthquake .35, tsunami .30, cyclone .20, severe_weather .15; proximity decay from (26.5N, 56.5E), full weight ≤500 km, decay to 1500 km |
| **Routing** | `routing_agent.py` | Isolation Forest (contamination 0.08) + transit z-score | threshold 0.55; min_rerouting_pct 10; weights: model_score .6, transit_zscore .4 |
| **News sentiment** | `news_agent.py` | Sentiment threshold (VADER) | negative_threshold −0.30, consensus 0.40, volume spike ×2; weights: sentiment .40, consensus .25, velocity .20, volume .15 |

Each domain agent supports `data_mode: "synthetic" | "csv" | "api"` (currently synthetic for geo/disaster/routing/news; shipping & market read real CSVs).

---

## 4. Data connectors (`src/ingestion/`)

All extend **`BaseConnector`** (ABC in `base_connector.py`): `shipping_connector`, `market_connector`, `geopolitical_connector`, `disaster_connector`, `routing_connector`, `news_connector`. Each produces the daily frame its agent consumes. Shipping + market run on a **merged daily frame**; the 4 domain agents each run on their **own connector frame** (Orchestrator `_frame_for_agent`). Source modes and CSV paths are set in `settings.yaml → ingestion` / per-agent `csv_path`.

---

## 5. Orchestrator (`src/orchestrator.py`)

`Orchestrator.run_full_pipeline()` is the single pipeline entrypoint:

1. `_build_enabled_agents()` auto-registers enabled agents **only when none were registered manually** (preserves `register_agent` unit tests).
2. Runs all 6 agents (graceful degradation: one failing agent/connector is logged and skipped).
3. Aggregates via `RiskEngine` (see §6).
4. Trains/caches a SHAP surrogate once per Orchestrator instance (`self._shap_explainer`) and attaches `output["explanation"]`.
5. Queries RAG after SHAP and populates `output["historical_context"]` (try/except guarded).
6. Output carries **both** legacy keys (`composite_score`, `agent_scores`) and rich keys (`risk_score`, `contributing_agents`), plus `metadata` (`agents_active`, `data_modes`, `weight_mode`).

`main.py run_pipeline()` delegates to `run_full_pipeline()` and prints JSON + a summary box. **Important:** `main.py` exits 0 even on pipeline failure (defensive catch) — the Airflow DAG greps stdout for `"PIPELINE FAILED"` and promotes it to exit 1. Do not remove that guard.

---

## 6. Risk aggregation (`src/aggregation/risk_engine.py`)

Weighted composite of the 6 agent scores. Inter-agent weights (settings.yaml `weights:`): shipping .25, geopolitical .25, market .15, routing .15, natural_disaster .10, news_sentiment .10. Risk bands (`thresholds:`): critical ≥0.8, high ≥0.6, medium ≥0.4, low ≥0.2. Includes a **multi-agent agreement bonus** (multiple agents firing raises the composite). `weight_mode: "hand_tuned"` is the default; `"optimized"` loads `config/optimized_weights.yaml`.

---

## 7. Explainability (`src/explainability/shap_explainer.py`)

- **`SurrogateShapExplainer`** + **`build_shap_training_data()`**: a surrogate model over the **20-feature, 6-agent space**, R² = 0.991 on 364 synthetic days. Lazy-trained once per Orchestrator instance.
- `output["explanation"]` carries `top_drivers` (top 3: feature / agent / shap_value), `text`, `surrogate_r2`, `expected_value`.
- Raw feature values come from **connector frames** (not scaled agent outputs); the market agent has a 1-row offset handled via `iloc[-n:]`.
- Phase-4 depth methods: `compare_explanations()`, `compute_faithfulness()`, `generate_comparison_plot()`.
- **Gotcha:** `compute_faithfulness` test fixtures need **disjoint feature sets per scenario type** — a shared feature becomes a universal RF split and tanks faithfulness to ~0.7.

---

## 8. RAG historical context (`src/rag/context_retriever.py`)

- **ChromaDB, fully local, no API keys** for retrieval. Embeddings: ChromaDB's `DefaultEmbeddingFunction` (ONNX all-MiniLM-L6-v2; cache at `~/.cache/chroma/`) — deliberately avoids a HuggingFace network dependency.
- Two collections: `disruption_cases` (**10 curated historical cases** covering all 6 signal domains, `data/knowledge_base/disruption_cases.json`) and `live_extracted_context` (**553 API-extracted docs**, backup at `data/knowledge_base/live_extracted_backup.json` — the canonical backup).
- Key methods: `build_index()` (rebuilds when case count changes), `query(signals_dict)` (6-domain profile → natural-language query), `query_gated()` (only fires when composite ≥ `rag.composite_threshold` 0.65, min cosine similarity 0.55), `format_context()`, `evaluate_retrieval_quality()`.

---

## 9. Live extractors (`src/extractors/`)

All extend **`BaseExtractor`**. Populate the live RAG collection via `scripts/populate_knowledge_base.py` (full backfill) or the Airflow DAG (daily quota-safe subset).

| Extractor | Source | Notes / known limitations |
|---|---|---|
| `newsapi_extractor` | NewsAPI | Free tier: no historical date filter (HTTP 426); current-news chokepoint queries work. 265 docs in backfill |
| `serpapi_extractor` | SerpAPI Google News | Date-unbounded — used for historical backfill (169 docs) |
| `ambee_extractor` | Ambee disasters | `/history` returns HTTP 400 on free tier (30-day cap); falls back to `/latest` (current events only). Categorical→numeric severity mapping in settings.yaml |
| `fred_extractor` | FRED economic data | 16 docs; official limit 120/min |
| `acled_extractor` | ACLED conflict data | 80 docs. **Fixed bug:** doc_id must be `f"{region}_{country}_{year}"` (was country_year — cross-region dedup silently dropped Saudi Arabia/Egypt docs) |
| `reliefweb_extractor` | ReliefWeb | Kept as fallback; gated by unapproved appname, not in enabled list |
| `aisstream_monitor` | aisstream.io WebSocket | Live-only vessel monitoring; not used for historical RAG. Disabled by default |

Backfill result (2026-07-04): 611 raw → 531 deduplicated → stored; daily Airflow runs have grown it to 553.

---

## 10. Optimization (`src/optimization/`)

- Files: `weight_optimizer.py`, `pipeline_evaluator.py`, `weight_config.py`, `data_split.py`, `optimization_analysis.py`.
- **Optuna TPE** (seed 42; split seeds 42/43/44 → fully deterministic, re-runs reproduce best trial 26 byte-identically), 100 trials, median pruner.
- Objective: `F1·0.5 + lead_time·0.3 − FPR·0.2`. Parameter space covers all 6 inter-agent weights, intra-agent weights, and thresholds (detection params off by default).
- Result: test objective 0.638 → **0.766**, lead time 2.67 → **5.0 days**, raw test F1 0.956 → 0.935 (accepted multi-objective trade-off).
- Outputs in `data/processed/` are **gitignored**; only `config/optimized_weights.yaml` is committed. JSON keys are `lead_time_days` / `lead_time_score` (NOT `lead_time`).

---

## 11. Evaluation (`src/evaluation/decision_effectiveness.py`)

Decision-effectiveness metrics for the thesis evaluation chapter: detection F1, lead time (days of early warning before a labeled disruption), false-positive rate, per-scenario breakdowns. `PipelineEvaluator._aggregate_daily` applies the agreement bonus consistently with the live engine.

---

## 12. FastAPI service (`src/api/endpoints.py`)

10 endpoints: `/health`, `/predict`, `/explain`, `/agents`, `/agents/toggle`, `/weights`, `/weights/switch`, `/optimization/results`, `/populate`, `/status`. Module-level state with `_get_orchestrator()` lazy init; CORS `allow_origins=["*"]`. Serves on host 0.0.0.0:8000 (settings.yaml `api:`). Tests mock via `patch.object(_ep, "_get_orchestrator", return_value=mock_orch)`.

---

## 13. Streamlit dashboard (`src/dashboard/`)

- `app.py` + `pages/1_Decision_View.py` + `pages/2_Analysis_View.py`, logic in `decision_view.py` / `analysis_view.py` / `core.py`.
- **Decision View**: executive risk summary — status, drivers, historical context, recommendations. Deliberately shows **no raw model scores** (a regex matcher `\d+\.\d{2,}` in tests enforces this — keep any numbers in the Decision view decimal-free).
- **Analysis View**: full per-agent scores, SHAP drivers, optimization results, retrieval diagnostics.
- Artifact loaders cache with `ttl=86400`; `core.read_last_updated()` renders a "Data refreshed:" badge from `data/processed/last_updated.txt` (written by the Airflow DAG's final task).

---

## 14. Airflow orchestration (`airflow/`, Phase 10)

- **Docker Compose**, image `apache/airflow:2.10.5` + pipeline deps, LocalExecutor, Postgres. UI at `localhost:8080` (airflow/airflow).
- DAG **`supply_chain_daily`**: `ingest_and_detect → rag_populate → evaluate → publish_marker`, cron `0 8 * * *` in `AIRFLOW_TZ` (airflow/.env, default Europe/Berlin). `retries=0` hard-stop — a task failure marks downstream `upstream_failed` (verified).
- `publish_marker` writes `data/processed/last_updated.txt` (ISO timestamp) — the dashboard freshness badge source. Marker only written when the whole chain succeeds.
- Project root is **bind-mounted read-write** at `/opt/airflow/supply-chain-dss` — container-written `data/processed/` + ChromaDB reach the host dashboard directly.
- Daily RAG uses quota-safe `DAILY_EXTRACTORS="fred,ambee"`; newsapi/serpapi/acled are **manual-backfill-only** (free-tier caps).
- Dockerfile preinstalls **CPU-only torch** (`--index-url` PyTorch CPU) — otherwise sentence-transformers drags in ~2.5 GB of CUDA wheels.
- `airflow/.env` is force-added past the `.env` gitignore pattern; it is secret-free by design.
- Known cosmetic artifact: the container rewrites `live_extracted_backup.json` with LF line endings, so git may show it "modified" after each run with no content change.

---

## 15. Configuration model (`config/settings.yaml`)

Single source of truth. Top-level keys: `weight_mode`, `optimization`, `ingestion`, `agents` (per-agent enable/method/thresholds/weights/data_mode/api), `weights` (inter-agent), `thresholds` (risk bands), `rag`, `api_keys` (all `${ENV_VAR}` placeholders — never commit real secrets), `extraction` (enabled extractors, chokepoint countries + bounding boxes, per-API rate limits, historical range 2007–2025), `aisstream`, `api`, `logging`. Monitoring chokepoints: **hormuz** (primary), red_sea, malacca, suez — each with named monitoring points and lat/lng.

---

## 16. Testing

`pytest` from the project root with the venv active — **247 tests, all must pass**. Modules: `test_agents`, `test_new_agents`, `test_ingestion`, `test_risk_engine`, `test_scenarios`, `test_optimization`, `test_shap_6agent`, `test_rag_6domain`, `test_api_6agent`, `test_phase4_depth`, `test_extractors`, `test_evaluation`, `test_dashboard`, `test_fred_api`.

---

## 17. How to run things

```bash
# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1

python main.py                          # full 6-agent pipeline, JSON + summary box
python main.py --optimize               # Optuna weight optimization
python scripts/populate_knowledge_base.py   # full RAG backfill (needs API keys in .env)
uvicorn src.api.endpoints:app --port 8000   # FastAPI service
streamlit run src/dashboard/app.py      # dashboard
pytest                                  # 247 tests
docker compose up -d                    # (in airflow/) start scheduler+webserver+postgres
```

---

## 18. Working agreements for the assistant

1. This is an **academic thesis deliverable** — prefer rigour, explainability, and reproducibility (fixed seeds) over production shortcuts.
2. Follow existing patterns: extend the relevant ABC (`BaseAgent` / `BaseConnector` / `BaseExtractor`), config through `settings.yaml`, Google-style docstrings, full type hints.
3. Never break the dual output contract of `run_full_pipeline()` (legacy + rich keys) or the `main.py` exit-0 guard.
4. Keep the Decision view free of raw decimal scores; keep `data/processed/` out of git.
5. All 247 tests must pass before a change is considered done.
