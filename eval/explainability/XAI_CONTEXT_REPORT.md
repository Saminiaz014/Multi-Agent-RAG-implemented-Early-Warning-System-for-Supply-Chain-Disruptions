# XAI Context Report — `supply-chain-dss`

Generated 2026-10-14. This was a read-only inspection: this file is the only thing created. ChromaDB was queried from a scratchpad **copy** of `data/knowledge_base/.chromadb`, and every model fit ran in memory.

> **Read this first: the brief's assumptions vs. the code**
> - There are **5 agents, not 6**. The Routing agent was deleted in `b2306ae` ("Remove routing agent").
> - The SHAP surrogate uses **17 features, not 20**. Re-running it now gives a train R² of **0.983, not 0.991**. The 0.991 figure dates from the old 20-feature, 6-agent version (`DEVELOPMENT_LOG.md:1091`).
> - No **P_LOW** scenario exists anywhere. P_CRIT, P_HIGH, N_QUIET and N_DECOY survive only in the archived `docs/eval01-archived/`; no live code uses them.
> - There are **no trained model files** (no `.pkl`, `.joblib` or `.onnx` model). Every agent and the surrogate are refitted when they run.
> - `thesis/figures/` **does not exist**.
> - The `anthropic` package is **not installed** in `.venv`, so every LLM explanation path silently falls back to composed text.

---

## 1. Repo snapshot

| Item | Value |
|---|---|
| Repo root | `C:\Users\samin\OneDrive\Documents\Thesis\supply-chain-dss` (the parent `Thesis/` is not a repo) |
| Branch | `main` |
| HEAD | `b2306aed0e4a72614f20f3c1440a5bee5264f4f5` — "Remove routing agent" (2026-10-11) |
| `88c4274` in history? | **Yes.** It is an ancestor of HEAD: "Extend method comparison: trivial controls, matrix profile, circularity tagging" |
| Uncommitted changes | `git status --short` shows only `?? experiments/` (untracked ablation_v2 grid). Nothing is modified. |
| Test count | `.venv/Scripts/python.exe -m pytest --collect-only -q` → **410 tests collected**. The system Python 3.14 run gives 121 tests and 4 errors because `python-dotenv` is missing there, so use `.venv` (Python 3.13.2). |

Recent log:
```
b2306ae Remove routing agent
98213f2 Re-run the 100-trial optimization; record results in 4.9 and close the defect
13ad7d9 Expand Methodology to full depth, verified against the codebase
b3a3546 Restructure README around thesis chapters; report measured results as found
2f6d1d9 Document Phases 13-15 in README; correct stale data-source claims
c02d088 Add thesis writing brief and risk scoring reference
88c4274 Extend method comparison: trivial controls, matrix profile, circularity tagging
```

### Trees (depth ≤ 3; `__pycache__` and Chroma internals omitted)
```
src/
  agents/        base_agent.py disaster_agent.py geopolitical_agent.py market_agent.py news_agent.py shipping_agent.py
  aggregation/   risk_engine.py
  api/           endpoints.py region_endpoints.py
  baselines/     (empty — only __pycache__)
  core/          config_manager.py regions.py
  dashboard/     app.py core.py decision_view.py analysis_view.py llm_explanations.py cache.py vessel_data.py
                 pages/1_Decision_View.py pages/2_Analysis_View.py
  evaluation/    decision_effectiveness.py
  explainability/shap_explainer.py
  extractors/    acled, aisstream_monitor, ambee, base, disaster_combined, fred, gdacs, knowledge_base_builder,
                 newsapi, portwatch, reliefweb, serpapi, usgs (_extractor.py)
  ingestion/     base_connector.py disaster_ geopolitical_ market_ news_ shipping_connector.py
  optimization/  data_split.py optimization_analysis.py pipeline_evaluator.py weight_config.py weight_optimizer.py
  rag/           context_retriever.py
  utils/         (empty — only __pycache__)
  orchestrator.py
eval/
  COMPARISON_REPORT.md
  method_comparison_results.csv
  graphs_ablation_tiers/   B_tier_progression_{bab_el_mandeb,hormuz,panama}.png
  graphs_method_comparison/ A1_auc_by_method.png A2_false_positive_harness.png C_ranking_{3 regions}.png
                            D_auc_heatmap.png E_f1_vs_fpr.png
data/
  eval/            {hormuz,bab_el_mandeb,panama,malacca}.csv + .manifest.json
  knowledge_base/  .chromadb/ (+ two _test_ dirs) decision_labels.json disruption_cases.json
                   live_extracted_backup.json live_extracted_backup.pre_region_migration_20260823_215314.json
  processed/       evaluation_results.json optimization_results*.json shap_*.png  (gitignored)
  raw/             brent_crude, disaster_events, freight_ppi, freight_services, geopolitical_events, market_data,
                   market_processed, news_sentiment, routing_data, shipping_hormuz, shipping_processed, shuaiba_arrivals (.csv)
thesis/
  1_0_Introduction.tex … 6_0_Conclusion_and_Outlook.tex, A_Appendix.tex, FAPS-Thesis.tex (+ build artefacts)
  Bilder/ (template logos/cover only), CV/, Verzeichnis_Literatur.bib
  (no figures/ directory)
Dashboard: src/dashboard/  (Streamlit multipage: app.py + pages/)
```

---

## 2. Ensemble score & alerts

The repo has **three different aggregation paths**, and they disagree. This matters for any decomposition.

| Path | Where | Agreement bonus? | Clip? | Used by |
|---|---|---|---|---|
| A. `RiskEngine.compute_risk` | `src/aggregation/risk_engine.py:239-365` | yes (1.15 / 1.25) | `min(·,1)` | `Orchestrator.run_full_pipeline` → `risk_score` (API `/predict`) |
| A′. `RiskEngine.aggregate` (legacy) | `risk_engine.py:142-195` | no | no | `run_full_pipeline` → `composite_score`, which **gates RAG** |
| B. `run_timeseries_analysis` | `src/orchestrator.py:445-458` | no | no | per-day series |
| C. `composite_series` (dashboard) / `PipelineEvaluator._aggregate_daily` | `src/dashboard/core.py:594-610`, `src/optimization/pipeline_evaluator.py:214-235` | yes | `min(·,1)` | Dashboard, Optuna, METRIC 1-8 |
| D. `tier_scores` (real-data eval) | `scripts/run_method_comparison.py:402-426` | **no** | no | `eval/method_comparison_results.csv` |

### Constants and weights
`src/aggregation/risk_engine.py:20-26`
```python
# Per-agent mean score above which an agent is considered "in agreement"
# that a disruption is underway. Drives the non-linear agreement bonus.
_AGREEMENT_THRESHOLD: float = 0.5
# Multiplicative amplification applied to the base weighted risk when
# multiple independent agents corroborate a disruption.
_AGREEMENT_BONUS_3PLUS: float = 1.15
_AGREEMENT_BONUS_5PLUS: float = 1.25
```
`config/settings.yaml:184-192`, with `weight_mode: "hand_tuned"` (line 1):
```yaml
weights:
  shipping: 0.294
  market: 0.176
  geopolitical: 0.294
  natural_disaster: 0.118
  news_sentiment: 0.118
```
The optimized alternative is in `config/optimized_weights.yaml` (Optuna trial 63, 2026-09-05): shipping 0.4015, market 0.0879, geopolitical 0.1092, natural_disaster 0.0948, news_sentiment 0.1766. It sets `agreement_bonus_3: 1.2767`, `agreement_bonus_5: 1.4426`, `risk_high: 0.7657` and `risk_medium: 0.2759`.

### Normalisation, aggregation, bonus, clip — `risk_engine.py:303-334`
```python
weight_total = sum(raw_weights.values())
norm_weights = {name: w / weight_total for name, w in raw_weights.items()}
...
for name, score in scores.items():
    weight = norm_weights[name]
    contribution = weight * score
    base_risk += contribution
...
agreement = sum(1 for s in scores.values() if s > _AGREEMENT_THRESHOLD)
amplification = 1.0
if agreement >= 5:
    amplification = self.agreement_bonus_5
elif agreement >= 3:
    amplification = self.agreement_bonus_3
base_risk *= amplification
risk_score = float(min(base_risk, 1.0))
```
Each agent's input is `float(np.mean(result.anomaly_scores))` (`risk_engine.py:287`). In `run_full_pipeline`, that is the **mean over the whole ingested window**, not the latest day.

### Is the score additively decomposable per agent?
- **Path A (`compute_risk`): No.** Lines 326-334 decide it (`agreement = …`, `base_risk *= amplification`, `min(base_risk, 1.0)`). The multiplier is a step function of how many agents exceed 0.5, and the result is clipped. The score is still **exactly** `amp × Σ wᵢsᵢ` when unclipped, so an exact decomposition is `φᵢ = amp·wᵢ·sᵢ`. The step change in `amp` is an interaction term with no unique per-agent owner. With 5 agents at most, the 5-agent bonus fires only in Hormuz, where all 5 are active.
- **Paths B and D (eval, timeseries): Yes.** The deciding line is `scripts/run_method_comparison.py:421` `total += weight * scores`, followed by `return total / mass`. That is a pure renormalised weighted mean.
- **Path C (dashboard): No**, for the same reason as A (`core.py:604-610`).
- **Inside each agent: No** (see §3). Shipping, disaster, market, geopolitical and news all use `max`, a clip or a sigmoid. A feature-level exact decomposition is therefore impossible without a local linearisation or Shapley over agent inputs.

### Alert definition
| Context | Threshold | Set where | Scope | Granularity |
|---|---|---|---|---|
| Risk bands (A/A′/B/C) | `risk_critical 0.90`, `risk_high 0.69`, `risk_medium 0.51`, `risk_low 0.30` | `settings.yaml:209-213` (`thresholds.*`). Calibrated as p97/p85/p60 of calm days pooled over 4 regions (comment at :194-208) | **Global.** Region overlays in `config/regions/*.yaml` do not override thresholds | Daily |
| `compute_risk` label | `high ≥ risk_high`, `medium ≥ risk_medium` (3 levels) | `risk_engine.py:233-237` | Global | Per run |
| Dashboard "spikes" | upward crossings of medium/high/critical | `core.py:666-710` `detect_risk_spikes` | Global | **Episode-like:** one event per upward crossing |
| Dashboard "sustained" | HIGH for > 5 consecutive days | `core.py:652-658` `sustained_high_flags(min_run=5)` | Global | Run-based |
| RAG trigger | `composite_threshold: 0.65`, applied to the **legacy** `composite_score` (no bonus) | `settings.yaml:221`; `orchestrator.py:340-344` | Global | Per run |
| **Real-data eval alerts** (what the CSV reports) | each method's own **90th percentile on the train window** | `scripts/run_method_comparison.py:131-137` | Per region **and** per method | Daily |

`scripts/run_method_comparison.py:131-137`
```python
scores = np.nan_to_num(np.asarray(scores, dtype=float))
reference = scores if train_scores is None else np.nan_to_num(train_scores)
threshold = float(np.quantile(reference, 0.90))
if len(np.unique(y_true)) < 2:
    return {"auc": np.nan, "f1": np.nan, "fpr": np.nan,
            "alert_rate": float((scores > threshold).mean())}
predicted = (scores > threshold).astype(int)
```

### Alert counts per region × scenario
**Scenario classes P_CRIT, P_HIGH, P_LOW and N_QUIET are NOT FOUND in live code.** I searched `P_CRIT|P_HIGH|P_LOW|N_QUIET|N_DECOY` in `*.py`, `*.yaml`, `*.md` and `*.csv`. Matches exist only in `docs/eval01-archived/` (archived benchmark, including `benchmark/scenario_generator.py`). `P_LOW` exists nowhere, and `config/benchmark/` no longer exists.

The live real-data equivalent is per region on the held-out test window. The counts below are derived from `alert_rate × n_test` and `fpr × n_neg` in `eval/method_comparison_results.csv`, using daily alerts at the train-p90 threshold:

| Region | n_test | test pos | Tier 1 (shipping) alerts / TP / FP | Tier 5 (all available) alerts / TP / FP / FN | B8 RF (best supervised) |
|---|---|---|---|---|---|
| hormuz | 731 | 133 | 102 / 4 / 98 | 181 / 22 / 159 / 111 | 105 / 45 / 60 |
| bab_el_mandeb | 731 | 244 | 588 / 230 / 358 | 495 / 163 / 332 / 81 | n/a: no train positives |
| panama | 730 | 149 | 236 / 139 / 97 | 197 / 106 / 91 / 43 | n/a: no train positives |
| malacca | 731 | **0** | 130 / – / 130 (all FP) | 188 / – / 188 (all FP) | n/a |

Synthetic stream (dashboard, METRIC 8): 365 days with scenarios A (days 60-74), B (150-170) and C (280-290), stored in `core.py:115-119`. Confusion counts are stored in `data/processed/evaluation_results.json` (see §7). That file is dated 2026-07-30, which **predates** routing removal and the shipping level-shift change, so it is stale.

### `DetectionResult` — `src/agents/base_agent.py:10-26`
```python
@dataclass
class DetectionResult:
    agent_name: str
    anomaly_scores: np.ndarray
    anomaly_flags: np.ndarray
    feature_names: list[str]
    metadata: dict = field(default_factory=dict)
```

---

## 3. Agents

Routing is **NOT FOUND**: `src/agents/routing_agent.py` and `src/ingestion/routing_connector.py` were deleted in `b2306ae`. `AGENT_KEYS` lists 5 agents (`src/core/regions.py`).

| Agent | file | class | output range | input features | internal sub-scores | muted / passive? | trained model file |
|---|---|---|---|---|---|---|---|
| Shipping | `src/agents/shipping_agent.py` | `ShippingAgent` | [0,1] | `vessel_count`, `avg_delay_hours`, `congestion_index` (+ optional `tanker_count`, `vessel_count_trend`) | `shock_score` = 0.7·`isolation_score` + 0.3·`max_zscore_norm`; `duration_score` = `level_shift_score` (magnitude × 14-day persistence); **final = max(shock, duration)** | active in all 4 regions | none. IsolationForest(200 trees, rs=42) is fitted at runtime |
| Market | `src/agents/market_agent.py` | `MarketAgent` | [0,1] | `brent_crude_usd`, `trade_volume_index`, `freight_rate_index` (+ `freight_services_pct_change`) | per-feature rolling z; weighted Σ\|z\| / z_threshold, clipped at 1 | **passive: malacca** | none |
| Geopolitical | `src/agents/geopolitical_agent.py` | `GeopoliticalAgent` | (0.047, 0.953) via sigmoid | `sanctions_severity`, `military_activity_index`, `diplomatic_incident_score`, `regime_stability_index` (inverted) | `raw_composite` (renormalised over present columns) → sigmoid(6·(raw−0.5)) | **passive: panama** | none |
| Natural Disaster | `src/agents/disaster_agent.py` | `DisasterAgent` | [0,1] | `earthquake_severity`, `tsunami_risk`, `cyclone_severity`, `severe_weather_index` | `raw_composite`, `max_single_severity`; final = clip(max(composite, max_single)) | **passive: bab_el_mandeb** | none |
| News | `src/agents/news_agent.py` | `NewsAgent` | [0,1] | `sentiment_score`, `sentiment_magnitude`, `source_consensus`, `article_volume`, `recency_weighted_score` | `neg_sentiment_norm`, `consensus_norm`, `velocity_norm`, `volume_spike_norm`; weighted, renormalised | active everywhere. Its data is **missing in the eval set for hormuz, bab_el_mandeb and malacca** (manifest `domains_missing`) | none |
| Routing | — | — | — | — | — | **removed** | — |

Passive and active status comes from `src.core.regions.REGIONS[...].active_agents()`. Separately, `settings.yaml` sets `data_mode: "synthetic"` for geopolitical, natural_disaster and news_sentiment (lines 76, 93, 152). The live orchestrator therefore runs those three agents on synthetic data unless a region overlay or the eval builder overrides it (`scripts/build_eval_dataset.py:148` sets `data_mode = "api"`).

Key scoring lines:

`shipping_agent.py:343-362`
```python
iforest_norm = np.clip(
    (iforest_score - self._iforest_low) / span, 0.0, 1.0
)
max_abs_z = np.max(np.abs(scaled), axis=1)
max_z_norm = np.minimum(max_abs_z / self._z_threshold, 1.0)
shock = self._if_weight * iforest_norm + self._zscore_weight * max_z_norm
...
    duration = self.level_shift_score(
        pd.Series(data["_raw_vessel_count"].to_numpy())
    ).to_numpy()
    combined = np.maximum(shock, duration)
```
`market_agent.py:324-326`
```python
            weighted += weights[col] * np.abs(z)

        out["anomaly_score"] = np.minimum(weighted / self._z_threshold, 1.0)
```
`geopolitical_agent.py:196`: `anomaly_score = 1.0 / (1.0 + np.exp(-6.0 * (raw - 0.5)))`

`disaster_agent.py:135-137`
```python
out["anomaly_score"] = np.clip(
    np.maximum(composite, max_single), 0.0, 1.0
).astype(float)
```
`news_agent.py:209,217`: `weighted = weighted / used` … `out["anomaly_score"] = np.clip(weighted, 0.0, 1.0)`

---

## 4. SHAP / surrogate

- **File:** `src/explainability/shap_explainer.py`. It defines `SurrogateShapExplainer` (`:208`), `build_shap_training_data` (`:509`), `compare_explanations` (`:750`), `compute_faithfulness` (`:1011`) and the legacy `ShapExplainer` (`:100`).
- **Surrogate:** `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)` (`:270`).
- **Features: 17, not 20.** The module docstring and `:37-38` still say "20" and "6 agents", which is stale. The list is `ALL_FEATURE_NAMES` at `:39-63` and the agent mapping is `FEATURE_AGENT_MAP` at `:65-83`:

| # | feature | agent |
|---|---|---|
| 1-3 | vessel_count, avg_delay_hours, congestion_index | shipping |
| 4-6 | brent_crude_usd, trade_volume_index, freight_rate_index | market |
| 7-10 | sanctions_severity, military_activity_index, diplomatic_incident_score, regime_stability_index | geopolitical |
| 11-14 | earthquake_severity, tsunami_risk, cyclone_severity, severe_weather_index | natural_disaster |
| 15-17 | sentiment_score, source_consensus, article_volume | news_sentiment |

  The surrogate does **not** see the inputs that actually drive the shipping and news scores: `_raw_vessel_count` and the level-shift / duration signal, `tanker_count`, `recency_weighted_score`, `sentiment_velocity` and `volume_rolling_30d`.
- **Training data:** fully **synthetic**, one 365-day realisation (2025-01-01 to 2025-12-31, seed 42, 47 disruption days). Market drops one row, leaving **n = 364**. `:550` `conn = ShippingConnector(source_mode="synthetic", config={})`, and likewise for every connector. The agents are `fit_detect`-ed on the same frame (`run_dataframe`).
- **Target:** a plain weighted mean with **no agreement bonus and no clip**. That is a different function from `compute_risk` (path A) and the dashboard's `composite_series` (path C). See `:648-662`:
```python
for agent_name, scored in scored_frames.items():
    ...
    w = float(weights.get(agent_name, _DEFAULT_WEIGHTS.get(agent_name, 1 / 6)))
    scores = scored["anomaly_score"].to_numpy(dtype=float)
    scores = scores[:n]
    ...
    risk_scores += w * scores
    total_weight += w
if total_weight > 0:
    risk_scores /= total_weight
```
- **How R² is computed: in-sample (train), global (one model, not per region).** See `:267-274`:
```python
X = self._align_features(features_df)
y = np.asarray(risk_scores, dtype=float)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

r2 = float(r2_score(y, rf.predict(X)))
self.r2 = r2
```
  **I re-measured it now (in memory, hormuz config):** train R² = **0.9827**. A chronological 70/30 hold-out gives R² = **0.528**, and 5-fold shuffled CV gives R² = **0.868**. The documented 0.991 (`DEVELOPMENT_LOG.md:1091`) belongs to the pre-`b2306ae` 20-feature model.
- **Explainer:** `shap.TreeExplainer(rf)` (`:289`) uses the default `tree_path_dependent` mode. **No background dataset is passed**, so the base value is the RF mean prediction (re-measured `expected_value` = 0.3335). Additivity holds: on the last row, Σφ + E = 0.24459, equal to the RF prediction 0.24459. The true target was 0.2186.
- **Per-alert (local) SHAP values stored? NOT FOUND.** `explain()` returns `shap_values`, `top_drivers` (top 3) and `expected_value` in memory only. I searched for `json.dump`, `to_csv` and `to_parquet` in `src/explainability`, `src/orchestrator.py` and `src/dashboard`; none of them persist SHAP values. `run_full_pipeline` puts `top_drivers`, `expected_value`, `text` and `surrogate_r2` into `output["explanation"]` (`orchestrator.py:316-321`).
- **Explained row vs. trained distribution:**
  - In the orchestrator, the explained row comes from the **live or CSV connector frames** (`_build_shap_features_row`, `orchestrator.py:720-748`, last row), while the model was trained on synthetic data.
  - In the dashboard (`decision_view.py:240`, `:622`; `analysis_view.py:164`), the explained row is `features_df.iloc[day-1]` of the **seed-42 synthetic training frame**. The risk timeline it sits next to is the **seed-44 test realisation** (`core.py:459-490`), so attributions and scores come from different series.
- **Existing SHAP plots** (`data/processed/`, gitignored):
  - `shap_beeswarm_6agent.png` and `shap_waterfall_6agent.png`, both 2026-06-20, `generate_shap_plot`, `:447-475`, dpi 100. Stale: 20-feature era.
  - `shap_comparison_waterfall.png` and `shap_comparison_importance.png`, both 2026-09-17, `generate_comparison_plot`, `:965-996`, dpi 120.
  - The dashboard also renders an on-the-fly per-day SHAP bar (`analysis_view.py:163-178`, export key `m2_shap_day_{day}`).
- **Stored faithfulness:** METRIC 2 = 0.9078 (hand_tuned) and 0.9149 (optimized), on 47 disruption days, from the stale July `evaluation_results.json`.

---

## 5. RAG retriever

- **Class:** `ContextRetriever` (`src/rag/context_retriever.py:66`). Its persistent client is `data/knowledge_base/.chromadb` (`:84-107`). It has two collections:
  - `disruption_cases` (static, from `rag.collection_name`)
  - `live_extracted_context` (from `rag.collections.live_context`)
- **Query functions:**
  - `retrieve(query_text: str, top_k: int | None = None) -> list[dict]` (`:222`)
  - `query(current_signals: dict[str, float], top_k: int | None = None) -> list[dict]` (`:274`). Static collection only; used by the dashboard Decision view (`decision_view.py:650`).
  - `query_gated(current_signals: dict[str, float], composite_risk_score: float, top_k: int | None = None, min_similarity: float | None = None) -> dict | None` (`:320`). Used by the orchestrator. Returns `None` below the `0.65` threshold.
- **Query text is built from agent scores, not features or region.** `_build_query_string` (`:596-613`) concatenates fixed phrases for every agent whose score is above `_SIGNAL_BASELINE = 0.40` (`:37`). The phrases (`_DOMAIN_DESCRIPTIONS`, `:40`) hard-code "Strait of Hormuz". No `where=` region or date filter is applied to any `collection.query` call.
- **Embedding and similarity:** Chroma `DefaultEmbeddingFunction` (ONNX all-MiniLM-L6-v2), collections created with `{"hnsw:space": "cosine"}`. **The similarity score is returned:** `similarity = round(1.0 - dist, 4)` (`:264` and `:381`), and results below `min_similarity` 0.55 are filtered out in `query_gated`.
- **Real example result** from `query_gated` on a copy of the DB, with signals `{shipping .82, market .40, geopolitical .71, natural_disaster .05, news .60}` and composite 0.70. All 3 returned matches came from `live`; no static case cleared 0.55.
```json
{"source": "live",
 "text": "Hormuz crisis side effect: a sharp rise in container shipping rates. Spot container freight rates continue their ascent ...",
 "similarity": 0.7136,
 "metadata": {"event_type": "news_article", "title": "Hormuz crisis side effect: a sharp rise in container shipping rates",
   "countries": "Iran,Oman,United Arab Emirates,Saudi Arabia,Qatar,Bahrain,Kuwait,Iraq", "event_date": "2026-05-30",
   "source_api": "newsapi", "primary_agents": "news_sentiment", "url": "https://www.lloydslist.com/LL1157327/...",
   "source_name": "Lloydslist.com", "severity": "medium", "region": "hormuz"}}
```
  Match #2 was the **same article** with `"region": "bab_el_mandeb"`. The same text is stored once per region, so duplicates take up top-k slots.
- **Legacy `query()` result shape:** `{"id", "document", "distance", "similarity", "metadata"}`. Example top hit: `id="hormuz_2019"` (Hormuz Tanker Attacks 2019-06).

### Metadata keys actually present
- **`disruption_cases` (10 docs):** `event, date, region, primary_agents, duration_days, recovery_days, geopolitical_risk_level, natural_disaster_involved, vessel_count_drop_pct, oil_price_spike_pct` (`_case_to_metadata`, `:578-593`). `region` is free text with 9 distinct values (e.g. "Strait of Hormuz", "Suez Canal") and **does not use** the `hormuz` / `bab_el_mandeb` keys. Dates run from 2007-06 to 2023-10.
- **`live_extracted_context` (4,375 docs):**
  - Every document has `region`, `source_api`, `event_type`, `event_date`, `severity`, `primary_agents` and `countries`.
  - Partial keys: `title`, `url`, `source_name` (3,175); `case_id`, `context` (1,176); `chokepoint` and PortWatch stats (336); `magnitude`, `usgs_event_id`, … (363); `military_events`, `year` (279); FRED `series_id`, `spike_pct` (160); GDACS `alert_level`, … (58).
  - `event_type` has 15 values, including news_article 1,999, historical_news 1,176, earthquake 387, chokepoint_traffic_summary 336, conflict_summary 279 and market_* 160.
  - `event_date` runs from 2007-01-01 to **2026-08-18**; 2,039 docs are dated 2026 or later.
- **`outcome`: NOT FOUND** in either collection. `source` exists as `source_api`, `date` as `event_date` and `event_type` as `event_type`. There is no relevance or ground-truth label.

### Doc count per region × source (`live_extracted_context`)
| region | acled | ambee | fred | gdacs | newsapi | portwatch | serpapi | usgs | **total** |
|---|---|---|---|---|---|---|---|---|---|
| bab_el_mandeb | 144 | – | 45 | 10 | 1035 | 84 | 382 | 153 | **1853** |
| hormuz | 99 | 1 | 49 | 9 | 571 | 84 | 714 | 108 | **1635** |
| malacca | 32 | 1 | 31 | 35 | 359 | 84 | 80 | 102 | **724** |
| panama | 8 | – | 35 | 2 | 34 | 84 | – | – | **163** |

Plus 10 static cases. (Memory notes from July say 531 or 2,865 docs. The DB now holds 4,375.)

---

## 6. LLM explanation

There are **two independent LLM paths**, both in the dashboard. The pipeline and API have none: `run_full_pipeline`'s `explanation.text` comes from a template in `SurrogateShapExplainer.generate_explanation_text`.

| | Spike explanation | Peak narrative |
|---|---|---|
| File | `src/dashboard/llm_explanations.py` (`explain_spike`, `:176-212`) | `src/dashboard/core.py` (`generate_risk_narrative`, `:918-958`) |
| Model | `_MODEL = "claude-opus-5"` (`:36`), `max_tokens=2000`, `output_config={"effort": "low"}` | `"claude-haiku-4-5-20251001"` (`:936`), `max_tokens=160` |
| Inputs | region display name, crossed band, **agent scores** (not SHAP), agreement count | **SHAP top-3 drivers** (from the mismatched seed-42 frame, see §4), agreement, status word, region |
| Caching | module dict `_CACHE[(region, day, level)]`, process lifetime (`:45`, `:193-211`) | `@st.cache_data` on the args |
| Fallback | `_compose()`, deterministic template (`:65-120`) | `_compose_narrative()` (`:898-915`), which still says "**of six** detection agents" (`:913`, `:946`) |
| Output | free text, 2-3 sentences, "no numbers" | free text, ≤2 sentences, "no numbers" |
| Checked against data? | **No.** Only the `stop_reason == "refusal"` path is handled (`:155`) | **No** |

System prompt for the spike explanation (`llm_explanations.py:47-55`):
```python
_SYSTEM = (
    "You explain supply-chain risk signals to a logistics manager. You are "
    "describing the output of a multi-agent anomaly-detection model running on "
    "SYNTHETIC evaluation data. Never name a real incident, organisation, "
    "vessel, or date, and never state that anything happened in the world — "
    "there is no real event behind this score. Explain which detection signals "
    "moved together and what that combination would mean operationally. Two to "
    "three sentences, plain language, no numbers, no jargon, no preamble."
)
```
User prompt (`:198-205`):
```python
prompt = (
    f"Chokepoint: {get_region(region).display_name}. "
    f"Risk crossed into the {level} band. "
    f"Detection agents and their anomaly scores, strongest first: "
    f"{_format_drivers(drivers) or 'none elevated'}. "
    f"{agreement} of the region's agents are above their alert level. "
    "Explain what this pattern of signals means."
)
```
Peak-narrative prompt (`core.py:940-947`, single user turn): "You write one short plain-language paragraph (2 sentences max, no numbers, no jargon) explaining a supply-chain risk alert to a logistics manager. Region: … Status: … Top anomaly drivers from an explainability model: <json top-3> … {agreement} of six detection agents agree. Write only the paragraph."

Status: `anthropic` is **not installed** in `.venv`, so both paths always return the composed text. The two model IDs have not been verified against the current API.

---

## 7. Decision effectiveness (METRIC 8)

`predict_action()` — `src/evaluation/decision_effectiveness.py:240-271` (rules documented at `:213-221`):
```python
level = str(risk_level or "low").lower()
if level == "low":
    return "no_action"
if level == "medium":
    return "monitor"
# risk_level == "high"
top_agent = ""
if top_shap_drivers:
    top_agent = str(top_shap_drivers[0].get("agent", "") or "").lower()
if sustained:
    return "escalate"
if top_agent in _PHYSICAL_AGENTS:          # {"shipping"}
    return "reroute"
if top_agent in _CRISIS_AGENTS:            # {"geopolitical", "natural_disaster"}
    ... if hist_sim > _ESCALATE_SIM_FLOOR and hist_action == "escalate":   # 0.75
        return "escalate"
    return "monitor"
return "monitor"
```
- **Action space:** `["no_action", "monitor", "reroute", "escalate"]` (`:33`).
- **`decision_labels.json` schema:** a flat `{case_id: action}` with **10 rows**: hormuz_2019→escalate, ever_given_2021→reroute, houthi_redsea_2024→escalate, hormuz_mine_threat_2010→monitor, somali_piracy_2011→reroute, japan_earthquake_2011→escalate, covid_port_congestion_2021→monitor, west_coast_port_strikes_2014→reroute, iran_sanctions_2012→escalate, cyclone_gonu_2007→reroute. The labels are **derived by rule** from case text (`derive_case_action`, `:50`), not by human annotation.
- **Day-level ground truth:** synthetic only. `scenario_correct_action` (`:170-198`) maps A→monitor, B→reroute then escalate after more than 5 HIGH days, C→monitor, everything else→no_action.
- **Accuracy per region: NOT FOUND.** METRIC 8 runs only on the single synthetic 365-day test series (`notebooks/evaluation.py:689-884`) and has no region dimension. The stored values (`data/processed/evaluation_results.json`, **2026-07-30, stale**) are:

| mode | overall (365 d) | A | B | C | per-case (10) | naive baseline |
|---|---|---|---|---|---|---|
| hand_tuned | 0.8658 | 0.0667 | 1.0 | 0.0909 | 0.70 | 0.7973 |
| optimized | 0.4904 | 0.1333 | 1.0 | 0.1818 | 0.70 | 0.7973 |

Most of the hand_tuned accuracy comes from 293 correct `no_action` days. Monitor-class recall is 2 of 26.

---

## 8. Evaluation data & splits

**Real-data eval** (`scripts/run_method_comparison.py`):
- Temporal split: the last 30% is test (`_TEST_FRACTION = 0.30`, `:61`; `_split`, `:111-113`).

| region | rows | train | test | train pos | test pos |
|---|---|---|---|---|---|
| hormuz | 2434 | 1703 (2019-01-01 → 2023-08-30) | 731 (2023-08-31 → 2025-08-30) | 26 | 133 |
| bab_el_mandeb | 2434 | 1703 (same) | 731 | **0** | 244 |
| panama | 2433 | 1703 | 730 (→ 2025-08-29) | **0** | 149 |
| malacca | 2434 | 1703 | 731 | 0 | **0** (FP harness) |

  There is **no validation split** on real data. The eval data ends 2025-08-30, so the **2026 Hormuz closure is not in the labelled set**.
- **Synthetic** (`src/optimization/data_split.py`): three independent 365-day realisations, seeds 42 (train), 43 (val) and 44 (test) (`settings.yaml:13-16`). The same scenario days are used in all three, and dates run 2025-01-01 → 2025-12-31.

**`eval/method_comparison_results.csv`:** 76 rows × 12 columns `region, method, kind, family, circularity, auc, f1, fpr, alert_rate, not_applicable, agents, n_agents`.
```
region,method,kind,family,circularity,auc,f1,fpr,alert_rate,not_applicable,agents,n_agents
bab_el_mandeb,M0 random,control,baseline,,0.5036186757329922,0.2018927444794952,0.0841889117043121,0.0998632010943912,,,
bab_el_mandeb,M1 always alert,control,baseline,,0.5,0.0,0.0,0.0,,,
hormuz,Tier 5,multi-agent,tier,high,0.400772,0.140127,0.265886,0.247606,,"shipping,market,geopolitical,natural_disaster",4
```
The last row is abbreviated. Note that "M1 always alert" shows `alert_rate 0.0`: with a constant score, the p90 threshold equals the score and the strict `>` test never fires. This row is mislabelled in effect.

**Ground-truth labels:**
- Real data: column `y_true` (bool) in `data/eval/{region}.csv`. The definition comes from the manifest: *"30-day mean at least 20% below the trailing 365-day median, sustained 14+ days"* on vessel_count. This is the same statistic as `ShippingAgent.level_shift_score`, and is tagged `circularity: high` for every tier.
- Synthetic data: `is_disruption` column from the connectors, exposed via `DataSplitManager.get_ground_truth`.
- Columns in `data/eval/*.csv`: `timestamp, y_true, <domain>__<feature>…`. Only panama has `news_sentiment__*`.

**How synthetic data is flagged:** there is **no column and no filename flag**. The mode comes from the connector config (`source_mode` for shipping and market, `data_mode` for the others) and is reported at runtime in `output["metadata"]["data_modes"]` (`orchestrator.py:248-251`). The dashboard flags it only in caption text (`core.py:507-510` `TIMELINE_SYNTHETIC_CAPTION`; `llm_explanations.py` system prompt). The real eval CSVs are real by construction (`build_eval_dataset.py` forces `data_mode="api"`).

---

## 9. Dashboard

- **Entry point:** `src/dashboard/app.py`. Pages are thin shims: `pages/1_Decision_View.py` → `decision_view.render()` and `pages/2_Analysis_View.py` → `analysis_view.render()`. Shared data and helpers live in `core.py`.
- **Data source:** **synthetic** seed-44 test split, 365 days (`core.compute_timeseries`, `:459-490`), re-indexed onto a rolling date axis ending today (`TIMELINE_AXIS_NOTE`, `:493-499`). It is **not** the real `data/eval` frames.

**Decision View alert display** (`decision_view.py`):
- `_render_trend_chart` (`:519-610`): Plotly line with threshold bands and clickable spike markers from `detect_risk_spikes`.
- `_render_spike_panel` (`:340-388`): LLM or composed text, a "Contributing agents" list of agent scores, and a provenance caption.
- `_render_peak_explanation` (`:619-635`): narrative from SHAP top-3, with a deep link to the Analysis view.
- `_render_decision` (`:638-676`): action chip from `decide_action` (with a rule string), a one-line rationale, and a RAG top-1 (`get_retriever().query`) used for the escalate rule. It names the event when similarity is above 0.6.
- `_render_agent_breakdown` (`:390-447`): per-agent score lines for active agents.
- Also: status words Critical / High / Low (`core.status_word`, `:360-371`), the route list, the map and the vessel panel.

**Analysis View** (`analysis_view.py`): one section per stored metric, M1 to M8 (`_section_1_detection` … `_section_8_decision`, `:106-366`), and `_section_9_explorer` (`:369-540`).
- Explanation-specific pieces:
  - M2 faithfulness bars.
  - The two stored SHAP comparison PNGs.
  - A per-day SHAP top-10 bar (`:163-178`).
  - M6 RAG similarity per scenario.
  - M8 confusion heatmap.
- `export_button` reads the stale `evaluation_results.json`.

**What is shown as explanation today:** agent scores, SHAP top-3 or top-10 (surrogate, different realisation), template or LLM text, rule name plus action, and the top RAG case name. **Not shown:** the surrogate R², the agreement multiplier's contribution, uncertainty or calibration, and counterfactuals.

**Figure export:** `core.export_button` → `fig_to_jpeg` → `plotly.io.to_image(format="jpeg", width=1100, height=620, scale=2)` with the `plotly_white` template and a caption baked in (`core.py:962-1000`; needs kaleido 1.3.0). This is Analysis view only. There is also a CSV download of the summary (`analysis_view.py:537`).

---

## 10. Figures & style

- **Plotting modules:**
  - `scripts/report_method_comparison.py` (matplotlib, Agg backend, **dpi 200**) → `eval/graphs_method_comparison/`, `eval/graphs_ablation_tiers/`
  - `src/explainability/shap_explainer.py` (matplotlib/shap, dpi 100 and 120) → `data/processed/`
  - `src/optimization/optimization_analysis.py` (Optuna plots) → `data/processed/`
  - `experiments/ablation_v2/fuse.py` (dpi 150, untracked)
  - Dashboard Plotly (`template="plotly_dark"` on screen, `plotly_white` for export)
- **Palette:**
  - Dashboard `core.AGENT_COLORS` (`:101-107`): shipping `#1f77b4`, market `#2ca02c`, geopolitical `#d62728`, natural_disaster `#ff7f0e`, news `#17becf` (matplotlib tab10 values).
  - `RISK_COLORS` / `STATUS_COLORS`: green `#2e9e5b`, amber `#d9a514`, red `#d64545`.
  - `analysis_view._MODE_COLORS`: hand_tuned `#1f77b4`, optimized `#ff7f0e`.
  - The matplotlib scripts use the default colour cycle.
- **Fonts:** no `rcParams`, `plt.style`, `sns.set` or font-family settings anywhere. Defaults are used, with ad-hoc `fontweight="bold"` and `fontsize=9`.
- **Shared style file: NOT FOUND.** I searched `*.mplstyle`, `style*.py`, `*plot_style*`, `rcParams` and `set_theme`.
- **`thesis/figures/`: NOT FOUND.** The thesis includes only template graphics (`LogoFAPS`, `BalkenFAPSGruen`, `Biomechgreifer.jpg`, cover). No result figure is `\includegraphics`-ed yet.

---

## 11. Thesis

| File | lines | words (approx, comments stripped) | section-level headings |
|---|---|---|---|
| `1_0_Introduction.tex` | 519 | 4,244 | 15 |
| `2_0_Literature_Review.tex` | 852 | 8,194 | 41 |
| `3_0_Methodology.tex` | 1,657 | 13,175 | 55 |
| `4_0_Evaluation.tex` | 126 | **~660** | 19 |
| `5_0_Discussion.tex` | 92 | 473 | 15 |
| `6_0_Conclusion_and_Outlook.tex` | 56 | 250 | 8 |
| `A_Appendix.tex` | 33 | 114 | 6 |

**`4_0_Evaluation.tex` headings** (12 `\TODO`s; this is a skeleton):
```
\section{Evaluation Framework}
  \subsection{Evaluation Objectives} \subsection{Experimental Setup}
  \subsection{Baseline Approaches}   \subsection{Validity and Threats to Validity}
\section{Performance Metrics}
  \subsection{Detection Lead Time} \subsection{Detection Accuracy}
  \subsection{Explainability Score} \subsection{Decision Effectiveness}
\section{Results}
  \subsection{Timeliness} \subsection{Accuracy} \subsection{Explainability}
  \subsection{Decision Support} \subsection{Ablation and Sensitivity Analysis}
\section{Summary of Results}
```
"Explainability Score" carries a TODO that says the score is undefined and asks for faithfulness to be separated from human judgement (`:76-85`).

Methodology §Layer Explainability (`3_0_Methodology.tex:1090-1115`) already states that the surrogate uses **17 features**, that R² is reported and a warning is raised below 0.85, and that **exact analytic attributions are "strictly better" and listed as future work**.

**Bibliography:** `thesis/Verzeichnis_Literatur.bib`, **50 entries**, style `ieeetr` (`Format_Bibliography.tex:22,26`). The Literature Review has TODOs flagging missing sources (`:193`, `:291`, `:338`).

---

## 12. Dependencies & config

Installed in `.venv` (Python 3.13.2):

| package | version | in requirements.txt |
|---|---|---|
| shap | 0.51.0 | `>=0.42` |
| scikit-learn | 1.8.0 | `>=1.3` |
| chromadb | 1.5.8 | `>=0.4` |
| sentence-transformers | 5.4.1 | `>=2.2` (not used by the retriever, which uses Chroma ONNX) |
| streamlit | 1.58.0 | `>=1.28` |
| matplotlib | 3.11.0 | `>=3.7` |
| plotly | 6.7.0 | `>=5.15` |
| kaleido | 1.3.0 | `>=0.2` |
| seaborn | **NOT INSTALLED** | not listed |
| anthropic | **NOT INSTALLED** | not listed |
| ragas | **NOT INSTALLED** | not listed |
| (numpy 2.4.4, pandas 3.0.2, optuna 4.9.0, onnxruntime 1.25.1) | | |

**Explainability-related `settings.yaml` keys:**
- `weight_mode` (:1)
- `weights.*` (:184-192)
- `thresholds.risk_critical / risk_high / risk_medium / risk_low` (:209-213). `agreement_bonus_3` and `agreement_bonus_5` are **not set** here, so the code defaults of 1.15 and 1.25 apply. They are set only in `optimized_weights.yaml`.
- `rag.collection_name`, `rag.top_k: 3`, `rag.composite_threshold: 0.65`, `rag.min_similarity: 0.55`, `rag.embedding_model: "all-MiniLM-L6-v2"` (informational only; the code ignores it and uses Chroma's default EF), `rag.collections.static_cases / live_context` (:215-226)
- `agents.news_sentiment.api.embedding_model` (:175)

There are **no** SHAP, surrogate or LLM keys. The SHAP hyperparameters, the LLM model IDs and the R² warning floor of 0.85 are all hard-coded.

---

## 13. Gaps & risks (assessment)

1. **The score being explained is ambiguous.**
   - Four aggregation paths exist (§2). The SHAP target (plain weighted mean, synthetic) is not `compute_risk` (bonus plus clip), and RAG is gated on the legacy `composite_score`, not `risk_score`.
   - An exact per-agent decomposition is possible at the **agent level**: `φᵢ = amp·wᵢ·sᵢ`, plus an interaction or clip residual when `amp > 1` or the score saturates. Choose one canonical path first.
   - The real-data eval path (D) is already purely additive.
2. **No exact feature-level decomposition.** Shipping uses `max(shock, duration)`, disaster uses `max(composite, single)`, geopolitical uses a sigmoid, and market and news use clips (§3).
   - The surrogate omits the inputs that actually drive shipping and news (`_raw_vessel_count` / level-shift, `recency_weighted_score`, velocity).
   - Feature attributions must therefore come either from per-agent analytic local linearisation or from Shapley over each agent's own inputs.
3. **The surrogate's claimed quality is in-sample and stale.** R² is a train-set number: 0.983 now, not 0.991. Chronological hold-out R² is only **0.53**.
   - The model is trained on one synthetic realisation and explains live, CSV or other-seed rows: distribution shift in the orchestrator, and a seed-42 vs. seed-44 mismatch in the dashboard.
   - Any thesis figure that cites 0.991 or "20 features" is wrong.
4. **Counterfactuals are feasible but costly.** No agents are persisted, so every counterfactual means re-running `fit` and `detect`. The IsolationForest uses `random_state=42`, so results are deterministic.
   - The rolling-window features (level shift = 30-day mean vs. 365-day median, 14-day persistence) mean that perturbing one day hardly moves the score. Counterfactuals have to be window-level ("traffic X% higher over 30 days").
5. **Calibrated probabilities are blocked by labels.** Train windows hold 0 positives in bab_el_mandeb, panama and malacca, and 26 in hormuz. Calibrating on train is impossible for 3 of 4 regions, and calibrating on test leaks.
   - Every real label ends 2025-08-30, so the 2026 Hormuz event is unlabelled.
   - The label is the shipping agent's own statistic (`circularity: high`), so "calibration" would largely calibrate the label against itself.
6. **RAG precision@k is blocked by metadata.**
   - There is no `outcome` or relevance label.
   - Static-case `region` is free text that does not match the region keys.
   - The live docs are **duplicated across regions**, and the gated query returns the same article twice.
   - Queries have **no region or date filter**: 2,039 docs from 2026 can be retrieved for a 2019-2025 alert, which is temporal leakage.
   - The query text is built only from which agents exceed 0.4, and always mentions "Strait of Hormuz".
   - A relevance-judgement set would have to be created from scratch.
7. **TP/FP triage — volume is fine, the composition is not.** Tier-5 test alerts at train-p90 are hormuz 181 (22 TP / 159 FP), bab_el_mandeb 495 (163 / 332), panama 197 (106 / 91) and malacca 188 (all FP). These are day-level counts, so clustering them into episodes will cut them sharply.
   - Hormuz TP = 22 is thin. The production `risk_high = 0.69` operating point has never been evaluated on real data, because CSV metrics use per-method p90.
   - In Hormuz the "Tier 5" row equals Tier 4: news data is missing.
8. **Eval and registry inconsistency.** `run_method_comparison.py` ignores `REGIONS[...].active_agents()`. For example, panama Tier 3/4 include geopolitical, which the registry marks passive. Explanations built on the registry would not match the CSV numbers.
9. **The decision and LLM layers are not verifiable.**
   - METRIC 8 is synthetic-only, has no regions, uses rule-derived labels and is stale (July; pre-routing removal). Its `top_drivers` come from the mismatched surrogate.
   - The LLM output is free text, unchecked against data, and has never run: `anthropic` is not installed. Fallback text still says "six agents".
10. **Risks of changing detection scores by accident.**
    - (a) Anything that touches `ShippingAgent._raw_vessel_count`, `preprocess` or the fit order changes scores.
    - (b) `RiskEngine` reads `self.weights` and `agreement_bonus_*`, which are mutable. `set_weights` exists for Optuna, so explainability code must never call it on a shared engine.
    - (c) `build_both_indexes` in `run_full_pipeline` **writes** to ChromaDB on every run, so a RAG evaluation run mutates the store. Use a copy, as this report did.
    - (d) `weight_mode: optimized` silently swaps the weights, the bonuses (1.28 / 1.44) and `risk_medium` (0.28). Explain under an explicit, recorded mode.
    - (e) Re-running `run_method_comparison.py` overwrites `eval/method_comparison_results.csv`. Write XAI outputs to `eval/explainability/` only.
