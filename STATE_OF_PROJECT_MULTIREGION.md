# State of Project: Multi-Region Architecture Assessment

**Date:** 2026-08-09
**Scope:** Current Hormuz-only implementation, verified readiness for multi-region expansion
**Method:** Direct code/config inspection (`scripts/audit_multiregion_dependencies.py` + manual verification), not inference from documentation or an external plan. Every claim below was checked against the running code.

## Executive Summary

This codebase is a **single-chokepoint system** (Strait of Hormuz / Shuaiba port) with **no live multi-region capability anywhere** in the ingestion → detection → aggregation → API path. However, it is not a blank slate: three separate, independently-evolved region concepts already exist in different subsystems, none of them wired to each other, and none matching the 5-region vocabulary (`hormuz`/`bab_el_mandeb`/`panama`/`suez`/`taiwan_strait`) that an external planning document has repeatedly assumed exists (see "External plan document" note below).

Building real multi-region support is a genuine architecture project — realistically 20–30 hours across 12 classes (6 connectors + 6 agents), the orchestrator, risk engine, config, dashboard, and the benchmark harness — not a config tweak. The good news: two of the three existing region concepts (the benchmark harness and the dashboard) already demonstrate the right extension pattern (`region: str` parameter + a registry keyed by region name, degrading gracefully for unpopulated regions). Any real implementation should extend that pattern, not invent a fourth.

**A note on the external plan document:** across this session, six prior "PROMPT N" documents (2, 2.5, 3, 3.5, 3-ALT, 4-ALT) each independently assumed a `hormuz`/`bab_el_mandeb`/`panama`/`suez`/`taiwan_strait` multi-region architecture with a `self.location` instance attribute, an `active_region` config key, and/or a `config["global"]["regions"]` block. None of that exists. This audit is the first pass to establish ground truth rather than trust that document's assumptions.

---

## Current Architecture

### What Exists (Region-Agnostic Foundations)

- [x] `BaseConnector` / `BaseAgent` ABCs — generic, no location coupling in the base classes themselves
- [x] `Orchestrator` framework — generic pipeline plumbing (`ingest()`, `fetch_domain()`, `_run_agents()`, `_build_enabled_agents()`); doesn't hardcode Hormuz *logic*, but has no location/region parameter either
- [x] `RiskEngine` — weight/threshold aggregation is config-driven, not location-coupled
- [x] **EVAL01 benchmark harness** (`src/benchmark/regions.py`) — a real `Region` dataclass + `REGION_REGISTRY` loaded from `config/benchmark/{name}.yaml`. Docstring is explicit: *"For R1 only hormuz is populated. Additional regions are added by dropping a new config/benchmark/{name}.yaml — no code changes here."* This is the closest thing to a designed-for-extension region system in the codebase.
- [x] **Dashboard region selector** (`src/dashboard/core.py`) — `AVAILABLE_REGIONS: dict[str, str] = {"Strait of Hormuz": "hormuz"}`, plus `get_routes(region)` / `get_news(region)` that already accept an arbitrary region key and return `[]` gracefully for anything unpopulated. `decision_view.py` already renders a `st.selectbox("Region", ...)` UI control. This is genuinely the most forward-compatible piece already in the codebase — extending `AVAILABLE_REGIONS` with a new key would surface a selector option immediately (though selecting it would show empty data until a real connector exists).

### What's Hormuz-Hardcoded

All 6 connectors (`src/ingestion/*_connector.py`) and all 6 agents (`src/agents/*_agent.py`) hardcode location as a **class or module constant**, not a constructor parameter:

| File | Constant | Value |
|---|---|---|
| `disaster_connector.py` | `LOCATION` | `"Strait of Hormuz"` |
| `geopolitical_connector.py` | `LOCATION` | `"Strait of Hormuz"` |
| `market_connector.py` | `LOCATION` | `"Global/Persian Gulf"` |
| `news_connector.py` | `LOCATION` | `"Strait of Hormuz"` |
| `routing_connector.py` | `LOCATION` | `"Strait of Hormuz"` |
| `shipping_connector.py` | `LOCATION` | `"Shuaiba Port, Persian Gulf"` |
| `disaster_agent.py` / `geopolitical_agent.py` / `news_agent.py` / `routing_agent.py` | `_LOCATION` | `"Strait of Hormuz"` |
| `market_agent.py` / `shipping_agent.py` | `_DEFAULT_LOCATION` / `_REAL_DATA_LOCATION` | `"Strait of Hormuz"` / region-specific real-data label |

None of the 12 `__init__` signatures accept a `location` or `region` parameter — confirmed by direct inspection, not inference:
```
BaseConnector.__init__(self, config: dict)
DisasterConnector.__init__(self, config: dict | None = None)
GeopoliticalConnector.__init__(self, config: dict | None = None)
MarketConnector.__init__(self, source_mode: str | None = None, config: dict | None = None)
NewsConnector.__init__(self, config: dict | None = None)
RoutingConnector.__init__(self, config: dict | None = None)
ShippingConnector.__init__(self, source_mode: str | None = None, config: dict | None = None)
BaseAgent.__init__(self, name: str, config: dict)
DisasterAgent / GeopoliticalAgent / MarketAgent / NewsAgent / RoutingAgent / ShippingAgent
    .__init__(self, config: dict[str, Any] | None = None)
```

`Orchestrator.__init__(self, config: dict)` — no location/region parameter. Agents are built via `_build_enabled_agents()`, which imports and constructs each of the 6 agent classes directly and registers them via `register_agent()` — there is no per-region agent factory.

`RiskEngine` has a `region: str = "hormuz"` parameter on one internal method, but its own docstring says *"used only for log messages"* — confirmed by reading the implementation: it's interpolated into three `logger.debug(...)` calls and nothing else. Not wired into thresholds, weights, or any decision logic.

`src/api/endpoints.py` — zero location/region mentions. `/predict`, `/explain`, etc. are single-region by construction.

`config/settings.yaml` — no `global.active_region` or `global.regions` key (confirmed absent; this is the thing every prior "PROMPT N" assumed existed).

### The Three Unreconciled Region Vocabularies

This is the most important and least obvious finding. Three parts of the codebase each independently invented a region concept, at different times, for different purposes, with **different names for the same or overlapping places**:

1. **EVAL01 benchmark harness** — `config/benchmark/*.yaml` + `config/benchmark/scenarios/*.yaml`. Only `hormuz` populated (1 region spec, 4 scenario classes: `N_QUIET`, `N_DECOY`, `P_HIGH`, `P_CRIT` — all real, all with 58 baseline result files each in `results/baselines/`, i.e. fully integrated into evaluation, not stubs).
2. **Dashboard** — `AVAILABLE_REGIONS = {"Strait of Hormuz": "hormuz"}` in `src/dashboard/core.py`. Only `hormuz` populated.
3. **RAG/KB extraction** — `config/settings.yaml: extraction.chokepoints` = `['hormuz', 'red_sea', 'malacca', 'suez']`. This is for historical-case backfill into the RAG knowledge base, not live monitoring — a genuinely different concern. Note the naming: `red_sea`, not `bab_el_mandeb`; `malacca` is present and used nowhere else in the codebase; `panama` and `taiwan_strait` are absent entirely.

**Implication:** a real multi-region design has to pick a canonical region-key vocabulary and reconcile these three, not just extend one of them in isolation. If the thesis ever adds a second chokepoint, "which 4 letters/words identify it" needs to be decided once, centrally — not per-subsystem the way it's evolved so far.

### Missing for Multi-Region (Live Pipeline)

- [ ] Location-parameterized connectors (all 6) and agents (all 6)
- [ ] Location-aware `Orchestrator` (construct the right connector/agent set for a given region)
- [ ] A canonical region registry reconciling the three vocabularies above, consumed by config, connectors, dashboard, and the benchmark harness alike
- [ ] Per-region thresholds/weights (today `config["weights"]`/`config["thresholds"]` are flat, single-region)
- [ ] `RiskEngine`'s `region` param made functionally meaningful (currently cosmetic)
- [ ] Region-aware API (`src/api/endpoints.py`)
- [ ] 4 more benchmark region specs (`config/benchmark/{region}.yaml`) + their scenario YAMLs, if EVAL01 is to cover more than Hormuz
- [ ] Real data sources for any new region (CSV/API equivalents of `shuaiba_arrivals.csv`, `brent_crude.csv`, etc. — connectors are useless without something to ingest)

## Dependency Chain

```
Connectors (6, LOCATION hardcoded)
    |
Agents (6, LOCATION hardcoded)
    |
DetectionResult (unified schema, no location field)
    |
Orchestrator.ingest() / fetch_domain()  (single-region only)
    |
RiskEngine.compute_risk()  (region= param exists but is cosmetic/log-only)
    |
FastAPI endpoints (/predict, /explain — no region param)
    |
Dashboard (region selector UI + region-parameterized fetch functions
           ALREADY EXIST here — the one place ahead of the rest)

Separately:
EVAL01 benchmark harness (src/benchmark/regions.py + scenario_generator.py)
    -- real Region registry, real per-region scenario YAMLs, but only
       "hormuz" populated; designed for exactly this kind of extension

Separately:
RAG/KB extraction (extraction.chokepoints: hormuz, red_sea, malacca, suez)
    -- a fourth, unrelated region list, historical-backfill only
```

## Sizing

This is not a 45–90 minute prompt. Realistic estimate for a functioning second region (data source, connectors, agents, orchestrator wiring, dashboard, one benchmark scenario set), assuming real evidence-grade data can be sourced:

| Phase | Work | Est. hours |
|---|---|---|
| 1. Core architecture | Parameterize connectors, agents, orchestrator, canonical region registry | 8–10 |
| 2. Risk & aggregation | Per-region thresholds/weights, make `RiskEngine`'s region param real | 4–6 |
| 3. API & dashboard | Region-aware endpoints; dashboard already ~half-done | 3–5 |
| 4. Benchmark harness | New region.yaml + 4 scenario YAMLs per region, grounded in real precedent (per the routing-research approach already used for Hormuz) | 2–3 per region |
| 5. Real data sourcing | Finding/building an evidence-grade dataset per new region (the actual bottleneck — Hormuz's Shuaiba CSV is real IMF PortWatch data; a second region needs its own) | highly variable, likely the largest cost |

See `MULTIREGION_IMPLEMENTATION_SEQUENCE.md` for the ordered prompt breakdown.
