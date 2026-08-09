# Multi-Region Implementation Sequence

Grounded in `STATE_OF_PROJECT_MULTIREGION.md` — read that first. These prompts describe the *what* and *where*, not full inline code: every prior externally-authored "PROMPT N" this session that shipped pre-written code blocks turned out to have wrong method names, wrong schemas, or logic bugs, because the code was written against an assumed architecture instead of the real one. Whoever executes a prompt below should read the actual current file first, then implement — these are specs, not copy-paste payloads.

## Prerequisites

- [x] Audit complete (`STATE_OF_PROJECT_MULTIREGION.md`, `scripts/audit_multiregion_dependencies.py`)
- [ ] **Explicit decision that multi-region is in scope for this thesis.** EVAL01 is currently "R1: Hormuz only" by design (see `src/benchmark/regions.py` docstring). Expanding it is a real scope change to a thesis deliverable, not a neutral technical upgrade — worth deciding deliberately before Prompt 1, not discovering as a side effect of executing this sequence.
- [ ] A real, evidence-grade data source identified for at least one additional region (Prompt 6 below). Without this, Prompts 2–5 build plumbing with nothing real to carry.

## Prompts (in order)

### Phase 0: Foundation

**Prompt 1 — Canonical Region Registry**
*Blocking: everything else depends on this.* Est. 3–4h.
Reconcile the three existing, divergent region vocabularies (EVAL01's `src/benchmark/regions.py`, the dashboard's `AVAILABLE_REGIONS` in `src/dashboard/core.py`, and the RAG extraction `chokepoints` in `config/settings.yaml`) into one canonical `{region_key: display_name}` mapping and one place connectors/agents/orchestrator/dashboard/benchmark all read it from. Decide the actual region-key strings now (e.g. does Bab el-Mandeb get called `bab_el_mandeb` or does it fold into the RAG layer's existing `red_sea`?) — this prevents a second round of drift. Extend `src/benchmark/regions.py`'s `Region` dataclass (or a new sibling module) to be that canonical source, since it already has the right shape (center coords, active_domains, reroutable, etc.) and the right extension philosophy. Files: new or extended `src/benchmark/regions.py`; `config/settings.yaml` (add a `regions:` block referencing the canonical keys — do *not* call it `global.active_region`/`global.regions`, that was invented by the external plan and never existed). No behavior change yet — this prompt only establishes the registry.

### Phase 1: Core Architecture

**Prompt 2 — Parameterize Connectors**
Est. 6–8h. Depends on: Prompt 1.
Add a `region: str` (or `location` — pick one name, consistently, per Prompt 1) constructor parameter to all 6 connectors (`src/ingestion/{shipping,market,geopolitical,disaster,routing,news}_connector.py`) and `BaseConnector`. Replace the hardcoded `LOCATION`/`_LOCATION` class constants with an instance attribute resolved from the registry, defaulting to `"hormuz"` so every existing caller (all 309 current tests, `Orchestrator.__init__`, `main.py`) keeps working unchanged. Do **not** change what data each connector fetches yet — a `region="panama"` `ShippingConnector` should still legitimately fail/fall back to synthetic if there's no Panama CSV configured (that's Prompt 6's job). This prompt is purely: make location a parameter instead of a constant, prove it via tests, change nothing about behavior for `region="hormuz"`.
Acceptance: full existing suite green with zero test changes; new tests confirm `region=` is accepted and defaults correctly.

**Prompt 3 — Parameterize Agents**
Est. 4–6h. Depends on: Prompt 2.
Same treatment for the 6 agents (`src/agents/*_agent.py`) and `BaseAgent` — replace `_LOCATION`/`_DEFAULT_LOCATION` module constants with a constructor parameter, default `"hormuz"`. Agents mostly use location for labeling (`DetectionResult` metadata, log lines), so this is lower-risk than Prompt 2, but touch it second anyway since agents are constructed by connectors' output shape, and getting connectors right first avoids rework.
Acceptance: same as Prompt 2.

**Prompt 4 — Location-Aware Orchestrator**
Est. 5–7h. Depends on: Prompts 2, 3.
Add a `region` parameter to `Orchestrator.__init__` (default `"hormuz"`, preserving every existing call site). Thread it into `_build_enabled_agents()` (currently hardcodes agent construction with no location arg) and into however connectors get built in `__init__` (`self._shipping_connector = ShippingConnector(...)` etc.). This is the prompt most likely to touch `ingest()`/`fetch_domain()`/`_frame_for_agent()` — read all of `src/orchestrator.py` fresh before starting, its internals changed twice already this session (timeline validator wiring, AIS fallback) and a plan written against an older read of this file will be wrong.
Acceptance: `Orchestrator(config)` (no region arg) behaves byte-identical to today; `Orchestrator(config, region="hormuz")` explicit-default also identical; a stub second region can be constructed without crashing (even if its connectors immediately fall back to synthetic for lack of real data).

**Prompt 5 — Per-Region Config Layer**
Est. 3–4h. Depends on: Prompt 1.
`config["weights"]` and `config["thresholds"]` are currently flat and single-region. Decide and implement the actual shape for per-region overrides (e.g. `config["regions"]["panama"]["weights"]` overriding the global default) and a resolution function the Orchestrator/RiskEngine call. Cross-check against `src/optimization/weight_config.py`'s `resolve_active_weights` — that's the existing precedent for "resolve an effective weight set from multiple possible sources" and per Phase 2F.1b (see project memory) is already Optuna-integrated; a second, incompatible resolution path would fragment that machinery.

**Prompt 6 — Second-Region Data Sourcing** *(research, not code)*
Est. highly variable — likely the largest real cost in this whole sequence.
Find or build an evidence-grade dataset for at least one additional chokepoint, equivalent in kind to `data/raw/shuaiba_arrivals.csv` (real IMF PortWatch vessel-arrival data) for Hormuz. This is not optional plumbing — Prompts 2–5 give you the *capacity* to ingest a second region; without real data behind it, every connector for that region silently falls back to synthetic forever, and any EVAL01 results for it would be synthetic-only (a materially weaker evidence claim for a thesis than what Hormuz already has). Worth explicitly deciding whether synthetic-only is acceptable for a second region, or whether real sourcing is required before proceeding to Phase 4.

### Phase 2: Risk & Aggregation

**Prompt 7 — Make RiskEngine Region-Aware for Real**
Est. 3–4h. Depends on: Prompts 4, 5.
`RiskEngine`'s `region` parameter currently only feeds `logger.debug` calls (confirmed by reading `src/aggregation/risk_engine.py` directly). Wire it into actual threshold/weight resolution via Prompt 5's per-region config, so `region="panama"` genuinely uses Panama's thresholds, not Hormuz's borrowed ones with a different label in the log line.

**Prompt 8 — Region-Specific Agreement Bonus**
Est. 2–3h. Depends on: Prompt 7.
The real agreement-bonus mechanism already exists and is Optuna-tunable: `agreement_bonus_3`/`agreement_bonus_5` in `RiskEngine` (`src/aggregation/risk_engine.py`), loaded from `config["thresholds"]`. Extend Prompt 5's per-region config so these can differ by region, instead of building a new, parallel `compute_agreement_bonus()` (an external plan proposed exactly that, as a new `src/aggregation/agreement_bonus.py` — that file doesn't exist and shouldn't; it would duplicate real, working, already-tuned logic).

### Phase 3: API & Dashboard

**Prompt 9 — Region-Aware API**
Est. 3–5h. Depends on: Prompt 4.
Add a `region` query/body parameter to the relevant `src/api/endpoints.py` routes (`/predict`, `/explain`, `/status` at minimum), defaulting to `"hormuz"`. `_get_orchestrator()`'s module-level lazy-init singleton will need to become keyed by region (a dict of orchestrators, not one). Watch the existing `asyncio.run()` landmine noted in project memory: `ShippingConnector.fetch_from_api()` (AIS live mode) uses `asyncio.run()` internally and would raise `RuntimeError` if reached from these `async def` handlers without a thread offload — don't accidentally make `source_mode="api"` reachable here for any region.

**Prompt 10 — Dashboard Region Wiring**
Est. 2–3h. Depends on: Prompt 4, 9.
This is the smallest phase-3 prompt because the dashboard is already half-built for this: `AVAILABLE_REGIONS` and the `st.selectbox("Region", ...)` control already exist in `src/dashboard/core.py`/`decision_view.py`. The work is: extend `AVAILABLE_REGIONS` with the new region key(s) from Prompt 1's registry, and wire the selected region through to whatever orchestrator/API call the Decision/Analysis views make (currently implicitly Hormuz-only underneath the already-generic-looking `get_routes(region)`/`get_news(region)` calls).

### Phase 4: Benchmark Harness Expansion

One prompt per new region, each depends on Prompts 1–8 and (for anything beyond synthetic-only) Prompt 6:

**Prompt 11 — Second Region: EVAL01 Scenario Set**
Est. 4–6h per region.
Per region: write `config/benchmark/{region}.yaml` (the `Region` spec — center coords, `active_domains`, `reroutable`, `loss_scaling`, `disaster_relevance`; see `src/benchmark/regions.py` for the required shape and `config/benchmark/hormuz.yaml` for a working example) plus 4 scenario YAMLs in `config/benchmark/scenarios/` (`{region}_P_CRIT`, `_P_HIGH`, `_N_QUIET`, `_N_DECOY`) — copy `hormuz_*.yaml`'s structure exactly (it's correct and has real baseline results computed against it) and validate every generated file with `load_scenario()` + `materialize_scenario()` before considering it done, not just "wrote to disk without erroring." Ground the event parameters in real documented precedent for that chokepoint the same way `data/routing_research/hormuz_rerouting_patterns.json` did for Hormuz's routing signal — actually research it (WebSearch or equivalent), don't invent plausible-looking numbers; label anything not independently sourced as "extrapolated," not "documented."
**Repeat this prompt once per additional region** — that's where the "15–20 prompts" count flexes: 1 region added = 11 prompts total in this sequence; 4 regions added = 14 prompts, plus Prompts 12–14 below.

### Phase 5: Validation

**Prompt 12 — Regression Suite**
Est. 2–3h.
Confirm the full existing test suite (309 tests as of this session) passes unchanged after Phases 0–3 — every parameterization prompt above should default to `region="hormuz"` and change nothing observable for existing callers. This should really be run *after every single prior prompt*, not saved to the end; listed here as the final gate.

**Prompt 13 — New Multi-Region Test Coverage**
Est. 4–6h.
Tests for: each connector/agent constructs correctly for a non-default region; `Orchestrator(config, region=...)` builds the right connector/agent set; `RiskEngine` actually applies region-specific thresholds (not just logs the region name); API endpoints respect the `region` parameter; dashboard's region selector reaches real data for the new region(s).

**Prompt 14 — End-to-End Validation**
Est. 3–4h.
Run `main.py` and the full FastAPI flow for every populated region, run EVAL01 against each new region's scenario set, and sanity-check results the way Phase 2F.1b's Optuna re-run was sanity-checked (per project memory: deterministic seeds, compare objective/F1/lead-time against expectations) — not just "it ran without crashing."

## Total Estimated Time

| Phase | Hours |
|---|---|
| 0: Foundation | 3–4 |
| 1: Core architecture | 18–25 (+ variable for data sourcing) |
| 2: Risk & aggregation | 5–7 |
| 3: API & dashboard | 5–8 |
| 4: Benchmark harness | 4–6 per region |
| 5: Validation | 9–13 |
| **Total (1 new region)** | **~44–63 hours** |
| **Total (4 new regions, matching the external plan's original ambition)** | **~56–100+ hours**, dominated by Prompt 6 (data sourcing) × 4 |

This is substantially larger than the external plan's original "~25–30 hours" estimate, because that estimate assumed the architecture already existed in the shape the plan wanted (parameterized connectors, a working region config layer, etc.) — none of which is true today, per `STATE_OF_PROJECT_MULTIREGION.md`.

## Success Criteria

- [ ] All 309+ existing Hormuz tests still pass, unmodified
- [ ] New region(s) have real (not fabricated) data behind at least the primary detection signal, or an explicit, documented decision that synthetic-only is acceptable for that region
- [ ] No hardcoded `LOCATION`/`_LOCATION` constants remain in connectors/agents — all resolved from the Prompt 1 registry
- [ ] `RiskEngine`'s `region` parameter is functionally load-bearing, not cosmetic
- [ ] Dashboard region selector reaches real (or honestly-labeled synthetic) data end-to-end
- [ ] EVAL01 scenario YAMLs for new regions pass `load_scenario()` + `materialize_scenario()` and have baseline results computed, matching Hormuz's existing coverage
- [ ] Every new scenario parameter is traceable to either a cited real-world source or an explicit "extrapolated" label — no unlabeled invented numbers, consistent with the standard already set for Hormuz's `hormuz_rerouting_patterns.json`
