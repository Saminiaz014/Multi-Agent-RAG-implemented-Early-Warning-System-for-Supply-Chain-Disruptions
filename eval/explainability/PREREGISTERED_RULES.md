# Pre-registered rules (XAI-02)

Rules are committed **before** the step they govern runs. The commit hash of each section is reported in the phase report. Later sections may be added; committed rules are not edited. Any change after the fact is appended as a dated amendment, with its reason.

---

## A. Knowledge-base backfill 2023-08 → 2025-08 (Phase 3)

**Purpose:** give precedent retrieval and FP characterisation uniform, alert-blind coverage of the real test window (2023-08-31 → 2025-08-30, plus the 7-day look-back).

**Sources** (validated 2026-10-14):
- ACLED event-level API (account active).
- SerpApi Google News with a custom date range (`engine=google`, `tbm=nws`, `tbs=cdr`).
- GDELT DOC is **skipped**: HTTP 429 on the first request, and its DOC API is documented to search only a rolling ~3-month window.

**Months:** 2023-08 through 2025-08, 25 calendar months × 4 regions = 100 region-months. Sampling ignores alert dates entirely.

**Cap:** at most **15 documents per region per month**. Nominal split: up to 10 news + up to 5 ACLED. If one source yields fewer, the other may fill the remaining slots, up to 15 in total.

### A.1 SerpApi (one search per region-month; 100 searches; quota checked at 250 left)

| Region | Query (`q`) |
|---|---|
| hormuz | `"Strait of Hormuz" (shipping OR tanker OR attack OR military OR seized)` |
| bab_el_mandeb | `("Bab el-Mandeb" OR "Red Sea") (shipping OR Houthi OR attack OR vessel)` |
| panama | `"Panama Canal" (shipping OR transit OR drought OR vessels)` |
| malacca | `("Strait of Malacca" OR "Malacca Strait") (shipping OR piracy OR vessel OR security)` |

- **Fixed parameters:** `engine=google`, `tbm=nws`, `num=20`, `gl=us`, `hl=en`, `tbs=cdr:1,cd_min:M/1/YYYY,cd_max:M/<last>/YYYY`.
- **Retries:** one retry on an HTTP error; none otherwise.
- **Kept:** results whose `date` parses to a day inside the query month, de-duplicated by URL and then by title, taken in the order returned. Undated or out-of-month results are dropped.
- **Caching:** raw responses are cached under `eval/explainability/backfill_raw/` (untracked) so a rebuild never spends quota again.

### A.2 ACLED (event level)

| Region | Countries (coastal states of the chokepoint) |
|---|---|
| hormuz | Iran, Oman, United Arab Emirates |
| bab_el_mandeb | Yemen, Djibouti, Eritrea |
| panama | Panama |
| malacca | Malaysia, Indonesia, Singapore |

- **Request:** one request per country-month (`event_date BETWEEN first|last day`, `limit=5000`).
- **Maritime filter:** an event is kept if its `notes` match (case-insensitive) `\b(ship|ships|shipping|vessel|vessels|tanker|tankers|boat|boats|port|ports|sea|maritime|strait|navy|naval|coast ?guard|canal|ferry|cargo|harbou?r|offshore|seiz\w*)\b`.
- **Selection:** candidates for a region-month are pooled over countries and sorted by (`event_date`, `event_id_cnty`). Up to the allotted number are taken at evenly spaced positions (`numpy.linspace` over the sorted list, rounded, de-duplicated). This is uniform within the month and independent of content.

### A.3 Schema and isolation

- **Metadata:** the same as `live_extracted_context`: `region` (registry key), `source_api` (`serpapi` | `acled`), `event_type` (`news_article` | `conflict_event`), `event_date` (YYYY-MM-DD), `severity`, `primary_agents`, `countries`, `title`, `url` / `source_name` where available. Plus **`backfill: true`**; ACLED docs also carry `event_id_cnty`, `fatalities`, `acled_event_type` and `location`.
- **Severity:** news is `medium`. ACLED fatalities map 0 → `low`, 1–9 → `medium`, ≥ 10 → `high`.
- **`primary_agents`:** news → `news_sentiment`; ACLED → `geopolitical`.
- **Text:** news is `"{title}. {snippet} Source: {source}."`; ACLED is the event `notes`.
- **Isolation:** written only to **`data/knowledge_base/.chromadb_xai/`**, collection **`backfill_2023_25`**. That directory starts as a copy of the production DB. Production `data/knowledge_base/.chromadb` is never opened for writing.

## B. Precedent retrieval evaluation (Phase 3)

- **Production retriever:** `ContextRetriever.query_gated(agent_scores, composite)` on a temporary copy of the production DB, unchanged (`top_k = 3`, `min_similarity = 0.55`, composite threshold 0.65). Diagnostics use the Path D alert score as the composite and report the trigger rate. Retrieval quality is measured on the ungated query.
- **Eval wrapper** (`src/explainability/precedent_retriever.py`):
  - **Query text:** `"{region display name}: {agent phrase 1}; {agent phrase 2}."` built from the two agents with the largest positive exact Shapley values that day; each phrase names the agent's top effective terms.
  - **Filters**, applied in Python after over-fetching 200 candidates per collection:
    1. `region == alert region` (static cases via the fixed mapping in the wrapper);
    2. `event_date < alert date` (static `YYYY-MM` dates read as the first of the month);
    3. de-duplication by text hash.
  - Then `min_similarity = 0.55`, `top_k = 3`.
- **Diagnostics:** unit = episode, query on the episode's first day. Collections: production (static + live) and production + backfill. Measures:
  - temporal-validity rate (`event_date < alert date`);
  - region-match rate;
  - duplicate rate in the top 3;
  - share of queries with any hit ≥ 0.55;
  - similarity distribution.
- **Annotation set:**
  - **Sampling:** 20 episodes, stratified by region × episode label. The 7 non-empty strata get 3 episodes each (all, if fewer), reduced to 20 by dropping one from the largest FP stratum. Drawn with `numpy.random.default_rng(20261014)`.
  - **Pairs:** the wrapper's top 3 (production + backfill) per episode, giving ≤ 60 pairs.
  - **Blinding:** TP/FP labels and similarities are withheld from the annotation file.
  - **Relevance scale:** 0 = not relevant; 1 = related context (same region, related risk, would not explain the alert); 2 = directly relevant (describes an event that plausibly explains the signals in the alert window).
  - **Scoring:** precision@3 is reported both lenient (relevance ≥ 1) and strict (= 2).
- **FP characterisation tags**, applied per FP episode with window = [start − 7 d, end] and a region match. ACLED yearly summaries are excluded.
  - `risk_without_traffic_drop`: at least one geopolitical/conflict document (ACLED event, or news matching `GEO_PATTERN`).
  - `unexplained`: documents exist in the window, but none is geopolitical/conflict.
  - `no_coverage`: no documents at all in the window.

## C. Phase 4 triage indicators

Each indicator is scored per alert episode as an AUC for FP (positive) vs TP. There is no fitting, so there is no leakage.

| # | Indicator | Definition | Expected direction (→ FP) |
|---|---|---|---|
| C1 | **Top exact driver = geopolitical** | 1 if the agent with the largest episode-summed exact Shapley value (Path D, calm baselines) is `geopolitical`, else 0 | 1 → FP |

**Disclosure for C1:** this indicator was proposed *after* seeing Phase 2 descriptives, where geopolitical led 29/49 Malacca FP episodes and 13/34 Hormuz FP episodes, but also 4/5 Hormuz TP episodes. It is registered now, before any triage AUC is computed. The remaining Phase 4 indicators and their cut-offs (margin, agreement count, k80_agent, agent dispersion, exact-vs-surrogate coherence, max precedent similarity, episode length) will be registered in this section before Phase 4 runs.

### C.0 Inputs (Option A, approved 2026-10-15)

`run_triage_inputs.py` writes `triage_inputs/` once. It is the only agent run in Phase 4.

- **Path D re-score:** before writing, it asserts that the Tier 5 metrics recomputed from the cached scores equal `eval/method_comparison_results.csv` within 1e-9.
- **Per region and day:** train- and test-window agent scores for both variants (shipping = shock score in `without_duration`), composites, thresholds, `y_true`.
- **Calm baselines** for both variants.
- **The production surrogate's agent attribution** on every test alert day of both variants.

Nothing else calls an agent, an API or a retriever. Max precedent similarity is read from the AERs' `precedents`; where an episode's first day has no AER (some `without_duration` episodes), C7 is missing for that episode and the count is reported.

### C.1 Unit, label, timing

- **Unit:** the Tier 5 alert episode (alert days merged across gaps ≤ 3 calendar days).
- **Label:** FP = 1 if no `y_true` day falls in the episode's span, else TP = 0.
- **Prospective indicators** are read on the episode's **first day**.
- **Retrospective indicators** (need the whole episode) are reported in a separate table and never enter a model.

"Train alert days" means train-window days whose composite exceeds the variant's train p90 threshold. "Train calm days" means train-window days with `y_true = 0`.

### C.2 Indicators

**Primary**, the only ones interpreted:

| # | Indicator (first day) | Direction → FP | Low-trust cut-off | Cut-off source |
|---|---|---|---|---|
| P1 = C1b | Top exact driver = `geopolitical` (largest first-day exact φ) | 1 → FP | = 1 | binary |
| P2 | Agreement: agents whose score exceeds that agent's own train-calm-day **q0.90** | low → FP | ≤ 1 | per-agent train calm-day q0.90 |
| P3 | Dispersion: population std of the composite's agent scores | high → FP | > train-alert-day **q0.75** of dispersion | train alert days |

**Baseline:** S0 = first-day composite score (low → FP). Margin is **dropped** as an indicator: within a region it is S0 minus a constant, so its AUC is identical to S0's.

**Secondary, EXPLORATORY** (reported, never interpreted; no multiplicity correction):

| # | Indicator (first day) | Direction → FP | Low-trust cut-off | Cut-off source |
|---|---|---|---|---|
| X1 | k80_agent (fewest agents covering 80% of Σ\|φ\|) | low → FP | ≤ train-alert-day **q0.25** of k80 | train alert days |
| X2 | Exact top agent ≠ production-surrogate top agent | 1 → FP | = 1 | binary |
| X3 | Max precedent similarity (AER `precedents`, wrapper prod + backfill; 0 if none) | low → FP | none (AUC only) | — |
| X4 | Noise robustness: share of 200 score-noise draws above threshold (σ per agent from train calm-day changes / √2; seed = day index + 1) | low → FP | < **0.95** | calm-day noise model, 5% tail |

**Retrospective** (separate table):

| # | Indicator (whole episode) | Direction → FP | Low-trust cut-off | Cut-off source |
|---|---|---|---|---|
| C1 | Top exact driver = `geopolitical` (largest episode-summed exact φ; registered `ae8c7b7`) | 1 → FP | = 1 | binary |
| R2 | Episode length in alert days | short → FP | < median (**q0.50**) train-window episode length | train-window episodes |

Directions are fixed hypotheses. Every AUC is "AUC for FP" in the registered direction, so a value below 0.5 means the hypothesis was wrong; no indicator is flipped after the fact.

### C.3 Single-indicator AUCs

- **What is computed:** per region × variant, the AUC for FP vs TP of S0, the primaries, the secondaries and (in its own table) the retrospective indicators.
- **CI:** stratified episode bootstrap, 1,000 resamples, vectorised, seed 20261014.
- **Primaries only:**
  - one-sided Mann–Whitney U p-value in the registered direction;
  - **Holm-adjusted** within each region × variant family (m = 3);
  - interpreted as evidence only if the adjusted p < 0.05 **and** the minimum-n rule holds.
- **Minimum-n rule:** an AUC is computed when a region has ≥ 1 TP and ≥ 1 FP episode. It is interpreted only with **≥ 5 TP and ≥ 5 FP episodes**; otherwise it is reported as "not interpreted (n)". Every Hormuz result carries "small-n (5 TP episodes with duration, 7 without)".

### C.4 Leave-one-region-out logistic model

- **Features:** the **three primaries only** (P1, P2, P3), z-scored with the training fold's mean and sd.
- **Model:** `LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000)`. No tuning, no feature selection.
- **Folds:** for each region R, fit on the other three regions' episodes and predict R.
  - AUC on R where R has both classes, with a bootstrap CI and the minimum-n rule.
  - Malacca is evaluated only by C.5.
- **Score-only comparator:** the same model with S0 alone.
- **Result:** ΔAUC (primaries − score-only) on the same resamples (paired bootstrap, 1,000).

### C.5 Malacca: share of FP episodes flagged low-trust

- **Flags measured:**
  - for P1, P2 and P3 (cut-offs above), the share of Malacca FP episodes flagged;
  - for the LORO models held out on Malacca (primaries; score-only), the share with predicted P(FP) ≥ **0.5**.
- **The flagging cost:** each rule's flag rate on the TP episodes of the other three regions. For the LORO models this uses each region's own held-out predictions.
- **CI:** episode bootstrap, 1,000 resamples.
- X1, X2 and X4 flag rates are reported as exploratory.

### C.6 Circularity-control variant

Everything in C.3–C.5 is repeated on the `without_duration` alert set. Shipping is reduced to its shock score, and the threshold, calm baselines, noise scales, episodes and labels are all recomputed from the C0 export.

Results are reported side by side, and a conclusion is drawn only if it holds in both variants.

### C.7 Outputs

All under `eval/explainability/`, every row tagged `data_origin = real_eval`:
- `triage_results.csv` (single indicators, S0, primaries with Holm);
- `triage_retrospective.csv`;
- `triage_loro.csv`;
- `triage_malacca.csv`.


## D. Phase 5: decision value (SYNTHETIC, seed 44)

**Scope (trimmed 2026-10-15):** METRIC 8 is refreshed, and exactly **one** comparison is run. P1, P2 and simulated reliance are not run.

### D.1 METRIC 8 refresh

`notebooks/evaluation.py::metric_8_decision` is run unchanged on the current code (5 agents):
- splits: seeds 42 / 43 / 44;
- agents fitted on train and scored on test (seed 44);
- both weight modes.

It is written to `eval/explainability/metric8_refresh.json` next to the stale 2026-07-30 values (`data/processed/evaluation_results.json`, not overwritten). Accuracy, **macro-F1** over the four actions, and per-class recall are reported. In the stock rule, `top_agent` is the argmax of inter-weight × score (`_day_top_agent`).

### D.2 The registered comparison: P0-surrogate vs P0-exact

The day records are the stock METRIC 8 records (hand-tuned primary; optimized reported too), identical in risk level, `sustained`, `historical_context = None` and correct action. They differ **only** in `top_drivers[0].agent`:

- **P0-surrogate:** what the dashboard does today. The production SHAP surrogate (`build_shap_training_data` on the default config, seed-42 synthetic year) is explained on **row i of its seed-42 training frame** for test day i (`min(i, n−1)`), and its top driver's agent is used. This reproduces the seed-42 / seed-44 mismatch deliberately.
- **P0-exact:** the agent with the largest exact Shapley value of that seed-44 day's composite.
  - The value function is the evaluator's aggregation: Path C, i.e. a renormalised mean over agents with weight > 0, the agreement bonus at 0.5 with the mode's bonus3/bonus5, and a clip at 1.
  - Baselines are each agent's median score on the **seed-42 train** realisation's non-disruption days, scored by agents fitted on seed-42 train with the same mode's parameters.

**Measures:**
- accuracy, macro-F1 and per-class recall of `predict_action` against `scenario_correct_action`, over the 365 test days;
- **McNemar's exact test** (two-sided binomial on the discordant days) of per-day correctness, P0-exact vs P0-surrogate;
- the number of days on which the two top agents differ, and on which the action differs. `top_agent` affects the action only on HIGH, non-sustained days, so most days are identical by construction.

Everything is labelled SYNTHETIC. The correct actions are rule-derived from the injected scenario windows, not observed decisions.

---

## Amendments

**2026-10-14, A.1 retry rule.** The first backfill run stopped on a SerpApi read timeout (60 s) at `hormuz` 2024-02, after 6 region-months had been cached. The account usage showed the timed-out request was not charged.

- A read timeout or connection error now counts as an HTTP error, so it gets the one permitted retry.
- The request timeout is raised to 120 s.

Queries, cap, selection and months are unchanged, and no fetched result had been inspected beyond the 2024-01 dry run.

**2026-10-14, A.1 double failure.** The resumed run stopped when SerpApi returned HTTP 503 on both attempts for `panama` 2024-11. Rule added:

- A region-month whose SerpApi request fails on both attempts gets no news documents. ACLED fills its slots under the normal allocation.
- The month is recorded as `serpapi_status = fetch_failed` in `backfill_coverage.csv` and is not cached, so the next run retries it.
- Error messages no longer include the request URL, which had exposed the API key in a local log.

**2026-10-15, post-hoc, EXPLORATORY: stricter FP-characterisation rule (B, one run, no tuning).**

The pre-registered tag (§B) is the primary outcome and stays as it is: a null result, with calm-day windows tagged at the same rate as FP windows. This amendment was written *after* seeing that result. It defines one stricter rule, run exactly once on the already-backfilled documents (collection `backfill_2023_25` only; no new fetching), and nothing in it may be adjusted after the run.

- **ACLED:** a document counts only if `acled_event_type` ∈ {`Battles`, `Explosions/Remote violence`, `Strategic developments`}.
  - "In the region's maritime area" is the backfill's own maritime-notes filter (§A.2), which every ACLED backfill document already passed.
  - The documents carry no coordinates, so no geographic box is applied.
- **News:** a document counts only if its title + text matches `INCLUDE` and does not match `EXCLUDE` (case-insensitive).
  - `INCLUDE`, either:
    - (a) a strong term: `houthi*`, `missile*`, `hijack*`, `pirate*`, `piracy`, `warship*`, `blockade*`, `airstrike*`, `air strike*`, `seized`, `seizure`, `seizes`, `seizing`, `drone strike*`, `drone attack*`; or
    - (b) both an attack verb (`attack*`, `strike*`, `struck`, `fired on`, `explosion*`, `blast*`) **and** a maritime object (`ship`, `ships`, `vessel*`, `tanker*`, `merchant`, `cargo`, `navy`, `naval`).
  - `EXCLUDE`: `shark*`, `heart attack*`, `panic attack*`, `cyber*`, `ransomware`, `hacker*`, `hunger strike*`, `strike action`, `dock* strike*`, `worker* … strike*` (within 20 characters), `union`, `price war`, `trade war`, `tariff war`, `football`, `soccer`, `cricket`.
- **Window:** 8 days ending on the given day, the same footing as the calm-day control. The windows are FP-episode first days, TP-episode first days (reported, not used in the lift), and calm days (test-window days that are neither alerts nor `y_true = 1`).
- **Metric:** lift = (share of FP windows with ≥ 1 counting document) / (share of calm windows with ≥ 1). It is reported per region and pooled. The 95% CI comes from a percentile bootstrap: 1,000 resamples, FP windows and calm windows resampled independently (stratified by region when pooled), seed 20261014. A region whose calm rate is 0 reports an undefined lift.
