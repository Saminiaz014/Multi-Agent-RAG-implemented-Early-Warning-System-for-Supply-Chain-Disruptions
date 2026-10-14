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
