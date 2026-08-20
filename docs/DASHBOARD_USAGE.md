# Dashboard Usage — Multi-Region Edition

Phase 12 opened the Streamlit dashboard and the FastAPI service to all four
Phase 11 chokepoints. This covers the region selector, what actually changes
when you switch, the caching behind it, and the HTTP endpoints for driving
region selection programmatically.

For the pipeline-side architecture — config merging, projection, and the
evidence behind each region's agent activation — see `docs/REGION_USAGE_GUIDE.md`
and the **Phase 11** section of `README.md`.

---

## Starting the dashboard

```bash
streamlit run src/dashboard/app.py
```

Page 1 (**Decision View**) carries the region selector. Page 2 (**Analysis
View**) renders the evaluation suite, which is not region-scoped.

---

## The region selector

Top-right of the Decision View header, beside the page title:

| Label | Region key |
|---|---|
| Strait of Hormuz *(default)* | `hormuz` |
| Panama Canal | `panama` |
| Bab el-Mandeb | `bab_el_mandeb` |
| Strait of Malacca | `malacca` |

The list is built from `src.core.regions.REGIONS` at import
(`core.AVAILABLE_REGIONS`), so a region added to the registry appears in the
dropdown with no UI change — and the two cannot drift apart.

### What changes when you switch

Selecting a region reloads the merged config for that chokepoint, which
changes:

- **Which agents run.** Panama drops geopolitical; Malacca drops market; Bab
  el-Mandeb drops natural disaster. Routing is muted everywhere.
- **AIS bounds** handed to `ShippingConnector`.
- **ACLED countries** handed to `GeopoliticalConnector`.
- **NewsAPI keywords** derived for `NewsConnector`.

### Corridors, vessels and the map (Phase 12.5)

All four chokepoints render the full Decision View. Each has three schematic
corridors, its own camera framing, and named waypoint buttons.

**The corridor polylines are schematic.** They trace each strait's real transit
axis between real named places — Gatún and Miraflores locks, Perim Island, One
Fathom Bank, Bandar Abbas — at a fidelity suitable for a dashboard overview.
They are **not official IMO TSS geometry**; do not read navigational meaning
into them.

Vessel markers carry their **corridor's** risk score, so vessels on one corridor
share a status. The pipeline has no per-vessel signal (daily aggregate arrivals,
`aisstream.enabled` false, `fetch_from_api` a stub), and spreading the corridor
score across ships with a per-vessel jitter would be invented detail. Vessel
identity fields are deterministic synthetic labels.

A region with no corridors still falls back to the activation summary panel
(active agents, plus each passive agent's recorded reason).

### The timeline's dates are a display window — read this

The trend chart's x-axis **ends today**, but the underlying series is the
365-day evaluation test split (seed 44), whose own timestamps run
**2025-01-01 → 2025-12-31**. The axis re-indexes that series onto a rolling
window so the right edge is always current.

**A date on this axis is a position in a synthetic series, not a calendar
event.** In particular it has nothing to do with the real April–May 2026 Hormuz
shutdown in `data/raw/shuaiba_arrivals.csv`. The note under every chart using
the axis (`core.TIMELINE_AXIS_NOTE`) says so; leave it in place if you restyle
the page.

To see a region's true, unrelabelled scoring, run it headlessly:

```bash
python main.py --region panama
```

### Spikes and explanations

Upward threshold crossings appear as clickable rings on the trend chart; only
the highest band crossed on a day is marked. Clicking one opens an explanation
plus the contributing agent scores, and a Close button.

Explanations come from Anthropic when `ANTHROPIC_API_KEY` is set, and otherwise
from a deterministic paragraph composed from the live agent scores — the caption
under each explanation says which. Both are cached per `(region, day, level)`,
so re-clicking never re-queries.

Because the axis is relabelled, the LLM is explicitly forbidden from naming a
real incident, actor or date, and is asked to explain the *signal* instead. A
safety refusal (plausible — these prompts name sanctions and military signals)
falls back to the composed text rather than erroring.

```python
from src.dashboard.llm_explanations import clear_explanation_cache
clear_explanation_cache()   # force a re-query
```

### Agent breakdown

Below the map, a full-width chart plots one line per **active** agent on the
same axis as the composite. Click a legend entry to toggle a line. Passive
agents are omitted rather than drawn flat at zero — a zero line would read as
"measured and quiet" rather than "not run in this region"; the caption lists
which agents are passive.

---

## Caching

`src/dashboard/cache.py` keeps one config and one Orchestrator per region:

```python
from src.dashboard.cache import get_cached_orchestrator, clear_region_caches

orchestrator = get_cached_orchestrator("panama")   # built once
result = orchestrator.run_full_pipeline()
```

- First visit to a region builds its Orchestrator (six connectors); subsequent
  visits and switches back are cache hits.
- `maxsize` equals the number of registered regions, so no region evicts
  another — all four stay warm once visited.
- `core.load_app_config(region)` is separately cached by Streamlit's
  `@st.cache_data`, keyed on the region argument.

Two caveats worth knowing:

- **The cache is process-global, not per-session.** Every Streamlit session in
  the process shares one Orchestrator per region. That suits this
  single-analyst deployment; a multi-user one would want session state.
- **Orchestrator is stateful** — it retains `_agents` and `_last_agent_frames`
  from its last run. That is what makes reuse cheap. Do not hand one cached
  instance to concurrent runs.

Editing a region YAML while the dashboard is running? Call
`clear_region_caches()` or restart — the caches will not notice the file
change on their own.

```python
from src.dashboard.cache import cache_info
cache_info()   # hit/miss counts for both caches, for debugging a slow switch
```

---

## HTTP API for region management

Start the service:

```bash
python main.py --serve      # or: uvicorn src.api.endpoints:app
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/regions/list` | Every region, with activation summary and the active flag |
| `GET` | `/api/regions/current` | The region the service is scoring now |
| `GET` | `/api/regions/info/{region}` | One region in detail (404 if unknown) |
| `POST` | `/api/regions/switch` | Change the active region (400 if unknown) |

```bash
curl localhost:8000/api/regions/list
curl localhost:8000/api/regions/info/panama
curl -X POST localhost:8000/api/regions/switch \
     -H 'Content-Type: application/json' -d '{"region": "malacca"}'
```

Each region payload carries its centre coordinates, `active_agents`,
`passive_agents`, and `passive_reasons` — so a client can explain *why* an
agent is missing rather than just showing a gap.

### `switch` is process-global, not per-client

`POST /api/regions/switch` mutates module state in `src/api/endpoints.py`: it
sets the active region, clears the cached config, and resets the Orchestrator,
so the next `/predict` or `/health` scores the new chokepoint. This is a real
switch, not a validation echo.

The consequence: **one caller switching changes what every caller sees.** That
suits the single-analyst thesis deployment. A multi-tenant service should take
the region as a request parameter instead of holding it in server state.

An unknown region is rejected *before* any state changes, so a bad request
cannot leave the service half-switched.

The service's starting region follows the same precedence as the CLI:
`SUPPLY_CHAIN_REGION`, then the `hormuz` default.

---

## Troubleshooting

### Only Hormuz appears in the dropdown

`AVAILABLE_REGIONS` is derived from the registry, so this means
`src/core/regions.py` itself has one region — check it imports cleanly
(`_validate_registry()` raises on a malformed entry).

### A region shows the agent-summary panel instead of the map

That is the no-corridors fallback. As of Phase 12.5 all four registered regions
have corridors, so this means `core._ROUTES` has no entry for the selected key —
check the region was added to `_ROUTES` and `_REGION_MAP`, not just to the
registry.

### The timeline says a date that means nothing to me

Correct — it is a display window ending today, not the data's own dates. See
*The timeline's dates are a display window* above.

### A corridor's trend looks wrong for the region

Corridor `agents` are intersected with the region's active agents at render
time, so a corridor listing a regionally-passive agent (Hormuz's eastbound
corridor lists `routing`, muted since Phase 11) does not fold that signal in.
If a trend looks like it includes a muted agent, check that the call site is
passing `region` to `route_risk_series`.

### Region switch is slow every time

The first visit to each region pays for building six connectors. If *every*
switch is slow, the cache is being cleared — check `cache_info()["orchestrator"]`
for a rising `misses` count with `hits` stuck at zero, which means the process
is restarting or something is calling `clear_region_caches()`.

### `/health` reports fewer than six active agents

Correct as of Phase 11. Routing is muted in every region, and each region has
its own passive domains. `GET /api/regions/current` returns the exact expected
set with a reason for each exclusion.

### Two regions show identical risk scores

Check `config["_active_region"]` first. If it is right, the scores may
legitimately be close — the regions share the same synthetic/CSV inputs and
differ mainly in which agents contribute. Compare `agent_scores` keys, not just
the composite.

---

## Known gaps

- ~~Route corridors, vessel markers and map: Hormuz-only.~~ **Closed in Phase
  12.5** — all four regions render the full Decision View. The corridors remain
  schematic, not sourced navigational geometry.
- **The timeline axis is relabelled onto a window ending today.** The series'
  own dates are 2025-01-01 → 2025-12-31. This is a deliberate presentation
  choice, labelled in the UI, but it means axis dates are not calendar events.
- **Vessel records are synthetic and share a corridor's score.** There is no
  per-vessel signal to draw on until `aisstream` is implemented.
- **Analysis View is not region-scoped.** It renders
  `data/processed/evaluation_results.json`, which is produced by a single
  evaluation run and carries no region dimension yet.
- **`agents.natural_disaster.monitoring_points` has no `panama` entry**, so a
  live Ambee query would have no point to poll there. Inert while `data_mode`
  is `synthetic`.
- **Live API modes remain stubs.** `fetch_api()` on the shipping, geopolitical,
  news, and routing connectors raises `NotImplementedError`, reporting the
  region settings it resolved.
