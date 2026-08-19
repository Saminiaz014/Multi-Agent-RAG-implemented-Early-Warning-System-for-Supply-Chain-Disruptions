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

### What does *not* change — read this before judging the output

The Decision View's **route corridors, vessel markers, news feed and map are
Hormuz-only**. `core._ROUTES` defines schematic corridor geometry for the
Strait of Hormuz alone; the other three chokepoints have none, because drawing
them would mean inventing polylines rather than sourcing them.

Selecting any region other than Hormuz therefore shows a summary panel — the
region's display name, its active agents, and each passive agent with the
registry's recorded reason for the exclusion — instead of the trend chart, map
and route list. The detection pipeline for that region is fully configured and
runnable; it is the *presentation* layer that is Hormuz-only.

To confirm a non-Hormuz region really is scoring, run it headlessly:

```bash
python main.py --region panama
```

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

### A non-Hormuz region shows no chart or map

Expected — see *What does not change* above. The panel listing active and
passive agents is the intended output, not an error.

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

- **Route corridors, vessel markers, news feed and map: Hormuz-only.** The
  largest remaining asymmetry, and the reason non-Hormuz regions show a summary
  panel rather than the full Decision View.
- **Analysis View is not region-scoped.** It renders
  `data/processed/evaluation_results.json`, which is produced by a single
  evaluation run and carries no region dimension yet.
- **`agents.natural_disaster.monitoring_points` has no `panama` entry**, so a
  live Ambee query would have no point to poll there. Inert while `data_mode`
  is `synthetic`.
- **Live API modes remain stubs.** `fetch_api()` on the shipping, geopolitical,
  news, and routing connectors raises `NotImplementedError`, reporting the
  region settings it resolved.
