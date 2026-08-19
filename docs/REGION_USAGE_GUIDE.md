# Region Usage Guide

How to run the pipeline against a specific chokepoint, what each region's
config controls, and what to check when a region behaves unexpectedly.

For the architecture behind this — the merge/projection pipeline and the
evidence behind each region's agent activation — see the **Phase 11** section
of `README.md`.

---

## Supported regions

| Key | Chokepoint |
|---|---|
| `hormuz` | Strait of Hormuz (Persian Gulf) — **default** |
| `panama` | Panama Canal (Central America) |
| `bab_el_mandeb` | Bab el-Mandeb (Red Sea entrance) |
| `malacca` | Strait of Malacca (Southeast Asia) |

Keys are case-insensitive and are stripped, so `"  Panama "` resolves to
`panama`. Anything else raises `ValueError` listing the valid options.

```python
from src.core.config_manager import available_regions

available_regions()   # ['hormuz', 'panama', 'bab_el_mandeb', 'malacca']
```

---

## Running a region

### Command line

```bash
python main.py                                # hormuz (default)
python main.py --region panama
python main.py --region malacca --mode synthetic
python main.py --region bab_el_mandeb --optimize --trials 50
```

`--region` is validated by argparse against the registry, so a typo is rejected
before anything loads.

### Environment variable

Consulted only when `--region` is absent — useful for a shell session or a
container, without touching the command:

```bash
export SUPPLY_CHAIN_REGION=bab_el_mandeb
python main.py
```

A typo here raises rather than silently falling back to Hormuz, so an
unnoticed misconfiguration cannot quietly produce Hormuz results labelled as
another region.

### Programmatically

The Orchestrator builds its own agents from the config — there is nothing to
register:

```python
from src.core.config_manager import load_config_for_region
from src.orchestrator import Orchestrator

config = load_config_for_region("panama")
result = Orchestrator(config=config).run_full_pipeline()

print(result["composite_score"])          # 0.382649
print(sorted(result["agent_scores"]))     # market, natural_disaster, news_sentiment, shipping
print(config["_active_region"])           # 'panama'
```

`load_config_for_region()` with no argument resolves the region the same way
the CLI does: `SUPPLY_CHAIN_REGION`, else `hormuz`.

Paths are resolved against the project root, not the process CWD, so this works
identically from `main.py`, from pytest, and from the Streamlit dashboard.

### Comparing regions

Each call returns an independent config, so regions can be run in sequence
without interference (`tests/test_region_isolation.py` asserts exactly this):

```python
for region in available_regions():
    config = load_config_for_region(region)
    result = Orchestrator(config=config).run_full_pipeline()
    print(f"{region:14s} {result['composite_score']:.6f} "
          f"{sorted(result['agent_scores'])}")
```

---

## What a region config controls

Each region has an overlay at `config/regions/<region>.yaml`, deep-merged onto
`config/settings.yaml`. Only the keys a region actually changes appear in the
overlay; everything else — thresholds, weights, detection parameters, ingestion
paths — comes from `settings.yaml` and stays shared across regions.

### Agent activation

```yaml
agents:
  shipping:
    enabled: true
  geopolitical:
    enabled: false      # passive: not built, not run, not weighted
```

This drives the *existing* `config["agents"][<key>]["enabled"]` flag rather than
a parallel mechanism. The flags must agree with `src/core/regions.py`'s registry
— `tests/test_region_configs.py` fails if they drift apart.

Agent keys are the pipeline's real config keys: **`natural_disaster`** and
**`news_sentiment`**, not `disaster`/`news`.

### Connector data sources

```yaml
extraction:
  chokepoint_key: panama                 # which extraction.chokepoints entry to fill
  countries: ["Panama"]                  # → agents.geopolitical.acled_countries
  bounding_box: { lat_min: 8.8, lat_max: 9.5, lon_min: -80.1, lon_max: -79.4 }

aisstream:
  bbox: [[8.8, -80.1], [9.5, -79.4]]     # → ingestion.shipping.ais_bounds

agents:
  news_sentiment:
    location_context:                    # → NewsConnector.newsapi_keywords
      primary_location: "Panama Canal"
      region: "Central America"
      countries: ["Panama"]
      topics: ["shipping", "drought", "Gatun Lake", "transit slots", ...]
```

Note `chokepoint_key` need not equal the region key: `bab_el_mandeb` maps onto
`settings.yaml`'s existing `red_sea` chokepoint entry.

News keywords are **derived**, not listed: `NewsConnector` builds them from
`location_context` as primary location → region → topics, de-duplicated in
order. To override, set `agents.news_sentiment.newsapi_keywords` explicitly.

### Inspecting what a region resolved

```python
config = load_config_for_region("malacca")

config["ingestion"]["shipping"]["ais_bounds"]        # [[0.5, 99.0], [4.0, 105.0]]
config["agents"]["geopolitical"]["acled_countries"]  # ['Malaysia', 'Indonesia', 'Singapore']
config["aisstream"]["monitor_regions"]               # [{'name': 'malacca', 'bbox': [...]}]

from src.ingestion import NewsConnector
NewsConnector(config=config["agents"]["news_sentiment"]).newsapi_keywords
```

Note the connectors receive *sub-blocks*, not the whole config — that is how the
Orchestrator wires them, and why the projection step exists.

---

## Adding a region

1. **Register it** — add a `RegionConfig` to `REGIONS` in `src/core/regions.py`,
   mapping all six agent keys and giving a `passive_reasons` entry for each
   agent set to `False`. `_validate_registry()` fails at import if the mapping
   is incomplete or names an unknown agent.
2. **Write the overlay** — `config/regions/<key>.yaml`, with `region`, `agents`,
   `extraction`, and `aisstream` blocks. Its `enabled` flags must match the
   registry, and its centre point must fall inside its own bounding box.
3. **Run the region tests** — no new test file is needed; the region suites loop
   over `list_regions()` and will pick the new region up automatically:

   ```bash
   pytest tests/test_regions.py tests/test_region_configs.py \
          tests/test_config_manager.py tests/test_region_specific_connectors.py \
          tests/test_region_config_completeness.py tests/test_region_isolation.py
   ```

Evidence standard: a domain gets activated only where a documented real-world
driver exists for that region. Plausibility is not sufficient — Malacca's
`market` domain was removed on exactly that ground. Record the reasoning in
`passive_reasons` so it travels with the data rather than living in commit
history.

---

## Troubleshooting

### `ValueError: Region 'x' not found. Valid regions: [...]`

The key is not in the registry. Check spelling — note `bab_el_mandeb` uses
underscores. If it came from the environment, the message names
`SUPPLY_CHAIN_REGION` explicitly.

### `FileNotFoundError: Base config file not found`

`config/settings.yaml` is missing. The message reports the resolved absolute
path — since paths resolve from the project root, this means the file is
genuinely absent rather than that you are in the wrong directory.

### An overlay is missing → warning, not an error

A missing `config/regions/<key>.yaml` logs a warning and returns `{}`, so the
run continues on base config alone. If a region looks like Hormuz, check that
warning first.

### An agent is missing from `agent_scores`

Expected if it is passive for that region — check the activation table in
`README.md` or:

```python
from src.core.regions import get_region
get_region("panama").passive_agents()          # ['geopolitical', 'routing']
get_region("panama").passive_reasons["geopolitical"]
```

Routing is passive in **all four** regions by a deliberate global decision, not
by per-region evidence.

### A connector logs "no ais_bounds / acled_countries / keywords configured"

Only `api` data mode needs these; CSV and synthetic modes are unaffected. If it
appears on a merged region config, the projection did not run — confirm the
config came from `load_config_for_region()` rather than a bare
`yaml.safe_load("config/settings.yaml")`.

### A region's results look like Hormuz's

Check `config["_active_region"]`, then that the connector received its
sub-block rather than the whole config. `test_settings_are_distinct_across_regions`
in `tests/test_region_config_completeness.py` guards against copy-pasted values
in the overlays themselves.

---

## Known gaps

- **`agents.natural_disaster.monitoring_points` has no `panama` entry.**
  `settings.yaml` covers `hormuz`, `red_sea`, `malacca`, and `suez` only, so
  `DisasterConnector.fetch_api()` has no point to query for Panama. Inert while
  `data_mode` is `synthetic`, which is the default.
- ~~`src/api/endpoints.py` and the Streamlit dashboard are still
  region-blind.~~ **Closed in Phase 12** — both now load through
  `load_config_for_region()`. The dashboard has a region selector and the API
  has `/api/regions/*`; see `docs/DASHBOARD_USAGE.md`. The dashboard's *route
  corridors, vessel markers, news feed and map* remain Hormuz-only, so other
  regions show an activation summary instead.
- **Live API modes are stubs.** `fetch_api()` on the shipping, geopolitical,
  news, and routing connectors raises `NotImplementedError`. The region settings
  are wired and reported in those messages, ready for the integrations.
