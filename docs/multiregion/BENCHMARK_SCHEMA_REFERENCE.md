# EVAL01 Benchmark Schema Reference

Source of truth for the region and scenario YAML schema as actually implemented by
`src/benchmark/regions.py` and `src/benchmark/scenario_generator.py`, cross-checked
against `config/benchmark/hormuz.yaml`, the four `config/benchmark/scenarios/hormuz_*.yaml`
files, `tests/test_benchmark_regions.py`, `tests/test_scenario_batch_generator.py`, and
the `results/baselines/` generation scripts. Every claim below was verified by reading
the code, not inferred from docstrings — where a docstring and the code disagree, that
disagreement is called out explicitly (see §6).

This document describes what exists today. It does not propose changes.

---

## §1 Region Spec Schema

Loaded by `load_region(name)` from `config/benchmark/{name}.yaml`. The file must have a
single top-level key equal to `name` (the filename stem) — `load_region` cross-checks
this and raises `ValueError` if it's missing or mismatched.

| Field | Type | Required | Default | Units | Validated? | Hormuz value |
|---|---|---|---|---|---|---|
| *(top-level key)* | str | required | — | region key, must equal filename stem | **VALIDATED** — `raw.get(name) is None` raises `ValueError` | `hormuz` |
| `center.lat` | float | required | — | decimal degrees latitude | **VALIDATED** (presence only — `KeyError` on `center["lat"]` is caught and re-raised as `ValueError`); no `-90..90` range check | `26.5` |
| `center.lng` | float | required | — | decimal degrees longitude | **VALIDATED** (presence only); no `-180..180` range check | `56.5` |
| `baseline_transits_per_day` | list | required | — | vessel transits / day | **VALIDATED** (presence only, via `KeyError`); no length/order check. See §6 — a bare scalar (not a list) crashes with an *uncaught* `TypeError`, not the friendly `ValueError` | `[60, 80]` |
| `active_domains` | list[str] | optional | `[]` | subset of `KNOWN_DOMAINS` (`shipping`, `market`, `geopolitical`, `routing`, `news`, `disaster`) | **VALIDATED** — any name outside `KNOWN_DOMAINS` raises `ValueError` listing the unknown names | `[shipping, market, geopolitical, routing, news, disaster]` (all six) |
| `reroutable` | bool | required | — | boolean | presence **VALIDATED**; value coerced via `bool(spec["reroutable"])` — no type check. See §6 for a string-quoting gotcha | `false` |
| `loss_scaling` | str | required | — | `"linear"` or `"superlinear"` per docstring | presence **VALIDATED**; value **NOT** checked against those two strings — any string is accepted and just `str()`-cast | `superlinear` |
| `disaster_relevance` | str | required | — | `"low"` / `"none"` / `"high"` per docstring | presence **VALIDATED**; value **NOT** checked against those three strings | `low` |

**Load sequence inside `load_region`:**
1. `path.exists()` check → `FileNotFoundError` if the YAML file is absent.
2. Top-level key match → `ValueError` if missing/mismatched.
3. `active_domains` subset check → `ValueError` if any entry is outside `KNOWN_DOMAINS`.
4. Everything else (`center.lat/lng`, `baseline_transits_per_day`, `reroutable`,
   `loss_scaling`, `disaster_relevance`) is built inside one `try`/`except KeyError` block
   → any of these missing raises `ValueError` naming the missing key (but only the
   *first* missing key encountered — it does not enumerate all of them the way the
   top-level scenario check does).

**`REGION_REGISTRY`:** populated at import time by globbing `config/benchmark/*.yaml`
(non-recursive — `config/benchmark/scenarios/*.yaml` is a different directory and is
never matched). A region YAML that fails `load_region` is **silently skipped** at import
time (`ValueError`/`FileNotFoundError` are caught in the module-level loop) — it will not
appear in `REGION_REGISTRY` and nothing will crash on import. Calling `load_region(name)`
directly still raises for callers who need the failure surfaced. See §6.

---

## §2 Scenario Spec Schema

Loaded by `load_scenario(yaml_path)`. Required top-level keys (checked in one pass,
`ValueError` lists **all** missing keys at once):

```
scenario_id, region, class, seed, days, event, signals, noise
```

| Field | Python attr | Type | Required | Units | Validated? | Hormuz P_CRIT value |
|---|---|---|---|---|---|---|
| `scenario_id` | `scenario_id` | str | yes | identifier | not validated for uniqueness/format vs filename | `hormuz_P_CRIT` |
| `region` | `region` | str | yes | region key | **NOT cross-checked** against `REGION_REGISTRY`/`load_region` inside `load_scenario` — a typo'd region only surfaces later, whenever a caller separately calls `load_region(spec.region)` | `hormuz` |
| `class` | `scenario_class` | str | yes | free-form; only `.strip().upper().startswith("P")` is ever inspected | **NOT** validated against `{P-CRIT, P-HIGH, N-QUIET, N-DECOY}` — any string starting with `P` is treated as a positive class | `P-CRIT` |
| `seed` | `seed` | int | yes | NumPy RNG seed | not validated; routinely overridden by `ScenarioBatchGenerator` per seed in the grid | `42` |
| `days` | `days` | int | yes | days | not validated (`>0` assumed); fixes the materialized series length and date range | `365` |
| `event` | `event` | dict | yes | see below | **VALIDATED** shape, with a gap — see below and §6 | see below |
| `signals` | `signals` | dict | yes (key must be present) | see below | value may be falsy (`{}`/`null`) and is silently accepted as "no domains configured" | see below |
| `noise` | `noise` | dict | yes (key must be present) | see below | value may be falsy and is silently accepted as all-defaults | see below |

### `event` block

Required sub-keys (`_REQUIRED_EVENT_KEYS`): `onset_day`, `ramp_days`, `duration_days`,
`peak_band`. Missing any of these raises `ValueError` listing the missing sub-keys —
**but only if the `event` key's value is itself a mapping.**

**The null/absent event code path — read exactly, do not assume:** `load_scenario` does
`event = dict(raw["event"])`. If the YAML has `event: null`, the key `"event"` *is*
present (so the top-level required-keys check passes), but `raw["event"]` is `None`, and
`dict(None)` raises an **uncaught `TypeError`** (`'NoneType' object is not iterable`) —
not the `ValueError` every other malformed-input path in this module produces. Nothing
catches this. **`event: null` is not a supported way to express "no event."**

Hormuz's actual "no event" scenario (`N_QUIET`) does **not** use `event: null`. It
supplies a fully-formed `event` mapping with all-zero/placeholder values:
```yaml
event:
  onset_day: 0
  ramp_days: 0
  duration_days: 0
  peak_band: none
```
This block is syntactically required and must satisfy the four-key check even though —
because `N_QUIET`'s `class` doesn't start with `P` and every domain's `effect` is
`null` — none of these four values are ever functionally used to shape output. They are
inert placeholders, not "the event that didn't happen."

| `event` sub-key | Type | Units | Validated? |
|---|---|---|---|
| `onset_day` | int-like | day-index, 0-based offset into the `days`-length series | not range-checked; can be negative or `>= days` — any effect window that falls entirely outside `[0, days)` is silently skipped (no error, no data change) |
| `ramp_days` | int-like | days | not range-checked; used for **both** the ramp-up length *and* the ramp-down (decay) length — one value controls both slopes symmetrically |
| `duration_days` | int-like | days; total window = ramp + plateau + decay | not range-checked; if `duration_days < 2 * ramp_days` the plateau collapses to zero and the decay phase truncates (`_intensity_curve`) |
| `peak_band` | str | free-form label copied verbatim into `y_band` across the disruption window, for `P-*` classes only | **NOT** validated against any enum here. `ScenarioBatchGenerator.BAND_TO_INT` only recognizes `{quiet, none, low, medium, high, critical}` — an unrecognized `peak_band` loads and materializes fine but silently becomes `NaN` in `y_band_int` at batch-generation time |

### `signals` block

Keyed by domain name; only keys inside `KNOWN_DOMAINS` (`shipping`, `market`,
`geopolitical`, `routing`, `news`, `disaster`) are ever read — `materialize_scenario`
iterates `KNOWN_DOMAINS` and does `spec.signals.get(domain)`, so a misspelled domain key
(e.g. `shiping`) is **silently ignored**, not an error (see §6).

A domain's column in the materialized output is **all-NaN (disabled)** if either:
- the domain is not in `region.active_domains`, **or**
- `spec.signals.get(domain)` is falsy (missing key, `null`, or an empty dict `{}`).

Otherwise the domain is enabled, and:

| `signals.<domain>` sub-key | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `baseline.mean` | float | optional | `0.0` | domain's steady-state level (units below) |
| `baseline.std` | float | optional | `0.0` | domain's steady-state noise scale (units below); `std <= 0` makes the AR(1) noise term identically zero |
| `effect` | dict or `null`/absent | optional | `null` → **enabled-but-silent**: baseline + noise only, no ramp | the event's effect on this domain |
| `effect.type` | str | required if `effect` present | — | `"multiplicative_ramp"` or `"additive_ramp"` — **VALIDATED** at `materialize_scenario` time (`ValueError: Unknown effect type`) for anything else |
| `effect.target` | float | required if `effect` present | — | ramp target (units/semantics depend on `type`, see below) — **NOT VALIDATED**: a missing `target` raises an **uncaught `KeyError`**, not a `ValueError` |
| `effect.lead_days` | int | optional | `0` | shifts the effect window relative to `event.onset_day`: positive = signal moves *before* the physical onset, negative = lags after it |

`effect.type` semantics:
- `multiplicative_ramp`: `value = value * (1 + (target - 1) * strength)`, where
  `strength ∈ [0, 1]` is the ramp/plateau/decay intensity curve. `target` is a
  **multiplier on the domain's own baseline**, e.g. `target: 0.40` on `shipping` ramps
  the signal *down* toward 40% of baseline at full intensity (a ~60% drop in vessel
  transits).
- `additive_ramp`: `value = value + target * strength`. `target` is added directly **in
  the domain's own units** — e.g. `geopolitical` baseline `0.05` + `target: 0.80` at full
  strength ≈ `0.85`. Nothing clips this back into `[0, 1]`; an additive target can push a
  0–1-scaled domain above 1 or below 0 if chosen carelessly.

**UNITS FOR EVERY NUMERIC SIGNAL FIELD** (this is the section that matters most — a
wrong unit here silently distorts a signal rather than raising anything):

| Domain | Unit convention | Evidence |
|---|---|---|
| `shipping` | **Raw domain quantity** — vessel-transit count/day, same units as `region.baseline_transits_per_day`. Hormuz baseline `mean: 70, std: 5` sits inside the region's `[60, 80]` range. **NOT** a 0–1 score. | Hormuz region spec + `tests/test_scenario_batch_generator.py::test_signals_not_clipped` ("Shipping is vessel counts (~70 +/- 5), not a [0,1] proportion") |
| `market` | **Raw domain quantity**, unbounded, z-score-like (Hormuz baseline `mean: 0.0, std: 1.0`). Legitimately goes negative. **NOT** a 0–1 score, **NOT** a percentage. | Hormuz YAMLs + `test_signals_not_clipped` ("Market is an unbounded z-score-like signal and does go negative") |
| `geopolitical` | **0–1 normalized risk score.** | `src/ingestion/geopolitical_connector.py` explicitly logs an error if a value leaves `[0, 1]` — the domain this scenario generator's docstring says it mirrors |
| `routing` | **0–1 normalized score** — same convention as `geopolitical`/`news` (baseline means/stds and effect targets in every Hormuz spec stay well inside `[0, 1]`) | inferred from consistent scale across all four Hormuz scenario files; not independently confirmed against `routing_connector.py` in this pass — see §6 |
| `news` | **0–1 normalized score**, same convention | as above |
| `disaster` | **0–1 normalized score**, same convention | as above |

A prior real incident put a raw percentage (`15`–`70`) into a field expecting a 0–1
score — a ~100x error. Nothing in `load_scenario` or `materialize_scenario` would catch
that: there is no per-domain range validation anywhere in this pipeline. The table above
is the only thing standing between a new-region author and that exact mistake.

### `noise` block

| `noise` sub-key | Type | Required | Default | Units | Validated? |
|---|---|---|---|---|---|
| `autocorrelation` | float | optional | `0.0` | dimensionless AR(1) coefficient (φ) | **clamped internally** to `[-0.99, 0.99]` regardless of the YAML value — an out-of-range value is silently clipped, never rejected |
| `seasonality.period` | float | optional | `0` | days (e.g. `365` = annual cycle) | not validated; `period <= 0` silently disables seasonality |
| `seasonality.amplitude` | float | optional | `0.0` | **dimensionless fraction of the domain's own baseline `std`** — NOT a fraction of the mean, NOT a raw percentage of the value. `amplitude: 0.08` means a seasonal swing of up to ±8% of that domain's `std`, scaled into the signal as `amplitude * std * sin(2πt/period)` | not validated |
| `missing_data_rate` | float | optional | `0.0` | fraction in `[0, 1]` — probability a given day is masked `NaN`, per domain | **NOT clamped**: a negative rate behaves as "never mask" and a rate `> 1` masks every day, neither raises |

All four Hormuz scenario files use the identical `noise` block
(`autocorrelation: 0.35`, `seasonality: {period: 365, amplitude: 0.08}`,
`missing_data_rate: 0.02`) — this is a convention, not an enforced constraint.

---

## §3 Scenario Class Semantics

All four Hormuz scenarios share `region: hormuz`, `seed: 42`, `days: 365`, and an
identical `noise` block. What differs:

| | `P_CRIT` | `P_HIGH` | `N_QUIET` | `N_DECOY` |
|---|---|---|---|---|
| Ground-truth label | **Positive** (`class` starts with `P`) | **Positive** | Negative | Negative |
| `onset_day` | 240 | 120 | 0 (inert) | 200 |
| `ramp_days` | 12 | 7 | 0 (inert) | 3 |
| `duration_days` | 60 | 25 | 0 (inert) | 15 |
| `peak_band` | `critical` | `high` | `none` (inert) | `none` |
| Domains with an active `effect` | shipping, market, geopolitical, routing, news (5 of 6) | same 5 | none | news only (1 of 6) |
| `disaster` domain | enabled-but-silent (`effect: null`) | enabled-but-silent | enabled-but-silent | enabled-but-silent |

**Why `disaster` is always silent for Hormuz:** every Hormuz scenario is a
geopolitical/blockage event, not a natural-disaster one — `disaster` stays
enabled-but-silent (baseline + noise, no ramp) in all four files, and
`test_silent_agent_handling` asserts its mean stays near baseline (`0.01 < mean < 0.04`).

**What makes `N_DECOY` a decoy:** it is the only negative-class scenario in which a
domain *does* ramp — `news` gets `effect: {type: additive_ramp, target: 0.55,
lead_days: 0}` over a 15-day window starting day 200 — while `shipping`, `market`,
`geopolitical`, `routing`, and `disaster` all stay flat. Ground truth still comes out
negative for the whole scenario, because `_ground_truth_labels` derives `y_disruption`
purely from `scenario_class.startswith("P")` — it never inspects `signals` at all. A
real, single-domain signal move with no corroboration in the other five domains and a
`class` that isn't `P-*` is *by construction* a guaranteed negative. The scenario exists
to catch detectors that over-trigger on one noisy domain.

**`N_QUIET` vs. `N_DECOY`:** `N_QUIET` is the pure negative control — every domain's
`effect` is `null`, nothing ramps anywhere, `event` is inert placeholder zeros.
`N_DECOY` is a harder negative — one domain genuinely moves, but the ground truth is
still negative.

**`P_CRIT` vs. `P_HIGH` — the exact intensity relationship** (same 5 domains ramp in
both; only `disaster` stays silent in both):

| Field | `P_CRIT` | `P_HIGH` | Direction |
|---|---|---|---|
| `duration_days` | 60 | 25 | CRIT is 2.4x longer |
| `ramp_days` | 12 | 7 | CRIT ramps in ~1.7x more days |
| `peak_band` | `critical` | `high` | one severity tier apart |
| `shipping` target (multiplicative) | `0.40` | `0.65` | CRIT drops capacity further (to 40% of baseline vs. 65%) |
| `market` target (additive) | `+2.5` | `+1.4` | CRIT spikes ~1.8x harder |
| `geopolitical` target (additive) | `+0.80` | `+0.45` | CRIT ~1.8x higher |
| `routing` target (additive) | `+0.60` | `+0.30` | CRIT exactly 2x higher |
| `news` target (additive) | `+0.75` | `+0.40` | CRIT ~1.9x higher |
| `market` lead_days | `-5` | `-3` | CRIT's market signal leads onset by 2 more days |
| `geopolitical` lead_days | `7` | `4` | CRIT lags onset by 3 more days |
| `routing` lead_days | `5` | `3` | CRIT lags onset by 2 more days |
| `news` lead_days | `10` | `5` | CRIT lags onset by 5 more days |

In every domain, `P_CRIT`'s effect target is roughly 1.8–2x `P_HIGH`'s, and the window
it plays out over is longer — a scaled-up, longer version of the identical shape and
mechanism, not a structurally different scenario. This is deliberate:
`test_materialize_hormuz_4_scenarios`-adjacent test comments describe the intent as
"detectors scale their response to event magnitude rather than firing identically on
every positive."

---

## §4 Validation Checklist

The exact call sequence a new region/scenario pair must survive:

1. **`load_region(region_name)`** — raises `FileNotFoundError` (no YAML) or `ValueError`
   (bad top-level key, unknown `active_domains`, or a missing required region field).
2. **`load_scenario(yaml_path)`** — raises `ValueError` for missing top-level or
   `event` sub-keys. **Exception:** `event: null` raises an uncaught `TypeError`
   instead (see §2, §6).
3. **`materialize_scenario(spec, region)`** — raises `ValueError` for an unrecognized
   `effect.type`, or an uncaught `KeyError` for a missing `effect.target`. Everything
   else (unit mismatches, unrecognized `peak_band`, misspelled signal-domain keys,
   out-of-range noise parameters) materializes *without* raising — it just produces
   silently wrong output. Passing this step is necessary but not sufficient evidence
   that a new YAML is correct.
4. **(optional, for full benchmark parity)**
   `ScenarioBatchGenerator.generate_scenario(scenario_id, seed, region_name)` — repeats
   steps 1–3 internally, then derives `y_band_int`/`y_action_int` from `y_band` +
   `region.reroutable` (silently `NaN` for an unrecognized `peak_band` — see §2) and
   writes a parquet file.

### Copy-pasteable validation snippet

Run from the project root (`supply-chain-dss/`):

```python
"""Validate a region + scenario pair through load_region -> load_scenario ->
materialize_scenario, and print a pass/fail summary plus per-domain stats.

Usage: python validate_scenario.py [region_name] [scenario_id]
Defaults to hormuz / hormuz_P_CRIT.
"""
import sys

from src.benchmark.regions import KNOWN_DOMAINS, load_region
from src.benchmark.scenario_generator import load_scenario, materialize_scenario


def validate(region_name: str, scenario_id: str) -> bool:
    try:
        region = load_region(region_name)
        print(f"[PASS] load_region({region_name!r}) -> "
              f"active_domains={region.active_domains}, reroutable={region.reroutable}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] load_region({region_name!r}): {exc}")
        return False

    yaml_path = f"config/benchmark/scenarios/{scenario_id}.yaml"
    try:
        spec = load_scenario(yaml_path)
        print(f"[PASS] load_scenario({yaml_path!r}) -> "
              f"class={spec.scenario_class}, days={spec.days}, seed={spec.seed}")
    except (ValueError, TypeError) as exc:
        print(f"[FAIL] load_scenario({yaml_path!r}): {exc!r}")
        return False

    if spec.region != region_name:
        print(f"[WARN] spec.region={spec.region!r} != requested region {region_name!r} "
              f"(load_scenario does not cross-check this)")

    try:
        df = materialize_scenario(spec, region)
        print(f"[PASS] materialize_scenario -> shape={df.shape}")
    except (ValueError, KeyError) as exc:
        print(f"[FAIL] materialize_scenario: {exc!r}")
        return False

    print(f"date range: {df['timestamp'].min()} .. {df['timestamp'].max()}")
    for domain in KNOWN_DOMAINS:
        col = df[domain]
        if col.isna().all():
            print(f"  {domain:12s} DISABLED (all-NaN)")
            continue
        print(f"  {domain:12s} min={col.min():.4f} max={col.max():.4f} "
              f"mean={col.mean():.4f} n_nan={int(col.isna().sum())}")

    print(f"y_disruption positives: {int(df['y_disruption'].sum())} / {len(df)} days")
    print(f"y_band values seen: {sorted(df['y_band'].unique())}")
    return True


if __name__ == "__main__":
    region_name = sys.argv[1] if len(sys.argv) > 1 else "hormuz"
    scenario_id = sys.argv[2] if len(sys.argv) > 2 else "hormuz_P_CRIT"
    ok = validate(region_name, scenario_id)
    print("RESULT:", "PASS" if ok else "FAIL")
```

---

## §5 Baseline Pipeline

**Generation (region → parquet):**
`scripts/generate_hormuz_benchmark.py` calls
`ScenarioBatchGenerator.generate_batch(DEFAULT_SCENARIOS, DEFAULT_SEEDS,
region_name="hormuz")` where (in `src/benchmark/scenario_batch_generator.py`):
- `DEFAULT_SCENARIOS = ("hormuz_P_CRIT", "hormuz_P_HIGH", "hormuz_N_QUIET",
  "hormuz_N_DECOY")`
- `DEFAULT_SEEDS = (42, 123, 456, 789, 999)`

This is a **hardcoded 4×5 grid, Hormuz-specific by name** — no CLI arguments, no region
parameter exposed at the script level. It writes 20 parquet files
(`{scenario_id}_seed_{seed}.parquet`) to `data/benchmark/scenarios_generated/`
(confirmed: 20 files present). Each file's `y_band_int`/`y_action_int` columns are
derived from `region.reroutable` via `ScenarioBatchGenerator.BAND_TO_INT` (§2, §6).

**Evaluation (parquet → JSON results):** three tier scripts + one ablation script, all
under `results/baselines/`, all reading **every** parquet file in
`data/benchmark/scenarios_generated/` via an unfiltered glob (`*.parquet`, or
`*_seed_42.parquet` for ablations) — none of them take a region argument or filter by
region (see §6). All four use the identical pre-declared split:
`val = [201, 281)` (80 days), `test = [281, 365)` (84 days), matching
`tests/test_benchmark_regions.py::test_scenario_split`.

| Script | Baselines | Grid | Result files | Confirmed count |
|---|---|---|---|---|
| `scripts/run_tier0_baselines.py` | `RandomBaseline`, `AlwaysAlarmBaseline`, `NeverAlarmBaseline` | 20 parquet × 3 baselines | `results/baselines/tier0/{scenario}_{baseline}_seed_{seed}.json` | 60 |
| `scripts/run_tier1_baselines.py` | `ZScoreBaseline`, `EWMABaseline`, `CUSUMBaseline`, `SARIMABaseline`, `PersistenceBaseline` | 20 parquet × 5 baselines | `results/baselines/tier1/...` | 100 |
| `scripts/run_tier2_baselines.py` | `IsolationForestBaseline(contamination=0.1, n_estimators=100)`, `MatrixProfileBaseline(m=30)` | 20 parquet × 2 baselines | `results/baselines/tier2/...` | 40 |
| `scripts/run_ablations.py` | `ABLATIONS` dict (A0–A7, 8 configs) from `src/baselines/ablations.py` | **seed=42 only** ("single seed for speed, per rule 6") × 4 scenarios × 8 configs | `results/baselines/ablations/{scenario}_{A#}_seed_42.json` + `ablation_summary.csv` | 32 + 1 = 33 |

None of these five scripts accept command-line arguments — all paths, grids, and seeds
are hardcoded module constants. Run each with `python scripts/<name>.py` from the
project root, in this order: `generate_hormuz_benchmark.py` → the three tier scripts
(any order) → `run_ablations.py`.

**Aggregation:** `scripts/aggregate_all_results.py` reads all of `tier0/`, `tier1/`,
`tier2/`, and `ablations/` and produces the top-level `results/*.csv` /
`results/benchmark_summary.md` files. It declares 8 measured metrics
(`D3_auc_pr`, `D4_auc_roc`, `D5_f1_tau`, `D6_best_f1`, `D7_precision_tau`,
`D8_recall_tau`, `D9_fpr_tau`, `D10_macro_f1`, computed by `BaselineEvaluator`) and 7
declared-but-unmeasured placeholders (`D1_vus_pr`, `D2_vus_roc`, `D11_event_f1`,
`D12_range_f1`, `D13_affiliation_f1`, `D14_pa_f1`, `D15_pa_k_0_2` — carried through as
`NaN`, not silently dropped). `scripts/generate_results_summary.py` is a further
downstream consumer of these CSVs; it was not read in full for this pass (out of the
region/scenario-schema scope) and its internals are not documented here.

**What a new region needs to reach parity with Hormuz's coverage:**
1. `config/benchmark/{region}.yaml` + four scenario YAMLs
   (`{region}_P_CRIT/_P_HIGH/_N_QUIET/_N_DECOY`) passing the §4 checklist.
2. A generation step producing the same 4×5 = 20 parquet grid — since
   `generate_hormuz_benchmark.py` hardcodes `DEFAULT_SCENARIOS`/`region_name="hormuz"`,
   this means either editing that script or calling
   `ScenarioBatchGenerator(...).generate_batch([...new scenario ids...], DEFAULT_SEEDS,
   region_name="{region}")` directly — there is no generic, parameterized entry point
   today.
3. Running the three tier scripts + `run_ablations.py` — as written, these will pick up
   **every** parquet file present in `data/benchmark/scenarios_generated/`, Hormuz's
   included, since none of them filter by region or scenario prefix (see §6).
4. Re-running `scripts/aggregate_all_results.py` to fold the new results into the
   top-level CSVs.

---

## §6 Gaps and Ambiguities

Flagged, not resolved. A new-region author would have to guess at or independently
decide each of these:

1. **`event: null` is not a supported "no event" spelling.** `load_scenario` does
   `dict(raw["event"])`; a literal `null` raises an uncaught `TypeError`, not the
   `ValueError` every other malformed-field path in this loader produces. The actual
   supported pattern (per `N_QUIET`) is a fully-populated `event` block with all-zero
   values, combined with every `signals.<domain>.effect: null`.

2. **`peak_band` has no enum validation at load time**, but a downstream consumer
   (`ScenarioBatchGenerator.BAND_TO_INT`) only recognizes
   `{quiet, none, low, medium, high, critical}`. An unrecognized value passes
   `load_scenario`/`materialize_scenario` cleanly and only manifests as a silent `NaN`
   in `y_band_int` two layers downstream, with no error anywhere in the chain.

3. **`class` is not validated against `{P-CRIT, P-HIGH, N-QUIET, N-DECOY}`.**
   Positivity is decided purely by `scenario_class.strip().upper().startswith("P")` —
   any string starting with `P` is a positive scenario; the rest of the string is
   cosmetic (used only in output filenames/labels).

4. **`region` inside a scenario YAML is never cross-checked against `REGION_REGISTRY`
   or the region actually passed into `materialize_scenario`.** `load_scenario` reads it
   and stores it, but nothing enforces `spec.region == region.name` at any point in the
   pipeline described here.

5. **`effect.target` has no presence guard.** A `signals.<domain>.effect` block with a
   `type` but no `target` raises a raw, uncaught `KeyError` from inside
   `_apply_effect`, not a `ValueError` — inconsistent with the rest of the module's
   error handling.

6. **`baseline_transits_per_day` accepting "a single value" (per the `Region`
   docstring) does not match the code.** `load_region` does
   `list(spec["baseline_transits_per_day"])`; a bare scalar (e.g. `baseline_transits_per_day: 70`)
   raises an uncaught `TypeError` (`'int' object is not iterable`), not a `ValueError`.
   Only a list/tuple value works in practice, contradicting the docstring's parenthetical.

7. **`reroutable: "false"` (quoted string) would silently invert.** The cast is
   `bool(spec["reroutable"])`; `bool("false")` is `True` in Python. This only matters if
   a YAML author quotes the boolean — unquoted `true`/`false` parse to real Python
   booleans via `yaml.safe_load` and are unaffected — but nothing guards against the
   quoted case.

8. **No per-domain unit/range validation anywhere in the loader.** The §2 units table
   (raw count for `shipping`, unbounded z-score-like for `market`, 0–1 for the other
   four) is a *convention* read off the Hormuz files and one corroborating check in
   `geopolitical_connector.py` — it is not enforced by `load_scenario` or
   `materialize_scenario`. A new-region author who puts a raw percentage into a 0–1
   field (the exact incident this document was commissioned to prevent a repeat of)
   gets no error, just a silently distorted signal.

9. **`routing`'s 0–1 convention was not independently re-confirmed against
   `routing_connector.py`.** It is inferred from consistent scale across all four Hormuz
   scenario files (baseline means/stds and effect targets that never leave `[0, 1]`),
   the same way `geopolitical`/`news`/`disaster` are, but `routing_connector.py` itself
   was outside this pass's required reading list and was not opened to independently
   verify the `[0, 1]`-clamp-and-log behavior confirmed for `geopolitical_connector.py`.

10. **A misspelled `signals` domain key is a silent no-op.**
    `materialize_scenario` only ever reads `spec.signals.get(domain)` for
    `domain in KNOWN_DOMAINS`; a key like `shiping` (typo) is never read, and the
    intended `shipping` domain — now absent from `signals` — is silently disabled
    (all-`NaN`) rather than raising a "did you mean" error or any error at all.

11. **An empty per-domain signal block (`{}`) is indistinguishable from "disabled."**
    `if domain not in region.active_domains or not sig:` — an empty dict is falsy in
    Python, so `signals: {shipping: {}}` disables `shipping` exactly the same as
    omitting it entirely, rather than being treated as "baseline mean=0, std=0."

12. **`noise.missing_data_rate` is not clamped to `[0, 1]`**, unlike `autocorrelation`
    (which *is* clamped to `[-0.99, 0.99]`). A value outside `[0, 1]` changes masking
    behavior at the extremes without any warning.

13. **`REGION_REGISTRY` silently drops malformed region YAMLs at import time.** A typo'd
    new `config/benchmark/{region}.yaml` won't crash any import — it will simply be
    absent from `REGION_REGISTRY`, discoverable only by explicitly calling
    `load_region(name)` and checking, or by a `KeyError` wherever code assumes
    `REGION_REGISTRY[name]` exists.

14. **`loss_scaling` and `disaster_relevance` are read once, stored, and never
    consumed again within `regions.py` or `scenario_generator.py`.** Whether anything
    else in the codebase (e.g. a risk-scoring module) actually reads these fields off
    the loaded `Region` object was out of scope for this pass and is an open question —
    per `STATE_OF_PROJECT_MULTIREGION.md`, `RiskEngine`'s own `region` parameter is
    documented as cosmetic/log-only, which raises the same question for these two
    fields specifically.

15. **None of the tier0/tier1/tier2/ablation scripts filter by region.** They glob
    every parquet file in `data/benchmark/scenarios_generated/` unconditionally. Adding
    a second region's parquet files to that directory means every subsequent baseline
    run evaluates across all regions mixed together in one pass — separable afterward
    only by the `{region}_{CLASS}` filename convention, which is incidental, not
    enforced.

16. **No generic "generate benchmark for region X" entry point exists.**
    `scripts/generate_hormuz_benchmark.py` is Hormuz-specific by name and by hardcoded
    `DEFAULT_SCENARIOS`/`region_name="hormuz"`. Per `MULTIREGION_IMPLEMENTATION_SEQUENCE.md`
    Prompt 11, a second region requires either editing this script or writing an
    equivalent, region-parameterized one — this does not exist today.
