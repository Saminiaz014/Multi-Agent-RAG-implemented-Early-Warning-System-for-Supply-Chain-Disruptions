# Ablation Rationale: Why Domain Scorers, Not Production Agents

## Problem

Production agents (`ShippingAgent`, `MarketAgent`, etc.) expect rich,
multi-column schemas:

- `ShippingAgent`: `vessel_count`, `avg_delay_hours`, `congestion_index`
- `MarketAgent`: `brent_crude_usd`, `trade_volume_index`, `freight_rate_index`
- `GeopoliticalAgent`: `sanctions_severity`, `military_activity_index`, `diplomatic_incident_score`, `regime_stability_index`
- `RoutingAgent`: `rerouting_percentage`, `avg_route_deviation_km`, `transit_volume_ratio`, `vessels_holding`, `alternative_route_traffic`
- `NewsAgent`: `sentiment_score`, `sentiment_magnitude`, `source_consensus`, `article_volume`, `recency_weighted_score`
- `DisasterAgent`: `earthquake_severity`, `tsunami_risk`, `cyclone_severity`, `severe_weather_index`

R3's synthetic scenario generator (`src/benchmark/scenario_generator.py`)
produces a *single float per domain* — one `shipping` value, one `market`
value, etc. There is no adapter that turns one float into 3-5 semantically
distinct sub-features without fabricating data that was never part of
R1's generative model.

## Solution

Use lightweight **domain scorers** (`src/baselines/domain_scorers.py`) for
ablations — the same rolling z-score pattern as the Tier 1 SPC baselines
(`ZScoreBaseline` in `tier1_statistical.py`).

## What Ablations Measure

**Aggregation strategy value**, not agent quality:

- A0 -> A7: does weighting, and then tuning that weighting, and then a
  consensus bonus on top, add value?
- Specifically: does the agreement bonus reduce false alarms on N-DECOY
  scenarios, where one domain spikes but the others stay quiet?

## Why This Is Scientifically Honest

1. **Schema stays consistent.** R3 produces one float per domain; domain
   scorers consume exactly that — no fabrication.
2. **Not reinventing agents.** Domain scorers are benchmarking proxies
   (the same pattern as CUSUM/EWMA in Tier 1), not attempts to replicate
   production logic.
3. **Focuses the question.** Ablations isolate aggregation strategy
   rather than confounding it with "how good is the real agent
   implementation?"
4. **Production agents are untouched.** They're tested separately
   (`tests/test_agents.py`, `tests/test_new_agents.py`) against their own
   multi-column schemas. This benchmark doesn't touch them.

## A5's Weights Are Actually Tuned, Not Just Labeled "Optuna"

A5/A6/A7 run a real Optuna search (`ablation_runner.tune_weights_optuna`)
per scenario, 50 trials, maximizing best-achievable F1 on the validation
split (days 201-280) — not a hand-typed placeholder presented as if it
were optimized. The one substitution: rule 5 calls for optimizing
VUS-PR (metric D1), but D1 isn't implemented in `BaselineEvaluator` (it
needs `tslearn`, not a project dependency — see R4's documented NaN
placeholders). Best-F1-on-validation is used instead, the same
substitute objective already used to tune EWMA's lambda and CUSUM's
threshold in R5.

## Domain Scorer Limitations

- Single rolling z-score per domain (simpler than production agents)
- Only detects univariate deviations, not multivariate patterns within a domain
- No domain-specific knowledge (seasonality, holiday effects, etc.)

**This is intentional.** Ablations test "does the aggregation strategy
work?" on a level playing field, not "did we implement shipping better
than market?"

## External Validity

Results on domain scorers do **not** claim to predict production-agent
performance in absolute terms. They're directional:

- If the agreement bonus helps here, it likely helps with real agents too.
- If tuned weighting adds value here, it likely adds value in production.
- Real agent performance will differ (plausibly better, given their
  domain-specific sophistication).

The ablation's value is *comparative* — "config A beats config B by X on
a controlled benchmark" — not an absolute claim about the deployed system.

## For Thesis Defense

**Examiner may ask:** "Why didn't you use the real agents in your ablations?"

**Answer:**

- Production agents require multi-column schemas that R3 doesn't
  generate; fabricating that mapping would mean inventing data.
- Ablations test aggregation strategy, not agent quality — lightweight
  domain scorers (the same pattern as the Tier 1 SPC baselines) keep the
  benchmark schema honest.
- Production agents are tested separately, against their own schemas.
  This benchmark isolates the value of weighting and consensus logic.
- Results are directional, not absolute: if the bonus helps on scorers,
  it likely helps with real agents.
