# Results: every method, per region

Source: `eval/method_comparison_results.csv`; regenerate with `python scripts/results_table.py`. This is the one results table: other documents point here.

- AUC: 95% moving-block bootstrap intervals (30-day blocks, 1,000 resamples). ΔAUC: the paired difference against Tier 1 on the same resampled blocks (tiers only); bold when its interval excludes 0. F1, FAR and alert rate are point estimates (intervals are in `results_main.csv`).
- FAR: false-alarm rate, FP/(FP+TN). Alert rate: share of scored days that alert. A day alerts when its score is at or above the method's training 90th percentile; M1, M2 and the ORACLE set alerts directly (M1 every day, M2 never, the ORACLE when yesterday's label is 1).
- Without positives (Malacca) AUC and F1 are undefined and every alert is false, so FAR equals the alert rate.
- `= Tier k`: copies the tier before it (the agent it would add is disabled in that region). `same numbers as Tier k`: a different agent set with identical metrics (an enabled agent that adds nothing, e.g. a constant one). n/a: B7 and B8 need positives in the training window.
- † fits on every feature column, geopolitical included (B3, B7, B8), so it moves when those columns do. Tiers 3-5 also include the geopolitical agent wherever it is enabled (every region except Panama); only B3, B7 and B8 are marked because only they fit on the raw columns.
- Best non-ceiling method: the highest AUC point estimate among methods and tiers, with the methods whose interval overlaps it named. Controls M0-M2 are listed but cannot be the best. The ORACLE reads yesterday's label: a ceiling row, never ranked.

### bab_el_mandeb — 244 positives among 857 scored test days (2023-07-01 .. 2025-11-21)

| Method | n agents | AUC [CI] | ΔAUC vs Tier 1 [CI] | F1 | FAR | Alert rate |
|---|---|---|---|---|---|---|
| M0 random | — | 0.494 [0.444, 0.541] | — | 0.192 | 9.3% | 10.4% |
| M1 always alert | — | 0.500 [0.500, 0.500] | — | 0.443 | 100.0% | 100.0% |
| M2 never alert | — | 0.500 [0.500, 0.500] | — | 0.000 | 0.0% | 0.0% |
| M3 persistence | — | 0.966 [0.920, 0.996] | — | 0.778 | 19.4% | 40.8% |
| M8 matrix profile | — | 0.489 [0.426, 0.564] | — | 0.111 | 7.3% | 7.2% |
| B1 rolling z | — | 0.525 [0.494, 0.557] | — | 0.133 | 10.6% | 10.2% |
| B2 MA crossover | — | 0.595 [0.499, 0.715] | — | 0.380 | 15.2% | 20.1% |
| B3 isolation forest † | — | 0.560 [0.454, 0.645] | — | 0.408 | 46.2% | 48.8% |
| B4 EWMA deviation | — | 0.544 [0.505, 0.594] | — | 0.258 | 17.5% | 18.6% |
| B5 AR residual | — | 0.509 [0.491, 0.521] | — | 0.066 | 7.7% | 6.7% |
| B6 CUSUM | — | 0.308 [0.113, 0.500] | — | 0.480 | 69.2% | 74.1% |
| B7 logistic regression † | — | n/a | — | n/a | n/a | n/a |
| B8 random forest † | — | n/a | — | n/a | n/a | n/a |
| Tier 1 | 1 | 0.747 [0.672, 0.802] | — | 0.517 | 71.9% | 79.3% |
| Tier 2 | 2 | 0.638 [0.526, 0.715] | **-0.109 [-0.190, -0.044]** | 0.477 | 51.2% | 57.1% |
| Tier 3 | 3 | 0.492 [0.384, 0.567] | **-0.255 [-0.349, -0.175]** | 0.422 | 68.8% | 70.0% |
| Tier 4 (= Tier 3) | 3 | 0.492 [0.384, 0.567] | **-0.255 [-0.349, -0.175]** | 0.422 | 68.8% | 70.0% |
| Tier 5 | 4 | 0.513 [0.399, 0.594] | **-0.234 [-0.333, -0.152]** | 0.450 | 68.0% | 71.1% |
| *ORACLE label t-1 (ceiling, not a method)* | — | 0.997 [0.991, 1.000] | — | 0.996 | 0.2% | 28.5% |

**Best non-ceiling method:** M3 persistence, the highest AUC point estimate, 0.966 [0.920, 0.996] (circularity high). Its interval overlaps no other method's. Best multi-agent tier: Tier 1, AUC 0.747 [0.672, 0.802]; its interval does not overlap the best's.

### hormuz — 133 positives among 857 scored test days (2023-07-01 .. 2025-11-21)

| Method | n agents | AUC [CI] | ΔAUC vs Tier 1 [CI] | F1 | FAR | Alert rate |
|---|---|---|---|---|---|---|
| M0 random | — | 0.517 [0.462, 0.571] | — | 0.153 | 9.9% | 10.4% |
| M1 always alert | — | 0.500 [0.500, 0.500] | — | 0.269 | 100.0% | 100.0% |
| M2 never alert | — | 0.500 [0.500, 0.500] | — | 0.000 | 0.0% | 0.0% |
| M3 persistence | — | 0.848 [0.785, 0.902] | — | 0.465 | 13.5% | 19.6% |
| M8 matrix profile | — | 0.600 [0.509, 0.696] | — | 0.223 | 9.1% | 10.6% |
| B1 rolling z | — | 0.478 [0.435, 0.518] | — | 0.104 | 11.9% | 11.4% |
| B2 MA crossover | — | 0.465 [0.374, 0.558] | — | 0.008 | 17.1% | 14.6% |
| B3 isolation forest † | — | 0.455 [0.359, 0.540] | — | 0.119 | 18.6% | 17.7% |
| B4 EWMA deviation | — | 0.490 [0.450, 0.533] | — | 0.090 | 14.0% | 13.1% |
| B5 AR residual | — | 0.492 [0.471, 0.513] | — | 0.077 | 12.6% | 11.7% |
| B6 CUSUM | — | 0.935 [0.865, 0.982] | — | 0.000 | 0.0% | 0.0% |
| B7 logistic regression † | — | 0.939 [0.877, 0.981] | — | 0.044 | 0.0% | 0.4% |
| B8 random forest † | — | 0.644 [0.540, 0.756] | — | 0.285 | 11.6% | 14.0% |
| Tier 1 | 1 | 0.574 [0.516, 0.633] | — | 0.028 | 19.9% | 17.3% |
| Tier 2 | 2 | 0.497 [0.384, 0.594] | **-0.078 [-0.161, -0.007]** | 0.032 | 15.2% | 13.3% |
| Tier 3 | 3 | 0.465 [0.350, 0.549] | **-0.109 [-0.207, -0.034]** | 0.129 | 25.8% | 24.4% |
| Tier 4 | 4 | 0.466 [0.350, 0.550] | **-0.108 [-0.206, -0.033]** | 0.129 | 25.8% | 24.4% |
| Tier 5 | 5 | 0.456 [0.344, 0.537] | **-0.118 [-0.217, -0.045]** | 0.137 | 24.7% | 23.6% |
| *ORACLE label t-1 (ceiling, not a method)* | — | 0.987 [0.969, 0.997] | — | 0.977 | 0.4% | 15.5% |

**Best non-ceiling method:** B7 logistic regression †, the highest AUC point estimate, 0.939 [0.877, 0.981] (circularity medium). Its interval overlaps M3 persistence, B6 CUSUM: not a clear win. Best multi-agent tier: Tier 1, AUC 0.574 [0.516, 0.633]; its interval does not overlap the best's.

### malacca — 0 positives among 857 scored test days (2023-07-01 .. 2025-11-21)

| Method | n agents | AUC [CI] | ΔAUC vs Tier 1 [CI] | F1 | FAR | Alert rate |
|---|---|---|---|---|---|---|
| M0 random | — | — | — | — | 10.4% | 10.4% |
| M1 always alert | — | — | — | — | 100.0% | 100.0% |
| M2 never alert | — | — | — | — | 0.0% | 0.0% |
| M3 persistence | — | — | — | — | 4.1% | 4.1% |
| M8 matrix profile | — | — | — | — | 10.7% | 10.7% |
| B1 rolling z | — | — | — | — | 10.3% | 10.3% |
| B2 MA crossover | — | — | — | — | 5.8% | 5.8% |
| B3 isolation forest † | — | — | — | — | 31.3% | 31.3% |
| B4 EWMA deviation | — | — | — | — | 6.9% | 6.9% |
| B5 AR residual | — | — | — | — | 12.4% | 12.4% |
| B6 CUSUM | — | — | — | — | 0.0% | 0.0% |
| B7 logistic regression † | — | n/a | — | n/a | n/a | n/a |
| B8 random forest † | — | n/a | — | n/a | n/a | n/a |
| Tier 1 | 1 | — | — | — | 27.1% | 27.1% |
| Tier 2 (= Tier 1) | 1 | — | — | — | 27.1% | 27.1% |
| Tier 3 | 2 | — | — | — | 36.5% | 36.5% |
| Tier 4 (same numbers as Tier 3) | 3 | — | — | — | 36.5% | 36.5% |
| Tier 5 | 4 | — | — | — | 35.5% | 35.5% |
| *ORACLE label t-1 (ceiling, not a method)* | — | — | — | — | 0.0% | 0.0% |

**Best non-ceiling method:** none. This region has no positives, so AUC is undefined and only false-alarm rates exist; a low alert rate alone does not rank methods (a method that never alerts scores 0).

### panama — 149 positives among 856 scored test days (2023-07-01 .. 2025-11-20)

| Method | n agents | AUC [CI] | ΔAUC vs Tier 1 [CI] | F1 | FAR | Alert rate |
|---|---|---|---|---|---|---|
| M0 random | — | 0.501 [0.446, 0.561] | — | 0.126 | 10.5% | 10.4% |
| M1 always alert | — | 0.500 [0.500, 0.500] | — | 0.297 | 100.0% | 100.0% |
| M2 never alert | — | 0.500 [0.500, 0.500] | — | 0.000 | 0.0% | 0.0% |
| M3 persistence | — | 0.976 [0.941, 0.995] | — | 0.789 | 10.6% | 25.8% |
| M8 matrix profile | — | 0.436 [0.354, 0.521] | — | 0.027 | 10.2% | 8.8% |
| B1 rolling z | — | 0.481 [0.422, 0.539] | — | 0.147 | 9.2% | 9.6% |
| B2 MA crossover | — | 0.520 [0.352, 0.713] | — | 0.189 | 10.0% | 11.0% |
| B3 isolation forest † | — | 0.706 [0.598, 0.788] | — | 0.258 | 19.0% | 20.6% |
| B4 EWMA deviation | — | 0.506 [0.433, 0.583] | — | 0.145 | 11.5% | 11.6% |
| B5 AR residual | — | 0.508 [0.485, 0.528] | — | 0.069 | 10.7% | 9.8% |
| B6 CUSUM | — | 0.219 [0.055, 0.377] | — | 0.345 | 80.1% | 83.5% |
| B7 logistic regression † | — | n/a | — | n/a | n/a | n/a |
| B8 random forest † | — | n/a | — | n/a | n/a | n/a |
| Tier 1 | 1 | 0.921 [0.872, 0.952] | — | 0.710 | 12.9% | 26.1% |
| Tier 2 | 2 | 0.895 [0.845, 0.929] | **-0.027 [-0.053, -0.0004]** | 0.600 | 11.6% | 21.1% |
| Tier 3 (= Tier 2) | 2 | 0.895 [0.845, 0.929] | **-0.027 [-0.053, -0.0004]** | 0.600 | 11.6% | 21.1% |
| Tier 4 (same numbers as Tier 3) | 3 | 0.895 [0.845, 0.929] | **-0.027 [-0.053, -0.0004]** | 0.600 | 11.6% | 21.1% |
| Tier 5 | 4 | 0.900 [0.850, 0.934] | -0.021 [-0.047, +0.005] | 0.623 | 11.7% | 22.0% |
| *ORACLE label t-1 (ceiling, not a method)* | — | 0.996 [0.986, 1.000] | — | 0.993 | 0.1% | 17.4% |

**Best non-ceiling method:** M3 persistence, the highest AUC point estimate, 0.976 [0.941, 0.995] (circularity high). Its interval overlaps Tier 1: not a clear win. Best multi-agent tier: Tier 1, AUC 0.921 [0.872, 0.952]; its interval overlaps the best's.
