# Method comparison and agent ablation

All figures are measured on a held-out window: 2023-07-01 to 2025-11-21, the same fixed dates for every region (settings.yaml `evaluation`); every method is fitted only on earlier days. Features are real connector output; labels are the level-shift rule. Nothing here is assumed.


## Per-region results


### bab_el_mandeb

| method | kind | AUC [95% CI] | F1 | alert rate |
|---|---|---|---|---|
| M3 persistence | unsupervised | 0.966 [0.920, 0.996] | 0.778 | 0.41 |
| Tier 1 | multi-agent | 0.748 [0.675, 0.802] | 0.517 | 0.79 |
| Tier 2 | multi-agent | 0.640 [0.529, 0.717] | 0.477 | 0.57 |
| B2 MA crossover | unsupervised | 0.595 [0.499, 0.715] | 0.380 | 0.20 |
| B3 isolation forest | unsupervised | 0.554 [0.450, 0.638] | 0.426 | 0.49 |
| B4 EWMA deviation | unsupervised | 0.544 [0.505, 0.594] | 0.258 | 0.19 |
| B1 rolling z | unsupervised | 0.525 [0.494, 0.557] | 0.133 | 0.10 |
| B5 AR residual | unsupervised | 0.509 [0.491, 0.521] | 0.066 | 0.07 |
| M2 never alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M1 always alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M0 random | control | 0.494 [0.444, 0.541] | 0.192 | 0.10 |
| Tier 5 | multi-agent | 0.493 [0.373, 0.581] | 0.438 | 0.67 |
| M8 matrix profile | unsupervised | 0.489 [0.426, 0.564] | 0.111 | 0.07 |
| Tier 4 | multi-agent | 0.472 [0.355, 0.560] | 0.414 | 0.65 |
| Tier 3 | multi-agent | 0.471 [0.354, 0.559] | 0.417 | 0.65 |
| B6 CUSUM | unsupervised | 0.308 [0.113, 0.500] | 0.480 | 0.74 |
| *ORACLE label t-1 (ceiling, not a method)* | oracle | 0.997 [0.991, 1.000] | 0.996 | 0.28 |

Ceiling: ORACLE label t-1 (0.997 [0.991, 1.000]) reads yesterday's true label, which no detector has at inference. It bounds what one-day-lagged knowledge of the answer would score.


Best baseline M3 persistence (0.966 [0.920, 0.996]) vs best tier Tier 1 (0.748 [0.675, 0.802]): **-0.218**. The intervals do not overlap: a clear difference. The oracle and the controls are reference rows, not candidates for best baseline.


### hormuz

| method | kind | AUC [95% CI] | F1 | alert rate |
|---|---|---|---|---|
| B7 logistic regression | supervised | 0.939 [0.877, 0.981] | 0.072 | 0.01 |
| B6 CUSUM | unsupervised | 0.935 [0.865, 0.982] | 0.000 | 0.00 |
| M3 persistence | unsupervised | 0.848 [0.785, 0.902] | 0.465 | 0.20 |
| B8 random forest | supervised | 0.648 [0.534, 0.780] | 0.260 | 0.13 |
| M8 matrix profile | unsupervised | 0.600 [0.509, 0.696] | 0.223 | 0.11 |
| Tier 1 | multi-agent | 0.556 [0.480, 0.624] | 0.028 | 0.17 |
| M0 random | control | 0.517 [0.462, 0.571] | 0.153 | 0.10 |
| M1 always alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M2 never alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| Tier 2 | multi-agent | 0.498 [0.372, 0.602] | 0.064 | 0.14 |
| B5 AR residual | unsupervised | 0.492 [0.471, 0.513] | 0.077 | 0.12 |
| B4 EWMA deviation | unsupervised | 0.490 [0.450, 0.533] | 0.090 | 0.13 |
| B1 rolling z | unsupervised | 0.478 [0.435, 0.518] | 0.104 | 0.11 |
| B2 MA crossover | unsupervised | 0.465 [0.374, 0.558] | 0.008 | 0.15 |
| B3 isolation forest | unsupervised | 0.462 [0.333, 0.575] | 0.112 | 0.18 |
| Tier 4 | multi-agent | 0.442 [0.303, 0.557] | 0.171 | 0.24 |
| Tier 3 | multi-agent | 0.441 [0.302, 0.558] | 0.169 | 0.25 |
| Tier 5 | multi-agent | 0.434 [0.298, 0.548] | 0.140 | 0.23 |
| *ORACLE label t-1 (ceiling, not a method)* | oracle | 0.987 [0.969, 0.997] | 0.977 | 0.16 |

Ceiling: ORACLE label t-1 (0.987 [0.969, 0.997]) reads yesterday's true label, which no detector has at inference. It bounds what one-day-lagged knowledge of the answer would score.


Best baseline B7 logistic regression (0.939 [0.877, 0.981]) vs best tier Tier 1 (0.556 [0.480, 0.624]): **-0.383**. The intervals do not overlap: a clear difference. The oracle and the controls are reference rows, not candidates for best baseline.


### panama

| method | kind | AUC [95% CI] | F1 | alert rate |
|---|---|---|---|---|
| M3 persistence | unsupervised | 0.976 [0.941, 0.995] | 0.800 | 0.25 |
| Tier 1 | multi-agent | 0.921 [0.871, 0.952] | 0.710 | 0.26 |
| Tier 2 | multi-agent | 0.895 [0.844, 0.929] | 0.598 | 0.21 |
| Tier 5 | multi-agent | 0.894 [0.842, 0.929] | 0.594 | 0.22 |
| Tier 3 | multi-agent | 0.888 [0.834, 0.924] | 0.576 | 0.21 |
| Tier 4 | multi-agent | 0.888 [0.834, 0.924] | 0.576 | 0.21 |
| B3 isolation forest | unsupervised | 0.706 [0.598, 0.788] | 0.258 | 0.21 |
| B2 MA crossover | unsupervised | 0.520 [0.352, 0.713] | 0.189 | 0.11 |
| B5 AR residual | unsupervised | 0.508 [0.485, 0.528] | 0.069 | 0.10 |
| B4 EWMA deviation | unsupervised | 0.506 [0.433, 0.583] | 0.145 | 0.12 |
| M0 random | control | 0.501 [0.446, 0.561] | 0.126 | 0.10 |
| M2 never alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M1 always alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| B1 rolling z | unsupervised | 0.481 [0.422, 0.539] | 0.147 | 0.10 |
| M8 matrix profile | unsupervised | 0.436 [0.354, 0.521] | 0.027 | 0.09 |
| B6 CUSUM | unsupervised | 0.219 [0.055, 0.377] | 0.345 | 0.84 |
| *ORACLE label t-1 (ceiling, not a method)* | oracle | 0.996 [0.986, 1.000] | 0.993 | 0.17 |

Ceiling: ORACLE label t-1 (0.996 [0.986, 1.000]) reads yesterday's true label, which no detector has at inference. It bounds what one-day-lagged knowledge of the answer would score.


Best baseline M3 persistence (0.976 [0.941, 0.995]) vs best tier Tier 1 (0.921 [0.871, 0.952]): **-0.055**. The intervals overlap, which on its own is inconclusive: it neither shows nor rules out a difference. The oracle and the controls are reference rows, not candidates for best baseline.


## What the numbers show


### Does adding agents improve AUC?

- **bab_el_mandeb**: 0.748 -> 0.640 -> 0.471 -> 0.472 -> 0.493  (best: Tier 1)
- **hormuz**: 0.556 -> 0.498 -> 0.441 -> 0.442 -> 0.434  (best: Tier 1)
- **panama**: 0.921 -> 0.895 -> 0.888 -> 0.888 -> 0.894  (best: Tier 1)

In every region the peak is Tier 1 or Tier 2, and adding the geopolitical agent at Tier 3 lowers AUC each time. These are point estimates. Tested against Tier 1 with the paired difference on the same resampled blocks, a tier is clearly below Tier 1 in 11 of 12 comparisons, clearly above in 0 of 12, and inconclusive (interval includes 0) in 1 of 12. This does not support the claim that each agent adds value.


#### Paired ΔAUC against Tier 1 (Tier k - Tier 1)

Bold: the paired interval excludes 0.

| region | Tier 2 | Tier 3 | Tier 4 | Tier 5 |
|---|---|---|---|---|
| bab_el_mandeb | **-0.108 [-0.189, -0.044]** | **-0.277 [-0.394, -0.178]** | **-0.276 [-0.392, -0.178]** | **-0.256 [-0.375, -0.153]** |
| hormuz | -0.058 [-0.140, +0.002] | **-0.115 [-0.250, -0.026]** | **-0.114 [-0.248, -0.025]** | **-0.123 [-0.255, -0.035]** |
| panama | **-0.027 [-0.052, -0.001]** | **-0.033 [-0.057, -0.008]** | **-0.033 [-0.057, -0.008]** | **-0.027 [-0.054, -0.002]** |

### Split-date sensitivity

Tier 1 and Tier 4 re-scored with the split moved; the end date, the masked days and the training-window threshold rule are unchanged. The main tables use the settings.yaml split.

| region | split | scored test days (positives) | Tier 1 AUC | Tier 4 AUC | Tier 4 - Tier 1 (paired) |
|---|---|---|---|---|---|
| bab_el_mandeb | 2023-01-01 | 1037 (244) | 0.732 [0.670, 0.782] | 0.467 [0.362, 0.552] | **-0.266 [-0.382, -0.160]** |
| bab_el_mandeb | 2023-07-01 | 857 (244) | 0.748 [0.675, 0.802] | 0.472 [0.355, 0.560] | **-0.276 [-0.392, -0.178]** |
| bab_el_mandeb | 2024-01-01 | 673 (244) | 0.471 [0.418, 0.526] | 0.364 [0.241, 0.458] | **-0.107 [-0.243, -0.016]** |
| hormuz | 2023-01-01 | 1037 (133) | 0.553 [0.476, 0.608] | 0.453 [0.321, 0.560] | -0.100 [-0.221, +0.001] |
| hormuz | 2023-07-01 | 857 (133) | 0.556 [0.480, 0.624] | 0.442 [0.303, 0.557] | **-0.114 [-0.248, -0.025]** |
| hormuz | 2024-01-01 | 673 (116) | 0.547 [0.353, 0.687] | 0.464 [0.275, 0.623] | -0.083 [-0.257, +0.017] |
| malacca | 2023-01-01 | 1037 (0) | — | — | — |
| malacca | 2023-07-01 | 857 (0) | — | — | — |
| malacca | 2024-01-01 | 673 (0) | — | — | — |
| panama | 2023-01-01 | 1036 (149) | 0.927 [0.883, 0.957] | 0.888 [0.836, 0.924] | **-0.039 [-0.066, -0.014]** |
| panama | 2023-07-01 | 856 (149) | 0.921 [0.871, 0.952] | 0.888 [0.834, 0.924] | **-0.033 [-0.057, -0.008]** |
| panama | 2024-01-01 | 672 (110) | 0.926 [0.868, 0.958] | 0.872 [0.803, 0.923] | **-0.054 [-0.082, -0.013]** |

bab_el_mandeb: Tier 1 AUC ranges 0.471-0.748; hormuz: Tier 1 AUC ranges 0.547-0.556; panama: Tier 1 AUC ranges 0.921-0.927.


### Tiers scoring below 0.5

Below 0.5 but indistinguishable from chance (the interval includes 0.5): bab_el_mandeb Tier 3 (0.471 [0.354, 0.559]), bab_el_mandeb Tier 4 (0.472 [0.355, 0.560]), bab_el_mandeb Tier 5 (0.493 [0.373, 0.581]), hormuz Tier 2 (0.498 [0.372, 0.602]), hormuz Tier 3 (0.441 [0.302, 0.558]), hormuz Tier 4 (0.442 [0.303, 0.557]), hormuz Tier 5 (0.434 [0.298, 0.548]).


### Where simple baselines score above the ensemble

- **bab_el_mandeb**: M3 persistence at 0.966 is above the best tier at 0.748. The intervals do not overlap: a clear difference.
- **hormuz**: B7 logistic regression at 0.939 is above the best tier at 0.556. The intervals do not overlap: a clear difference.
- **panama**: M3 persistence at 0.976 is above the best tier at 0.921. The intervals overlap, which on its own is inconclusive: it neither shows nor rules out a difference.

### False-positive harness (Malacca)

Malacca has zero labelled disruptions in the evaluated data, so every alert is a false alarm. Mean alert rate: **tiers 29%**, **baselines 10%** (detection methods only; the oracle and controls are left out). The multi-agent tiers are the noisier of the two.


## Limits of this evaluation

- **Supervised baselines are untrainable in 3 of 4 regions.** Most labelled disruptions are recent (Houthi 2024, Gatun drought 2023-24), so the temporal split leaves no positives in training there. Those rows are reported as not-applicable rather than as a 0.5 score, which would have read as 'no better than chance'.
- **Intervals are 95% moving-block bootstrap CIs** (30-day blocks, 1,000 resamples) on the test window, with the training threshold held fixed. Every method in a region gets the same resamples, so intervals are paired. Non-overlapping intervals show a clear difference; overlapping ones are inconclusive on their own. Tier differences use the paired interval of the difference (the same resampled blocks for both tiers); a paired interval that excludes 0 is a clear difference. AUC and F1 intervals use only resamples that contain both classes: at least 998 of 1,000 for every scored method. They cover test-sample uncertainty only, not the choice of split or label, and they are conditional on the few disruption episodes in each test window, so they do not measure skill on a new kind of disruption.
- **Days GDELT never published are masked:** 18 test days in every region are left out of every method's score (point estimates and intervals) and out of every training threshold. They are not filled.
- **Malacca cannot be evaluated for detection** — no positives at all.
- **News features come from GDELT 1.0 event exports** (corridor box filter v2) in every region; volume is articles per 100k GDELT events that day. The news agent's tone means skip days without articles instead of counting them as neutral tone.
- **Values enter on the day they were known, where the source says.** FRED market series are stamped on their first-release date (ALFRED) and carried forward until the next release; gaps in other domains are forward-filled, never interpolated. Shipping (PortWatch) and conflict (ACLED) values stay on their event dates: their roughly weekly publication lag is not modelled. A few scaling constants (the freight index's rebase, the trade-volume and conflict-intensity scales) are still taken over the whole sample.
- **The label is one definition among several.** The level-shift rule recovers two documented events, but agents built against a different definition of disruption will score poorly whatever their merit.
