# Method comparison and agent ablation

All figures are measured on a held-out window: 2023-07-01 to 2025-11-21, the same fixed dates for every region (settings.yaml `evaluation`); every method is fitted only on earlier days. Features are real connector output; labels are the level-shift rule. Nothing here is assumed.


## Per-region results


### bab_el_mandeb

| method | kind | AUC [95% CI] | F1 | alert rate |
|---|---|---|---|---|
| M3 persistence | unsupervised | 0.966 [0.920, 0.996] | 0.778 | 0.41 |
| Tier 1 | multi-agent | 0.747 [0.672, 0.802] | 0.517 | 0.79 |
| Tier 2 | multi-agent | 0.638 [0.526, 0.715] | 0.477 | 0.57 |
| B2 MA crossover | unsupervised | 0.595 [0.499, 0.715] | 0.380 | 0.20 |
| B3 isolation forest | unsupervised | 0.554 [0.450, 0.638] | 0.426 | 0.49 |
| B4 EWMA deviation | unsupervised | 0.544 [0.505, 0.594] | 0.258 | 0.19 |
| B1 rolling z | unsupervised | 0.525 [0.494, 0.557] | 0.133 | 0.10 |
| B5 AR residual | unsupervised | 0.509 [0.491, 0.521] | 0.066 | 0.07 |
| M2 never alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M1 always alert | control | 0.500 [0.500, 0.500] | 0.443 | 1.00 |
| M0 random | control | 0.494 [0.444, 0.541] | 0.192 | 0.10 |
| Tier 5 | multi-agent | 0.491 [0.372, 0.579] | 0.439 | 0.67 |
| M8 matrix profile | unsupervised | 0.489 [0.426, 0.564] | 0.111 | 0.07 |
| Tier 4 | multi-agent | 0.470 [0.353, 0.557] | 0.413 | 0.65 |
| Tier 3 | multi-agent | 0.469 [0.353, 0.556] | 0.416 | 0.65 |
| B6 CUSUM | unsupervised | 0.308 [0.113, 0.500] | 0.480 | 0.74 |
| *ORACLE label t-1 (ceiling, not a method)* | oracle | 0.997 [0.991, 1.000] | 0.996 | 0.28 |

Ceiling: ORACLE label t-1 (0.997 [0.991, 1.000]) reads yesterday's true label, which no detector has at inference. It bounds what one-day-lagged knowledge of the answer would score.


Best baseline M3 persistence (0.966 [0.920, 0.996]) vs best tier Tier 1 (0.747 [0.672, 0.802]): **-0.219**. The intervals do not overlap: a clear difference. The oracle and the controls are reference rows, not candidates for best baseline.


### hormuz

| method | kind | AUC [95% CI] | F1 | alert rate |
|---|---|---|---|---|
| B7 logistic regression | supervised | 0.939 [0.877, 0.981] | 0.072 | 0.01 |
| B6 CUSUM | unsupervised | 0.935 [0.865, 0.982] | 0.000 | 0.00 |
| M3 persistence | unsupervised | 0.848 [0.785, 0.902] | 0.465 | 0.20 |
| B8 random forest | supervised | 0.648 [0.534, 0.780] | 0.260 | 0.13 |
| M8 matrix profile | unsupervised | 0.600 [0.509, 0.696] | 0.223 | 0.11 |
| Tier 1 | multi-agent | 0.574 [0.516, 0.633] | 0.028 | 0.17 |
| M0 random | control | 0.517 [0.462, 0.571] | 0.153 | 0.10 |
| M1 always alert | control | 0.500 [0.500, 0.500] | 0.269 | 1.00 |
| M2 never alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| Tier 2 | multi-agent | 0.497 [0.384, 0.594] | 0.032 | 0.13 |
| B5 AR residual | unsupervised | 0.492 [0.471, 0.513] | 0.077 | 0.12 |
| B4 EWMA deviation | unsupervised | 0.490 [0.450, 0.533] | 0.090 | 0.13 |
| B1 rolling z | unsupervised | 0.478 [0.435, 0.518] | 0.104 | 0.11 |
| B2 MA crossover | unsupervised | 0.465 [0.374, 0.558] | 0.008 | 0.15 |
| B3 isolation forest | unsupervised | 0.462 [0.333, 0.575] | 0.112 | 0.18 |
| Tier 4 | multi-agent | 0.440 [0.304, 0.555] | 0.161 | 0.24 |
| Tier 3 | multi-agent | 0.439 [0.303, 0.554] | 0.160 | 0.24 |
| Tier 5 | multi-agent | 0.431 [0.298, 0.544] | 0.143 | 0.22 |
| *ORACLE label t-1 (ceiling, not a method)* | oracle | 0.987 [0.969, 0.997] | 0.977 | 0.16 |

Ceiling: ORACLE label t-1 (0.987 [0.969, 0.997]) reads yesterday's true label, which no detector has at inference. It bounds what one-day-lagged knowledge of the answer would score.


Best baseline B7 logistic regression (0.939 [0.877, 0.981]) vs best tier Tier 1 (0.574 [0.516, 0.633]): **-0.365**. The intervals do not overlap: a clear difference. The oracle and the controls are reference rows, not candidates for best baseline.


### panama

| method | kind | AUC [95% CI] | F1 | alert rate |
|---|---|---|---|---|
| M3 persistence | unsupervised | 0.976 [0.941, 0.995] | 0.789 | 0.26 |
| Tier 1 | multi-agent | 0.921 [0.872, 0.952] | 0.710 | 0.26 |
| Tier 2 | multi-agent | 0.895 [0.845, 0.929] | 0.600 | 0.21 |
| Tier 5 | multi-agent | 0.894 [0.842, 0.929] | 0.594 | 0.22 |
| Tier 3 | multi-agent | 0.888 [0.835, 0.924] | 0.574 | 0.21 |
| Tier 4 | multi-agent | 0.888 [0.835, 0.924] | 0.574 | 0.21 |
| B3 isolation forest | unsupervised | 0.706 [0.598, 0.788] | 0.258 | 0.21 |
| B2 MA crossover | unsupervised | 0.520 [0.352, 0.713] | 0.189 | 0.11 |
| B5 AR residual | unsupervised | 0.508 [0.485, 0.528] | 0.069 | 0.10 |
| B4 EWMA deviation | unsupervised | 0.506 [0.433, 0.583] | 0.145 | 0.12 |
| M0 random | control | 0.501 [0.446, 0.561] | 0.126 | 0.10 |
| M2 never alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M1 always alert | control | 0.500 [0.500, 0.500] | 0.297 | 1.00 |
| B1 rolling z | unsupervised | 0.481 [0.422, 0.539] | 0.147 | 0.10 |
| M8 matrix profile | unsupervised | 0.436 [0.354, 0.521] | 0.027 | 0.09 |
| B6 CUSUM | unsupervised | 0.219 [0.055, 0.377] | 0.345 | 0.84 |
| *ORACLE label t-1 (ceiling, not a method)* | oracle | 0.996 [0.986, 1.000] | 0.993 | 0.17 |

Ceiling: ORACLE label t-1 (0.996 [0.986, 1.000]) reads yesterday's true label, which no detector has at inference. It bounds what one-day-lagged knowledge of the answer would score.


Best baseline M3 persistence (0.976 [0.941, 0.995]) vs best tier Tier 1 (0.921 [0.872, 0.952]): **-0.055**. The intervals overlap, which on its own is inconclusive: it neither shows nor rules out a difference. The oracle and the controls are reference rows, not candidates for best baseline.


## What the numbers show


### Does adding agents improve AUC?

- **bab_el_mandeb**: 0.747 -> 0.638 -> 0.469 -> 0.470 -> 0.491  (best: Tier 1)
- **hormuz**: 0.574 -> 0.497 -> 0.439 -> 0.440 -> 0.431  (best: Tier 1)
- **panama**: 0.921 -> 0.895 -> 0.888 -> 0.888 -> 0.894  (best: Tier 1)

In every region the peak is Tier 1 or Tier 2, and adding the geopolitical agent at Tier 3 lowers AUC each time. These are point estimates. Tested against Tier 1 with the paired difference on the same resampled blocks, a tier is clearly below Tier 1 in 12 of 12 comparisons, clearly above in 0 of 12, and inconclusive (interval includes 0) in 0 of 12. This does not support the claim that each agent adds value.


#### Paired ΔAUC against Tier 1 (Tier k - Tier 1)

Bold: the paired interval excludes 0.

| region | Tier 2 | Tier 3 | Tier 4 | Tier 5 |
|---|---|---|---|---|
| bab_el_mandeb | **-0.109 [-0.190, -0.044]** | **-0.278 [-0.394, -0.178]** | **-0.278 [-0.393, -0.178]** | **-0.256 [-0.376, -0.155]** |
| hormuz | **-0.078 [-0.161, -0.007]** | **-0.135 [-0.267, -0.041]** | **-0.134 [-0.265, -0.039]** | **-0.143 [-0.271, -0.049]** |
| panama | **-0.027 [-0.053, -0.000]** | **-0.033 [-0.058, -0.008]** | **-0.033 [-0.058, -0.008]** | **-0.027 [-0.054, -0.002]** |

### Split-date sensitivity

Tier 1 and Tier 4 re-scored with the split moved; the end date, the masked days and the training-window threshold rule are unchanged. The main tables use the settings.yaml split.

| region | split | scored test days (positives) | Tier 1 AUC | Tier 4 AUC | Tier 4 - Tier 1 (paired) |
|---|---|---|---|---|---|
| bab_el_mandeb | 2023-01-01 | 1037 (244) | 0.732 [0.669, 0.782] | 0.463 [0.359, 0.550] | **-0.269 [-0.383, -0.161]** |
| bab_el_mandeb | 2023-07-01 | 857 (244) | 0.747 [0.672, 0.802] | 0.470 [0.353, 0.557] | **-0.278 [-0.393, -0.178]** |
| bab_el_mandeb | 2024-01-01 | 673 (244) | 0.682 [0.632, 0.736] | 0.413 [0.280, 0.510] | **-0.270 [-0.422, -0.167]** |
| hormuz | 2023-01-01 | 1037 (133) | 0.547 [0.475, 0.599] | 0.440 [0.307, 0.550] | **-0.106 [-0.225, -0.001]** |
| hormuz | 2023-07-01 | 857 (133) | 0.574 [0.516, 0.633] | 0.440 [0.304, 0.555] | **-0.134 [-0.265, -0.039]** |
| hormuz | 2024-01-01 | 673 (116) | 0.638 [0.583, 0.683] | 0.485 [0.323, 0.621] | **-0.153 [-0.308, -0.027]** |
| malacca | 2023-01-01 | 1037 (0) | — | — | — |
| malacca | 2023-07-01 | 857 (0) | — | — | — |
| malacca | 2024-01-01 | 673 (0) | — | — | — |
| panama | 2023-01-01 | 1036 (149) | 0.928 [0.884, 0.958] | 0.890 [0.838, 0.924] | **-0.038 [-0.065, -0.014]** |
| panama | 2023-07-01 | 856 (149) | 0.921 [0.872, 0.952] | 0.888 [0.835, 0.924] | **-0.033 [-0.058, -0.008]** |
| panama | 2024-01-01 | 672 (110) | 0.927 [0.868, 0.959] | 0.889 [0.803, 0.936] | **-0.038 [-0.081, -0.007]** |

bab_el_mandeb: Tier 1 AUC ranges 0.682-0.747; hormuz: Tier 1 AUC ranges 0.547-0.638; panama: Tier 1 AUC ranges 0.921-0.928.


### Tiers scoring below 0.5

Below 0.5 but indistinguishable from chance (the interval includes 0.5): bab_el_mandeb Tier 3 (0.469 [0.353, 0.556]), bab_el_mandeb Tier 4 (0.470 [0.353, 0.557]), bab_el_mandeb Tier 5 (0.491 [0.372, 0.579]), hormuz Tier 2 (0.497 [0.384, 0.594]), hormuz Tier 3 (0.439 [0.303, 0.554]), hormuz Tier 4 (0.440 [0.304, 0.555]), hormuz Tier 5 (0.431 [0.298, 0.544]).


### Where simple baselines score above the ensemble

- **bab_el_mandeb**: M3 persistence at 0.966 is above the best tier at 0.747. The intervals do not overlap: a clear difference.
- **hormuz**: B7 logistic regression at 0.939 is above the best tier at 0.574. The intervals do not overlap: a clear difference.
- **panama**: M3 persistence at 0.976 is above the best tier at 0.921. The intervals overlap, which on its own is inconclusive: it neither shows nor rules out a difference.

### False-positive harness (Malacca)

Malacca has zero labelled disruptions in the evaluated data, so every alert is a false alarm. Mean alert rate: **tiers 29%**, **baselines 10%** (detection methods only; the oracle and controls are left out). The multi-agent tiers are the noisier of the two.


## Limits of this evaluation

- **Supervised baselines are untrainable in 3 of 4 regions.** Most labelled disruptions are recent (Houthi 2024, Gatun drought 2023-24), so the temporal split leaves no positives in training there. Those rows are reported as not-applicable rather than as a 0.5 score, which would have read as 'no better than chance'.
- **Intervals are 95% moving-block bootstrap CIs** (30-day blocks, 1,000 resamples) on the test window, with the training threshold (or a row's direct alerts) held fixed. Every method in a region gets the same resamples, so intervals are paired. Non-overlapping intervals show a clear difference; overlapping ones are inconclusive on their own. Tier differences use the paired interval of the difference (the same resampled blocks for both tiers); a paired interval that excludes 0 is a clear difference. AUC and F1 intervals use only resamples that contain both classes: at least 998 of 1,000 for every scored method. They cover test-sample uncertainty only, not the choice of split or label, and they are conditional on the few disruption episodes in each test window, so they do not measure skill on a new kind of disruption.
- **Alert rule:** a day alerts when its score >= the method's 90th percentile of training scores (masked days left out). A score tied with that threshold counts as an alert. These rows define their alerts directly instead: M1 always alert (every day); M2 never alert (no day); ORACLE label t-1 (yesterday's label is 1).
- **Days GDELT never published are masked:** 18 test days in every region are left out of every method's score (point estimates and intervals) and out of every training threshold. They are not filled.
- **Malacca cannot be evaluated for detection** — no positives at all.
- **News features come from GDELT 1.0 event exports** (corridor box filter v2) in every region; volume is articles per 100k GDELT events that day. The news agent's tone means skip days without articles instead of counting them as neutral tone.
- **Values enter on the day they were known, where the source says.** FRED market series are stamped on their first-release date (ALFRED) and carried forward until the next release; gaps in other domains are forward-filled, never interpolated. Shipping (PortWatch) and conflict (ACLED) values stay on their event dates: their roughly weekly publication lag is not modelled. A few scaling constants (the freight index's rebase, the trade-volume and conflict-intensity scales) are still taken over the whole sample.
- **The label is one definition among several.** The level-shift rule recovers two documented events, but agents built against a different definition of disruption will score poorly whatever their merit.
