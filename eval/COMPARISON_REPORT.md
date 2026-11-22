# Method comparison and agent ablation

All figures are measured on a held-out window: 2023-07-01 to 2025-11-21, the same fixed dates for every region (settings.yaml `evaluation`); every method is fitted only on earlier days. Features are real connector output; labels are the level-shift rule. Nothing here is assumed.


## Per-region results


### bab_el_mandeb

| method | kind | AUC [95% CI] | F1 | alert rate |
|---|---|---|---|---|
| ORACLE label t-1 | oracle | 0.997 [0.991, 1.000] | 0.996 | 0.28 |
| M3 persistence | unsupervised | 0.965 [0.919, 0.996] | 0.766 | 0.41 |
| Tier 1 | multi-agent | 0.746 [0.673, 0.799] | 0.510 | 0.79 |
| Tier 2 | multi-agent | 0.643 [0.562, 0.712] | 0.469 | 0.58 |
| B2 MA crossover | unsupervised | 0.585 [0.488, 0.699] | 0.368 | 0.21 |
| B3 isolation forest | unsupervised | 0.566 [0.451, 0.668] | 0.422 | 0.56 |
| B4 EWMA deviation | unsupervised | 0.542 [0.503, 0.590] | 0.254 | 0.19 |
| B1 rolling z | unsupervised | 0.524 [0.492, 0.557] | 0.132 | 0.10 |
| B5 AR residual | unsupervised | 0.509 [0.493, 0.522] | 0.066 | 0.07 |
| M2 never alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M1 always alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M0 random | control | 0.495 [0.444, 0.542] | 0.192 | 0.10 |
| M8 matrix profile | unsupervised | 0.487 [0.424, 0.563] | 0.110 | 0.07 |
| Tier 5 | multi-agent | 0.481 [0.374, 0.567] | 0.420 | 0.71 |
| Tier 3 | multi-agent | 0.472 [0.370, 0.554] | 0.399 | 0.71 |
| Tier 4 | multi-agent | 0.471 [0.369, 0.553] | 0.396 | 0.71 |
| B6 CUSUM | unsupervised | 0.300 [0.106, 0.493] | 0.470 | 0.75 |

Best baseline M3 persistence (0.965 [0.919, 0.996]) vs best tier Tier 1 (0.746 [0.673, 0.799]): **-0.219**. The intervals do not overlap: a clear difference. The oracle and the controls are reference rows, not candidates for best baseline.


### hormuz

| method | kind | AUC [95% CI] | F1 | alert rate |
|---|---|---|---|---|
| ORACLE label t-1 | oracle | 0.987 [0.969, 0.997] | 0.977 | 0.15 |
| B7 logistic regression | supervised | 0.971 [0.934, 0.992] | 0.676 | 0.10 |
| B6 CUSUM | unsupervised | 0.936 [0.866, 0.983] | 0.000 | 0.00 |
| M3 persistence | unsupervised | 0.851 [0.789, 0.905] | 0.465 | 0.19 |
| B8 random forest | supervised | 0.684 [0.583, 0.799] | 0.394 | 0.24 |
| M8 matrix profile | unsupervised | 0.600 [0.510, 0.695] | 0.220 | 0.11 |
| Tier 1 | multi-agent | 0.555 [0.478, 0.624] | 0.028 | 0.17 |
| M0 random | control | 0.517 [0.463, 0.571] | 0.152 | 0.10 |
| M1 always alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M2 never alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| B5 AR residual | unsupervised | 0.491 [0.469, 0.512] | 0.076 | 0.12 |
| B4 EWMA deviation | unsupervised | 0.490 [0.450, 0.532] | 0.090 | 0.13 |
| Tier 2 | multi-agent | 0.485 [0.387, 0.575] | 0.068 | 0.15 |
| B1 rolling z | unsupervised | 0.477 [0.434, 0.518] | 0.103 | 0.11 |
| B2 MA crossover | unsupervised | 0.466 [0.378, 0.558] | 0.008 | 0.14 |
| Tier 4 | multi-agent | 0.435 [0.314, 0.535] | 0.129 | 0.27 |
| Tier 3 | multi-agent | 0.434 [0.313, 0.536] | 0.135 | 0.27 |
| Tier 5 | multi-agent | 0.433 [0.314, 0.532] | 0.128 | 0.26 |
| B3 isolation forest | unsupervised | 0.403 [0.307, 0.501] | 0.067 | 0.19 |

Best baseline B7 logistic regression (0.971 [0.934, 0.992]) vs best tier Tier 1 (0.555 [0.478, 0.624]): **-0.415**. The intervals do not overlap: a clear difference. The oracle and the controls are reference rows, not candidates for best baseline.


### panama

| method | kind | AUC [95% CI] | F1 | alert rate |
|---|---|---|---|---|
| ORACLE label t-1 | oracle | 0.996 [0.986, 1.000] | 0.993 | 0.17 |
| M3 persistence | unsupervised | 0.976 [0.942, 0.995] | 0.789 | 0.25 |
| Tier 1 | multi-agent | 0.921 [0.873, 0.952] | 0.702 | 0.26 |
| Tier 2 | multi-agent | 0.900 [0.840, 0.942] | 0.615 | 0.22 |
| Tier 5 | multi-agent | 0.895 [0.833, 0.941] | 0.607 | 0.23 |
| Tier 4 | multi-agent | 0.894 [0.833, 0.939] | 0.603 | 0.21 |
| Tier 3 | multi-agent | 0.894 [0.833, 0.939] | 0.603 | 0.21 |
| B3 isolation forest | unsupervised | 0.714 [0.616, 0.786] | 0.265 | 0.20 |
| B2 MA crossover | unsupervised | 0.518 [0.348, 0.713] | 0.189 | 0.11 |
| B5 AR residual | unsupervised | 0.508 [0.485, 0.528] | 0.068 | 0.10 |
| B4 EWMA deviation | unsupervised | 0.507 [0.436, 0.583] | 0.142 | 0.12 |
| M0 random | control | 0.502 [0.446, 0.564] | 0.126 | 0.10 |
| M1 always alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| M2 never alert | control | 0.500 [0.500, 0.500] | 0.000 | 0.00 |
| B1 rolling z | unsupervised | 0.483 [0.424, 0.541] | 0.146 | 0.10 |
| M8 matrix profile | unsupervised | 0.439 [0.359, 0.524] | 0.027 | 0.09 |
| B6 CUSUM | unsupervised | 0.214 [0.054, 0.372] | 0.338 | 0.84 |

Best baseline M3 persistence (0.976 [0.942, 0.995]) vs best tier Tier 1 (0.921 [0.873, 0.952]): **-0.055**. The intervals overlap, which on its own is inconclusive: it neither shows nor rules out a difference. The oracle and the controls are reference rows, not candidates for best baseline.


## What the numbers show


### Does adding agents improve AUC?

- **bab_el_mandeb**: 0.746 -> 0.643 -> 0.472 -> 0.471 -> 0.481  (best: Tier 1)
- **hormuz**: 0.555 -> 0.485 -> 0.434 -> 0.435 -> 0.433  (best: Tier 1)
- **panama**: 0.921 -> 0.900 -> 0.894 -> 0.894 -> 0.895  (best: Tier 1)

In every region the peak is Tier 1 or Tier 2, and adding the geopolitical agent at Tier 3 lowers AUC each time. These are point estimates: consecutive tiers' intervals are clearly apart in 1 of 12 steps, and overlap alone is inconclusive. This does not support the claim that each agent adds value.


### Tiers scoring below 0.5

Below 0.5 but indistinguishable from chance (the interval includes 0.5): bab_el_mandeb Tier 3 (0.472 [0.370, 0.554]), bab_el_mandeb Tier 4 (0.471 [0.369, 0.553]), bab_el_mandeb Tier 5 (0.481 [0.374, 0.567]), hormuz Tier 2 (0.485 [0.387, 0.575]), hormuz Tier 3 (0.434 [0.313, 0.536]), hormuz Tier 4 (0.435 [0.314, 0.535]), hormuz Tier 5 (0.433 [0.314, 0.532]).


### Where simple baselines score above the ensemble

- **bab_el_mandeb**: M3 persistence at 0.965 is above the best tier at 0.746. The intervals do not overlap: a clear difference.
- **hormuz**: B7 logistic regression at 0.971 is above the best tier at 0.555. The intervals do not overlap: a clear difference.
- **panama**: M3 persistence at 0.976 is above the best tier at 0.921. The intervals overlap, which on its own is inconclusive: it neither shows nor rules out a difference.

### False-positive harness (Malacca)

Malacca has zero labelled disruptions in the evaluated data, so every alert is a false alarm. Mean alert rate: **tiers 31%**, **baselines 10%** (detection methods only; the oracle and controls are left out). The multi-agent tiers are the noisier of the two.


## Limits of this evaluation

- **Supervised baselines are untrainable in 3 of 4 regions.** Most labelled disruptions are recent (Houthi 2024, Gatun drought 2023-24), so the temporal split leaves no positives in training there. Those rows are reported as not-applicable rather than as a 0.5 score, which would have read as 'no better than chance'.
- **Intervals are 95% moving-block bootstrap CIs** (30-day blocks, 1,000 resamples) on the test window, with the training threshold held fixed. Every method in a region gets the same resamples, so intervals are paired. Non-overlapping intervals show a clear difference; overlapping ones are inconclusive on their own. AUC and F1 intervals use only resamples that contain both classes: at least 998 of 1,000 for every scored method. They cover test-sample uncertainty only, not the choice of split or label, and they are conditional on the few disruption episodes in each test window, so they do not measure skill on a new kind of disruption.
- **Malacca cannot be evaluated for detection** — no positives at all.
- **News features come from GDELT 1.0 event exports** (corridor box filter v2) in every region; volume is articles per 100k GDELT events that day.
- **The label is one definition among several.** The level-shift rule recovers two documented events, but agents built against a different definition of disruption will score poorly whatever their merit.
