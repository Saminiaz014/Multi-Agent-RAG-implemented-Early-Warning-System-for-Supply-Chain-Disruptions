# Method comparison and agent ablation

All figures are measured on a held-out window (last 30% of each region's series, temporal split). Features are real connector output; labels are the level-shift rule. Nothing here is assumed.


## Per-region results


### bab_el_mandeb

| method | kind | AUC | F1 | alert rate |
|---|---|---|---|---|
| Tier 1 | multi-agent | 0.679 | 0.553 | 0.80 |
| Tier 2 | multi-agent | 0.620 | 0.511 | 0.60 |
| B2 MA crossover | unsupervised | 0.581 | 0.386 | 0.23 |
| B4 EWMA deviation | unsupervised | 0.540 | 0.267 | 0.20 |
| B1 rolling z | unsupervised | 0.524 | 0.138 | 0.10 |
| B5 AR residual | unsupervised | 0.509 | 0.075 | 0.07 |
| B3 isolation forest | unsupervised | 0.500 | 0.410 | 0.61 |
| Tier 3 | multi-agent | 0.449 | 0.440 | 0.69 |
| Tier 4 | multi-agent | 0.449 | 0.441 | 0.68 |
| Tier 5 | multi-agent | 0.449 | 0.441 | 0.68 |
| B6 CUSUM | unsupervised | 0.263 | 0.498 | 0.76 |

Best baseline B2 MA crossover (0.581) vs best tier Tier 1 (0.679): **+0.098**


### hormuz

| method | kind | AUC | F1 | alert rate |
|---|---|---|---|---|
| B7 logistic regression | supervised | 0.968 | 0.682 | 0.12 |
| B6 CUSUM | unsupervised | 0.958 | 0.000 | 0.00 |
| B8 random forest | supervised | 0.747 | 0.378 | 0.14 |
| Tier 1 | multi-agent | 0.502 | 0.034 | 0.14 |
| B4 EWMA deviation | unsupervised | 0.496 | 0.108 | 0.12 |
| B5 AR residual | unsupervised | 0.493 | 0.084 | 0.11 |
| B1 rolling z | unsupervised | 0.479 | 0.112 | 0.11 |
| B2 MA crossover | unsupervised | 0.470 | 0.009 | 0.13 |
| Tier 2 | multi-agent | 0.465 | 0.071 | 0.12 |
| B3 isolation forest | unsupervised | 0.434 | 0.085 | 0.17 |
| Tier 4 | multi-agent | 0.401 | 0.140 | 0.25 |
| Tier 5 | multi-agent | 0.401 | 0.140 | 0.25 |
| Tier 3 | multi-agent | 0.400 | 0.141 | 0.25 |

Best baseline B7 logistic regression (0.968) vs best tier Tier 1 (0.502): **-0.465**


### panama

| method | kind | AUC | F1 | alert rate |
|---|---|---|---|---|
| Tier 1 | multi-agent | 0.909 | 0.722 | 0.32 |
| Tier 5 | multi-agent | 0.884 | 0.613 | 0.27 |
| Tier 2 | multi-agent | 0.884 | 0.620 | 0.25 |
| Tier 4 | multi-agent | 0.876 | 0.612 | 0.24 |
| Tier 3 | multi-agent | 0.876 | 0.612 | 0.24 |
| B3 isolation forest | unsupervised | 0.814 | 0.435 | 0.23 |
| B2 MA crossover | unsupervised | 0.523 | 0.192 | 0.12 |
| B4 EWMA deviation | unsupervised | 0.510 | 0.145 | 0.14 |
| B5 AR residual | unsupervised | 0.508 | 0.071 | 0.10 |
| B1 rolling z | unsupervised | 0.484 | 0.154 | 0.10 |
| B6 CUSUM | unsupervised | 0.145 | 0.372 | 0.89 |

Best baseline B3 isolation forest (0.814) vs best tier Tier 1 (0.909): **+0.095**


## What the numbers show


### Adding agents does not monotonically improve AUC

- **bab_el_mandeb**: 0.679 -> 0.620 -> 0.449 -> 0.449 -> 0.449  (best: Tier 1)
- **hormuz**: 0.502 -> 0.465 -> 0.400 -> 0.401 -> 0.401  (best: Tier 1)
- **panama**: 0.909 -> 0.884 -> 0.876 -> 0.876 -> 0.884  (best: Tier 1)

In every region the peak is Tier 1 or Tier 2, and adding the geopolitical agent at Tier 3 lowers AUC each time. This does not support the claim that each agent adds value.


### Some multi-agent scores are anti-correlated with the label

- bab_el_mandeb Tier 3: AUC 0.449
- bab_el_mandeb Tier 4: AUC 0.449
- bab_el_mandeb Tier 5: AUC 0.449
- hormuz Tier 2: AUC 0.465
- hormuz Tier 3: AUC 0.400
- hormuz Tier 4: AUC 0.401
- hormuz Tier 5: AUC 0.401

An AUC below 0.5 is not noise — it means the score runs opposite to the label. The likely cause is a mismatch of definitions: these agents are shock detectors (z-scores and isolation forests against rolling baselines), while the label marks *sustained* level shifts. Once traffic has settled at a lower level, a shock detector sees a stable series and reports calm. The two are measuring different things, and that is an architectural finding rather than a bug.


### Where simple baselines beat the ensemble

- **hormuz**: B7 logistic regression at 0.968 beats the best tier at 0.502

### False-positive harness (Malacca)

Malacca has zero labelled disruptions across 2019-2026, so every alert is a false alarm. Mean alert rate: **tiers 23%**, **baselines 9%**. The multi-agent tiers are the noisier of the two.


## Limits of this evaluation

- **Supervised baselines are mostly untrainable here.** Every labelled disruption is recent (Houthi 2024, Gatun drought 2023-24), so a temporal split leaves no positives in training for three of four regions. Those rows are reported as not-applicable rather than as a 0.5 score, which would have read as 'no better than chance'.
- **One split, no confidence intervals.** Positive counts are 133-244 days; treat differences under roughly 0.05 AUC as noise.
- **Malacca cannot be evaluated for detection** — no positives at all.
- **Panama alone has news features** (18 vs 14), because GDELT only answered for that region. Its tier 5 is therefore not directly comparable with the others'.
- **The label is one definition among several.** The level-shift rule recovers two documented events, but agents built against a different definition of disruption will score poorly whatever their merit.
