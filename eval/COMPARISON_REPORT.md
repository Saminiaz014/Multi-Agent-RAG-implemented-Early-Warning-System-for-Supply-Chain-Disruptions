# Method comparison and agent ablation

All figures are measured on a held-out window (last 30% of each region's series, temporal split). Features are real connector output; labels are the level-shift rule. Nothing here is assumed.


## Per-region results


### bab_el_mandeb

| method | kind | AUC | F1 | alert rate |
|---|---|---|---|---|
| B2 MA crossover | unsupervised | 0.581 | 0.386 | 0.23 |
| Tier 2 | multi-agent | 0.557 | 0.443 | 0.46 |
| Tier 1 | multi-agent | 0.548 | 0.459 | 0.65 |
| B4 EWMA deviation | unsupervised | 0.540 | 0.267 | 0.20 |
| B1 rolling z | unsupervised | 0.524 | 0.138 | 0.10 |
| B5 AR residual | unsupervised | 0.509 | 0.075 | 0.07 |
| B3 isolation forest | unsupervised | 0.500 | 0.410 | 0.61 |
| Tier 3 | multi-agent | 0.405 | 0.339 | 0.54 |
| Tier 4 | multi-agent | 0.405 | 0.341 | 0.54 |
| Tier 5 | multi-agent | 0.405 | 0.341 | 0.54 |
| B6 CUSUM | unsupervised | 0.263 | 0.498 | 0.76 |

Best baseline B2 MA crossover (0.581) vs best tier Tier 2 (0.557): **-0.025**


### hormuz

| method | kind | AUC | F1 | alert rate |
|---|---|---|---|---|
| B7 logistic regression | supervised | 0.968 | 0.682 | 0.12 |
| B6 CUSUM | unsupervised | 0.958 | 0.000 | 0.00 |
| B8 random forest | supervised | 0.747 | 0.378 | 0.14 |
| B4 EWMA deviation | unsupervised | 0.496 | 0.108 | 0.12 |
| B5 AR residual | unsupervised | 0.493 | 0.084 | 0.11 |
| B1 rolling z | unsupervised | 0.479 | 0.112 | 0.11 |
| B2 MA crossover | unsupervised | 0.470 | 0.009 | 0.13 |
| B3 isolation forest | unsupervised | 0.434 | 0.085 | 0.17 |
| Tier 2 | multi-agent | 0.357 | 0.091 | 0.12 |
| Tier 5 | multi-agent | 0.338 | 0.086 | 0.26 |
| Tier 4 | multi-agent | 0.338 | 0.086 | 0.26 |
| Tier 3 | multi-agent | 0.338 | 0.086 | 0.26 |
| Tier 1 | multi-agent | 0.327 | 0.034 | 0.14 |

Best baseline B7 logistic regression (0.968) vs best tier Tier 2 (0.357): **-0.610**


### panama

| method | kind | AUC | F1 | alert rate |
|---|---|---|---|---|
| Tier 1 | multi-agent | 0.905 | 0.714 | 0.33 |
| Tier 5 | multi-agent | 0.841 | 0.574 | 0.26 |
| Tier 2 | multi-agent | 0.837 | 0.579 | 0.25 |
| Tier 4 | multi-agent | 0.828 | 0.572 | 0.25 |
| Tier 3 | multi-agent | 0.828 | 0.572 | 0.25 |
| B3 isolation forest | unsupervised | 0.814 | 0.435 | 0.23 |
| B2 MA crossover | unsupervised | 0.523 | 0.192 | 0.12 |
| B4 EWMA deviation | unsupervised | 0.510 | 0.145 | 0.14 |
| B5 AR residual | unsupervised | 0.508 | 0.071 | 0.10 |
| B1 rolling z | unsupervised | 0.484 | 0.154 | 0.10 |
| B6 CUSUM | unsupervised | 0.145 | 0.372 | 0.89 |

Best baseline B3 isolation forest (0.814) vs best tier Tier 1 (0.905): **+0.092**


## What the numbers show


### Adding agents does not monotonically improve AUC

- **bab_el_mandeb**: 0.548 -> 0.557 -> 0.405 -> 0.405 -> 0.405  (best: Tier 2)
- **hormuz**: 0.327 -> 0.357 -> 0.338 -> 0.338 -> 0.338  (best: Tier 2)
- **panama**: 0.905 -> 0.837 -> 0.828 -> 0.828 -> 0.841  (best: Tier 1)

In every region the peak is Tier 1 or Tier 2, and adding the geopolitical agent at Tier 3 lowers AUC each time. This does not support the claim that each agent adds value.


### Some multi-agent scores are anti-correlated with the label

- bab_el_mandeb Tier 3: AUC 0.405
- bab_el_mandeb Tier 4: AUC 0.405
- bab_el_mandeb Tier 5: AUC 0.405
- hormuz Tier 1: AUC 0.327
- hormuz Tier 2: AUC 0.357
- hormuz Tier 3: AUC 0.338
- hormuz Tier 4: AUC 0.338
- hormuz Tier 5: AUC 0.338

An AUC below 0.5 is not noise — it means the score runs opposite to the label. The likely cause is a mismatch of definitions: these agents are shock detectors (z-scores and isolation forests against rolling baselines), while the label marks *sustained* level shifts. Once traffic has settled at a lower level, a shock detector sees a stable series and reports calm. The two are measuring different things, and that is an architectural finding rather than a bug.


### Where simple baselines beat the ensemble

- **bab_el_mandeb**: B2 MA crossover at 0.581 beats the best tier at 0.557
- **hormuz**: B7 logistic regression at 0.968 beats the best tier at 0.357

### False-positive harness (Malacca)

Malacca has zero labelled disruptions across 2019-2026, so every alert is a false alarm. Mean alert rate: **tiers 31%**, **baselines 9%**. The multi-agent tiers are the noisier of the two.


## Limits of this evaluation

- **Supervised baselines are mostly untrainable here.** Every labelled disruption is recent (Houthi 2024, Gatun drought 2023-24), so a temporal split leaves no positives in training for three of four regions. Those rows are reported as not-applicable rather than as a 0.5 score, which would have read as 'no better than chance'.
- **One split, no confidence intervals.** Positive counts are 133-244 days; treat differences under roughly 0.05 AUC as noise.
- **Malacca cannot be evaluated for detection** — no positives at all.
- **Panama alone has news features** (18 vs 14), because GDELT only answered for that region. Its tier 5 is therefore not directly comparable with the others'.
- **The label is one definition among several.** The level-shift rule recovers two documented events, but agents built against a different definition of disruption will score poorly whatever their merit.
