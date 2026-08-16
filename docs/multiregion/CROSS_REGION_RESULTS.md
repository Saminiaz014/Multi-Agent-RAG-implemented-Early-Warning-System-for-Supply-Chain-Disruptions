# Cross-Region Results, Predictions, and Thesis Narrative (A7)

Synthesis only — no new baseline/ablation/scenario computation was run to produce
this document. Every number below is read from a results file already generated
by A6 (`scripts/aggregate_all_results.py`, `scripts/aggregate_ablation_results.py`,
region-suffixed CSVs under `results/`) or from
`docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md`'s already-recorded gaps 18–22. The
one exception, noted where it appears, is the `results_all_runs_{region}.csv`
per-scenario filtering used in §1c/§3 item 4 — reading and filtering an existing
CSV, not re-running anything.

---

## 1a. Results table

**Tier 0–2 baselines, best performer per region** (source: `results/results_by_baseline_{region}.csv`, mean over 4 scenarios × 5 seeds = 20 runs per baseline):

| region | best AUC-PR (D3) | value | best Best-F1 (D6) | value |
|---|---|---|---|---|
| hormuz | ewma | 0.6970 | ewma | 0.4273 |
| bab_el_mandeb | ewma | 0.7008 | ewma | 0.4344 |
| panama | ewma | 0.7497 | ewma | 0.4985 |
| suez | ewma | 0.6515 | iforest | 0.4273 |
| malacca | iforest | 0.6255 | iforest | 0.3783 |

EWMA (a Tier 1 single-signal control-chart baseline reading `shipping` only) has the
best or tied-best AUC-PR in four of five regions; Isolation Forest (Tier 2,
genuinely multivariate) wins outright only in malacca and ties on F1 in suez. Full
16-metric × 10-baseline tables: `results/results_by_baseline_{hormuz,bab_el_mandeb,panama,suez,malacca}.csv`.

**Best-F1 by scenario type** (source: `results/results_by_scenario_{region}.csv`):

| baseline | metric | hormuz | bab_el_mandeb | panama | suez | malacca |
|---|---|---|---|---|---|---|
| ewma | P_CRIT | 0.7741 | 0.8100 | 0.9986 | 0.6034 | 0.4425 |
| ewma | P_HIGH | 0.9350 | 0.9277 | 0.9953 | 0.8814 | 0.2275 |
| iforest | P_CRIT | 0.8365 | 0.8206 | 0.9407 | 0.9091 | 0.8427 |
| iforest | P_HIGH | 0.7933 | 0.7920 | 0.9100 | 0.8000 | 0.6704 |
| always_alarm / never_alarm | N_QUIET, N_DECOY | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

N_QUIET and N_DECOY show `0.0` Best-F1 for every baseline in every region — this is
correct by construction (both windows have zero positive test-window days for every
negative-control scenario) and is **not** a defect; see Limitations item 1 for why
this column cannot be read as "no false alarms."

**Ablation A0–A7 aggregate** (source: `results/ablation_findings_{region}.csv`,
mean over 4 scenarios, seed=42): reproduced in full in §1d below.

**Region-level detection floor** (source: `results/results_by_baseline_{region}.csv`,
`always_alarm`/`never_alarm` rows, `D3_auc_pr_mean`): the trivial-baseline AUC-PR
floor is not `0.5` for every region as it would be under balanced classes — it
tracks each region's actual test-window positive-day fraction (panama 0.6652,
hormuz/bab_el_mandeb ~0.548, malacca 0.5327, suez 0.5134) — panama's P_CRIT alone
occupies 69 of 84 test days (§1a scenario table above), pulling `always_alarm`'s
AUC-PR far above the other four regions'. Any AUC-PR comparison across regions must
be read relative to this per-region floor, not against a shared `0.5` baseline.

---

## 1b. Predictions vs. outcomes

Source: `docs/multiregion/A6_PREDICTIONS.md`, written and committed before any
bab_el_mandeb/panama/suez/malacca result existed. **Methodology used to measure
"multi-agent gain" throughout this section:** ablation A0 (`shipping_only`, the
single-domain floor) vs. A6 (`6_+bonus` — all of that region's active domains,
Optuna-tuned, with the agreement bonus), read from `results/ablation_findings_{region}.csv`.
This is the one comparison in the generated data that isolates exactly the
"aggregating multiple domains" variable the predictions document's own
falsification language (§1: "fused-system FPR/TPR/F1 improvement over the best
single domain baseline") describes, without conflating it with which detection
algorithm is used. Two axes are reported because they disagree with each other (see
§1c) — reducing "gain" to one number would hide that disagreement, not simplify it.

| region | A0 FPR (D9) | A6 FPR (D9) | ΔFPR (A0−A6, + = improvement) | A0 F1 (D6) | A6 F1 (D6) | ΔF1 (A6−A0, + = improvement) |
|---|---|---|---|---|---|---|
| hormuz | 0.7262 | 0.3374 | **+0.3888** | 0.2444 | 0.1663 | **−0.0781** |
| bab_el_mandeb | 0.7262 | 0.3798 | +0.3464 | 0.2337 | 0.1756 | −0.0581 |
| panama | 0.7202 | 0.5179 | +0.2023 | 0.3922 | 0.3922 | +0.0000 |
| suez | 0.6274 | 0.2964 | +0.3310 | 0.0485 | 0.2500 | **+0.2015** |
| malacca | 0.4537 | 0.3003 | +0.1534 | 0.1158 | 0.2619 | +0.1461 |

### §1 Complementarity ranking

**Predicted:** Panama > Bab el-Mandeb ≳ Hormuz > Malacca > Suez (highest to lowest
complementarity, expected to track gain size).

**Measured:**
- ΔFPR ranking (largest gain to smallest): hormuz (0.389) > bab_el_mandeb (0.346) >
  suez (0.331) > panama (0.202) > malacca (0.153).
- ΔF1 ranking (largest gain to smallest): suez (+0.202) > malacca (+0.146) > panama
  (0.000) > bab_el_mandeb (−0.058) > hormuz (−0.078).

**Verdict: FAILED**, on the prediction document's own named falsification triggers.
Section 1's falsification condition was: *"falsified if... Suez shows a larger gain
than Hormuz or Bab el-Mandeb, or if Malacca's gain exceeds Panama's."* On ΔF1, Suez's
gain (+0.2015) is larger than both Hormuz's (−0.0781) and Bab el-Mandeb's (−0.0581) —
the first named trigger fires exactly as written. Malacca's ΔF1 gain (+0.1461)
exceeds Panama's (0.0000) — the second named trigger also fires exactly as written.
Panama, predicted highest-complementarity, shows the *smallest* F1 gain of any
region with a positive floor to fall from (Panama's already-high 0.90+ Best-F1 on
P_CRIT — see §1a — likely leaves less room for fusion to add F1 headroom at all,
a ceiling effect the prediction did not anticipate). This is reported as a failed
prediction, not reframed after the fact.

### §2 Per-region predicted direction of gain

| region | prediction | measured (ΔFPR / ΔF1) | verdict |
|---|---|---|---|
| hormuz | small gain, ~12% (stable re-run of an existing figure) | +0.3888 / −0.0781 | **Could not verify baseline.** No file in this repository sources a "~12%" hormuz fusion-FPR figure to compare against (searched the full repo for "12%" and "FPR improvement" — the only hits are `A6_PREDICTIONS.md` itself). Under the consistent A0-vs-A6 methodology applied to every region here, hormuz's own ΔFPR is a 53.5% *relative* reduction (0.3888/0.7262) and ΔF1 is a 32% *relative worsening* — either reading is a material difference from "~12%" under any interpretation, so this prediction is reported as **falsified**, with the caveat that the original "~12%" methodology is unsourced and may not be the same comparison.
| suez | weak gain, comparable to or smaller than Hormuz's | ΔFPR 0.331 (3rd of 5, below hormuz's 0.389) but ΔF1 +0.2015 (**largest of all five regions**) | **FAILED** on F1, matching the prediction's own named trigger ("falsified if Suez shows a strong fusion gain despite this domain-sparse, no-lead structure"). Ambiguous-to-passing on FPR alone.
| bab_el_mandeb | moderate gain, at or slightly above Hormuz's | ΔFPR 0.346 (slightly *below* hormuz's 0.389); ΔF1 −0.058 (slightly *less negative* than hormuz's −0.078, i.e. marginally "above" in the sense both are negative) | **Marginally failed on FPR, technically not-failed on F1** (both regions show fusion *hurting* F1, so "at or above" holds trivially in a direction the prediction did not intend). Not a clean pass either way — reported as-is rather than rounded to a verdict.
| panama | largest gain of the five regions | ΔFPR 4th of 5 (0.202); ΔF1 exactly 0.0000, middle of the pack | **FAILED**, matching the prediction's own named trigger exactly ("falsified if Panama's gain is not the largest").
| malacca | small gain, second-smallest after Suez | ΔFPR *smallest* of all five (0.153, i.e. worst, not second-smallest); ΔF1 **second-largest** of all five (+0.1461) | **FAILED** on F1, matching the prediction's own named trigger ("falsified if Malacca's gain is comparable to Panama's or Bab el-Mandeb's") — it is not merely comparable, it exceeds both.

### §3 Lead-time ranking

**Not available.** No lead-time / onset-detection metric is computed anywhere in
this pipeline. `scripts/generate_results_summary.py`'s own `_UNMEASURED_METRICS`
table lists `"E1 Lead time / MTTD"` with the reason `"no onset-detection logic
exists"` — this was true before A6 and remains true after it; A6 added regions and
fixed evaluation bugs, it did not add lead-time evaluation. The predicted ranking
(Panama best, Hormuz ≈ Bab el-Mandeb moderate, Suez worst, Malacca uncertain vs.
Suez) is a structural/onset-timing argument, not a measured one, and cannot be
marked held or failed here. It remains an open prediction for future work (see
`docs/ABLATION_RATIONALE.md`/roadmap item on lead-time evaluation).

### §4 False-positive-rate (decoy) ranking

**Predicted:** Malacca best (lowest FPR) > {Hormuz, Bab el-Mandeb, Panama} clustered
> Suez worst.

**Measured** (N_DECOY-scenario-specific `D9_fpr_tau`, filtered from
`results/results_all_runs_{region}.csv`, config `A6`):

| region | A6 N_DECOY FPR |
|---|---|
| hormuz | **0.4762** (lowest) |
| bab_el_mandeb | 0.5238 |
| malacca | 0.5238 |
| panama | 0.5238 |
| suez | 0.5238 |

**Verdict: FAILED**, on every named component. Hormuz is lowest, not Malacca.
Malacca, bab_el_mandeb, panama, and suez are in a four-way *exact tie*, not "three
clustered, one worst" — Suez is not distinctly worst, and Malacca is not distinctly
best.

**Traced cause, not retrofitted reasoning — a mechanical artifact of the tuning
procedure, not evidence against the underlying decoy-design argument.** N_DECOY's
validation window (days 201–280) has zero positive days in every region by
construction (negative-control scenarios have zero positives everywhere). Per
`tune_weights_optuna`'s own docstring (`src/baselines/ablation_runner.py`): *"Falls
back to equal weights if no trial beat an all-zero F1"* — exactly the condition
every region's N_DECOY hits. Confirmed directly from the ablation result JSONs'
`metadata.weights`: bab_el_mandeb/malacca/panama/suez (each 5 active domains) all
tune to `{domain: 0.2 for each of 5 domains}`; hormuz (6 active domains) tunes to
`{domain: 0.1667 for each of 6 domains}`. With every 5-domain region landing on
identical equal weights and sharing seed-42 noise draws (documented in gap 19c),
their composite decoy scores collapse to the same discrete FPR value at the
F1-tuned threshold. A6, as configured, cannot discriminate the region-specific
decoy-orthogonality argument the prediction was built on — **not because that
argument about domain orthogonality is wrong, but because the equal-weights
fallback this specific scenario always triggers erases whatever differentiation
per-region decoy design would otherwise produce.** Cross-checked against A4
(static hand-tuned weights, not subject to the Optuna fallback): hormuz 0.5000
(best) < bab_el_mandeb = suez 0.5238 (tied) < malacca 0.5357 < panama 0.5595
(worst) — still does not match the predicted ranking (Malacca not best; Panama,
not Suez, is worst). No ablation configuration checked (A0, A3, A4, A6) reproduces
the predicted ranking.

---

## 1c. Complementarity analysis

**The thesis claim under test:** multi-agent value is conditional on signal
complementarity, not on domain count alone.

**Does measured gain track predicted complementarity? No — stated plainly, per the
task's own instruction that a negative result here is a finding, not a problem.**
Neither of the two gain axes (ΔFPR, ΔF1) reproduces the predicted ordering (Panama >
Bab el-Mandeb ≳ Hormuz > Malacca > Suez), and — more consequentially for the
methodology itself — **the two axes are close to inverses of each other**: the
region with the largest FPR gain (hormuz) has the smallest (most negative) F1 gain,
and the region with the largest F1 gain (suez) has only the third-largest FPR gain.
Fusion in this benchmark is not well-described by a single "gain" scalar at all — it
trades recall/F1 for false-alarm suppression differently per region, and the
direction and size of that trade does not line up with the domain-count-and-lead-
structure reasoning the predictions document used to rank complementarity. The
domain-sparse, structurally-distinct regions (suez: 3 near-simultaneous domains;
malacca: 2 domains, one lagging) show the multi-agent system's fusion (A6) *adding*
detection power (F1) over the single-domain floor, while the domain-rich regions
with a common escalating-tension root cause (hormuz, bab_el_mandeb) show fusion
*reducing* F1 while also reducing FPR — i.e. the agreement bonus is suppressing
alarms broadly enough to cost some true positives along with false ones. This is
the opposite of what "more domains, more complementarity, more gain" would predict,
and it is reported as such rather than reinterpreted to fit.

---

## 1d. Ablation table

Source: `results/ablation_findings_{region}.csv`, mean over 4 scenarios, seed=42.
Degeneracy labels per `docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md` §6 gap 21/22
(`scripts/run_ablations.py`'s `_compute_degeneracy`, recorded in each result's
`metadata.degenerate_of`).

**hormuz** (`region.active_domains`: 6 — shipping, market, geopolitical, routing,
news, disaster) — **4/8 distinct configs**:

| config | D3_auc_pr | D6_best_f1 | D8_recall_tau | D9_fpr_tau | degenerate_of |
|---|---|---|---|---|---|
| A0 | 0.4467 | 0.2444 | 0.5000 | 0.7262 | — |
| A1 | 0.4301 | 0.2102 | 0.5000 | 0.7381 | — |
| A2 | 0.3118 | 0.1607 | 0.0385 | 0.3926 | — |
| A3 | 0.3097 | 0.1607 | 0.0132 | 0.4208 | — |
| A4 | 0.3088 | 0.1599 | 0.0000 | 0.4282 | A3 |
| A5 | 0.3172 | 0.1663 | 0.0000 | 0.3374 | A3 |
| A6 | 0.3188 | 0.1663 | 0.0000 | 0.3374 | A3 |
| A7 | 0.3188 | 0.1663 | 0.0000 | 0.3374 | A3 |

**bab_el_mandeb** (5 domains: shipping, market, geopolitical, routing, news) —
**4/8 distinct**:

| config | D3_auc_pr | D6_best_f1 | D8_recall_tau | D9_fpr_tau | degenerate_of |
|---|---|---|---|---|---|
| A0 | 0.4363 | 0.2337 | 0.5000 | 0.7262 | — |
| A1 | 0.4242 | 0.2053 | 0.5000 | 0.7381 | — |
| A2 | 0.3114 | 0.1607 | 0.0385 | 0.3926 | — |
| A3 | 0.3112 | 0.1599 | 0.0000 | 0.4587 | — |
| A4 | 0.3097 | 0.1592 | 0.0132 | 0.4628 | A3 |
| A5 | 0.3242 | 0.1784 | 0.0132 | 0.3658 | A3 |
| A6 | 0.3279 | 0.1756 | 0.0324 | 0.3798 | A3 |
| A7 | 0.3279 | 0.1756 | 0.0324 | 0.3798 | A3 |

**panama** (5 domains: shipping, market, routing, news, disaster) — **4/8 distinct**:

| config | D3_auc_pr | D6_best_f1 | D8_recall_tau | D9_fpr_tau | degenerate_of |
|---|---|---|---|---|---|
| A0 | 0.6240 | 0.3922 | 0.4762 | 0.7202 | — |
| A1 | 0.6232 | 0.3957 | 0.4881 | 0.7202 | — |
| A2 | 0.5943 | 0.3922 | 0.4405 | 0.7202 | — |
| A3 | 0.5789 | 0.3950 | 0.3095 | 0.6131 | — |
| A4 | 0.5878 | 0.3950 | 0.3869 | 0.6845 | A3 |
| A5 | 0.6137 | 0.3922 | 0.3155 | 0.5179 | A3 |
| A6 | 0.6150 | 0.3922 | 0.3214 | 0.5179 | A3 |
| A7 | 0.6150 | 0.3922 | 0.3214 | 0.5179 | A3 |

**suez** (5 domains: shipping, market, routing, geopolitical, news) — **4/8
distinct**:

| config | D3_auc_pr | D6_best_f1 | D8_recall_tau | D9_fpr_tau | degenerate_of |
|---|---|---|---|---|---|
| A0 | 0.2652 | 0.0485 | 0.3333 | 0.6274 | — |
| A1 | 0.2677 | 0.0571 | 0.5000 | 0.6967 | — |
| A2 | 0.2814 | 0.0786 | 0.4167 | 0.6202 | — |
| A3 | 0.3161 | 0.1493 | 0.2083 | 0.3442 | — |
| A4 | 0.3056 | 0.1417 | 0.2083 | 0.3475 | A3 |
| A5 | 0.3904 | 0.2667 | 0.2083 | 0.2777 | A3 |
| A6 | 0.3852 | 0.2500 | 0.2083 | 0.2964 | A3 |
| A7 | 0.3852 | 0.2500 | 0.2083 | 0.2964 | A3 |

**malacca** (5 domains: shipping, disaster, news, geopolitical, routing — no
`market`) — **3/8 distinct**, the only region below 4/8:

| config | D3_auc_pr | D6_best_f1 | D8_recall_tau | D9_fpr_tau | degenerate_of |
|---|---|---|---|---|---|
| A0 | 0.2984 | 0.1158 | 0.0857 | 0.4537 | — |
| A1 | 0.2984 | 0.1158 | 0.0857 | 0.4537 | **A0** |
| A2 | 0.3151 | 0.1445 | 0.0833 | 0.3357 | — |
| A3 | 0.3899 | 0.2258 | 0.1571 | 0.2999 | — |
| A4 | 0.3565 | 0.1870 | 0.1024 | 0.3178 | A3 |
| A5 | 0.4663 | 0.2291 | 0.1071 | 0.2720 | A3 |
| A6 | 0.4565 | 0.2619 | 0.2119 | 0.3003 | A3 |
| A7 | 0.4565 | 0.2619 | 0.2119 | 0.3003 | A3 |

Malacca's A1 (`shipping`, `market`) has no `market` domain to score once
domain-scoped to `region.active_domains`, so it collapses to exactly A0's
single-`shipping` configuration — confirmed identical in the actual numbers above,
not just the domain sets, matching `BENCHMARK_SCHEMA_REFERENCE.md` §6 gap 22's
report of this case.

A4–A7's degeneracy onto A3 (all five/six active domains, differing only in
weighting/bonus strategy) is present in **every** region including hormuz, and is a
pre-existing property of the `ABLATIONS` design (A3–A7 always declare the same
domain list), not a scoping artifact introduced by gap 21's fix.

---

## 2. Limitations

Every accumulated limitation qualifying the cross-region comparison above, in one
place, not buried in individual result files.

1. **Tier 0 reads no domain values at all; Tier 1's five baselines read only
   `shipping`.** Confirmed directly from `src/baselines/tier0_controls.py` /
   `tier1_statistical.py`: every Tier-1 baseline's `run()` opens with
   `_fill_missing(df["shipping"].to_numpy())`. `N_DECOY` — a decoy on `news`,
   `market`, or `geopolitical` depending on region — is structurally invisible to
   four of the five Tier-1 baselines and to all three Tier-0 controls, regardless of
   decoy magnitude. This means `N_DECOY`'s `0.0` Best-F1 for those baselines (§1a)
   reflects "the decoy domain was never read," not "the detector correctly ignored
   it" — the two are indistinguishable in this metric. Only Tier 2 (Isolation
   Forest, Matrix Profile) and the ablation configs that include the decoy's domain
   are a like-for-like negative test across method tiers.
2. **Tier 2's fixed `contamination=0.1` imposes a structural false-flag floor on
   any `N_QUIET` scenario, independent of decoy design**, per
   `BENCHMARK_SCHEMA_REFERENCE.md` §6 gap 19b: `hormuz_N_QUIET` flags 20/84 test
   days (23.8%); `bab_el_mandeb_N_QUIET` flags 13/84 (15.5%). Hormuz — the
   already-committed reference region — has the *higher* of the two measured rates,
   confirming this is a property of the Tier 2 method, not a defect specific to a
   new region's scenario authoring.
3. **All regions share `seed: 42`** for their primary comparison runs — the same
   noise realization offset by each region's own baseline mean, not independent
   draws (gap 19c). No cross-region confidence interval or significance test is
   computed anywhere in this pipeline; every table above reports point estimates
   only. A region-to-region difference of a few hundredths in any metric should not
   be read as statistically distinguishable from noise.
4. **Decoy threshold-crossing is not uniform across regions** — `N_DECOY` tests a
   structurally different thing per region (gap 18's threshold-crossing table,
   reproduced here for context): hormuz/bab_el_mandeb/panama's `news` decoys clear
   their single-domain agent's `0.40` threshold by ~2× (peaks `0.8411`/`0.8311`/
   `0.8949`); malacca's `geopolitical` decoy peaks at `0.4259` against a `0.50`
   threshold — sub-threshold, the only region's decoy that would not fire a
   single-domain detector reading that domain alone; suez's `market` decoy
   (`5.7947`) has no comparable-units threshold to compare against at all (the real
   market agent gates a z-scored, further-processed statistic, not this raw
   additive index). Any FPR comparison across regions inherits this asymmetry.
5. **Event onset is now near-uniform across regions** — six of the seven scenarios
   edited in gap 22 now start on day 266–279 (only panama's 83-day P_HIGH, onset
   240, is an outlier by construction). Regions differ from each other in signal
   structure, duration, and magnitude (per §1a/§1d), not in *when* the labeled
   window begins — a side effect of the gap-22 fix's mechanical goal (straddle day
   281), not a design choice about representativeness.
6. **Ablation weights are shared/Optuna-tuned per scenario, not per region beyond
   what that tuning search does automatically.** No separate per-region
   hyperparameter search or held-out region-specific validation exists; A5–A7's
   weights come from the same `tune_weights_optuna` procedure applied identically
   everywhere (see §1b §4's equal-weights-fallback finding for one concrete
   consequence of this).
7. **Suez's P_CRIT (Ever Given) exercises only 3 of the region's 5 active
   domains** (`shipping`, `market`, `news`; `routing` and `geopolitical` are
   `effect: null` in that file). `config/benchmark/suez.yaml`'s region-level
   strength classifications (`routing`: DOMINANT, `geopolitical`: WEAK) describe the
   region across *both* its documented events (Ever Given and the 2023–2024 Red Sea
   knock-on); only the latter, unmodeled event earns `routing`'s DOMINANT rating.
   Any reading of "suez's routing signal should be strong" against this benchmark's
   actual `suez_P_CRIT` scenario is not supported by what that scenario models.
8. **Suez has few positive days: `P_CRIT` 5 val-window / 6 test-window; `P_HIGH` 2 /
   3** (§1a scenario table). Its metrics — computed over single-digit test-day
   counts — carry substantially higher variance than hormuz's 19/panama's 69
   test-window positives. Small numeric differences between suez and other regions
   in any table above should not be over-read as meaningful.
9. **Ablation depth is not constant across regions** — malacca supports 3
   distinct configurations after domain-scoping (§1d); every other region supports
   4. A malacca-vs-other-region ablation comparison is comparing a 3-point
   progression against a 4-point one.
10. **Suez's positive days are causally independent of bab_el_mandeb's**, despite
    both regions sitting on the same Red Sea/Bab el-Mandeb shipping lane: the
    2023–2024 Red Sea knock-on event that drives bab_el_mandeb's `P_CRIT` is
    deliberately *not* blended into suez's `P_CRIT` (which models only the 2021
    Ever Given grounding) — recorded explicitly in `suez_P_CRIT.yaml`'s own header.
    The two regions' benchmark results are not a before/after view of one shared
    disruption.
11. **Two evaluation-invalidating bugs were found and fixed during A6.** Gap
    20/21: the aggregation layer (`scripts/aggregate_all_results.py`,
    `scripts/aggregate_ablation_results.py`) pooled every region's results under a
    header hardcoded to say "mean over Hormuz 4 scenarios," and separately, `A2`–`A7`
    scored domains a region didn't have active, silently poisoning the composite
    score with `NaN`. Gap 22: the `"P_HIGH onset = P_CRIT onset × 0.5"` convention
    placed every region's `P_HIGH` event entirely before the validation window, so
    `P_HIGH` was evaluated nowhere, in any region, including hormuz — **hormuz's own
    committed `results/baselines/tier0/hormuz_P_HIGH_*.json` results, checked into
    git before this pass, were structurally void** (confirmed: `D6_best_f1` was
    exactly `0.0` for all 10 baselines including `always_alarm`, which would score
    nonzero trivially if any positive test day existed) and have since been
    regenerated. Every number in this document postdates both fixes; anyone citing
    a pre-2026-08-16 `hormuz_P_HIGH` or malacca/panama/suez ablation result should
    treat it as superseded.

---

## 3. Claim tiering

Splitting results by comparability, so nothing here is read as a like-for-like
comparison across incomparable datasets.

- **Tier A — controlled internal comparison.** Multi-agent (ablation A6) vs.
  single-domain (A0) and vs. Tier 0–2 baselines, all on this benchmark's own
  synthetic scenarios, same 4×5 grid, same evaluation protocol, same compute
  budget. **This is the only tier where a defensible quantitative claim can be
  made** — everything in §1a–§1d above lives here. The complementarity-ranking
  failure (§1c) is itself a Tier A finding: a controlled, internally-comparable
  result that did not match the pre-registered hypothesis.
- **Tier B — protocol adoption.** The evaluation *methods* (AUC-PR/AUC-ROC/F1-at-
  threshold/best-F1, EWMA/CUSUM/SARIMA control charts, Isolation Forest, Matrix
  Profile) are drawn from published anomaly-detection literature and re-implemented
  here (`src/baselines/`), but run on this project's own synthetic data, not the
  papers' original datasets. Comparable in *method* to published work; the
  resulting *numbers* are not comparable to those papers' reported numbers, because
  the underlying data differs.
- **Tier C — contextual reference range.** Any published F1/AUC figure from
  external anomaly-detection literature, cited only to give a reader a sense of
  scale for what "good" looks like in that literature — never as a like-for-like
  comparison against Tier A's numbers, since Tier C figures are measured on
  different (often proprietary) datasets with different label definitions,
  different class balance, and no shared evaluation protocol with this benchmark.

**The optimism-gap point, recorded explicitly:** published anomaly-detection
figures on proprietary datasets are commonly reported in the ~0.89–0.94 F1 range.
**This specific figure could not be traced to a source file in this repository** —
searching the codebase for it surfaces two unrelated numbers: this project's own
Phase 4 Optuna weight-optimization result (`README.md`, hand-tuned F1 0.92 →
optimized F1 0.94, honest train/validation/test split, on the *production*
pipeline's synthetic training data, not a proprietary external dataset), and an
unrelated RAG historical-precedent cosine-similarity score (`0.89`, not an F1
metric at all). The ~0.89–0.94 figure is recorded here as asserted context (likely
sourced from a literature-review chapter maintained outside this code repository),
not as a verified in-repo citation — flagged per this document's own no-invented-
numbers rule rather than presented as sourced. Against that reference range, this
benchmark's Tier A honest, chronological-split (no future-leakage), controls-
included, decoy-tested evaluation reports substantially lower headline numbers
(Best-F1 in the 0.05–0.50 range across regions and baselines, §1a/§1d) — **framed
as a methodological contribution** (a harder, more honest evaluation protocol that
does not let a detector see its own test period during tuning, and that is scored
against explicit floor/ceiling controls and adversarial decoys), not as a shortfall
against Tier C's numbers, which are not measuring the same thing.

---

## 4. Thesis narrative

Across the five regions evaluated, multi-agent fusion (ablation A6, all of a
region's active domains plus a multi-domain agreement bonus) did not deliver a
uniform benefit over a single-domain detector (A0, `shipping` alone), and its
benefit was not well-described by a single number. In the two domain-rich regions
that share hormuz's structural shape — hormuz itself and bab_el_mandeb, where five
or six domains move as largely correlated reflections of one escalating-tension
root cause — fusion reduced the false-positive rate substantially (hormuz: FPR
0.726 → 0.337; bab_el_mandeb: 0.726 → 0.380) but did so by suppressing true
detections along with false ones, costing 6–8 points of Best-F1 relative to the
single-domain floor. In the two domain-sparse, structurally distinct regions —
suez (three near-simultaneous domains reacting to one zero-warning physical
blockage) and malacca (two domains, one of which lags the physical event by three
days) — fusion instead *improved* Best-F1 substantially (suez: 0.049 → 0.250;
malacca: 0.116 → 0.262) while still reducing FPR, though by less than in the
domain-rich regions. Panama, predicted to show the strongest fusion benefit of any
region on the strength of its long domain lead times (a leading disaster indicator
75 days ahead of the physical onset), instead showed essentially no F1 change at
all (0.392 → 0.392) — its single-domain floor was already close to ceiling on this
benchmark's own P_CRIT scenario, leaving little headroom for fusion to add.

The pre-registered prediction was that multi-agent value is conditional on signal
complementarity — that domains carrying genuinely independent information should
show a larger fusion benefit than domains that are many in count but redundant in
substance. The measured results do not support that specific ranking: the region
predicted to have the highest complementarity (panama) showed the smallest F1 gain
of the five; the two regions predicted to have the lowest complementarity (suez,
malacca) showed the two largest F1 gains. What the results do support, more
narrowly, is that complementarity affects the *character* of fusion's benefit, not
merely its *size*: fusion behaves differently — trading recall for false-alarm
suppression in the domain-rich regions, and adding detection power outright in the
domain-sparse ones — in a way that tracks something about domain structure, just
not the specific ranking this study pre-registered. Four of the five per-region
predictions and the decoy-FPR ranking prediction failed against their own named
falsification conditions, most sharply for suez (predicted the weakest fusion gain
of any region; measured the strongest) and malacca (predicted second-weakest;
measured second-strongest). The lead-time prediction could not be evaluated at all,
because no lead-time metric exists in this pipeline.

This is reported as a failed hypothesis about the *specific mechanism* proposed —
domain count and lead-time structure predicting fusion gain — not as evidence that
multi-agent fusion has no value. Fusion measurably changed detection behavior in
every region tested; it simply did not change it in the direction, or for the
reason, this study predicted in advance. A benchmark built specifically to test
that mechanism, and that found it did not hold, is a stronger result for a thesis
than a benchmark that only reports numbers consistent with its own hypothesis —
the finding here is that whatever is actually gating fusion's benefit across these
five regions is not yet identified, and the domain-count/lead-time story proposed
in `A6_PREDICTIONS.md` is not it.
