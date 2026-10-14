# Chapter 4 writing notes (XAI-02)

Handoff from the explainability work to `thesis/4_0_Evaluation.tex` and the SRQ3 discussion. Numbers come from `EXPLAINABILITY_REPORT.md` and its CSVs. This file grows phase by phase; the full mapping to the Chapter 4 headings is completed in Phase 8.

---

## SRQ3: the current answer is superseded

**Current answer (stale):** "Can the score be explained faithfully? → Yes, with a stated surrogate caveat" (README SRQ table). It rests on the SHAP surrogate's fit, quoted as **R² = 0.991** (`DEVELOPMENT_LOG.md`, 20-feature era). That number is:
1. a *training-set* R²;
2. on one synthetic year;
3. of a model that no longer exists (20 features, 6 agents). The current 17-feature surrogate gives train R² 0.983.

**Why it is superseded:** Phase 2 measured the surrogate against the real evaluation scores.
- Out of time it fits poorly: R² 0.528 on a chronological hold-out of its own synthetic year.
- On the real test windows it does worse than predicting a constant: R² −0.32 to −4.39.
- It ranks agents no better than chance: deletion AUC 0.49–0.59 vs 0.50 for random order.
- It names the same top agent as the true decomposition on only 18–42% of alert days (0.9% in Panama).
- It puts 15–17% of its attribution on agents that are not in the score.

Meanwhile the score is explicit arithmetic over five agent scores, so it can be decomposed exactly.

### Proposed replacement: README SRQ table, Verdict cell

> **Yes, exactly, at the agent level.** The composite is decomposed by exact Shapley values over the agent scores (efficiency residual ≤ 1.1e-16; METRIC 9, E1). Feature-level attributions inside each agent are exact as well. The SHAP surrogate is kept only as a comparator: on real data it does not reproduce the score (R² −0.32 to −4.39) and agrees with the exact top driver on 18–42% of alert days.

### Proposed replacement: thesis SRQ3 answer (`1_0_Introduction.tex` §SRQ3 framing, `5_0_Discussion.tex` §Explainability and Interpretation)

> Signals are translated into interpretable insight by decomposing each alert's risk score exactly. Because the composite is an explicit function of five agent scores, its Shapley values can be computed over all coalitions rather than estimated. Each agent's contribution is measured against that agent's own calm-day level. The contributions sum to the score up to floating-point error (≤ 1.1 × 10⁻¹⁶). Inside each agent, the same exact decomposition attributes the agent's score to its inputs, with the agent's non-linearity (maximum, clipping, sigmoid, isolation forest) kept inside the game.
>
> A SHAP explanation of a random-forest surrogate, the approach the system originally used, was retained as a comparator and evaluated against the exact decomposition on the real test windows.
> - **Fit:** the surrogate fitted its own synthetic training year closely (R² 0.983), but not a held-out period of it (0.528). On the real evaluation scores it was worse than a constant (R² −0.32 to −4.39).
> - **Ranking:** its agent ranking removed score no faster than a random ordering (deletion AUC 0.49–0.59 against 0.50; exact ordering 0.06–0.17).
> - **Agreement:** it named the same leading agent as the exact decomposition on 18–42% of alert days.
> - **Refitting does not close the gap:** a surrogate refitted on each region's real training window narrows it but does not close it (deletion AUC 0.13–0.34; top-agent agreement 53–94%).
>
> The finding is therefore not that SHAP is uninformative. Where the scoring function is known, a post-hoc surrogate adds an approximation error that an exact decomposition does not have, and that error is large enough to change which driver a user is shown.
>
> The attributions remain associational. They state how the score was computed, not what caused the disruption, and the evaluation label is itself the shipping level-shift statistic. Every result was therefore repeated without the shipping duration branch; no conclusion changed.

**Allowed claim:** "exact agent attributions are faithful by construction (residual ≤ 1.1e-16); the production surrogate's out-of-time R² is 0.53 and its R² on the real test windows is negative."

**Not allowed:**
- "explanations are faithful" without saying *which* explanation;
- any user-trust or comprehension claim (there is no human study);
- "the surrogate is wrong". It is an approximation of a known function, and its error is what was measured.

### Downstream effects
- `3_0_Methodology.tex:1090-1115` says exact analytic attribution is "future work". It is now implemented, so the subsection should be rewritten as the primary method, with the surrogate as comparator.
- The Ch. 4 "Explainability Score" TODO (`4_0_Evaluation.tex:76-85`) is resolved by METRIC 9 (E1–E9). The full text follows in Phase 8.

---

## Results → Decision Support: triage framing (Phase 4)

**Framing:** *score alone ≥ explanation features; a region-specific agreement effect in Panama; its direction reverses in Hormuz.*

**Suggested text:**

> Whether the explanation helps a user decide when to trust an alert was tested with three pre-registered first-day indicators, each scored as the AUC for separating false-positive from true-positive alert episodes: whether the leading exact driver is geopolitical, how many agents are individually elevated, and how much the agents disagree.
>
> One result survived Holm correction. In Panama, false-positive episodes had fewer individually elevated agents (AUC 0.89 [0.74, 0.98], adjusted p = 0.0002), identically with and without the shipping duration branch. The same indicator pointed the other way in Hormuz (0.34 [0.18, 0.47]) and was uninformative in Bab el-Mandeb (0.42). The geopolitical-driver hypothesis held in no labelled region; in Hormuz, geopolitical-led openings were more common among true alerts.
>
> A leave-one-region-out logistic model on the three indicators did worse than the alert score alone in every held-out region (ΔAUC −0.10 to −0.50; Hormuz and Panama CIs exclude zero), because it transfers an indicator's direction from regions where that direction is reversed.
>
> The explanation therefore does not yield a transferable trust rule. Its relationship to alert correctness is region-specific, and with 5–8 true-positive episodes per region the Panama effect is a finding to report, not a rule to deploy.

**Literature links** (bib entries verified in Phase 8):
- **Bansal et al. 2021** ("Does the whole exceed its parts?"): explanations need not improve complementary human-AI performance beyond showing the AI's confidence. This is the human-side analogue of "score alone ≥ explanation features".
- **Schemmer et al. 2022** (meta-analysis of XAI effects on human-AI decision making) and **Haag et al. 2026** (meta-analysis): the effects of explanations on appropriate reliance are small and heterogeneous across tasks and contexts. This is consistent with an effect that appears in one region and reverses in another.

The framing claims nothing about humans. It is a functionally grounded result: no user study was run.

**Allowed:** "explanation features did not improve on the score as a triage signal (LORO); one region-specific effect (Panama agreement) survived correction."
**Not allowed:** "explanations help users calibrate trust"; "geopolitical-led alerts are false alarms".

## Limitations: agreement cut-off quirks (Phase 4, §C P2)

The agreement indicator counts agents whose first-day score exceeds that agent's own train calm-day 90th percentile. Two mechanical properties follow and were left unaltered, because the rule was pre-registered:
1. **Market never counts.** Its calm-day q0.90 is exactly 1.0, the agent's ceiling (the market agent is clipped at 1.0 on ~25% of days, calm ones included; §4.3), so its score cannot exceed it.
2. **Any non-zero natural-disaster score counts.** That agent's calm-day q0.90 is 0.

Agreement is therefore effectively a count over shipping, geopolitical and news, plus any disaster activity. A rule using a strict quantile below the ceiling would behave differently; it was not tried, to avoid post-hoc tuning.

## Outlook

- **Simulated and real reliance.** Measuring appropriate reliance (RAIR/RSR) needs operator-accuracy estimates from a human-subject study. Simulating them with assumed accuracies was dropped, because it would only illustrate the mechanism.
- **Retrieval relevance at scale.** Precision@3 rests on one annotator (the author). A second annotator would give inter-rater agreement.

## Results → Decision Support: METRIC 8 (Phase 5, SYNTHETIC) — canonical version

**Canonical METRIC 8 for Chapter 4 = P0-exact.** In the day-level action rule, `top_agent` is the agent with the largest **exact Shapley** contribution to that day's composite, measured from the seed-42 calm baselines.

| Weight mode | Accuracy | Macro-F1 | Recall: no action / monitor / reroute / escalate |
|---|---|---|---|
| Hand-tuned | 0.937 | 0.708 | 0.96 / 0.77 / 0.17 / 1.00 |
| Optimized | 0.578 | 0.726 | 0.52 / 1.00 / 1.00 / 1.00 |

**Footnote text:** "With the evaluation notebook's original driver rule, the agent with the largest weight × score, hand-tuned accuracy is 0.948 (macro-F1 0.855) and optimized 0.578 (0.726). The exact-driver rule is reported in the text because it is the attribution the system shows its user; the two differ only on the 7 (hand-tuned) / 5 (optimized) high-risk, non-sustained days on which the rule consults the driver."

**Why the numbers changed since the July run** (stale: hand-tuned 0.866, optimized 0.490; `data/processed/evaluation_results.json`, 2026-07-30). That run predates two code changes that alter every agent score feeding the rule:
1. **Removal of the routing agent** (`b2306ae`): six agents became five, and the inter-agent weights were renormalised.
2. **The shipping agent's sustained level-shift (duration) branch** and its absolute isolation-forest normalisation (`f7b87ff`). These changed when risk crosses the HIGH band in the injected scenarios.

Scenario A accuracy rose from 0.07 to 0.73 (hand-tuned), because the shipping signal now holds through the moderate-tension window instead of flickering. The July values must not be cited.

**The optimized weights fall below the naive baseline.** Under the Optuna-optimized weights, METRIC 8 accuracy is 0.578, below the naive 2σ threshold baseline's 0.797. The optimized thresholds lower `risk_medium` to 0.28, so 153 of 318 calm days are pushed into "monitor" (no_action recall 0.52). The optimizer's objective (F1, lead time and FPR on detection) rewards early alerting, and the decision layer pays for it in false monitoring.

This is reportable: optimizing detection did not optimize decisions. It joins the negative findings on the optimized weights (README §4.9).

## Results → Decision Support: METRIC 8 (Phase 5, SYNTHETIC)

**Refreshed METRIC 8 (current code, seed 44):**

| | Accuracy | Macro-F1 | Reroute recall |
|---|---|---|---|
| Hand-tuned | 0.948 | 0.855 | 0.83 |
| Optimized | 0.578, below the naive baseline 0.797 | 0.726 | 1.00 |

The July values (0.866 / 0.490) are stale and must not be cited.

**P0-surrogate vs P0-exact (pre-registered):**
- The two disagree on the top agent on ~80% of days, but the decision rule consults it on only 5–7 days. McNemar p = 1.00 (hand-tuned) / 0.063 (optimized).
- The visible effect: the surrogate-driven dashboard never recommends rerouting (reroute recall 0.00).

Report this as a defect found and fixed, not as a significant improvement.
