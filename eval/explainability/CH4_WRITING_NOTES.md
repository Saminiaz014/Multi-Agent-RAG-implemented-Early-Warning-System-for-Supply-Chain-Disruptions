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
