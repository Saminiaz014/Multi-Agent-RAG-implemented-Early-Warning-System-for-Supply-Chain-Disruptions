# Chapter 4 writing notes (XAI-02, final)

This is the handoff from the explainability work to `thesis/4_0_Evaluation.tex`, organised by that file's own headings (labels in brackets). Every number comes from `EXPLAINABILITY_REPORT.md` and its CSVs, rebuilt by `python eval/explainability/make_figures.py`. Figures are in `thesis/figures/ch4/`; each has a `.caption.txt` with the exact numbers shown.

**`.tex` files are not edited here.** The text blocks below are drafts for the author.

---

## 0. Main framing for the chapter

> **The value of explanation in this system is diagnostic auditability, not trust triage.**

The exact decomposition is faithful by construction. Its usefulness showed up as a set of **defects it exposed** in the system itself:

1. **The production explainer was unfaithful.** The SHAP surrogate has negative R² on the real data, ranks agents no better than chance, and named a mismatched seed-42 row in the dashboard (§Results → Explainability; F4, F5, F10).
2. **The retrieval layer was defective.** It returned wrong-region, duplicated and future-dated "precedents", with a similarity floor that every configuration passes (F7).
3. **The market agent is saturated.** It sits at its ceiling on ~30% of days, calm ones included, so it cannot contribute to alerts (F15).
4. **The label masks the detector's signal.** In Hormuz the only signal that tracks the label is the label's own statistic, and it is no stronger than the other agents' day-to-day noise (F13).

What the explanation did **not** deliver is a way to tell good alerts from bad ones:
- explanation features do not beat the score as a triage signal;
- the only surviving effect (agreement, Panama) reverses in Hormuz;
- uncertainty does not separate TP from FP;
- the knowledge base shows no more conflict around false alarms than around calm days (F8, F9, F16).

---

## 1. §Evaluation Framework → Evaluation Objectives [`Abschnitt_Evaluation_Objectives`]

Add one falsifiable expectation for explainability:

> *E-X1:* An exact decomposition of the composite is more faithful to the scored function than the deployed SHAP surrogate. (Falsified if the surrogate matches exact on deletion AUC or top-driver agreement.)
>
> *E-X2:* Explanation features identify unreliable alerts better than the score alone. (Falsified if the leave-one-region-out model fails to beat score-only.)

E-X1 is supported. E-X2 is falsified.

## 2. §Experimental Setup [`Abschnitt_Experimental_Setup`]

- **Data.** Real evaluation frames for four chokepoints, 2019-01-01 → 2025-08-30.
  - Temporal 70/30 split: train to 2023-08-30, test 2023-08-31 → 2025-08-30.
  - Label `y_true`: 30-day mean transits ≥ 20% below the trailing 365-day median for 14+ days.
- **Explained score.** The Tier 5 composite of the method comparison: a renormalised weighted mean of the agent scores, *Path D*. It is reproduced bit-for-bit (frozen-score test, 1.1e-16).
- **Alerts.** Days above the train-window 90th percentile of the composite.
- **Episodes.** Alert days merged across gaps of ≤ 3 days. TP episodes: **5 / 7 / 8** (Hormuz / Bab el-Mandeb / Panama); Malacca has none.
- **Unit of evidence.** The episode, with 1,000-resample episode bootstrap CIs.
- **Dashboard and METRIC 8** run on a separate synthetic series (seed 44), with the agreement bonus and cap (*Path C*); these are labelled SYNTHETIC wherever they appear.
- **Pre-registration.** Every Phase 3–5 rule was committed before it ran (`PREREGISTERED_RULES.md`; commits `ae8c7b7`, `f89f445`, `f0f2253`, amendments `d8cdc62`, `ea1e5bb`, `0c2cfff`).

## 3. §Baseline Approaches [`Abschnitt_Baselines`]

The explainability baselines are:
- the **production SHAP surrogate** (random forest on one synthetic year, as deployed);
- a **refitted surrogate** (same recipe on each region's real train window), the strongest form of the surrogate approach;
- a **random attribution order** for the deletion/insertion tests.

For triage, the baseline is the **alert score alone**.

## 4. §Validity and Threats to Validity [`Abschnitt_Validity`]

Bullets for the threats paragraph:
- **Label circularity.** The label is the shipping agent's own level-shift statistic. Every result was repeated with the shipping duration branch removed; no conclusion changed.
- **No human-subject study.** All explainability measures are functionally grounded. No claim about user understanding, trust or decision quality is made.
- **Calibration is not estimable.** 3 of 4 regions have zero train-window positives. A score-noise band is reported instead, and it is not a probability.
- **Synthetic decision labels.** METRIC 8's correct actions are rule-derived from injected scenario windows, not observed decisions.
- **Single annotator.** Retrieval relevance (E8) is annotated by the author alone, blind to TP/FP.
- **Small n.** 5–8 TP episodes per region; Hormuz has 5. Every triage AUC carries a CI, and interpretation requires ≥ 5 TP and ≥ 5 FP episodes.
- **Registry vs evaluation agent sets.** The method-comparison tiers include agents the region registry marks passive (geopolitical in Panama, market in Malacca, natural disaster in Bab el-Mandeb). Explanations use the evaluation's agent sets, so their numbers match the CSV.
- **Knowledge-base coverage.** Before the bounded 2023-25 backfill, 123/123 FP episodes had no knowledge-base documents at all in their windows.

---

## 5. §Performance Metrics → Explainability Score [`Abschnitt_Metric_Explainability`]

**This replaces the TODO at `4_0_Evaluation.tex:76-85`.**

> Explainability is measured by METRIC 9, a scorecard of nine properties after Nauta et al.'s Co-12 framework~\cite{Nauta.2023}. Each is computed on every alert day and averaged first within each alert episode, then across episodes, with an episode-level bootstrap 95% CI.
>
> - **Correctness:** E1 faithfulness (efficiency residual; for surrogates, R² and local error) and E2 deletion/insertion AUC~\cite{Petsiuk.2018}, in which agents are moved to their calm baseline in attribution order.
> - **Compactness:** E3, the number of agents and features covering 80% of |attribution|.
> - **Consistency / continuity:** E4, rank stability across consecutive days and under agent-score noise.
> - **Coherence:** E5, agreement of the leading agent across methods.
> - **Contrastivity:** E6, the share of alerts with a feasible single-agent counterfactual.
> - **Confidence:** E7, whether the score-noise band separates false from true positive episodes.
> - **Context:** E8, precision@3 of retrieved precedents, from a blind relevance annotation.
> - **Narrative faithfulness:** E9, supported claims over claims, checked by a rule-based checker independent of the template.
>
> E1–E7 and E9 are *functionally grounded* properties of the attribution. E8 measures *grounding* in external evidence. Decision value is measured separately (METRIC 8). Whether people understand or trust the explanations is out of scope (§Outlook).

## 6. §Performance Metrics → Decision Effectiveness [`Abschnitt_Metric_Decision`]

**This replaces the TODO at `4_0_Evaluation.tex:87-95`:**

> Decision effectiveness (METRIC 8) is measured on the synthetic test series (365 days) by a transparent rule that maps risk level, sustained-HIGH status, the leading exact driver and a historical analogue onto one of four actions (no action, monitor, reroute, escalate). It is compared with actions derived from the injected scenario windows. Accuracy, macro-F1 and per-class recall are reported. No participant study was run; the rubric is the rule set itself.

---

## 7. §Results → Explainability [`Abschnitt_Results_Explainability`]

### 7.1 Scorecard table (booktabs; with duration; episode-level; TP episodes 5 / 7 / 8)

```latex
\begin{table}[htbp]
  \centering
  \caption[Explainability scorecard]{METRIC 9 explainability scorecard per region: exact
    attribution vs the production SHAP surrogate (episode means; 95\% CIs in brackets where
    shown; TP episodes 5 / 7 / 8 for Hormuz / Bab el-Mandeb / Panama, none for Malacca).}
  \label{Tabelle_Scorecard}
  \small
  \begin{tabular}{llcccc}
    \toprule
    & & Hormuz & Bab el-Mandeb & Panama & Malacca \\
    \midrule
    E1 & Efficiency residual (exact) & 5.6e-17 & 5.6e-17 & 1.1e-16 & 8.3e-17 \\
       & R$^2$ on real test window (surrogate) & $-0.37$ & $-4.39$ & $-1.10$ & $-0.32$ \\
       & Local error $|f-g|$ (surrogate) & 0.23 & 0.23 & 0.04 & 0.19 \\
    E2 & Deletion AUC, exact $\downarrow$ & 0.15 [0.12, 0.18] & 0.12 [0.09, 0.16] & 0.06 [0.01, 0.09] & 0.17 [0.14, 0.19] \\
       & Deletion AUC, surrogate & 0.53 & 0.53 & 0.59 & 0.49 \\
       & Deletion AUC, refit surrogate & 0.20 & 0.34 & 0.13 & 0.23 \\
       & Deletion AUC, random & 0.50 & 0.50 & 0.50 & 0.50 \\
    E3 & k80 agents, exact / surrogate & 2.00 / 2.35 & 2.05 / 2.26 & 1.65 / 2.59 & 1.94 / 2.42 \\
    E4 & Rank stability $\rho$ ($\sigma=0.05$) & 0.97 & 0.97 & 0.86 & 0.95 \\
    E5 & Surrogate top agent = exact & 0.34 & 0.18 & 0.01 & 0.42 \\
    E6 & Feasible single-agent counterfactual & 0.97 & 0.97 & 1.00 & 0.97 \\
    E7 & AUC of noise band, FP vs TP & 0.38 [0.12, 0.70] & 0.87 [0.66, 1.00] & 0.57 [0.38, 0.73] & n/a \\
    E8 & Precision@3 (annotation) & \multicolumn{4}{c}{pending} \\
    E9 & Narrative faithfulness & 1.00 & 1.00 & 1.00 & 1.00 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### 7.2 Text

> The exact decomposition reproduces the composite to floating-point precision in every region (E1 ≤ 1.1e-16).
>
> **Deletion.** Removing agents in its order collapses the score departure fastest: deletion AUC 0.06–0.17, against 0.50 for a random order (E2, Figure F4). The deployed SHAP surrogate does no better than random (0.49–0.59). On the real test windows it predicts the composite worse than a constant (R² −0.32 to −4.39; Figure F5), although its training-set R² on its own synthetic year is 0.983. It names the same leading agent as the exact decomposition on 1–42% of alert days (E5).
>
> **Refitting the surrogate on real data** narrows the gap (deletion AUC 0.13–0.34; agreement 53–94%) but does not close it. The gap is therefore a property of approximating a known function, not only of the training-distribution mismatch.
>
> **Compactness and stability.** Exact explanations are compact: two agents typically cover 80% of the attribution (E3). They are stable under realistic agent-score noise (ρ ≥ 0.86 at σ = 0.05; E4, Figure F6), and 97–100% of alerts have a single-agent counterfactual that stays above calm levels (E6).
>
> **Narrative.** The deterministic narrative states only values in the record (E9 = 1.00). An independent claim checker rejects 96.9% of single-number corruptions, so a future LLM paraphrase can be admitted only when it checks out.

### 7.3 Diagnostic findings

These belong in this subsection: they are what the explanation exposed.

> **Retrieval.** The production retriever returned precedents from the wrong region in Hormuz, Panama and Malacca (region match 0.00), duplicated one article within the top 3 (duplicate rate 0.33), and was not triggered at all for Panama. A region- and date-filtered, de-duplicated wrapper removes all three defects (Figure F7). The 0.55 similarity floor is passed by every configuration, including the defective one, so similarity is not evidence of relevance; E8 is therefore the annotated precision@3.
>
> **Market saturation.** The market agent's calm-day median is 0.77–0.78, and its score is ≥ 0.95 on 29% of test days (clipped at 1.0 on 25%), calm days included. Measured from its calm level it contributes 12–16% of the attribution on alerts and leads no episode (Figure F15).
>
> **The Hormuz case.** Shipping's duration branch, the label's own statistic, wins on 82 of 133 labelled days. Yet shipping's exact contribution is no larger on labelled days (mean 0.044) than on unlabelled ones (0.052), and labelled days are alerted less often (16.5%) than unlabelled ones (26.6%). The decomposition shows *why* the composite does not track the label (Figure F13).
>
> **False-positive drivers.** Geopolitical leads 29 of 49 Malacca FP episodes. In Hormuz, shipping leads the FPs (21/34) while geopolitical leads 4 of 5 TP episodes (Figure F14), so the leading driver is not an FP marker.

### 7.4 Figure blocks

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/ch4/F2_alert_anatomy}
  \caption[Alert anatomy]{Exact agent attribution of one alert per region (Path D, real
    evaluation data). TP episodes: 5 / 7 / 8 (Hormuz / Bab el-Mandeb / Panama).}
  \label{Bild_F2_Alert_Anatomy}
\end{figure}
```

Use the same pattern for:

| Figure | Label | Width |
|---|---|---|
| F1 pipeline | `Bild_F1_Pipeline` | 0.95 |
| F3 attribution timeline | `Bild_F3_Timeline` | 1.0 |
| F4 deletion / insertion | `Bild_F4_Deletion` | 1.0 |
| F5 surrogate honesty | `Bild_F5_Surrogate` | 1.0 |
| F6 compactness / stability | `Bild_F6_Compactness` | 1.0 |
| F7 retrieval diagnostics | `Bild_F7_Retrieval` | 1.0 |
| F12 scorecard heatmap | `Bild_F12_Scorecard` | 1.0 |
| F13 Hormuz case | `Bild_F13_Hormuz` | 1.0 |
| F14 FP drivers | `Bild_F14_Drivers` | 0.9 |
| F15 market saturation | `Bild_F15_Market` | 1.0 |

Take each caption's first sentence from its `.caption.txt`, and keep the TP-episode line.

## 8. §Results → Decision Support [`Abschnitt_Results_Decision`]

### 8.1 Triage

**Framing:** *score alone ≥ explanation features; a region-specific agreement effect in Panama; its direction reverses in Hormuz* (Figures F8, F9).

> Whether the explanation helps decide when to trust an alert was tested with three pre-registered first-day indicators: the leading exact driver being geopolitical, the number of individually elevated agents, and agent dispersion. Each is scored as an AUC for separating false- from true-positive episodes.
>
> One result survived Holm correction. In Panama, false-positive episodes had fewer elevated agents (AUC 0.89 [0.74, 0.98], adjusted p = 0.0002), identically without the shipping duration branch. The same indicator pointed the other way in Hormuz (0.34 [0.18, 0.47]) and was uninformative in Bab el-Mandeb. The geopolitical-driver hypothesis held nowhere. Agent dispersion and the score-noise band did not separate the classes (Figure F9).
>
> A leave-one-region-out logistic model on the three indicators did worse than the alert score alone in every held-out region (ΔAUC −0.10 to −0.50; Hormuz and Panama CIs exclude zero).
>
> This mirrors human-subject findings that explanations need not improve on showing the model's confidence~\cite{Bansal.2021}, and that their effects on reliance are small and heterogeneous across contexts~\cite{Schemmer.2022,Haag.2026,Atf.2025}. Here the heterogeneity appears without any human in the loop.

**Knowledge-base check on false positives** (Figure F16):

> False-positive episode windows contain conflict or shipping-attack documents no more often than calm-day windows. The pre-registered rule gives a pooled lift of 1.07 [0.97, 1.15]; a stricter, post-hoc exploratory rule gives 1.07 [0.88, 1.27]. The false positives cannot be explained as "real risk without a traffic drop".

### 8.2 METRIC 8: canonical version (SYNTHETIC)

**Canonical METRIC 8 for Chapter 4 = P0-exact.** The rule's driver is the agent with the largest exact Shapley contribution to that day's composite.

```latex
\begin{table}[htbp]
  \centering
  \caption[Decision effectiveness (synthetic)]{METRIC 8 on the synthetic test series
    (seed 44, 365 days); correct actions are rule-derived from injected scenarios.
    SYNTHETIC.}
  \label{Tabelle_Metric8}
  \small
  \begin{tabular}{lcccc}
    \toprule
    Weight mode & Accuracy & Macro-F1 & Recall reroute & Naive baseline \\
    \midrule
    Hand-tuned & 0.937 & 0.708 & 0.17 & 0.797 \\
    Optimized  & 0.578 & 0.726 & 1.00 & 0.797 \\
    \bottomrule
  \end{tabular}
\end{table}
```

**Footnote** (stock driver rule):
> "With the evaluation notebook's original driver rule (largest weight × score), hand-tuned accuracy is 0.948 (macro-F1 0.855) and optimized 0.578 (0.726). The exact-driver rule is reported because it is the attribution the system shows its user; the two differ only on the 7 (hand-tuned) / 5 (optimized) high-risk, non-sustained days on which the rule consults the driver."

**Why the numbers differ from the July run** (stale: 0.866 / 0.490). That run predates two changes:
1. the removal of the routing agent (`b2306ae`; six agents became five, weights renormalised);
2. the shipping agent's sustained level-shift branch and absolute isolation-forest normalisation (`f7b87ff`).

Both change when risk crosses the HIGH band in the injected scenarios. Hand-tuned Scenario A accuracy changed from 0.07 to 0.73 between the runs. The July values must not be cited.

**The optimized weights fall below the naive baseline.** Optimized accuracy is 0.578 vs the naive 2σ baseline's 0.797. Their lower `risk_medium` (0.28) pushes 153 of 318 calm days into "monitor" (no-action recall 0.52). Optimizing detection did not optimize decisions; this joins the negative findings on the optimized weights.

**P0-surrogate vs P0-exact** (pre-registered; Figure F10):
- The two drivers differ on 295/365 days, but the rule consults the driver on only 7/5 days.
- McNemar p = 1.00 (hand-tuned) / 0.063 (optimized).
- The surrogate-driven version never recommends rerouting (reroute recall 0.00).

Report this as a defect found and fixed, not as a significant improvement.

**Not run:** P1 (uncertainty downgrade), P2 (precedent-gated escalation) and simulated reliance. Their inputs tested null or non-selective in Phases 2–4. Simulated reliance belongs under Outlook.

## 9. §Results → Ablation and Sensitivity Analysis [`Abschnitt_Results_Ablation`]

> **Circularity control.** Every scorecard row, triage indicator and episode count was recomputed with shipping reduced to its shock branch, with threshold, baselines and noise recalibrated on train. No conclusion changed:
> - scorecard values moved by at most a few hundredths;
> - Panama's agreement AUC was identical (0.89);
> - the leave-one-region-out model still lost to the score alone.
>
> **Surrogate strength.** Refitting the surrogate on real data improved it but left exact attribution ahead on every E2 row.

## 10. §Summary of Results [`Abschnitt_Results_Summary`]

> Exact attribution is faithful, compact, stable and auditable. The deployed surrogate was not.
>
> The explanation layer's value was diagnostic: it exposed an unfaithful explainer, a defective retriever, a saturated market agent, and a label that the detector's signals do not track.
>
> It did not provide a transferable trust signal: explanation features did not beat the score for triage, and false positives were not associated with contemporaneous conflict evidence.

---

## 11. SRQ3: replacement answer

**Superseded:** "Yes, with a stated surrogate caveat" (README SRQ table), resting on a training-set R² of 0.991 from a retired 20-feature model.

**README Verdict cell:**
> **Yes, exactly, at the agent level.** The composite is decomposed by exact Shapley values over the agent scores (efficiency residual ≤ 1.1e-16; METRIC 9 E1); feature-level attributions inside each agent are exact as well. The SHAP surrogate is kept only as a comparator: on real data it does not reproduce the score (R² −0.32 to −4.39) and agrees with the exact top driver on 1–42% of alert days.

**Thesis paragraph** (`1_0_Introduction.tex` §SRQ3 framing; `5_0_Discussion.tex` §SRQ3):
> Signals are translated into interpretable insight by decomposing each alert's risk score exactly. The composite is an explicit function of five agent scores, so its Shapley values~\cite{Shapley.1953} are computed over all coalitions rather than estimated, with each agent measured from its own calm-day level. They sum to the score up to floating-point error. Inside each agent the same decomposition attributes the score to its inputs, keeping the agent's non-linearity inside the game.
>
> A SHAP explanation of a random-forest surrogate~\cite{Lundberg.2017}, the approach the system originally used, was retained as a comparator. It fitted its own synthetic training year (R² 0.983) but not a held-out part of it (0.528), and on the real test windows it was worse than a constant. Its agent ranking removed score no faster than a random ordering, and it named the same leading agent on 1–42% of alert days.
>
> The finding is not that SHAP is uninformative. Where the scoring function is known, a post-hoc surrogate adds an approximation error that an exact decomposition does not have, large enough to change which driver a user is shown. The attributions remain associational: they state how the score was computed, not what caused a disruption~\cite{Miller.2019}.

---

## 12. Allowed and not-allowed claims

**Allowed:**
- "Exact agent attributions are faithful by construction (residual ≤ 1.1e-16); the production surrogate's out-of-time R² is 0.53 and its R² on the real test windows is negative."
- "Exact attribution removes the score departure fastest (deletion AUC 0.06–0.17 vs 0.50 random); the surrogate is at chance (0.49–0.59)."
- "The explanation layer exposed four system defects: an unfaithful explainer, a defective retriever, a saturated market agent, and a masking label."
- "Explanation features did not improve on the score as a triage signal (leave-one-region-out); one region-specific effect (Panama agreement) survived correction, and it reverses in Hormuz."
- "False-positive alerts were not associated with more conflict evidence than calm days (lift 1.07 [0.97, 1.15])."
- "On the synthetic series, the surrogate-driven decision badge never recommended rerouting; the exact-driver version did (optimized reroute recall 1.00)." Always label as SYNTHETIC.
- "The deterministic narrative is fully faithful to its record; the claim checker rejects 96.9% of single-number corruptions."

**Not allowed:**
- "Explanations increase user trust", "help users calibrate reliance", or "improve decisions" (there is no human study).
- "Explanations are faithful" without saying which explanation.
- "Geopolitical-led alerts are false alarms."
- "The false positives were real risk without a traffic drop."
- "The surrogate is wrong." It is an approximation of a known function; its error is what was measured.
- Any METRIC 8 claim without the SYNTHETIC qualifier, or citing the July values (0.866 / 0.490).
- "Uncertainty / the noise band indicates alert reliability" (E7 CIs span 0.5; F9 null).
- Any precision@3 or retrieval-relevance claim before the annotation is complete.

## 13. Limitations: agreement cut-off quirks (Phase 4, P2)

The agreement indicator counts agents above their own train calm-day 90th percentile. As pre-registered:
1. **Market can never count.** Its calm q0.90 is exactly 1.0, its ceiling.
2. **Any non-zero natural-disaster score counts.** That agent's calm q0.90 is 0.

The indicator is therefore effectively a count over shipping, geopolitical and news, plus any disaster activity. No alternative cut-off was tried, to avoid post-hoc tuning.

## 14. Outlook

- **A human-subject study** of understanding and appropriate reliance on the exact explanations: the RAIR/RSR framework~\cite{Schemmer.2023}, cognitive forcing~\cite{Bucinca.2021}, and the Hoffman et al. measures~\cite{Hoffman.2023,Hoffman.2023b}. Trust in automation~\cite{LeeSee.2004} cannot be inferred from functionally grounded metrics~\cite{DoshiVelez.2017}.
- **Simulated reliance** under assumed operator accuracies was dropped; it would only illustrate a mechanism.
- **A second annotator for E8**, to report inter-rater agreement; RAG-specific metrics~\cite{Es.2024} could complement precision@3.
- **Region-specific triage rules.** The Panama agreement effect needs more TP episodes before it could be used.

**Supply-chain AI context for the introduction to the chapter:** Baryannis et al.~\cite{Baryannis.2019}.

**Bibliography:** `thesis/references_xai.bib` has 15 new entries, all with Crossref-verified DOIs. The one exception is Petsiuk 2018 (BMVC, no DOI; arXiv ID verified). DoshiVelez.2017, Bansal.2021 and Lundberg.2017 already exist in `Verzeichnis_Literatur.bib`.
