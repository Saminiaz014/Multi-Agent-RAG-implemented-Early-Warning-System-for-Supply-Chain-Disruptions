# Writing changes (thesis and paper)

Branch `docs/writing-fixes`, from `v1.0.1-thesis`. Every change in `thesis/` and `Paper/` is red in the
text: `\add{}` new, `\del{}` removed (struck through), `\todo{}` open, `addblock` for a new passage that
spans paragraphs. `\markchangesfalse` is not set. `thesis/` and `Paper/` are git-ignored on purpose, so
the `.tex` edits are not in any commit; a backup of both folders from before the edits is in
`../thesis_backup_2026-11-24/`. Numbers come from `eval/RESULTS_FROZEN.md`, the committed CSVs,
`eval/case_metrics.json` and `eval/FIGURES_ANALYSIS.md`; `scripts/check_added_numbers.py` checks every
number and date inside new text against them.

Issues closed in writing: 1, 2, 6, 7, 8, 10, 12, 13, 16, 17, 18. The thesis has no abstract file, so the
abstract change (issues 1, 2) is in the paper only.

| file | section | issue # | type | summary |
|---|---|---|---|---|
| thesis/Makros.tex, Packages.tex | preamble | n/a | add | red-mark macros `\add`, `\del`, `\todo`, environment `addblock` (an `\add` cannot span paragraphs), `ulem` with `normalem` |
| thesis/1_0_Introduction.tex | Research objective and questions | RQ | add, del | primary research question replaced by the amended text (`docs/rq_amendments.md`) |
| thesis/1_0_Introduction.tex | SRQ2 | RQ, 1, 7 | add, del | SRQ2 reworded; the answer now states 8 of 9 distinct tier comparisons clearly below Tier 1, none above, against a traffic-derived label |
| thesis/1_0_Introduction.tex | SRQ3 | RQ, 13 | add, del | SRQ3 reworded to surrogate fidelity per region; the failure in one region is announced |
| thesis/1_0_Introduction.tex | SRQ5 | RQ, 10 | add, del | SRQ5 reworded to how label and protocol decide the measured value of fusion; an editorial remark removed |
| thesis/2_0_Literature_Review.tex | Data sources and methods used in the evaluation (new section) | 18 | add | related work: ACLED and GDELT, CUSUM, EWMA and matrix profile, SHAP, and the manufacturing link |
| thesis/2_0_Literature_Review.tex | Maritime chokepoints | 8 | add, todo | IMF 74 % is "above the previous year's level"; the ITF Cape detour in days could not be retrieved (HTTP 403), so `\todo{verify citation}` |
| thesis/2_0_Literature_Review.tex | Feature attribution and SHAP | 8 | add, todo | tree variant cited to Lundberg et al. 2020; page range of Lundberg and Lee 2017 (4765-4774) taken from the review, `\todo{verify citation}` |
| thesis/Verzeichnis_Literatur.bib, references_xai.bib | all entries | 8, 18 | bib (not red-markable) | DOIs added for 15 entries (Crossref or DataCite), DOI coverage 49 of 90 to 70 of 97 entries (6 of the 7 new entries have one); Jahin et al. moved to the journal version; EIA "last updated" date; 7 new entries (Lundberg 2020, Dietterich 2000, Raleigh 2010, GDELT, Ivanov and Dolgui 2020, Al-Talib et al. 2020, Mediavilla et al. 2022) |
| thesis/3_0_Methodology.tex | Layer 1: interfaces table, news connector | 10, 12 | add, del | news source is the GDELT 1.0 daily event export, not the DOC API |
| thesis/3_0_Methodology.tex | Layer 1: market connector | 12 | add, todo | one global FRED series, first-release dating, median lags 6, 42 and 72 days; series IDs differ from the evaluation manifest (`\todo`) |
| thesis/3_0_Methodology.tex | Layer 1: validation, derived quantities | 10, 12 | add | the two-day gap fill belongs to the ingestion path; the evaluation uses the dataset builder; average delay is computed from `vessel_count` |
| thesis/3_0_Methodology.tex | Layer 2 introduction, natural disaster agent | 10 | add | validation gates never run in the evaluation; distance decay acts only in the synthetic scenarios |
| thesis/3_0_Methodology.tex | Implementation: weight determination | 10 | add | Optuna runs on synthetic data and its weights are not used; the evaluation uses the hand-set weights |
| thesis/3_0_Methodology.tex | Layer 4: method | n/a | add, del | "first 70 %" replaced by the fixed split date 2023-07-01 |
| thesis/3_0_Methodology.tex | Implementation: verification | n/a | todo | module and test counts predate the post-freeze changes (`\todo{number?}`) |
| thesis/3_0_Methodology.tex | Data basis and ground truth; News data; Split, history and agent flags | 1, 10 | add, del | window from 2019-01-01 to 2025-11-21 (ACLED limit); bands are fixed display thresholds; GDELT filter, volume per 100,000 events, 20 masked days; fixed split; agents scored with full history; flags set 18 August 2026 |
| thesis/3_0_Methodology.tex | Choices Made After Results Were Seen (new) | 5 | add | Eq. 1, split date, news normalisation and gap rule were fixed after earlier results; the Eq. 1 ablation and the split sensitivity test two of them |
| thesis/3_5_Demonstration_Cases.tex | intro, cases 1 to 4, cross-case table | 17 | add, del, fig, table | all numbers recomputed from `eval/case_metrics.json`; screenshots recaptured and three repointed (`decision_case1_2024-10-02`, `decision_case3_2023-12-17`, `decision_case3_2024-01-19`); captions `[UPDATED]`; "k of n agents" counted as the page counts them (case 3: three of four); shipping data ends 2026-10-25 stated once |
| thesis/4_0_Evaluation.tex | Evaluation objectives, experimental setup, explainability metric | 13, 10 | add, del | split and window, alert counts, no pre-registered 0.60 bar |
| thesis/4_0_Evaluation.tex | Results: accuracy, detection table, dilution | 2, 7 | table, add, del | table regenerated from `eval/results_main.csv` with intervals and a Tier 5 - Tier 1 row; persistence and CUSUM described as strong, the shipping agent's score as weak; lead-agent recommendation dropped |
| thesis/4_0_Evaluation.tex | Figures R1 to R4 (new) | 1, 7, 10 | fig | paired change against Tier 1, Eq. 1 ablation, agent contribution, split sensitivity, each `[NEW]` |
| thesis/4_0_Evaluation.tex | D1 caption | 16 | fig | `[UPDATED]` (regenerated by `make_all_figures.py`) |
| thesis/4_0_Evaluation.tex | Results: timeliness | 6 | add, del, fig | every method leads every onset; the leads reflect the lagged label; `D2_lead_time_all_tiers` replaces `D2_lead_time_tier5_3x1` (figure audit) |
| thesis/4_0_Evaluation.tex | Results: explainability, surrogate table | 13 | table, add, del | held-out R-squared per region; Bab el-Mandeb negative R-squared in one sentence, no figure; top-3 agreement and the examples have no frozen source (`\todo{number?}`) |
| thesis/4_0_Evaluation.tex | Summary of results table | 2, 6, 13 | add, del | verdicts for E-A, E-T and E-X follow the new results |
| thesis/5_0_Discussion.tex | Interpretation of findings (SRQ1 to SRQ5) | 1, 2, 6, 7, 13 | add, del | numbers updated; ensemble claim cited to Dietterich 2000; timeliness paragraph rewritten; the answer to the main question restated |
| thesis/5_0_Discussion.tex | Why Fusion Does Not Help Here (new) | 1, 7 | add | no-signal agents, the anti-informative geopolitical agent, fixed weights; circularity (persistence, Eq. 1 ablation); Panama as the clean case; Malacca false alarms |
| thesis/5_0_Discussion.tex | Limitations, implications | 1, 6, 13 | add, del | split, labelling, missing-domain and false-alarm statements updated |
| thesis/6_0_Conclusion_and_Outlook.tex | Summary, answers, contributions | 1, 2, 6, 13 | add, del | each amended SRQ answered in one or two sentences; test count `\todo{number?}` |
| thesis/6_0_Conclusion_and_Outlook.tex | Outlook | 18 | add, del | limitations a to i that cannot be fixed now (news coverage excluded, learned fusion and lagged-feature surrogates included) |
| thesis/A_Appendix.tex | metric list, coverage table | 13, 10 | add, del | reference values, not a pre-registered bar; GDELT covers all four corridors; window end ; the old AUC table is flagged as predating the frozen results (`	odo`) |
| thesis/1_0, 3_0, 4_0, 5_0, 6_0, A_Appendix.tex | various | ARS review | add, del | factual fixes after the methodology review: alerts at or above the training 90th percentile; the distinct-comparison rule names Panama Tier 4; no pooled surrogate; the reference values are met only in Panama; lead-time statements limited to the tiers and the random control; fixed nominal weights, not fixed shares; Panama "least affected"; the percentile sentence on the risk bands removed |
| Paper/procir_cirpcms2027_sami.tex | various | ARS review | add, del | the same fixes where the paper says the same thing; news "interval entirely above chance" instead of "above chance" |
| Paper/procir_cirpcms2027_sami.tex | preamble | n/a | add | red-mark macros |
| Paper/procir_cirpcms2027_sami.tex | Abstract | 1, 2 | add, del | new message and frozen numbers; "best monitored through evidence from several channels" removed |
| Paper/procir_cirpcms2027_sami.tex | Introduction | 1, 8, 18 | add, del, todo | ITF `\todo{verify citation}`; IMF "above last year's level"; the unsupported 3 % clause removed; main finding restated |
| Paper/procir_cirpcms2027_sami.tex | Related work (new) | 18 | add | data sources, detectors, SHAP, manufacturing link |
| Paper/procir_cirpcms2027_sami.tex | Methodology | 10, 12 | add, del | gates and decay off the evaluation path; one global market series with lags; GDELT described; bands fixed display thresholds; Optuna weights unused; split date and window; knowledge base and dashboard text cut |
| Paper/procir_cirpcms2027_sami.tex | Results: detection, table, figure | 1, 2, 7 | table, fig, add, del | results table with intervals (`[NEW]`), R1 as the main figure (`[NEW]`), old Fig. 2 removed |
| Paper/procir_cirpcms2027_sami.tex | Results: timeliness, figure | 6 | del, fig | subsection and figure removed, red note left |
| Paper/procir_cirpcms2027_sami.tex | Results: explainability example | 13 | add, del | R-squared per region without a "pre-set bar"; Fig. 4 flagged as not regenerated |
| Paper/procir_cirpcms2027_sami.tex | Discussion | 1, 7 | add, del | achievements rewritten; threats to validity added (circularity, post-hoc choices, few onsets, source limits) |
| Paper/procir_cirpcms2027_sami.tex | References | 8, 18 | add, del | DOIs, Jahin journal version, Lundberg pages with `\todo`, Dietterich in place of Kuncheva, 8 new entries |

## Decisions taken without asking

- The number check (`scripts/check_added_numbers.py`, 533 numbers and dates in new text, none unmatched) is necessary, not sufficient: it catches a figure that no source contains, but a wrong number that equals another source value passes, and whole numbers from 0 to 100 always pass. The methodology review compared about 190 numbers by hand and found no transcription error.
- `\add` and `\del` cannot span paragraphs (their arguments are not long), so a longer new passage uses an `addblock` environment, defined beside the macros.
- Table cells are marked cell by cell with `\add{}`, and the old rows are kept as `% [OLD]` comments, because `\sout` must not wrap an environment or a row.
- A footnote cannot sit inside `\del{}`; the one old footnote that was deleted appears as bracketed struck text.
- The "R-squared 0.991" claim is not in the thesis or the paper text, so there was nothing to remove there.
- The distance decay, validation gates, Optuna weights and two-day gap fill are marked as not on the evaluation path (checked in `src/ingestion/disaster_connector.py`, `scripts/build_eval_dataset.py` and the harness).
- The "k of n agents" in the screenshots counts every region agent with a score on the assessed day, not only the route's agents (the first recount of mine used the route and was wrong; checked against the pictures of cases 1, 3 and 4).
- The paper's Fig. 4 (a dashboard screenshot) was not regenerated; it carries a red `\todo`.
- The paper builds to 6 pages with marks off and 7 with marks on (the struck text takes space), so the page limit is met by the final text, not the red text.
- latexdiff could not be run (see the report); the red-marked PDFs are the redline.

## ARS methodology review (academic-paper-reviewer, methodology-focus protocol)

24 findings; about 190 numbers in the new text were checked against the sources and all matched at printed precision. The errors were in interpretation, counting rules and stale or contradicting text.

Fixed in red (factual errors): the surrogate verdict (the 0.60 and 0.70 reference values are met only in Panama; Hormuz and Malacca meet the Spearman value but not R-squared); alerts at or above the training 90th percentile; the distinct-comparison rule (Panama Tier 4 repeats Tier 3 because its agent is constant); no pooled surrogate in the frozen results; fidelity measured on the test window, not "the last 30 %"; the never-alert accuracy statement; "every method" leading every onset narrowed to the tiers and the random control (the lead-time file holds no baselines); "the same share" narrowed to the same nominal weight; the 60th/85th/97th percentile sentence for the bands removed; the stale Appendix AUC table flagged; news "above chance" narrowed to "interval entirely above chance".

Reported, not changed (interpretation, left to the author): the cause of the Bab el-Mandeb failure is stated as history-dependence, an untested hypothesis (the diagnostic shows a target-mean shift and out-of-range inputs on 20.3 % of days); "clearly below" means the day-level block-bootstrap interval excludes zero, and Bab el-Mandeb and Panama each have a single labelled run; with Eq. 1 off the AUC rises from Tier 1 to Tier 4 in Hormuz, so the fall with added agents there depends on the circular term; "every evaluable corridor AUC falls as agents are added" and "the mechanism is dilution" are stronger than the leave-one-out cells (3 of 13 raise AUC clearly on removal); the shipping score's "max() masking" is not isolated by any frozen run; "Panama is the clean case" had to be softened because market hurts there too; persistence is built from the label's own statistic, so its AUC is near-tautological; "does not outperform any of them" (Discussion, SRQ5) is too strong against the weaker baselines; Hevner et al. 2004 is cited for the relevance and rigour cycles, which are Hevner 2007; the "April-May 2026" closure against the 2026-03-07 break in the case data.

Checked after the review: the Jahin et al. journal version (SN Business & Economics 6(11), 440, 2026, DOI 10.1007/s43546-026-01436-7) is on Crossref and the arXiv record names it, so the reviewer's doubt about its venue is resolved; the ITF figures and the Lundberg and Lee page range stay unverified and carry `	odo{verify citation}`.

## Open items

- Open `	odo`: thesis 8 (ITF days, Lundberg pages, market series IDs, test counts twice, the top-3 agreement and the explanation examples of the earlier run, the Appendix AUC table), paper 3 (ITF days, Lundberg pages, Fig. 4).
- Unverified citations: ITF 2024 (Cape detour in days, the 55 % and per-vessel declines; HTTP 403), Lundberg and Lee 2017 page range (taken from the internal review; the publisher page shows none).
- latexdiff was not run: the machine lacks the Perl module `Algorithm::Diff` and loading a downloaded copy was refused by the permission check. The red PDFs are the redline.
- `Paper_Issues_Sami.docx` items 3, 4, 5, 9, 11, 14 and 15 are rerun and harness items closed by the earlier code work, not by writing.
