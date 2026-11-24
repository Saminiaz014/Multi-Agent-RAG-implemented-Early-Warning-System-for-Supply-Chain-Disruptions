"""The README and the two reference docs point at the results instead of restating them."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_ROOT / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")


README = _read("README.md")


def _section(start: str, end: str) -> str:
    return README[README.index(start):README.index(end)]


# ------------------------------------------------------------- headline table
def test_the_headline_table_points_at_the_results_and_types_no_result() -> None:
    head = README[:README.index("\n---\n")]
    assert "eval/results_main.md" in head and "eval/RESULTS_FROZEN.md" in head
    for stale in ("410 passed", "0.909", "0.502", "(chance)", "reduces** AUC in every"):
        assert stale not in head, stale
    assert not re.search(r"\b\d\.\d{3}\b", head), "a measured value is typed into the headline"


# ------------------------------------------------------------- sections 4.6-4.10
_SECTIONS = _section("## 4.6 The central negative finding", "## 4.11 Known defects")


def test_sections_4_6_to_4_10_state_no_measured_value() -> None:
    """No decimal and no percentage: the one decimals are the optimizer objective's weights."""
    body = "\n".join(ln for ln in _SECTIONS.splitlines()
                     if not ln.startswith(("**Objective:**", "## ", "### ")))
    body = re.sub(r"\]\(#[^)]*\)", "]", body)                  # link targets
    body = re.sub(r"§\s*\d+(?:\.\d+)*", "", body)               # section references
    body = re.sub(r"[Ss]ections? \d+(?:\.\d+)*", "", body)
    body = re.sub(r"`[^`]*·[^`]*`", "", body)                   # the objective's own weights (terms)
    found = re.findall(r"\b\d+\.\d+\b|\b\d+(?:\.\d+)? ?%", body)
    assert not found, found
    for stale in ("2025-08-30", "731 days", "57 completed", "trial 63", "0.7491", "38 %"):
        assert stale not in _SECTIONS.replace("test window 2023-08-31 to 2025-08-30", ""), stale


@pytest.mark.parametrize("start, end", [
    ("## 4.6 ", "## 4.7 "), ("## 4.7 ", "## 4.8 "), ("## 4.10 ", "## 4.11 ")])
def test_the_findings_sections_point_at_the_results_table(start: str, end: str) -> None:
    assert "results_main.md" in _section(start, end)


def test_section_4_5_says_the_later_sections_restate_no_numbers() -> None:
    four_five = _section("## 4.5 Results", "## 4.6 ")
    assert "without restating numbers" in four_five and "RESULTS_FROZEN.md" in four_five
    assert "come from earlier runs" not in four_five


def test_the_linked_result_files_exist() -> None:
    for rel in ("eval/results_main.md", "eval/results_main.csv", "eval/RESULTS_FROZEN.md",
                "eval/COMPARISON_REPORT.md", "scripts/calibrate_risk_bands.py"):
        assert (_ROOT / rel).exists(), rel


# ------------------------------------------------------------- the band claim
@pytest.mark.parametrize("rel", ["README.md", "docs/SCORING_REFERENCE.md", "docs/THESIS_BRIEF.md"])
def test_the_bands_are_display_thresholds_with_provenance_not_calibrated(rel: str) -> None:
    text = _read(rel)
    flat = " ".join(text.split())
    assert "fixed display thresholds, not calibrated" in flat.replace("**", "")
    for gone in ("empirically calibrated", "9,183", "calibrated, not chosen", "calibrated, not picked"):
        assert gone not in flat, gone
    assert "f7b87ff" in text and "calibrate_risk_bands.py" in text


# ------------------------------------------------------------- internal links
def _slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    return re.sub(r"[^\w\- ]", "", heading.lower().replace("`", "")).replace(" ", "-")


def test_every_internal_readme_link_has_a_heading() -> None:
    body = re.sub(r"```.*?```", "", README, flags=re.S)           # headings inside code fences
    anchors = {_slug(m.group(2)) for m in re.finditer(r"^(#{1,6}) (.+)$", body, flags=re.M)}
    links = set(re.findall(r"\]\(#([^)]+)\)", README))
    assert not links - anchors, sorted(links - anchors)


# ------------------------------------------------- review fixes: claims that must stay true
_BAND_DOCS = ("README.md", "docs/SCORING_REFERENCE.md", "docs/THESIS_BRIEF.md")


@pytest.mark.parametrize("rel", _BAND_DOCS)
def test_the_band_claim_is_scoped_to_the_real_data_evaluation(rel: str) -> None:
    flat = " ".join(_read(rel).split()).lower()
    assert "do not enter the evaluation" not in flat          # too broad: the optimizer reads them
    assert "do not enter the real-data" in flat
    assert "synthetic weight optimizer" in flat and "risk_high" in flat and "risk_medium" in flat


def test_no_other_readme_line_calls_the_bands_calibrated_or_quotes_their_old_percentiles() -> None:
    flat = " ".join(README.split())
    for gone in ("calibrated risk bands", "p97 of calm", "p85 of calm", "p60 of calm",
                 "recorded only in the commit message"):
        assert gone not in flat, gone
    assert "fixed display threshold" in flat


def test_the_readme_no_longer_says_the_test_suite_overwrites_the_plots() -> None:
    flat = " ".join(README.split())
    assert "silently overwrites" not in flat
    assert "Overwritten by the *test suite* on every run" not in flat
    assert "resolved in the freeze commit" in flat and "Do not cite" in flat


def test_the_dashboard_labels_the_shap_comparison_images_as_legacy() -> None:
    source = _read("src/dashboard/analysis_view.py")
    assert source.count("legacy fixture plot") == 2
    assert "generated by `generate_comparison_plot()` (tests/test_phase4_depth.py)" not in source
    assert "no script regenerates them now" in source


def test_section_4_9_says_where_the_study_numbers_live_in_the_tag() -> None:
    section = _section("## 4.9 ", "## 4.10 ")
    assert "not in the tag" in section
    for rel in ("docs/SCORING_REFERENCE.md", "docs/THESIS_BRIEF.md"):
        assert rel in section
    assert "Best run on record" in _read("docs/SCORING_REFERENCE.md")


# ------------------------------------------------- commit 17: the stale hand-typed numbers
BRIEF = _read("docs/THESIS_BRIEF.md")

_MEASURED = re.compile(
    r"\b\d+\.\d+\b"                                   # a decimal
    r"|\b\d+(?:\.\d+)? ?%"                            # a percentage
    r"|\b\d{4}-\d{2}-\d{2}\b"                         # a date
    r"|\b\d[\d,]* (?:passed|commits|tracked|source|test modules|modules|positives|rows|days)\b")


def _measured(text: str) -> list[str]:
    """Typed measurements in ``text``: decimals, percentages, dates and counts.

    Code fences (the label's definition), link targets, section references and
    heading numbers are not measurements.
    """
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"GDELT \d\.\d", "GDELT", text)                # a product version
    text = re.sub(r"\bv\d+(?:\.\d+)*-thesis\b", "TAG", text)     # a git tag's name
    text = re.sub(r"\]\(#[^)]*\)", "]", text)
    text = re.sub(r"§\s*\d+(?:\.\d+)*|[Ss]ections? \d+(?:\.\d+)*", "", text)
    text = re.sub(r"(?m)^#+ \d+(?:\.\d+)* ", "", text)
    return _MEASURED.findall(text)


def _brief(start: str, end: str) -> str:
    return BRIEF[BRIEF.index(start):BRIEF.index(end)]


def test_the_measurement_finder_finds_what_it_should() -> None:
    assert _measured("AUC 0.909, 6.53 %, 38%, 2025-08-30, 420 passed, 159 positives")
    assert not _measured("see §4.7 and Section 3.2, [x](#47-root) ```\n0.5 20%\n```")
    assert not _measured("## 4.2 Dataset\nSeeds 42 / 43 / 44")


def test_readme_dataset_section_types_no_dataset_figure() -> None:
    section = _section("## 4.2 Dataset and ground truth", "## 4.3 ")
    synthetic = section[section.index("### Synthetic splits"):]
    assert not _measured(section.replace(synthetic, "")), _measured(section.replace(synthetic, ""))
    for stale in ("2025-08-30", "missing from three", "did not answer", "cut transits"):
        assert stale not in section, stale
    assert "RESULTS_FROZEN.md" in section and "manifest.json" in section
    assert "evaluation.end_date" in section


def test_readme_research_question_verdicts_type_no_number() -> None:
    table = _section("## 2. Research Questions", "\n---\n\n# 3. Methodology")
    assert not _measured(table), _measured(table)
    for stale in ("0.909", "0.502", "1.67", "0.039", "0.0031"):
        assert stale not in table, stale
    assert "RESULTS_FROZEN.md" in table


def test_readme_section_3_5_1_types_no_old_shock_measurement() -> None:
    section = _section("### 3.5.1 ShippingAgent", "### 3.5.2 MarketAgent")
    for stale in ("−0.06", "−0.05", "0.55 on calm", "0.44 on disrupted", "no signal at all"):
        assert stale not in section, stale
    flat = " ".join(section.split())
    assert "not re-measured on the fixed split" in flat
    assert "forced the risk-band recalibration" not in flat


def test_readme_test_suite_table_types_no_count() -> None:
    section = _section("### The suite", "**What the suite does not verify")
    assert not re.search(r"\|\s*\d+\s*\|", section), "a test count is typed in the table"
    assert "410" not in section and "23 modules" not in section
    assert "pytest --collect-only" in section


def test_section_4_5_does_not_type_the_results_row_count() -> None:
    assert "76 rows" not in _section("## 4.5 Results", "## 4.6 ")


_BRIEF_PARTS = [
    ("the header", "# Thesis Writing Brief", "# PART I "),
    ("the data section", "## 4.1 Real evaluation dataset", "## 4.3 Synthetic splits"),
    ("the results", "# PART V — RESULTS", "## 5.5 Weight optimization"),
    ("what you cannot claim and the narrative", "# PART VIII", "# APPENDIX"),
]


@pytest.mark.parametrize("name, start, end", _BRIEF_PARTS)
def test_the_brief_types_no_measured_value_in(name: str, start: str, end: str) -> None:
    part = _brief(start, end)
    assert not _measured(part), (name, _measured(part))
    for stale in ("0.909", "0.502", "0.968", "159", "244", "2025-08-30"):
        assert stale not in part, (name, stale)


def test_the_brief_names_where_the_numbers_are() -> None:
    head = _brief("# Thesis Writing Brief", "# PART I ")
    for needle in ("eval/RESULTS_FROZEN.md", "eval/results_main.md", "v1.0.1-thesis"):
        assert needle in head, needle
    assert "Every number in it was measured" not in head
    results = _brief("# PART V — RESULTS", "## 5.5 Weight optimization")
    assert "results_main.md" in results and "RESULTS_FROZEN.md" in results


def test_the_brief_dataset_section_describes_the_current_frame() -> None:
    part = _brief("## 4.1 Real evaluation dataset", "## 4.3 Synthetic splits")
    flat = " ".join(part.split())
    assert "missing from three of four" not in flat
    assert "cut transits" not in flat
    assert "evaluation.end_date" in flat


def test_the_readme_does_not_make_the_brief_the_authority_on_results() -> None:
    flat = " ".join(README.split())
    assert "the brief wins" not in flat and "**Source of record**" not in flat
    assert "states no measured value" in flat
    assert "which win over any sentence in a doc" in flat


# ------------------------------------------------- commit 17, after review
def test_no_doc_says_the_label_misses_panamas_drought() -> None:
    for rel in ("README.md", "docs/SCORING_REFERENCE.md", "docs/THESIS_BRIEF.md",
                "scripts/build_eval_dataset.py"):
        flat = " ".join(_read(rel).split())
        for stale in ("flags almost none", "almost entirely unlabelled", "cut transits by"):
            assert stale not in flat, (rel, stale)


def test_the_readme_describes_the_news_source_the_evaluation_reads() -> None:
    flat = " ".join(README.split())
    for stale in ("GDELT DOC API v2", "Answered for Panama only", "answered for only one of four",
                  "ends 2025-08-30", "diagnosed mechanism", "root cause of the central negative",
                  "did fail on two of four"):
        assert stale not in flat, stale
    assert "GDELT 1.0 daily event export" in flat and "build_gdelt_news_csv.py" in flat


def test_the_readme_types_no_module_or_test_count() -> None:
    flat = " ".join(README.split())
    for stale in ("410 tests", "410 passing", "48 source modules", "24 test modules", "4m45s"):
        assert stale not in flat, stale


def test_the_usgs_pointer_names_the_code_default_not_the_config() -> None:
    for rel in ("README.md", "docs/THESIS_BRIEF.md"):
        flat = " ".join(_read(rel).split())
        assert "USGSExtractor.min_magnitude" in flat, rel
        assert "magnitude threshold in `config/settings.yaml`" not in flat, rel


def test_the_evaluation_window_not_the_frame_ends_at_end_date() -> None:
    for rel in ("README.md", "docs/THESIS_BRIEF.md"):
        flat = " ".join(_read(rel).split())
        assert "The frame ends at `evaluation.end_date`" not in flat, rel
        assert "frame may run a few days past it" in flat, rel
        assert "ends at `evaluation.end_date`" in flat or "ends at `evaluation.end_date`" in flat.replace("**", ""), rel


def test_the_circularity_pointer_goes_to_the_section_that_has_the_ratings() -> None:
    section = _section("## 4.2 Dataset and ground truth", "## 4.3 ")
    assert "(#43-baselines-tiers-and-circularity)" not in section
    assert "(#circularity-rating--the-decision-that-makes-the-evaluation-credible)" in section


def test_the_brief_scopes_its_no_number_claim_to_the_real_data_evaluation() -> None:
    head = _brief("# Thesis Writing Brief", "# PART I ")
    flat = " ".join(head.split())
    assert "no measured value of the real-data evaluation" in flat
    for part in ("§5.5", "Part VI", "Part VII"):
        assert part in flat, part
    assert "§5.1 to §5.4 state no number" in BRIEF
    assert "This part states no number" not in BRIEF
    # the README says it twice (the document table and the reference table)
    assert " ".join(README.split()).count("no measured value of the real-data evaluation") == 2


def test_the_brief_does_not_claim_a_generation_time_the_figures_do_not_have() -> None:
    assert "Generated 2026-08-30 23:24" not in BRIEF
    assert "eval/FIGURES.md" in _brief("# PART VI", "# PART VII")
