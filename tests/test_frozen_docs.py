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
