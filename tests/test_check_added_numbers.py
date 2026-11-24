"""scripts/check_added_numbers.py: every number inside new (red) text traces to a committed source."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_added_numbers", _ROOT / "scripts" / "check_added_numbers.py")
can = importlib.util.module_from_spec(_spec)
sys.modules["check_added_numbers"] = can
_spec.loader.exec_module(can)


# ------------------------------------------------------------- what counts as new text
def test_added_text_takes_add_and_addblock_but_not_del() -> None:
    tex = r"old \del{gone 0.111} and \add{new 0.222} then \begin{addblock}block 0.333\end{addblock} end"
    pieces = can.added_text(tex)
    assert [p for p, _ in pieces] == ["new 0.222", "block 0.333"]


def test_added_text_follows_nested_braces_and_commands() -> None:
    tex = r"\add{a \textbf{0.5} b \emph{c}} tail 0.9"
    assert can.added_text(tex)[0][0] == r"a \textbf{0.5} b \emph{c}"


def test_added_text_reports_the_line_it_starts_on() -> None:
    tex = "line one\nline two \\add{x 1.5}\n"
    assert can.added_text(tex)[0][1] == 2


# ------------------------------------------------------------------ number tokens
def test_numbers_reads_decimals_percentages_and_thousands() -> None:
    text = r"AUC 0.574 [0.516, 0.633], 57.1\,\%, 1{,}344 alerts, $-0.021$, $+0.005$, 857 days"
    assert can.numbers(text) == ["0.574", "0.516", "0.633", "57.1", "1344", "0.021", "0.005", "857"]


def test_numbers_ignores_latex_plumbing_and_references() -> None:
    text = (r"\includegraphics[width=0.62\textwidth,height=0.7\textheight]{figures/R1}"
            r" \label{Bild_R1} \ref{Abschnitt_3} \cite{Lundberg.2020} \footnote{x}")
    assert can.numbers(text) == []


def test_numbers_ignores_years_and_iso_dates_which_are_checked_separately() -> None:
    text = "from 2019-01-01 to 2025-11-21, in 2024, and 0.5"
    assert can.numbers(text) == ["0.5"]
    assert can.dates(text) == ["2019-01-01", "2025-11-21"]


# ---------------------------------------------------------------- allowed numbers
def test_allowed_numbers_come_from_every_source() -> None:
    frame = pd.DataFrame({"auc": [0.57412], "n": [857]})
    allowed = can.allowed_numbers(frames=[frame], texts=["R2 is -1.376 here, 57.1% share"], objects=[{"x": [0.8123, 3]}])
    assert {"0.574", "0.57", "857", "1.376", "57.1", "0.812", "3"} <= allowed


def test_a_percentage_is_allowed_when_its_fraction_is_in_a_source() -> None:
    allowed = can.allowed_numbers(frames=[pd.DataFrame({"share": [0.5707]})], texts=[], objects=[])
    assert "57.1" in allowed and "57" in allowed


def test_a_signed_value_is_allowed_by_its_magnitude() -> None:
    allowed = can.allowed_numbers(frames=[pd.DataFrame({"d": [-0.255]})], texts=[], objects=[])
    assert "0.255" in allowed


# ------------------------------------------------------------------------ check
def test_a_typed_wrong_number_is_flagged_and_a_sourced_one_passes() -> None:
    allowed = can.allowed_numbers(frames=[pd.DataFrame({"auc": [0.574]})], texts=[], objects=[])
    bad = can.check(r"Tier 1 reaches \add{0.575}", allowed, set())
    good = can.check(r"Tier 1 reaches \add{0.574}", allowed, set())
    assert [b["token"] for b in bad] == ["0.575"] and good == []


def test_numbers_outside_new_text_are_not_checked() -> None:
    allowed = can.allowed_numbers(frames=[], texts=[], objects=[])
    assert can.check(r"old 0.999 \del{0.888} and \add{words only}", allowed, set()) == []


def test_small_whole_numbers_need_to_appear_somewhere_in_the_sources() -> None:
    allowed = can.allowed_numbers(frames=[pd.DataFrame({"k": [9]})], texts=[], objects=[])
    assert can.check(r"\add{9 distinct}", allowed, set()) == []
    assert [b["token"] for b in can.check(r"\add{7 distinct}", allowed, set())] == ["7"]


def test_a_date_must_be_in_the_sources_or_the_allowed_dates() -> None:
    allowed = can.allowed_numbers(frames=[], texts=["split 2023-07-01"], objects=[])
    dates = can.allowed_dates(texts=["split 2023-07-01"], frames=[], objects=[])
    assert can.check(r"\add{on 2023-07-01}", allowed, dates) == []
    assert [b["token"] for b in can.check(r"\add{on 2023-07-02}", allowed, dates)] == ["2023-07-02"]


def test_the_committed_sources_load_and_contain_the_headline_numbers() -> None:
    allowed = can.load_allowed(_ROOT)
    assert {"0.574", "0.292", "0.966", "0.976", "1.376", "57.1"} <= allowed[0]


def test_a_plain_thousands_comma_is_one_number() -> None:
    assert can.numbers("per 100,000 events and 1,344 alerts") == ["100000", "1344"]


def test_the_bibliography_is_not_checked_for_numbers() -> None:
    tex = (r"\add{words 0.574}" "\n" r"\begin{thebibliography}{00}" "\n"
           r"\bibitem{k} \add{Doe, 2020. Journal 12, 345--356. doi:10.1000/xyz}" "\n" r"\end{thebibliography}")
    allowed = can.allowed_numbers(frames=[pd.DataFrame({"auc": [0.574]})], texts=[], objects=[])
    assert can.check(tex, allowed, set()) == []


# ------------------------------------------------ after the code review of the checker
def test_a_decimal_that_starts_like_a_year_is_still_a_number() -> None:
    assert can.numbers("AUC 0.1901 and 0.2005") == ["0.1901", "0.2005"]


def test_a_count_between_1900_and_2099_is_not_mistaken_for_a_year() -> None:
    assert can.numbers("2{,}048 alerts, 1,950 days and 2087") == ["2048", "1950", "2087"]
    assert can.numbers("in 2024 and since 1954") == []


def test_a_number_after_a_name_and_a_comma_is_not_swallowed() -> None:
    assert can.numbers("Table~2, 7777 alerts") == ["7777"]
    assert can.numbers("Tier~5 and 7777 alerts") == ["7777"]
    assert can.numbers("Tier 1, 0.9876") == ["0.9876"]
    assert can.numbers("Tiers~1 to 5") == []


def test_add_followed_by_a_space_or_a_newline_is_still_new_text() -> None:
    assert [p for p, _ in can.added_text(r"\add {7777 words}")] == ["7777 words"]
    assert [p for p, _ in can.added_text(r"\add" "\n{7777 words}")] == ["7777 words"]


def test_a_brace_in_a_comment_does_not_close_the_new_text() -> None:
    tex = r"\add{foo % close }" "\n 7777 bar}"
    assert can.check(tex, set(), set())[0]["token"] == "7777"


def test_a_commented_bibliography_start_does_not_hide_the_text_after_it() -> None:
    tex = (r"%\begin{thebibliography}{00}" "\n" r"\add{7777 words in the body}" "\n"
           r"\begin{thebibliography}{00}" "\n" r"\end{thebibliography}")
    assert [p["token"] for p in can.check(tex, set(), set())] == ["7777"]


def test_dates_in_string_columns_are_collected_whatever_the_pandas_string_dtype() -> None:
    frame = pd.DataFrame({"test_start": ["2023-07-01"], "auc": [0.5]})
    assert can.allowed_dates(texts=[], frames=[frame], objects=[]) == {"2023-07-01"}


def test_the_appendix_is_among_the_checked_files() -> None:
    assert "A_Appendix.tex" in can._TEX_FILES


def test_a_todo_with_nested_braces_and_a_cite_option_are_ignored() -> None:
    assert can.numbers(r"\todo{number?: 70/30 \texttt{eval/results\_main.csv}} and \cite[p.~7777]{k}") == []


def test_main_exits_1_on_a_wrong_number_and_0_on_clean_text(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.tex"
    bad.write_text(r"\add{AUC ≥ 0.9999999 here}" "\n", encoding="utf-8")
    good = tmp_path / "good.tex"
    good.write_text(r"\add{words only}" "\n", encoding="utf-8")
    assert can.main([str(bad)]) == 1
    assert "0.9999999" in capsys.readouterr().out
    assert can.main([str(good)]) == 0
