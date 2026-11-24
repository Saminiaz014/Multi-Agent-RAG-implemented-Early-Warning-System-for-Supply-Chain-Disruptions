"""Check that every number in new (red) LaTeX text traces to a committed source.

The writing fixes mark new text with ``\\add{...}`` (or ``\\begin{addblock}...\\end{addblock}``) and
removed text with ``\\del{...}``. This script reads only the new text and tests each number in it
against the committed results: ``eval/RESULTS_FROZEN.md``, the result CSVs under ``eval/``,
``eval/case_metrics.json``, ``eval/FIGURES_ANALYSIS.md``, ``outputs/explanations_summary.json`` and
``config/settings.yaml`` (for the split date and the fixed thresholds).

A number is accepted when it equals a source value at the precision the text prints it (a source
value is tried at 0 to 4 decimals, as its absolute value, and times 100 for a percentage). Years
(written bare, 1950 to 2039) and ISO dates are not measurements: a date must appear in a source.
LaTeX plumbing (``\\includegraphics`` options, ``\\label``, ``\\ref``, ``\\cite``, names such as
``Tier~5``), ``\\todo{...}`` (an open item by definition), comments and the reference list are ignored.

**What this check can and cannot do.** It catches a number that no source contains, which is the
mistake of typing or remembering a figure. It cannot catch a wrong number that happens to equal some
other source value: the allowed set is one pool, so a typo in the last digit of a three-decimal value
often still lands on a value from somewhere in the sources, and every whole number from 0 to 100 is
always allowed. A reader (or the methodology review) still has to check that a number is the right one.

Usage::

    python scripts/check_added_numbers.py                    # thesis chapters and the paper
    python scripts/check_added_numbers.py path/to/file.tex   # chosen files

Exit status is 1 when a number or date is not found.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: Design constants named in new text that are not results: the chance level of an AUC and the
#: reference values (stated as not pre-registered), and the per-100,000 normalisation of news volume.
CONSTANTS = {"0.5": "AUC of a coin flip (chance level)",
             "0.60": "reference value for the surrogate R2, stated as not pre-registered",
             "0.70": "reference value for the Spearman correlation, stated as not pre-registered",
             "100000": "news volume is normalised per 100,000 GDELT events (build_gdelt_news_csv.py)"}

_SOURCE_CSV = ("results_main.csv", "method_comparison_no_eq1.csv", "agent_ablation.csv",
               "agent_ablation_no_eq1.csv", "split_sensitivity.csv", "lead_time.csv",
               "method_comparison_all_agents.csv", "method_comparison_results.csv", "bem_surrogate.csv")
_TEX_FILES = ("1_0_Introduction.tex", "2_0_Literature_Review.tex", "3_0_Methodology.tex",
              "3_5_Demonstration_Cases.tex", "4_0_Evaluation.tex", "5_0_Discussion.tex",
              "6_0_Conclusion_and_Outlook.tex", "A_Appendix.tex")
_PAPER = Path("Paper") / "PROCIR_CIRP CMS 2027_Tex_Template" / "procir_cirpcms2027_sami.tex"

_DATE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
#: A bare four-digit year: not part of a longer number, a decimal or a thousands-separated count.
_YEAR = re.compile(r"(?<![\d.,])(?:19[5-9]\d|20[0-3]\d)(?!\d|,\d{3}|\.\d)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_STRIP_ARG = re.compile(r"\\(?:label|ref|eqref|cref|autoref|cite[a-z]*|bibitem|url)(?:\[[^\]]*\])?\s*\{[^{}]*\}")
_STRIP_OPT = re.compile(r"\\includegraphics\[[^\]]*\]")
#: A name and its number; a range after it is swallowed only when it is made of one or two digit numbers.
_NAME_NUM = re.compile(r"\b(?:Tiers?|Eq\.|Equation|Figs?\.|Figure|Table|Section|Chapter|SRQ|RQ|METRIC|E-[A-Z])"
                       r"[~\s]*\d+(?:\s*(?:to|and|--)\s*\d{1,2}(?![\d.]))*")
_ID = re.compile(r"\b[A-Za-z]\d+\b")        # R1, T5, B7, M3, F12


# ------------------------------------------------------------------ new text
def _uncomment(tex: str) -> str:
    """Remove LaTeX comments (an unescaped ``%`` to the end of the line); line numbers are kept."""
    return re.sub(r"(?<!\\)%[^\n]*", "", tex)


def _match_brace(s: str, i: int) -> int:
    """Index just past the brace that closes the one opened at ``s[i - 1]``."""
    depth = 1
    while i < len(s) and depth:
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            i += 2
            continue
        depth += (c == "{") - (c == "}")
        i += 1
    return i


def _strip_macro(text: str, name: str) -> str:
    """Remove ``\\name{...}`` including its balanced argument, however many braces it holds."""
    out, pos = [], 0
    for m in re.finditer(r"\\" + name + r"\s*\{", text):
        if m.start() < pos:
            continue
        out.append(text[pos:m.start()])
        pos = _match_brace(text, m.end())
    out.append(text[pos:])
    return " ".join(out)


def added_text(tex: str) -> list[tuple[str, int]]:
    """The new text: bodies of ``\\add{...}`` and of ``addblock`` environments, with their line numbers."""
    tex = _uncomment(tex)
    out = []
    for m in re.finditer(r"\\add\s*\{", tex):
        end = _match_brace(tex, m.end())
        out.append((m.start(), tex[m.end():end - 1]))
    for m in re.finditer(r"\\begin\{addblock\}(.*?)\\end\{addblock\}", tex, flags=re.S):
        out.append((m.start(), m.group(1)))
    out.sort(key=lambda t: t[0])
    return [(body, tex.count("\n", 0, pos) + 1) for pos, body in out]


# ------------------------------------------------------------------ tokens
def _clean(text: str) -> str:
    text = _uncomment(text)
    text = _strip_macro(text, "todo")
    text = _STRIP_OPT.sub(" ", text)
    text = _STRIP_ARG.sub(" ", text)
    return text


def dates(text: str) -> list[str]:
    """ISO dates in ``text``."""
    return _DATE.findall(_clean(text))


def numbers(text: str) -> list[str]:
    """Numbers in ``text`` (dates and bare years left out), thousands separators removed."""
    text = _DATE.sub(" ", _clean(text))
    text = _YEAR.sub(" ", text)                        # before thousands separators go: 1,950 is a count
    text = text.replace("{,}", "")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)      # 100,000 -> 100000
    text = _NAME_NUM.sub(" ", text)
    text = _ID.sub(" ", text)
    return _NUMBER.findall(text)


# ---------------------------------------------------------------- the sources
def _variants(value: float) -> set[str]:
    v = abs(float(value))
    out = {f"{v:.{k}f}" for k in range(0, 5)}
    out |= {f"{v * 100:.{k}f}" for k in range(0, 2)}
    if float(v).is_integer():
        out.add(str(int(v)))
    return out


def _walk(obj, sink: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        sink.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, sink)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, sink)


def allowed_numbers(frames, texts, objects) -> set[str]:
    """Every number string a source can justify (see the module docstring)."""
    out: set[str] = set()
    for frame in frames:
        for col in frame.select_dtypes("number").columns:
            for value in frame[col].dropna():
                out |= _variants(value)
    for text in texts:
        for token in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
            token = token.replace(",", "")
            out.add(token)
            try:
                out |= _variants(float(token))
            except ValueError:
                pass
    for obj in objects:
        values: list[float] = []
        _walk(obj, values)
        for value in values:
            out |= _variants(value)
    return out


def allowed_dates(texts, frames, objects) -> set[str]:
    """Every ISO date that appears in a source."""
    found: set[str] = set()
    for text in texts:
        found |= set(_DATE.findall(text))
    for frame in frames:
        for col in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[col]):         # object or the newer string dtype
                for value in frame[col].dropna().astype(str):
                    found |= set(_DATE.findall(value))
    found |= set(_DATE.findall(json.dumps(objects, default=str)))
    return found


def load_allowed(root: Path = ROOT) -> tuple[set[str], set[str]]:
    """``(numbers, dates)`` justified by the committed sources under ``root``."""
    frames = [pd.read_csv(root / "eval" / name) for name in _SOURCE_CSV if (root / "eval" / name).exists()]
    texts = [(root / p).read_text(encoding="utf-8") for p in
             ("eval/RESULTS_FROZEN.md", "eval/FIGURES_ANALYSIS.md", "config/settings.yaml")
             if (root / p).exists()]
    objects = [json.loads((root / p).read_text(encoding="utf-8")) for p in
               ("eval/case_metrics.json", "outputs/explanations_summary.json") if (root / p).exists()]
    nums = allowed_numbers(frames, texts, objects) | set(CONSTANTS)
    return nums, allowed_dates(texts, frames, objects)


# ---------------------------------------------------------------------- check
def _without_bibliography(tex: str) -> str:
    """Drop comments and the reference list: its volumes, pages and DOIs are checked against Crossref instead."""
    return re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", _uncomment(tex), flags=re.S)


def check(tex: str, allowed: set[str], allowed_date_set: set[str]) -> list[dict]:
    """Numbers and dates in the new text of ``tex`` that no source justifies."""
    tex = _without_bibliography(tex)
    problems = []
    for body, line in added_text(tex):
        for token in numbers(body):
            if token not in allowed and token.lstrip("0") not in allowed:
                problems.append({"line": line, "token": token, "context": " ".join(body.split())[:90]})
        for date in dates(body):
            if date not in allowed_date_set:
                problems.append({"line": line, "token": date, "context": " ".join(body.split())[:90]})
    return problems


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):                 # a context may hold a character the console cannot print
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = list(sys.argv[1:] if argv is None else argv)
    files = [Path(a) for a in args] or ([ROOT / "thesis" / f for f in _TEX_FILES] + [ROOT / _PAPER])
    allowed, date_set = load_allowed()
    bad = 0
    checked = 0
    print("| file | line | not found in a source | context |")
    print("|---|---|---|---|")
    for path in files:
        if not path.exists():
            print(f"| {path} | | file missing | |")
            bad += 1
            continue
        tex = _without_bibliography(path.read_text(encoding="utf-8"))
        checked += sum(len(numbers(b)) + len(dates(b)) for b, _ in added_text(tex))
        for p in check(tex, allowed, date_set):
            bad += 1
            print(f"| {path.name} | {p['line']} | {p['token']} | {p['context'].replace('|', '/')} |")
    print(f"\n{checked} numbers and dates in new text checked, {bad} not found in a committed source.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
