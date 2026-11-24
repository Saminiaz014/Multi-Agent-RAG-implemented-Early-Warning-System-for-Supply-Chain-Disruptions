"""eval/ch4/fig_results.py: figures R1-R4 and their generated analysis.

The figures are drawn from committed CSVs only, and every sentence of
``eval/FIGURES_ANALYSIS.md`` is built from those CSVs, so the tests check
(a) the data selection on small hand-made frames, (b) the committed numbers the
thesis quotes, and (c) that the text moves when the data does.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fr = _load("fig_results", "eval/ch4/fig_results.py")
mf = _load("make_all_figures", "scripts/make_all_figures.py")
frozen = _load("make_results_frozen", "scripts/make_results_frozen.py")

MAIN = pd.read_csv(_ROOT / "eval" / "results_main.csv")
NO_EQ1 = pd.read_csv(_ROOT / "eval" / "method_comparison_no_eq1.csv")
ABLATION = pd.read_csv(_ROOT / "eval" / "agent_ablation.csv")
SPLIT = pd.read_csv(_ROOT / "eval" / "split_sensitivity.csv")


# ---------------------------------------------------------------- helpers
def _tiers(region: str, deltas: dict[str, tuple[float, float, float]], *,
           copy_of: dict[str, str] | None = None,
           same_as: dict[str, str] | None = None) -> list[dict]:
    """Tier 1..5 rows of ``results_main.csv`` shape; ``deltas`` maps a tier to (d, lo, hi)."""
    rows = []
    for k in range(1, 6):
        name = f"Tier {k}"
        d, lo, hi = deltas.get(name, (np.nan, np.nan, np.nan))
        rows.append({"region": region, "method": name, "row_type": "tier", "n_agents": k,
                     "copy_of": (copy_of or {}).get(name), "same_numbers_as": (same_as or {}).get(name),
                     "auc": 0.6 + d if d == d else np.nan, "auc_ci_low": 0.5, "auc_ci_high": 0.7,
                     "delta_auc": d, "delta_auc_ci_low": lo, "delta_auc_ci_high": hi})
    return rows


def _main(*regions: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([row for rows in regions for row in rows])


@pytest.fixture
def small_main() -> pd.DataFrame:
    return _main(
        _tiers("hormuz", {"Tier 2": (-0.08, -0.16, -0.01), "Tier 3": (-0.10, -0.20, -0.03),
                          "Tier 4": (-0.11, -0.21, -0.04), "Tier 5": (-0.12, -0.22, -0.05)}),
        _tiers("panama", {"Tier 2": (-0.03, -0.05, -0.01), "Tier 3": (-0.03, -0.05, -0.01),
                          "Tier 4": (-0.03, -0.05, -0.01), "Tier 5": (-0.02, -0.05, 0.01)},
               copy_of={"Tier 3": "Tier 2"}, same_as={"Tier 4": "Tier 3"}),
        _tiers("malacca", {}),                              # no positives: nothing to compare
    )


# ------------------------------------------------------------------ R1 data
def test_forest_rows_keep_tiers_2_to_5_of_evaluable_regions_only(small_main) -> None:
    rows = fr.forest_rows(small_main)
    assert set(rows["region"]) == {"hormuz", "panama"}           # malacca has no delta
    assert "Tier 1" not in set(rows["method"])
    assert len(rows) == 8


def test_forest_rows_mark_copied_tiers_and_say_of_what(small_main) -> None:
    rows = fr.forest_rows(small_main).set_index(["region", "method"])
    assert rows.loc[("panama", "Tier 3"), "copy"] == "copy of Tier 2"
    assert rows.loc[("panama", "Tier 4"), "copy"] == "same numbers as Tier 3"
    assert rows.loc[("panama", "Tier 2"), "copy"] == ""
    assert rows.loc[("hormuz", "Tier 4"), "copy"] == ""


def test_forest_verdict_is_read_from_the_interval(small_main) -> None:
    rows = fr.forest_rows(small_main).set_index(["region", "method"])
    assert rows.loc[("hormuz", "Tier 2"), "verdict"] == "below"
    assert rows.loc[("panama", "Tier 5"), "verdict"] == "spans zero"
    up = small_main.copy()
    up.loc[(up.region == "hormuz") & (up.method == "Tier 2"),
           ["delta_auc", "delta_auc_ci_low", "delta_auc_ci_high"]] = [0.05, 0.01, 0.09]
    assert fr.forest_rows(up).set_index(["region", "method"]).loc[("hormuz", "Tier 2"),
                                                                  "verdict"] == "above"


def test_tier_counts_on_a_small_frame(small_main) -> None:
    counts = fr.tier_counts(small_main)
    assert counts == {"compared": 8, "distinct": 6, "below": 5, "above": 0, "spans": 1}


def test_tier_counts_on_the_frozen_results() -> None:
    """The headline the abstract quotes: 8 of 9 distinct comparisons below Tier 1, none above."""
    counts = fr.tier_counts(MAIN)
    assert counts == {"compared": 12, "distinct": 9, "below": 8, "above": 0, "spans": 1}


# ------------------------------------------------------------------ R2 data
def test_eq1_rows_pair_each_tier_with_its_change() -> None:
    rows = fr.eq1_rows(NO_EQ1, MAIN)
    assert set(rows["region"]) == {"hormuz", "bab_el_mandeb", "panama"}      # malacca: empty
    assert len(rows) == 15
    one = rows.set_index(["region", "method"]).loc[("hormuz", "Tier 1")]
    assert one["auc_on"] == pytest.approx(
        MAIN[(MAIN.region == "hormuz") & (MAIN.method == "Tier 1")]["auc"].iloc[0])
    assert one["delta"] == pytest.approx(one["auc_off"] - one["auc_on"], abs=1e-9)


def test_eq1_with_values_equal_the_main_results() -> None:
    """The 'on' side of R2 is the main table, so the two figures cannot disagree."""
    rows = fr.eq1_rows(NO_EQ1, MAIN)
    main = MAIN[MAIN.row_type == "tier"].set_index(["region", "method"])["auc"]
    for _, r in rows.iterrows():
        assert r["auc_on"] == pytest.approx(main[(r["region"], r["method"])], abs=1e-9)


# ------------------------------------------------------------------ R3 data
def test_agent_rows_split_single_agents_and_leave_one_out() -> None:
    single, loo, full = fr.agent_rows(ABLATION)
    assert set(single["region"]) == {"hormuz", "bab_el_mandeb", "panama"}    # malacca: not evaluable
    assert set(loo["region"]) == set(single["region"])
    assert set(full["region"]) == set(single["region"])
    assert len(single) == 5 + 4 + 4 and len(loo) == 5 + 4 + 4


def test_agents_come_in_one_fixed_order_within_every_region() -> None:
    single, loo, _ = fr.agent_rows(ABLATION)
    for frame in (single, loo):
        for _, sub in frame.groupby("region"):
            order = [fr.AGENT_ORDER.index(a) for a in sub["agent"]]
            assert order == sorted(order)


def test_leave_one_out_delta_is_without_minus_full() -> None:
    """Positive means removing the agent raised AUC, i.e. the agent was hurting."""
    single, loo, full = fr.agent_rows(ABLATION)
    f = full.set_index("region")["auc"]
    for _, r in loo.iterrows():
        assert r["delta"] == pytest.approx(r["auc_without"] - f[r["region"]], abs=1e-6)


def test_every_agent_has_one_colour_and_the_colours_differ() -> None:
    assert set(fr.AGENT_COLOR) == set(fr.AGENT_ORDER)
    assert len(set(fr.AGENT_COLOR.values())) == len(fr.AGENT_ORDER)


# ------------------------------------------------------------------ R4 data
def test_split_rows_are_tier_1_and_tier_4_of_evaluable_regions_at_three_dates() -> None:
    rows = fr.split_rows(SPLIT)
    assert set(rows["region"]) == {"hormuz", "bab_el_mandeb", "panama"}
    assert set(rows["method"]) == {"Tier 1", "Tier 4"}
    assert rows["split_date"].nunique() == 3
    assert len(rows) == 3 * 3 * 2


def test_split_main_date_is_the_evaluation_split_in_settings() -> None:
    import yaml
    settings = yaml.safe_load((_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    assert fr.main_split(SPLIT, MAIN) == str(settings["evaluation"]["split_date"])[:10]


# --------------------------------------------------------------- formatting
@pytest.mark.parametrize("x", [0.0, -0.0004, 0.0004, -0.078, 0.1394, 0.005, -0.000001])
def test_signed_format_matches_the_frozen_page(x) -> None:
    assert fr.signed(x) == frozen._signed(x)


def test_plain_format_matches_the_frozen_page() -> None:
    assert fr.plain(0.57412) == frozen._f(0.57412) == "0.574"


# ---------------------------------------------------------------- the text
def test_r1_text_states_the_headline_from_the_data() -> None:
    text = fr.analysis_r1(MAIN)
    assert "12 tier comparisons" in text
    assert "8 of 9 distinct" in text and "none above" in text
    assert "Panama Tier 5" in text                                # the one that spans zero
    assert "Bab el-Mandeb Tier 3" in text                         # the largest drop


def test_r1_text_follows_the_data(small_main) -> None:
    a = fr.analysis_r1(small_main)
    flipped = small_main.copy()
    flipped.loc[(flipped.region == "panama") & (flipped.method == "Tier 5"),
                "delta_auc_ci_high"] = -0.001
    b = fr.analysis_r1(flipped)
    assert a != b
    assert "5 of 6 distinct" in a and "6 of 6 distinct" in b


def test_r2_text_names_the_hormuz_collapse_and_the_panama_no_change() -> None:
    text = fr.analysis_r2(NO_EQ1, MAIN)
    assert "12 of 12 distinct" in text and "9 of 12" in text    # copies are not counted twice
    assert "Hormuz" in text and "below 0.5" in text
    assert "Panama" in text and "about 0.0005" in text           # the largest point change, not a rounded bound


def test_r3_text_names_the_best_single_agent_and_the_geopolitical_agent() -> None:
    text = fr.analysis_r3(ABLATION)
    assert "shipping" in text and "geopolitical" in text
    assert "Bab el-Mandeb" in text


def test_r4_text_counts_the_dates_and_pairs() -> None:
    text = fr.analysis_r4(SPLIT)
    assert "2023-01-01" in text and "2023-07-01" in text and "2024-01-01" in text
    assert "9 of 9" in text and "8 of 9" in text                  # point estimates; paired interval below zero
    assert "At each of" not in text                               # no universal claim the data do not guard


def test_every_number_in_the_analysis_is_in_a_committed_csv() -> None:
    """Numbers are generated, not typed: each one appears, as formatted, in a source file."""
    doc = fr.render_analysis()
    sources = {
        "results_main.csv": MAIN, "method_comparison_no_eq1.csv": NO_EQ1,
        "agent_ablation.csv": ABLATION, "split_sensitivity.csv": SPLIT,
        "bem_surrogate.csv": pd.read_csv(_ROOT / "eval" / "bem_surrogate.csv"),
    }
    seen: set[str] = set()
    for frame in sources.values():
        for col in frame.select_dtypes("number").columns:
            for v in frame[col].dropna():
                seen.add(frozen._f(v))
                seen.add(frozen._signed(v).lstrip("+"))
                seen.add(frozen._signed(v))
                seen.add(frozen._signed(abs(v)).lstrip("+"))
                seen.add(f"{v * 100:.1f}%")
                seen.add(f"{int(v)}" if float(v).is_integer() else "")
    body = re.sub(r"`[^`]*`", "", doc)                            # file names and code spans
    body = re.sub(r"\d{4}-\d{2}-\d{2}", "", body)                 # dates are checked below
    body = re.sub(r"\b(Tier|R|T) ?\d\b", "", body)                # tier and figure names
    body = re.sub(r"\b\d+ of \d+\b|\b\d+ (tier|distinct|region)", "", body)   # counts are derived
    chance = {"0.5"}                                              # AUC of a coin flip: a constant, not a result
    for num in re.findall(r"[+-]?\d+\.\d+%?", body):
        assert num in seen or num.lstrip("+") in seen or num in chance, num


def test_analysis_dates_are_split_dates_of_the_csv() -> None:
    doc = fr.render_analysis()
    dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", doc))
    assert dates <= set(SPLIT["split_date"]) | set(MAIN["test_start"]) | set(MAIN["test_end"])


# ---------------------------------------------------------------- drawing
@pytest.mark.parametrize("name, draw", [
    ("R1_delta_auc_forest", lambda out: fr.fig_r1(MAIN, out)),
    ("R2_eq1_ablation", lambda out: fr.fig_r2(NO_EQ1, MAIN, out)),
    ("R3_agent_contribution", lambda out: fr.fig_r3(ABLATION, out)),
    ("R4_split_sensitivity", lambda out: fr.fig_r4(SPLIT, out)),
])
def test_each_figure_writes_a_pdf_and_a_png(tmp_path, name, draw) -> None:
    written = draw(tmp_path)
    assert {p.name for p in written} == {f"{name}.pdf", f"{name}.png"}
    assert all(p.exists() and p.stat().st_size > 1000 for p in written)


def test_r3_has_the_line_at_one_half_on_the_auc_panels() -> None:
    fig = fr.figure_r3(ABLATION)
    try:
        auc_axes = [ax for ax in fig.axes if ax.get_xlabel().startswith("AUC")]
        assert len(auc_axes) == 3
        for ax in auc_axes:
            xs = [ln.get_xdata()[0] for ln in ax.lines if len(set(ln.get_xdata())) == 1]
            assert 0.5 in xs
    finally:
        fr.plt.close(fig)


def test_r1_marks_copied_tiers_with_open_markers() -> None:
    from matplotlib.collections import PathCollection
    fig = fr.figure_r1(MAIN)
    try:
        faces = [tuple(c.get_facecolor()[0]) for ax in fig.axes for c in ax.collections
                 if isinstance(c, PathCollection)]
        hollow = [f for f in faces if f == (1.0, 1.0, 1.0, 1.0)]
        assert len(hollow) == 3, "Bab el-Mandeb Tier 4 and Panama Tiers 3 and 4 copy an earlier tier"
        assert len(faces) == 12
    finally:
        fr.plt.close(fig)


def test_figures_use_only_the_declared_palette() -> None:
    assert set(fr.PALETTE.values()) >= set(fr.AGENT_COLOR.values())
    for colour in list(fr.PALETTE.values()) + [fr.CONTROL_GREY]:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", colour)
    assert fr.CONTROL_GREY not in fr.AGENT_COLOR.values()          # controls are grey, agents are not


# ---------------------------------------------------------------- registry
def test_registry_lists_r1_to_r4_with_committed_sources_and_two_files_each() -> None:
    by_id = {f.figure.split(" ", 1)[0]: f for f in mf.FIGURES}
    for fig_id in ("R1", "R2", "R3", "R4"):
        fig = by_id[fig_id]
        assert len(fig.files) == 2 and {Path(p).suffix for p in fig.files} == {".pdf", ".png"}
        assert fig.script.startswith("eval/ch4/fig_results.py")
    assert by_id["R1"].sources == ("eval/results_main.csv",)
    assert by_id["R2"].sources == ("eval/method_comparison_no_eq1.csv", "eval/results_main.csv")
    assert by_id["R3"].sources == ("eval/agent_ablation.csv",)
    assert by_id["R4"].sources == ("eval/split_sensitivity.csv", "eval/results_main.csv")


def test_there_is_no_figure_for_malacca_alert_rates_or_the_bem_surrogate() -> None:
    names = " ".join(f.figure.lower() for f in mf.FIGURES if f.figure.startswith("R"))
    assert "malacca" not in names and "surrogate" not in names and "bem" not in names


def test_the_analysis_file_is_generated_and_current() -> None:
    doc = (_ROOT / "eval" / "FIGURES_ANALYSIS.md").read_text(encoding="utf-8")
    assert doc == fr.render_analysis()


def test_write_analysis_flag_writes_the_file(tmp_path, monkeypatch) -> None:
    out = tmp_path / "FIGURES_ANALYSIS.md"
    monkeypatch.setattr(mf, "ANALYSIS_MD", out)
    assert mf.main(["--write-analysis"]) == 0
    assert out.read_text(encoding="utf-8") == fr.render_analysis()


def test_the_analysis_says_which_values_have_no_figure() -> None:
    doc = fr.render_analysis()
    assert "No figure" in doc and "Malacca" in doc and "surrogate" in doc.lower()


# ------------------------------------------------ after the code review
def test_r2_headline_counts_distinct_cells_only() -> None:
    fig = fr.figure_r2(NO_EQ1, MAIN)
    try:
        assert "9 of 12 distinct" in fig._suptitle.get_text()
    finally:
        fr.plt.close(fig)


def test_r1_title_follows_the_data(small_main) -> None:
    fig = fr.figure_r1(small_main)
    try:
        assert "lowers AUC" in fig._suptitle.get_text()
    finally:
        fr.plt.close(fig)
    up = small_main.copy()
    up.loc[(up.region == "hormuz") & (up.method == "Tier 2"),
           ["delta_auc", "delta_auc_ci_low", "delta_auc_ci_high"]] = [0.05, 0.01, 0.09]
    fig = fr.figure_r1(up)
    try:
        assert "lowers AUC" not in fig._suptitle.get_text()      # one tier is above Tier 1
    finally:
        fr.plt.close(fig)


def test_r1_text_says_when_a_clear_comparison_only_narrowly_excludes_zero() -> None:
    text = fr.analysis_r1(MAIN)
    assert "narrowly" in text and "Panama Tier 2" in text        # upper bound -0.0004


def test_the_surrogate_share_names_how_many_features_it_covers() -> None:
    text = fr.render_analysis()
    assert re.search(r"at least one of the 5 features checked", text)


def test_the_committed_metrics_file_is_exempt_from_newline_conversion() -> None:
    """core.autocrlf would turn it into CRLF on checkout and break the hash in the screenshot manifest."""
    import subprocess
    out = subprocess.run(["git", "check-attr", "text", "eval/case_metrics.json"], cwd=_ROOT,
                         capture_output=True, text=True).stdout
    assert out.strip().endswith("text: unset"), out


def _scaled_copy(src: Path, dst: Path, factor: float) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("results_main.csv", "method_comparison_no_eq1.csv", "agent_ablation.csv",
                 "split_sensitivity.csv", "bem_surrogate.csv"):
        frame = pd.read_csv(src / name)
        for col in frame.select_dtypes("float").columns:
            frame[col] = frame[col] * factor
        frame.to_csv(dst / name, index=False)


def test_no_number_in_the_analysis_survives_a_change_of_the_data(tmp_path) -> None:
    """A typed number would not move when every float in the sources is scaled; a generated one does.

    Two different scale factors are used so that a coincidental equal string is not mistaken for a
    typed number.
    """
    number = r"[+-]?\d+\.\d+"

    def numbers(text: str) -> set[str]:
        return set(re.findall(number, re.sub(r"`[^`]*`|\d{4}-\d{2}-\d{2}", "", text)))

    _scaled_copy(_ROOT / "eval", tmp_path / "up", 1.37)
    _scaled_copy(_ROOT / "eval", tmp_path / "down", 0.61)
    survivors = (numbers(fr.render_analysis()) & numbers(fr.render_analysis(tmp_path / "up"))
                 & numbers(fr.render_analysis(tmp_path / "down")))
    survivors = {n for n in survivors if abs(float(n)) >= 0.001 and n.lstrip("+-") != "0.5"}
    assert not survivors, survivors


def _drawn_colours(fig) -> set[str]:
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.colors import to_hex
    out = set()
    for ax in fig.axes:
        for line in ax.lines:
            out.add(to_hex(line.get_color()).lower())
        for coll in ax.collections:
            if isinstance(coll, LineCollection):
                out |= {to_hex(c).lower() for c in coll.get_colors()}
            elif isinstance(coll, PathCollection):
                out |= {to_hex(c).lower() for c in list(coll.get_facecolor()) + list(coll.get_edgecolor())}
    return out


@pytest.mark.parametrize("make", [lambda: fr.figure_r1(MAIN), lambda: fr.figure_r2(NO_EQ1, MAIN),
                                  lambda: fr.figure_r3(ABLATION), lambda: fr.figure_r4(SPLIT, MAIN)])
def test_every_colour_a_figure_draws_is_declared(make) -> None:
    fig = make()
    try:
        allowed = {c.lower() for c in fr.PALETTE.values()} | {fr.CONTROL_GREY.lower(), "#ffffff"}
        drawn = _drawn_colours(fig)
        assert drawn <= allowed, drawn - allowed
        assert fr.CONTROL_GREY.lower() in drawn            # reference lines are grey
    finally:
        fr.plt.close(fig)
