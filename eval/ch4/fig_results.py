"""Thesis figures R1-R4 and the analysis text that goes with them, from committed CSVs only.

====  =========================================================  ===============================
R1    paired change in AUC against Tier 1, per region and tier    ``eval/results_main.csv``
R2    Eq. 1 duration term on and off, per region and tier         ``eval/method_comparison_no_eq1.csv``
R3    each agent alone, and the change when it is removed         ``eval/agent_ablation.csv``
R4    Tier 1 and Tier 4 AUC at three split dates                  ``eval/split_sensitivity.csv``
====  =========================================================  ===============================

Nothing here needs the git-ignored evaluation frames. Malacca has no positive
test day, so no AUC exists for it and it is not drawn; the Malacca false-alarm
rates and the Bab el-Mandeb surrogate error have no figure on purpose and appear
in the analysis as values only.

``render_analysis()`` writes ``eval/FIGURES_ANALYSIS.md``: two or three
sentences per figure, every number taken from the CSV it describes (nothing is
typed). ``scripts/make_all_figures.py`` draws the figures and, with
``--write-analysis``, writes that file.

Usage: ``python eval/ch4/fig_results.py`` (draws R1-R4 into ``thesis/figures/ch4``).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
THESIS_FIGURES = ROOT / "thesis" / "figures" / "ch4"
CSV_DIR = ROOT / "eval"

#: Regions with positives in the test window; Malacca has none, so no AUC exists.
REGIONS = ("hormuz", "bab_el_mandeb", "panama")
SHORT = {"hormuz": "Hormuz", "bab_el_mandeb": "Bab el-Mandeb", "panama": "Panama",
         "malacca": "Malacca"}
TIERS = tuple(f"Tier {k}" for k in range(1, 6))

#: Okabe-Ito, the colourblind-safe set the thesis figures already use (D1's blue is #0072B2).
PALETTE = {"blue": "#0072B2", "orange": "#E69F00", "vermillion": "#D55E00",
           "green": "#009E73", "purple": "#CC79A7"}
#: Controls and reference lines are grey, never one of the agent colours.
CONTROL_GREY = "#777777"
GRID = "#E6E6E6"
AGENT_ORDER = ("shipping", "market", "geopolitical", "natural_disaster", "news_sentiment")
AGENT_COLOR = dict(zip(AGENT_ORDER, PALETTE.values()))
TIER_COLOR = {"Tier 1": PALETTE["blue"], "Tier 4": PALETTE["vermillion"]}
VERDICT_COLOR = {"below": PALETTE["blue"], "spans zero": PALETTE["vermillion"],
                 "above": PALETTE["green"]}

_W = 6.3        # figure width in inches: the text width of the thesis


# ------------------------------------------------------------------ formatting
def plain(x: float) -> str:
    """Three decimals, as ``eval/RESULTS_FROZEN.md`` prints a value."""
    return f"{x:.3f}"


def signed(x: float) -> str:
    """Signed, three decimals; a nonzero value that would print as zero gets four (as the frozen page)."""
    return f"{x:+.4f}" if x != 0 and abs(x) < 5e-4 else f"{x:+.3f}"


def interval(x: float, lo: float, hi: float, *, delta: bool = False) -> str:
    """``x [lo, hi]``; signed when it is a change."""
    f = signed if delta else plain
    return f"{f(x)} [{f(lo)}, {f(hi)}]"


def _agent_name(agent: str) -> str:
    return agent.replace("_", " ")


def _style() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True})


def _footer(fig, text: str) -> None:
    """Small grey source line at the bottom, wrapped so none of it is cut off."""
    fig.text(0.02, 0.006, textwrap.fill(text, width=118), fontsize=6.6, color=CONTROL_GREY,
             va="bottom")


def _save(fig, out_dir: Path | None, name: str) -> list[Path]:
    out_dir = Path(out_dir) if out_dir is not None else THESIS_FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / name)


def _region_order(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_r"] = frame["region"].map({r: i for i, r in enumerate(REGIONS)})
    return frame


# ------------------------------------------------------------------ R1 data
def _verdict(lo: float, hi: float) -> str:
    return "below" if hi < 0 else "above" if lo > 0 else "spans zero"


def forest_rows(main: pd.DataFrame) -> pd.DataFrame:
    """Tiers 2-5 of the regions that have a paired change against Tier 1.

    Columns: region, method, delta, lo, hi, copy (``"copy of Tier 2"``,
    ``"same numbers as Tier 3"`` or ``""``), verdict (``below`` / ``above`` /
    ``spans zero``, from the interval).
    """
    t = main[(main["row_type"] == "tier") & main["delta_auc"].notna()
             & main["region"].isin(REGIONS)]
    rows = []
    for r in t.itertuples():
        copy = (f"copy of {r.copy_of}" if isinstance(r.copy_of, str)
                else f"same numbers as {r.same_numbers_as}" if isinstance(r.same_numbers_as, str)
                else "")
        rows.append({"region": r.region, "method": r.method, "delta": r.delta_auc,
                     "lo": r.delta_auc_ci_low, "hi": r.delta_auc_ci_high, "copy": copy,
                     "verdict": _verdict(r.delta_auc_ci_low, r.delta_auc_ci_high)})
    out = _region_order(pd.DataFrame(rows))
    return out.sort_values(["_r", "method"]).drop(columns="_r").reset_index(drop=True)


def tier_counts(main: pd.DataFrame) -> dict[str, int]:
    """How many tier comparisons there are, how many are distinct, and how they fall.

    A tier whose numbers copy an earlier tier is a comparison but not a distinct one.
    """
    rows = forest_rows(main)
    distinct = rows[rows["copy"] == ""]
    return {"compared": len(rows), "distinct": len(distinct),
            "below": int((distinct["verdict"] == "below").sum()),
            "above": int((distinct["verdict"] == "above").sum()),
            "spans": int((distinct["verdict"] == "spans zero").sum())}


# ------------------------------------------------------------------ R2 data
def eq1_rows(no_eq1: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Per region and tier: AUC with Eq. 1 on and off, and the paired change (off minus on).

    ``auc_on`` is the main table's value (carried in the ablation file as
    ``auc_with_eq1``), so R2 and R1 cannot disagree. ``copy`` marks tiers whose
    numbers repeat an earlier tier in the main table.
    """
    flags = {(r.region, r.method): (f"copy of {r.copy_of}" if isinstance(r.copy_of, str)
                                    else f"same numbers as {r.same_numbers_as}"
                                    if isinstance(r.same_numbers_as, str) else "")
             for r in main[main["row_type"] == "tier"].itertuples()}
    t = no_eq1[no_eq1["auc"].notna() & no_eq1["region"].isin(REGIONS)]
    out = pd.DataFrame({
        "region": t["region"], "method": t["method"],
        "auc_on": t["auc_with_eq1"], "auc_off": t["auc"],
        "off_lo": t["auc_ci_low"], "off_hi": t["auc_ci_high"],
        "delta": t["eq1_delta_auc"], "dlo": t["eq1_delta_auc_ci_low"],
        "dhi": t["eq1_delta_auc_ci_high"],
        "copy": [flags.get((r, m), "") for r, m in zip(t["region"], t["method"])]})
    out = _region_order(out)
    return out.sort_values(["_r", "method"]).drop(columns="_r").reset_index(drop=True)


def _on_interval(main: pd.DataFrame) -> pd.DataFrame:
    t = main[main["row_type"] == "tier"].set_index(["region", "method"])
    return t[["auc", "auc_ci_low", "auc_ci_high"]]


# ------------------------------------------------------------------ R3 data
def agent_rows(ablation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(single, loo, full)``: each agent alone, each agent left out, the full enabled set.

    ``single``: region, agent, auc, lo, hi. ``loo``: region, agent, delta, lo, hi,
    auc_without (``delta`` is the AUC without the agent minus the full set's, so a
    positive value means removing the agent helped, i.e. the agent was hurting).
    ``full``: region, auc, lo, hi. Regions without an AUC (Malacca) are dropped.
    """
    a = ablation[ablation["region"].isin(REGIONS)]

    def order(frame: pd.DataFrame) -> pd.DataFrame:
        frame = _region_order(frame)
        frame["_a"] = frame["agent"].map({x: i for i, x in enumerate(AGENT_ORDER)}) \
            if "agent" in frame else 0
        return frame.sort_values(["_r", "_a"]).drop(columns=["_r", "_a"]).reset_index(drop=True)

    s = a[(a["ablation"] == "single") & a["auc"].notna()]
    single = order(pd.DataFrame({"region": s["region"], "agent": s["agent"], "auc": s["auc"],
                                 "lo": s["auc_ci_low"], "hi": s["auc_ci_high"]}))
    l = a[(a["ablation"] == "leave-one-out") & a["loo_delta_auc"].notna()]
    loo = order(pd.DataFrame({"region": l["region"], "agent": l["agent"], "delta": l["loo_delta_auc"],
                              "lo": l["loo_delta_auc_ci_low"], "hi": l["loo_delta_auc_ci_high"],
                              "auc_without": l["auc"]}))
    f = a[(a["ablation"] == "full") & a["auc"].notna()]
    full = _region_order(pd.DataFrame({"region": f["region"], "auc": f["auc"],
                                       "lo": f["auc_ci_low"], "hi": f["auc_ci_high"]}))
    full = full.sort_values("_r").drop(columns="_r").reset_index(drop=True)
    return single, loo, full


# ------------------------------------------------------------------ R4 data
def split_rows(split: pd.DataFrame) -> pd.DataFrame:
    """Tier 1 and Tier 4 per region and split date; regions without an AUC are dropped.

    Columns: split_date, region, method, auc, lo, hi, delta, dlo, dhi (the paired
    Tier 4 - Tier 1 change, on the Tier 4 rows), n_test, test_positives.
    """
    t = split[split["method"].isin(["Tier 1", "Tier 4"]) & split["auc"].notna()
              & split["region"].isin(REGIONS)]
    out = pd.DataFrame({"split_date": t["split_date"].astype(str), "region": t["region"],
                        "method": t["method"], "auc": t["auc"], "lo": t["auc_ci_low"],
                        "hi": t["auc_ci_high"], "delta": t["delta_auc"],
                        "dlo": t["delta_auc_ci_low"], "dhi": t["delta_auc_ci_high"],
                        "n_test": t["n_test"], "test_positives": t["test_positives"]})
    out = _region_order(out)
    return (out.sort_values(["_r", "split_date", "method"]).drop(columns="_r")
            .reset_index(drop=True))


def main_split(split: pd.DataFrame, main: pd.DataFrame) -> str:
    """The split date the main results use: the test window's first day in ``results_main.csv``."""
    start = str(main.loc[main["row_type"] == "tier", "test_start"].dropna().iloc[0])
    if start not in set(split["split_date"].astype(str)):
        raise ValueError(f"the main split {start} is not among the sensitivity dates")
    return start


# ------------------------------------------------------------------ R1 figure
def figure_r1(main: pd.DataFrame):
    """Forest plot: paired change in AUC against Tier 1, one row per tier and region."""
    _style()
    rows = forest_rows(main)
    counts = tier_counts(main)
    ys: list[float] = []
    heads: list[tuple[float, str]] = []
    y = 0.0
    for region in REGIONS:
        if not (rows["region"] == region).any():
            continue
        heads.append((y, SHORT[region]))
        y -= 1.0
        for _ in range((rows["region"] == region).sum()):
            ys.append(y)
            y -= 1.0
        y -= 0.5
    fig, ax = plt.subplots(figsize=(_W, 5.2))
    lo_min = min(rows["lo"].min(), 0)
    ax.axvline(0, color=CONTROL_GREY, lw=1.1, ls="--", zorder=1)
    for (_, r), yy in zip(rows.iterrows(), ys):
        colour = VERDICT_COLOR[r["verdict"]]
        ax.hlines(yy, r["lo"], r["hi"], color=colour, lw=2.0, zorder=2)
        ax.scatter([r["delta"]], [yy], s=46, zorder=3, edgecolor=colour, linewidth=1.6,
                   facecolor="white" if r["copy"] else colour)
    for (yy, name) in heads:
        ax.text(lo_min - 0.02, yy, name, ha="left", va="center", fontsize=10.5,
                fontweight="bold")
    ax.set_yticks(ys, [r["method"] + (" †" if r["copy"] else "")
                       for _, r in rows.iterrows()], fontsize=9)
    ax.set_ylim(min(ys) - 0.8, 0.7)
    ax.set_xlim(lo_min - 0.03, max(rows["hi"].max(), 0) + 0.03)
    ax.set_xlabel("Change in AUC against Tier 1 (paired, 95% interval)", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], marker="o", ls="", color=VERDICT_COLOR[v], label=lab)
               for v, lab in [("below", "interval entirely below 0"),
                              ("spans zero", "interval spans 0")]
               if (rows["verdict"] == v).any()]
    if (rows["verdict"] == "above").any():
        handles.append(Line2D([], [], marker="o", ls="", color=VERDICT_COLOR["above"],
                              label="interval entirely above 0"))
    if (rows["copy"] != "").any():
        handles.append(Line2D([], [], marker="o", ls="", mfc="white", color=CONTROL_GREY,
                              label="† same numbers as an earlier tier"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)
    above = "none above" if counts["above"] == 0 else f"{counts['above']} above"
    lead = ("Adding agents lowers AUC against Tier 1:" if (rows["delta"] < 0).all()
            else "Adding agents, against Tier 1:")
    fig.suptitle(f"{lead}\n{counts['below']} of "
                 f"{counts['distinct']} distinct comparisons clearly below, {above}",
                 x=0.02, ha="left", fontsize=11.5, fontweight="bold")
    _footer(fig, "data: eval/results_main.csv · 95% moving-block bootstrap, paired with "
                 "Tier 1 on the same days · Malacca has no positive test day: not drawn")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    return fig


def fig_r1(main: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """Draw R1; returns the files written."""
    return _save(figure_r1(main), out_dir, "R1_delta_auc_forest")


# ------------------------------------------------------------------ R2 figure
def figure_r2(no_eq1: pd.DataFrame, main: pd.DataFrame):
    """Dumbbells per region: AUC with Eq. 1 on and off for each tier, with the paired change."""
    _style()
    rows = eq1_rows(no_eq1, main)
    on_ci = _on_interval(main)
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(_W, 7.4), sharex=True)
    for ax, region in zip(axes, REGIONS):
        sub = rows[rows["region"] == region].reset_index(drop=True)
        ys = list(range(len(sub), 0, -1))
        for (_, r), yy in zip(sub.iterrows(), ys):
            lo_on, hi_on = on_ci.loc[(region, r["method"]), ["auc_ci_low", "auc_ci_high"]]
            ax.plot([r["auc_off"], r["auc_on"]], [yy, yy], color=CONTROL_GREY, lw=1.2, zorder=1)
            ax.hlines(yy + 0.16, lo_on, hi_on, color=PALETTE["blue"], lw=1.1, alpha=0.55)
            ax.hlines(yy - 0.16, r["off_lo"], r["off_hi"], color=PALETTE["vermillion"], lw=1.1,
                      alpha=0.55)
            ax.scatter([r["auc_on"]], [yy + 0.16], s=34, color=PALETTE["blue"], zorder=3)
            ax.scatter([r["auc_off"]], [yy - 0.16], s=34, color=PALETTE["vermillion"],
                       marker="D", zorder=3)
            ax.text(1.015, yy, interval(r["delta"], r["dlo"], r["dhi"], delta=True),
                    transform=ax.get_yaxis_transform(), fontsize=7.4, va="center")
        ax.axvline(0.5, color=CONTROL_GREY, lw=1, ls=":")
        ax.set_yticks(ys, [m + (" †" if c else "") for m, c in zip(sub["method"], sub["copy"])],
                      fontsize=8.5)
        ax.set_ylim(0.4, len(sub) + 0.7)
        ax.set_xlim(0.15, 1.0)
        ax.set_title(SHORT[region], loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8.5)
    axes[0].text(1.015, 1.12, "change, off − on", transform=axes[0].transAxes, fontsize=7.4,
                 color=CONTROL_GREY)
    axes[-1].set_xlabel("ROC AUC", fontsize=9.5)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=PALETTE["blue"], label="Eq. 1 duration term on"),
        Line2D([], [], marker="D", ls="", color=PALETTE["vermillion"],
               label="Eq. 1 duration term off"),
        Line2D([], [], color=CONTROL_GREY, ls=":", label="AUC 0.5 (chance)")],
        loc="upper left", ncol=3, fontsize=7.6, frameon=False, bbox_to_anchor=(0.02, 0.93))
    distinct = rows[rows["copy"] == ""]
    clear = int((distinct["dhi"] < 0).sum())
    fig.suptitle(f"Without Eq. 1's duration term the AUC falls clearly in {clear} of {len(distinct)} "
                 "distinct cells", x=0.02, ha="left", fontsize=11.5, fontweight="bold")
    block = no_eq1["bootstrap_block"].dropna().iloc[0]
    n_boot = no_eq1["bootstrap_n"].dropna().iloc[0]
    _footer(fig, f"data: eval/method_comparison_no_eq1.csv, eval/results_main.csv · 95% "
                 f"moving-block bootstrap ({int(block)}-day blocks, {int(n_boot)} resamples) · "
                 "† same numbers as an earlier tier · Malacca: not drawn")
    fig.tight_layout(rect=(0, 0.05, 0.80, 0.90), h_pad=1.6)
    return fig


def fig_r2(no_eq1: pd.DataFrame, main: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """Draw R2; returns the files written."""
    return _save(figure_r2(no_eq1, main), out_dir, "R2_eq1_ablation")


# ------------------------------------------------------------------ R3 figure
def figure_r3(ablation: pd.DataFrame):
    """Per region: each agent's AUC alone (line at 0.5) and the change when it is removed."""
    _style()
    single, loo, full = agent_rows(ablation)
    fig, axes = plt.subplots(len(REGIONS), 2, figsize=(_W, 7.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    for row, region in enumerate(REGIONS):
        left, right = axes[row]
        s = single[single["region"] == region].reset_index(drop=True)
        l = loo[loo["region"] == region].reset_index(drop=True)
        f = full[full["region"] == region].iloc[0]
        ys = list(range(len(s), 0, -1))
        names = [_agent_name(a) for a in s["agent"]]
        for (_, r), yy in zip(s.iterrows(), ys):
            c = AGENT_COLOR[r["agent"]]
            left.hlines(yy, r["lo"], r["hi"], color=c, lw=2.0)
            left.scatter([r["auc"]], [yy], s=40, color=c, zorder=3)
        left.axvline(0.5, color=CONTROL_GREY, lw=1.1, ls=":")
        left.axvline(f["auc"], color=CONTROL_GREY, lw=1.0, ls="--")
        left.set_yticks(ys, names, fontsize=8.5)
        left.set_ylim(0.4, len(s) + 0.6)
        left.set_xlim(0.15, 1.0)
        left.set_xlabel("AUC, agent alone", fontsize=8.5)
        left.set_title(SHORT[region], loc="left", fontsize=11, fontweight="bold")
        ys_l = list(range(len(l), 0, -1))
        for (_, r), yy in zip(l.iterrows(), ys_l):
            c = AGENT_COLOR[r["agent"]]
            right.hlines(yy, r["lo"], r["hi"], color=c, lw=2.0)
            right.scatter([r["delta"]], [yy], s=40, color=c, zorder=3)
        right.axvline(0, color=CONTROL_GREY, lw=1.1, ls="--")
        right.set_yticks(ys_l, [_agent_name(a) for a in l["agent"]], fontsize=8.5)
        right.set_ylim(0.4, len(l) + 0.6)
        right.set_xlabel("ΔAUC when the agent is removed", fontsize=8.5)
        for ax in (left, right):
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("What each agent contributes: alone, and when removed", x=0.02, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.legend(handles=[
        Line2D([], [], color=CONTROL_GREY, ls=":", label="AUC 0.5 (chance)"),
        Line2D([], [], color=CONTROL_GREY, ls="--",
               label="all enabled agents (left); no change (right)")],
        loc="upper left", ncol=2, fontsize=7.6, frameon=False, bbox_to_anchor=(0.02, 0.945))
    block = ablation["bootstrap_block"].dropna().iloc[0]
    n_boot = ablation["bootstrap_n"].dropna().iloc[0]
    _footer(fig, f"data: eval/agent_ablation.csv · 95% moving-block bootstrap "
                 f"({int(block)}-day blocks, {int(n_boot)} resamples) · right: above 0 means "
                 "the AUC rose without the agent, i.e. it was hurting · the right-hand scale differs by "
                 "region · Malacca: not drawn")
    fig.tight_layout(rect=(0, 0.05, 1, 0.915), h_pad=1.6, w_pad=1.2)
    return fig


def fig_r3(ablation: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """Draw R3; returns the files written."""
    return _save(figure_r3(ablation), out_dir, "R3_agent_contribution")


# ------------------------------------------------------------------ R4 figure
def figure_r4(split: pd.DataFrame, main: pd.DataFrame | None = None):
    """Tier 1 and Tier 4 AUC at each split date, one panel per region."""
    _style()
    rows = split_rows(split)
    dates = sorted(rows["split_date"].unique())
    base = main_split(split, main if main is not None else _read("results_main.csv"))
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(_W, 3.5), sharey=True)
    for ax, region in zip(axes, REGIONS):
        for tier, shift in (("Tier 1", -0.07), ("Tier 4", 0.07)):
            sub = rows[(rows["region"] == region) & (rows["method"] == tier)]
            xs = [dates.index(d) + shift for d in sub["split_date"]]
            c = TIER_COLOR[tier]
            ax.plot(xs, sub["auc"], color=c, lw=1.6, marker="o", ms=5, label=tier, zorder=3)
            ax.vlines(xs, sub["lo"], sub["hi"], color=c, lw=1.4, alpha=0.6, zorder=2)
        ax.axhline(0.5, color=CONTROL_GREY, lw=1.0, ls=":")
        ax.axvspan(dates.index(base) - 0.3, dates.index(base) + 0.3, color=GRID, alpha=0.6, zorder=0)
        ax.set_xticks(range(len(dates)), dates, rotation=35, ha="right", fontsize=7.6)
        ax.set_xlim(-0.5, len(dates) - 0.5)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(SHORT[region], loc="left", fontsize=10.5, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8.5)
    axes[0].set_ylabel("ROC AUC", fontsize=9.5)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    pairs = rows[rows["method"] == "Tier 4"]
    worse = int((pairs["delta"] < 0).sum())
    fig.suptitle(f"Tier 4's AUC is below Tier 1's in {worse} of {len(pairs)} region-date pairs",
                 x=0.02, ha="left", fontsize=11.5, fontweight="bold")
    block = split["bootstrap_block"].dropna().iloc[0]
    n_boot = split["bootstrap_n"].dropna().iloc[0]
    _footer(fig, f"data: eval/split_sensitivity.csv, eval/results_main.csv · grey band: the "
                 f"split of the main results · 95% moving-block bootstrap ({int(block)}-day "
                 f"blocks, {int(n_boot)} resamples) · Malacca: not drawn")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92), w_pad=1.0)
    return fig


def fig_r4(split: pd.DataFrame, out_dir: Path | None = None,
           main: pd.DataFrame | None = None) -> list[Path]:
    """Draw R4; returns the files written."""
    return _save(figure_r4(split, main), out_dir, "R4_split_sensitivity")


def draw_all(out_dir: Path | None = None) -> list[Path]:
    """Draw R1-R4 from the committed CSVs; returns the files written."""
    main = _read("results_main.csv")
    return (fig_r1(main, out_dir)
            + fig_r2(_read("method_comparison_no_eq1.csv"), main, out_dir)
            + fig_r3(_read("agent_ablation.csv"), out_dir)
            + fig_r4(_read("split_sensitivity.csv"), out_dir, main))


# ------------------------------------------------------------------ analysis text
def _where(region: str, method: str | None = None) -> str:
    return SHORT[region] + (f" {method}" if method else "")


def analysis_r1(main: pd.DataFrame) -> str:
    """Two or three sentences on R1, every number from ``results_main.csv``."""
    rows = forest_rows(main)
    c = tier_counts(main)
    above = "none above" if c["above"] == 0 else f"{c['above']} above"
    first = (f"Of the {c['compared']} tier comparisons against Tier 1, {c['distinct']} are distinct "
             f"(the others repeat the numbers of an earlier tier), and {c['below']} of "
             f"{c['distinct']} distinct comparisons are clearly below Tier 1 (the paired 95% "
             f"interval lies entirely below zero), {above}.")
    spans = rows[(rows["copy"] == "") & (rows["verdict"] == "spans zero")]
    low = rows.loc[rows["delta"].idxmin()]
    out = [first]
    narrow = rows[(rows["copy"] == "") & (rows["verdict"] == "below") & (rows["hi"] > -0.001)]
    if len(spans) or len(narrow):
        parts = []
        if len(spans):
            listed = "; ".join(f"{_where(r.region, r.method)} ({interval(r.delta, r.lo, r.hi, delta=True)})"
                               for r in spans.itertuples())
            parts.append(f"the distinct comparison{'s' if len(spans) > 1 else ''} whose interval "
                         f"includes zero: {listed}")
        if len(narrow):
            listed = "; ".join(f"{_where(r.region, r.method)} (upper bound {signed(r.hi)})"
                               for r in narrow.itertuples())
            parts.append(f"{'those' if len(narrow) > 1 else 'the one'} that excludes zero only "
                         f"narrowly: {listed}")
        out.append((parts[0][0].upper() + parts[0][1:] + ("; " + parts[1] if len(parts) > 1 else "")) + ".")
    out.append(f"The largest drop is {_where(low['region'], low['method'])} "
               f"({interval(low['delta'], low['lo'], low['hi'], delta=True)}).")
    return " ".join(out)


def analysis_r2(no_eq1: pd.DataFrame, main: pd.DataFrame) -> str:
    """Two or three sentences on R2, every number from the ablation file."""
    rows = eq1_rows(no_eq1, main)
    distinct = rows[rows["copy"] == ""]
    down = distinct[distinct["delta"] < 0]
    big = rows.loc[rows["delta"].idxmin()]
    clear = int((distinct["dhi"] < 0).sum())
    first = (f"Switching Eq. 1's duration term off lowers the AUC point estimate in {len(down)} of "
             f"{len(distinct)} distinct tier-region cells (a tier that copies an earlier one is not counted "
             f"twice), clearly (paired interval below zero) in {clear} of "
             f"{len(distinct)}; the largest change is {_where(big['region'], big['method'])}, "
             f"{plain(big['auc_on'])} to {plain(big['auc_off'])} "
             f"({interval(big['delta'], big['dlo'], big['dhi'], delta=True)}).")
    parts = []
    for region in REGIONS:
        sub = rows[rows["region"] == region]
        if len(sub) and (sub["off_hi"] < 0.5).all():
            parts.append(f"In {SHORT[region]} the interval of the AUC without Eq. 1 lies below 0.5 "
                         f"at every tier (highest upper bound {plain(sub['off_hi'].max())})")
        if len(sub) and sub["delta"].abs().max() < 0.005:
            parts.append(f"in {SHORT[region]} no tier's paired change exceeds about "
                         f"{sub['delta'].abs().max():.4f} in magnitude")
    return first + (" " + "; ".join(parts) + "." if parts else "")


def analysis_r3(ablation: pd.DataFrame) -> str:
    """Two or three sentences on R3, every number from ``agent_ablation.csv``."""
    single, loo, full = agent_rows(ablation)
    best = single.loc[single.groupby("region", sort=False)["auc"].idxmax()]
    named = ", ".join(f"{SHORT[r.region]} {plain(r.auc)}" for r in best.itertuples())
    if best["agent"].nunique() == 1:
        first = (f"{_agent_name(best['agent'].iloc[0]).capitalize()} alone is the best single "
                 f"agent in every region ({named}).")
    else:
        first = "The best single agent differs by region: " + ", ".join(
            f"{SHORT[r.region]} {_agent_name(r.agent)} {plain(r.auc)}" for r in best.itertuples()) + "."
    below = single[single["hi"] < 0.5]
    others = single[~single.index.isin(best.index)]
    clear = others[others["lo"] > 0.5]
    second = []
    if len(below):
        second.append("Single agents whose interval lies entirely below 0.5: " + "; ".join(
            f"{_agent_name(r.agent)} in {SHORT[r.region]} ({interval(r.auc, r.lo, r.hi)})"
            for r in below.itertuples()))
    if len(clear):
        second.append("other single agents with an interval entirely above 0.5: " + "; ".join(
            f"{_agent_name(r.agent)} in {SHORT[r.region]} ({interval(r.auc, r.lo, r.hi)})"
            for r in clear.itertuples()))
    up = loo[loo["lo"] > 0]
    dn = loo[loo["hi"] < 0]

    def listing(frame: pd.DataFrame) -> str:
        items = [f"{_agent_name(r.agent)} in {SHORT[r.region]} "
                 f"({interval(r.delta, r.lo, r.hi, delta=True)})" for r in frame.itertuples()]
        if not items:
            return "no agent"
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    third = (f"Removing an agent raises the fused AUC (interval above zero) for {listing(up)}; "
             f"it lowers it (interval below zero) for {listing(dn)}.")
    middle = []
    if second:
        text = "; ".join(second)
        middle.append(text[0].upper() + text[1:] + ".")
    return " ".join([first] + middle + [third])


def analysis_r4(split: pd.DataFrame) -> str:
    """Two or three sentences on R4, every number from ``split_sensitivity.csv``."""
    rows = split_rows(split)
    dates = sorted(rows["split_date"].unique())
    pairs = rows[rows["method"] == "Tier 4"]
    n_regions = rows["region"].nunique()
    below_pt = int((pairs["delta"] < 0).sum())
    below_ci = int((pairs["dhi"] < 0).sum())
    first = (f"Across the {len(dates)} split dates ({', '.join(dates)}) and the {n_regions} evaluable "
             f"regions, Tier 4's AUC is below Tier 1's in {below_pt} of {len(pairs)} region-date pairs, "
             f"clearly so (paired interval below zero) in {below_ci} of {len(pairs)}.")
    t1 = rows[rows["method"] == "Tier 1"]
    spread = t1.groupby("region")["auc"].agg(["min", "max"])
    spread["range"] = spread["max"] - spread["min"]
    wide = spread["range"].idxmax()
    second = (f"Tier 1's AUC moves by at most {plain(spread.loc[wide, 'range'])} across the "
              f"dates ({SHORT[wide]}, {plain(spread.loc[wide, 'min'])} to "
              f"{plain(spread.loc[wide, 'max'])}).")
    third = (f"The paired Tier 4 − Tier 1 change runs from {signed(pairs['delta'].min())} "
             f"({_where(pairs.loc[pairs['delta'].idxmin(), 'region'])}, "
             f"{pairs.loc[pairs['delta'].idxmin(), 'split_date']}) to "
             f"{signed(pairs['delta'].max())} ({_where(pairs.loc[pairs['delta'].idxmax(), 'region'])}, "
             f"{pairs.loc[pairs['delta'].idxmax(), 'split_date']}).")
    return " ".join([first, second, third])


def _no_figure_values(main: pd.DataFrame, bem: pd.DataFrame) -> str:
    """The Malacca false-alarm rates and the Bab el-Mandeb surrogate error, as text values."""
    m = main[(main["region"] == "malacca") & (main["row_type"] == "tier")
             & main["alert_rate"].notna()]
    lo, hi = m.loc[m["alert_rate"].idxmin()], m.loc[m["alert_rate"].idxmax()]
    days = int(m["n_test"].iloc[0])
    q = bem[bem["feature"].isna()].set_index("quantity")["value"]
    n_feat = bem.loc[bem["quantity"] == "outside", "feature"].nunique()
    return "\n".join([
        f"- **Malacca false alarms.** {days} test days, "
        f"{int(m['test_positives'].iloc[0])} positives, so every alert is a false alarm and no AUC "
        f"exists. The tiers alert on {plain(lo['alert_rate'])} of days at the lowest "
        f"({lo['method']}, interval [{plain(lo['alert_rate_ci_low'])}, "
        f"{plain(lo['alert_rate_ci_high'])}]) and {plain(hi['alert_rate'])} at the highest "
        f"({hi['method']}, [{plain(hi['alert_rate_ci_low'])}, {plain(hi['alert_rate_ci_high'])}]).",
        f"- **Bab el-Mandeb surrogate.** Held-out R² {plain(q['r2'])}; the bias is "
        f"{q['bias_share'] * 100:.1f}% of the squared error, and R² after subtracting the mean "
        f"bias is {plain(q['r2_debiased'])}. The target's mean moves from {plain(q['train_mean'])} "
        f"in training to {plain(q['target_mean'])} in the test window, and {q['any_outside'] * 100:.1f}% "
        f"of test days have at least one of the {n_feat} features checked outside its training range."])


def render_analysis(csv_dir: Path | None = None) -> str:
    """``eval/FIGURES_ANALYSIS.md``: the analysis of R1-R4, generated from the committed CSVs."""
    d = Path(csv_dir) if csv_dir is not None else CSV_DIR
    main = pd.read_csv(d / "results_main.csv")
    no_eq1 = pd.read_csv(d / "method_comparison_no_eq1.csv")
    ablation = pd.read_csv(d / "agent_ablation.csv")
    split = pd.read_csv(d / "split_sensitivity.csv")
    bem = pd.read_csv(d / "bem_surrogate.csv")
    return "\n".join([
        "# Figure analysis",
        "",
        "Generated by `python scripts/make_all_figures.py --write-analysis` from the committed "
        "CSVs named under each figure; no number here is typed. Do not edit: rerun the script. "
        "The figures are R1 to R4 in `eval/FIGURES.md`.",
        "",
        "## R1: change in AUC against Tier 1 (`eval/results_main.csv`)",
        "",
        analysis_r1(main),
        "",
        "## R2: Eq. 1 ablation (`eval/method_comparison_no_eq1.csv`)",
        "",
        analysis_r2(no_eq1, main),
        "",
        "## R3: agent contribution (`eval/agent_ablation.csv`)",
        "",
        analysis_r3(ablation),
        "",
        "## R4: split sensitivity (`eval/split_sensitivity.csv`)",
        "",
        analysis_r4(split),
        "",
        "## No figure, values for the text",
        "",
        "By decision these two have no figure; quote them from here or from "
        "`eval/RESULTS_FROZEN.md`.",
        "",
        _no_figure_values(main, bem),
        "",
    ])


def main() -> int:
    for path in draw_all():
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
