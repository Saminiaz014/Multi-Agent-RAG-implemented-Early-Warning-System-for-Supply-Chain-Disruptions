"""Graphs and report for the method comparison, read from measured results.

Every number here comes from ``eval/method_comparison_results.csv``. Nothing
is asserted that the CSV does not contain, and the findings section is written
from what the numbers show rather than from what the architecture predicts.

Usage::

    python scripts/report_method_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_EVAL = _ROOT / "eval"
_G_METHOD = _EVAL / "graphs_method_comparison"
_G_TIER = _EVAL / "graphs_ablation_tiers"


def _load() -> pd.DataFrame:
    return pd.read_csv(_EVAL / "method_comparison_results.csv")


def _has_ci(results: pd.DataFrame) -> bool:
    return {"auc_ci_low", "auc_ci_high"} <= set(results.columns)


def _ci_err(rows: pd.DataFrame) -> np.ndarray | None:
    """Asymmetric AUC error bars (2 x n) from the CI columns, or None."""
    if not _has_ci(rows):
        return None
    low = (rows["auc"] - rows["auc_ci_low"]).clip(lower=0).fillna(0).to_numpy()
    high = (rows["auc_ci_high"] - rows["auc"]).clip(lower=0).fillna(0).to_numpy()
    return np.vstack([low, high])


def _auc_cell(row: pd.Series) -> str:
    if pd.notna(row.get("auc_ci_low")) and pd.notna(row.get("auc_ci_high")):
        return f"{row['auc']:.3f} [{row['auc_ci_low']:.3f}, {row['auc_ci_high']:.3f}]"
    return f"{row['auc']:.3f}"


#: Reference rows rather than methods. The oracle reads yesterday's label and
#: the controls are constants or noise, so none is a method a tier can be
#: judged against.
_NOT_METHODS = ("oracle", "control")


def _real_baselines(rows: pd.DataFrame) -> pd.DataFrame:
    """Baseline rows that are detection methods: no oracle, no controls."""
    return rows[(rows["family"] == "baseline") & ~rows["kind"].isin(_NOT_METHODS)]


def _bootstrap_settings() -> dict | None:
    """settings.yaml ``evaluation.bootstrap``, or None if it cannot be read."""
    try:
        from src.evaluation.window import bootstrap_config
        return bootstrap_config()
    except (ImportError, KeyError, FileNotFoundError):
        return None


def _ci_pct() -> str:
    cfg = _bootstrap_settings()
    return f"{cfg['ci']:.0%}" if cfg else "95%"


def _separation(a: pd.Series, b: pd.Series) -> str:
    """How two methods' AUC intervals relate.

    Non-overlap shows a clear difference. Overlap does not show the two are
    the same: with paired resamples the interval of the *difference* can be
    much narrower than either marginal one.
    """
    keys = ("auc_ci_low", "auc_ci_high")
    if not all(pd.notna(r.get(k)) for r in (a, b) for k in keys):
        return ""
    apart = a["auc_ci_low"] > b["auc_ci_high"] or b["auc_ci_low"] > a["auc_ci_high"]
    return (" The intervals do not overlap: a clear difference." if apart else
            " The intervals overlap, which on its own is inconclusive: it neither "
            "shows nor rules out a difference.")


def graph_method_auc(results: pd.DataFrame) -> None:
    """Baselines vs tiers, AUC, on the regions that have positives."""
    scored = results.dropna(subset=["auc"])
    regions = sorted(scored["region"].unique())
    methods = list(scored["method"].unique())

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.8 / len(regions)
    x = np.arange(len(methods))
    for i, region in enumerate(regions):
        picked = [scored[(scored.region == region) & (scored.method == m)].head(1)
                  for m in methods]
        values = [p["auc"].iloc[0] if len(p) else np.nan for p in picked]
        err = None
        if _has_ci(scored):
            err = np.hstack([_ci_err(p) if len(p) else np.zeros((2, 1)) for p in picked])
        ax.bar(x + i * width, values, width, label=region, alpha=0.85,
               yerr=err, capsize=2, error_kw={"linewidth": 0.8})

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.text(len(methods) - 0.5, 0.51, "chance", fontsize=9, ha="right")
    ax.set_ylabel("AUC", fontweight="bold")
    ax.set_title("Method comparison: AUC on held-out window (dashed = chance"
                 + (f"; bars = {_ci_pct()} block-bootstrap CI)" if _has_ci(scored) else ")"),
                 fontweight="bold")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_G_METHOD / "A1_auc_by_method.png", dpi=200)
    plt.close(fig)


def graph_alert_rate(results: pd.DataFrame) -> None:
    """False-positive harness: how often each method fires in Malacca."""
    malacca = results[results.region == "malacca"].dropna(subset=["alert_rate"])
    if malacca.empty:
        return
    malacca = malacca.sort_values("alert_rate")
    # Controls and the ceiling are reference rows, not baselines: grey.
    colors = ["#e53935" if f == "tier"
              else "#90a4ae" if f == "ceiling" or k in _NOT_METHODS else "#42a5f5"
              for f, k in zip(malacca["family"], malacca["kind"])]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(range(len(malacca)), malacca["alert_rate"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(malacca)))
    ax.set_yticklabels(malacca["method"], fontsize=9)
    ax.set_xlabel("Fraction of days alerting (Malacca: zero real disruptions)",
                  fontweight="bold")
    ax.set_title("False-positive harness — every alert here is a false alarm",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in ("#42a5f5", "#e53935", "#90a4ae")]
    ax.legend(handles, ["baseline", "multi-agent tier", "control / ceiling (oracle)"],
              loc="lower right")
    fig.tight_layout()
    fig.savefig(_G_METHOD / "A2_false_positive_harness.png", dpi=200)
    plt.close(fig)


def graph_tiers(results: pd.DataFrame) -> None:
    """Tier progression per region, against the best other method there."""
    tiers = results[results.family == "tier"].dropna(subset=["auc"])
    baselines = _real_baselines(results.dropna(subset=["auc"]))

    for region in sorted(tiers["region"].unique()):
        region_tiers = tiers[tiers.region == region].sort_values("method")
        best = baselines[baselines.region == region]["auc"].max()

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar(range(len(region_tiers)), region_tiers["auc"],
                    yerr=_ci_err(region_tiers), marker="o", linewidth=2.5,
                    markersize=9, capsize=5, color="#e53935",
                    label="multi-agent tier"
                    + (f" ({_ci_pct()} block-bootstrap CI)" if _has_ci(region_tiers) else ""))
        if pd.notna(best):
            ax.axhline(best, color="#42a5f5", linestyle="-", linewidth=2,
                       label=f"best other method ({best:.3f}; no ceiling, no controls)")
        top = results[(results.region == region) & _is_ceiling(results)]["auc"].dropna()
        if not top.empty:
            ax.axhline(float(top.iloc[0]), color="#90a4ae", linestyle=":", linewidth=2,
                       label=f"ceiling: oracle, yesterday's label ({top.iloc[0]:.3f})")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")

        ax.set_xticks(range(len(region_tiers)))
        ax.set_xticklabels(region_tiers["method"], fontsize=9)
        ax.set_ylabel("AUC", fontweight="bold")
        ax.set_title(f"Agent ablation — {region}", fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(_G_TIER / f"B_tier_progression_{region}.png", dpi=200)
        plt.close(fig)


#: Bar colour by how circular a method is with the label. A ranking that
#: shows only height invites reading the most label-adjacent method as the
#: best one, which is exactly backwards.
_CIRC_COLOUR = {
    "high": "#e53935", "medium": "#fb8c00", "low": "#43a047",
    "n/a": "#90a4ae", "unknown": "#90a4ae",
}


def graph_region_rankings(results: pd.DataFrame) -> None:
    """One ranking per region, coloured by circularity with the label."""
    scored = results.dropna(subset=["auc"])
    for region in sorted(scored["region"].unique()):
        in_region = scored[scored.region == region]
        # The ceiling is a reference line, not a ranked bar.
        top = in_region[_is_ceiling(in_region)]
        rows = in_region[~_is_ceiling(in_region)].sort_values("auc")
        colours = [_CIRC_COLOUR.get(c, "#90a4ae") for c in rows["circularity"]]

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(range(len(rows)), rows["auc"], color=colours, alpha=0.9,
                xerr=_ci_err(rows), capsize=2, error_kw={"linewidth": 0.8})
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows["method"], fontsize=9)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        for _, row in top.iterrows():
            ax.axvline(row["auc"], color="#90a4ae", linestyle=":", linewidth=2)
            ax.text(row["auc"], len(rows) - 0.4, f" ceiling: {row['method']} "
                    f"({row['auc']:.3f})", fontsize=8, ha="right", color="#607d8b")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("AUC (dashed = chance)", fontweight="bold")
        ax.set_title(
            f"Method ranking — {region}\n"
            "colour = circularity with the label, not quality",
            fontweight="bold",
        )
        right = rows["auc_ci_high"].fillna(rows["auc"]) if _has_ci(rows) else rows["auc"]
        for i, (value, edge) in enumerate(zip(rows["auc"], right)):
            ax.text(max(value, edge) + 0.01, i, f"{value:.3f}", va="center", fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=_CIRC_COLOUR[k])
                   for k in ("high", "medium", "low", "n/a")]
        ax.legend(handles, ["high circularity", "medium", "low", "control/oracle"],
                  loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(_G_METHOD / f"C_ranking_{region}.png", dpi=200)
        plt.close(fig)


def graph_heatmap(results: pd.DataFrame) -> None:
    """AUC across every method and region, matplotlib only (no seaborn)."""
    scored = results.dropna(subset=["auc"])
    methods = sorted(scored["method"].unique())
    regions = sorted(results["region"].unique())
    grid = np.full((len(methods), len(regions)), np.nan)
    for i, method in enumerate(methods):
        for j, region in enumerate(regions):
            match = scored[(scored.method == method) & (scored.region == region)]
            if not match.empty:
                grid[i, j] = match["auc"].mean()

    fig, ax = plt.subplots(figsize=(8, 10))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eceff1")
    image = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    for i in range(len(methods)):
        for j in range(len(regions)):
            label = "n/a" if np.isnan(grid[i, j]) else f"{grid[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    ax.set_title("AUC by method and region\n"
                 "(malacca: no positives, AUC undefined)",
                 fontweight="bold")
    fig.colorbar(image, ax=ax, label="AUC")
    fig.tight_layout()
    fig.savefig(_G_METHOD / "D_auc_heatmap.png", dpi=200)
    plt.close(fig)


def graph_f1_fpr(results: pd.DataFrame) -> None:
    """F1 against FPR per region: the operating-point trade-off."""
    scored = results.dropna(subset=["f1", "fpr"])
    regions = sorted(scored["region"].unique())
    if not regions:
        return
    fig, axes = plt.subplots(1, len(regions), figsize=(6 * len(regions), 5.5),
                             squeeze=False)
    for ax, region in zip(axes[0], regions):
        rows = scored[scored.region == region]
        for _, row in rows.iterrows():
            ax.scatter(row["f1"], row["fpr"], s=90, alpha=0.85,
                       color=_CIRC_COLOUR.get(row["circularity"], "#90a4ae"))
            ax.annotate(row["method"], (row["f1"], row["fpr"]),
                        fontsize=7, alpha=0.8,
                        xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("F1", fontweight="bold")
        ax.set_ylabel("FPR", fontweight="bold")
        ax.set_title(region, fontweight="bold")
        ax.grid(alpha=0.3)
    fig.suptitle("Operating point: F1 vs false-positive rate (top-left is better)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(_G_METHOD / "E_f1_vs_fpr.png", dpi=200)
    plt.close(fig)


def _has_delta(rows: pd.DataFrame) -> bool:
    return {"delta_auc", "delta_auc_ci_low", "delta_auc_ci_high"} <= set(rows.columns)


def _delta_cell(row: pd.Series) -> str:
    if pd.isna(row.get("delta_auc")):
        return "—"
    cell = (f"{row['delta_auc']:+.3f} [{row['delta_auc_ci_low']:+.3f}, "
            f"{row['delta_auc_ci_high']:+.3f}]")
    return f"**{cell}**" if _delta_verdict(row) != "inconclusive" else cell


def _delta_verdict(row: pd.Series) -> str:
    """'below' / 'above' when the paired interval excludes 0, else 'inconclusive'."""
    if row["delta_auc_ci_high"] < 0:
        return "below"
    if row["delta_auc_ci_low"] > 0:
        return "above"
    return "inconclusive"


def _settings_used(results: pd.DataFrame) -> dict | None:
    """The bootstrap settings the CSV was produced with, else the current ones."""
    cols = {"bootstrap_block", "bootstrap_n", "bootstrap_ci"}
    if cols <= set(results.columns) and results["bootstrap_n"].notna().any():
        row = results.dropna(subset=["bootstrap_n"]).iloc[0]
        return {"block": int(row["bootstrap_block"]), "n_resamples": int(row["bootstrap_n"]),
                "ci": float(row["bootstrap_ci"])}
    return _bootstrap_settings()


def _ci_note(results: pd.DataFrame) -> str:
    cfg = _settings_used(results)
    how = (f"{cfg['block']}-day blocks, {cfg['n_resamples']:,} resamples" if cfg
           else "settings.yaml `evaluation.bootstrap`")
    valid = ""
    scored = results.dropna(subset=["auc"])
    if cfg and "bootstrap_valid" in scored.columns and not scored.empty:
        valid = (" AUC and F1 intervals use only resamples that contain both "
                 f"classes: at least {int(scored['bootstrap_valid'].min()):,} of "
                 f"{cfg['n_resamples']:,} for every scored method.")
    pct = f"{cfg['ci']:.0%}" if cfg else "95%"
    paired = (" Tier differences use the paired interval of the difference "
              "(the same resampled blocks for both tiers); a paired interval "
              "that excludes 0 is a clear difference." if _has_delta(results) else "")
    return (
        f"- **Intervals are {pct} moving-block bootstrap CIs** ({how}) on "
        "the test window, with the training threshold (or a row's direct "
        "alerts) held fixed. Every method "
        "in a region gets the same resamples, so intervals are paired. "
        "Non-overlapping intervals show a clear difference; overlapping ones "
        "are inconclusive on their own." + paired + valid + " They cover test-sample "
        "uncertainty only, not the choice of split or label, and they are "
        "conditional on the few disruption episodes in each test window, so "
        "they do not measure skill on a new kind of disruption.\n"
    )


def _is_ceiling(rows: pd.DataFrame) -> pd.Series:
    return (rows["family"] == "ceiling") | (rows["kind"] == "oracle")


def _dating_note(results: pd.DataFrame) -> str:
    """The data-dating policy, stated only when the results CSV records it.

    The harness writes ``frame_policy`` after checking each frame's manifest,
    so a CSV produced from older frames makes no such claim.
    """
    if "frame_policy" not in results.columns or results["frame_policy"].isna().any():
        return ""
    return (
        "- **Values enter on the day they were known, where the source says.** "
        "FRED market series are stamped on their first-release date (ALFRED) "
        "and carried forward until the next release; gaps in other domains are "
        "forward-filled, never interpolated. Shipping (PortWatch) and conflict "
        "(ACLED) values stay on their event dates: their roughly weekly "
        "publication lag is not modelled. A few scaling constants (the freight "
        "index's rebase, the trade-volume and conflict-intensity scales) are "
        "still taken over the whole sample.\n"
    )


def _alert_rule_note(results: pd.DataFrame) -> str:
    """The alert rule, the controls defined directly, and tie-driven rows.

    Read from the CSV's ``alert_rule`` column, not hard-coded: rows whose
    rule starts ``direct:`` (the controls) set their alerts themselves. Of
    the others, under ``>=`` a constant score (or one at its floor on 90%+ of
    training days) alerts every day, so its F1 and alert rate are
    always-alert values; those are listed, not special-cased.
    """
    rule = (results["alert_rule"].fillna("") if "alert_rule" in results.columns
            else pd.Series("", index=results.index))
    is_direct = rule.str.startswith("direct:")
    direct = ""
    if is_direct.any():
        first = results[is_direct].groupby("method", sort=False)["alert_rule"].first()
        shown = [f"{m} ({r.removeprefix('direct:').strip()})" for m, r in first.items()]
        direct = (" These rows define their alerts directly instead: "
                  + "; ".join(shown) + ".")
    always = results[~is_direct & (results["alert_rate"] >= 1.0)]
    listed = ""
    if not always.empty:
        names = []
        for method, rows in always.groupby("method", sort=False):
            where = sorted(rows["region"].unique())
            span = ("all regions" if len(where) == results["region"].nunique()
                    else ", ".join(where))
            names.append(f"{method} ({span})")
        listed = (" These rows alert on every scored test day, so their F1 and "
                  "alert rate are always-alert values: " + "; ".join(names) + ".")
    return (
        "- **Alert rule:** a day alerts when its score >= the method's 90th "
        "percentile of training scores (masked days left out). A score tied "
        "with that threshold counts as an alert." + direct + listed + "\n"
    )


def _tier_k(method: str) -> int:
    return int(str(method).split()[-1])


def _repeats(results: pd.DataFrame) -> set[tuple[str, str]]:
    """``(region, method)`` of tiers that fuse exactly the previous tier's agents.

    A tier whose added agent is disabled in its region is a copy of the tier
    before it. It is marked in the tables and left out of every count and
    mean, so a copy is not read as more evidence. Needs ``agents``.
    """
    if "agents" not in results.columns:
        return set()
    out: set[tuple[str, str]] = set()
    tiers = results[results["family"] == "tier"]
    for region, rows in tiers.groupby("region"):
        rows = rows.iloc[np.argsort([_tier_k(m) for m in rows["method"]], kind="stable")]
        agents = rows["agents"].fillna("").tolist()
        out |= {(region, m) for m, a, prev in
                zip(rows["method"].iloc[1:], agents[1:], agents[:-1]) if a == prev}
    return out


def _label(row: pd.Series, repeats: set[tuple[str, str]]) -> str:
    """A row's method name, marked when it repeats the tier before it."""
    name = row["method"]
    if (row["region"], name) in repeats:
        return f"{name} (= Tier {_tier_k(name) - 1})"
    return name


def _disabled_note(results: pd.DataFrame) -> str:
    """Agents each region disables, and the tiers that therefore repeat.

    Read from the tier rows' ``agents`` / ``agents_disabled`` columns. A tier
    is named when it fuses exactly the agents of the tier before it.
    """
    if "agents_disabled" not in results.columns:
        return ""
    tiers = results[results["family"] == "tier"]
    disabled = []
    for region in sorted(tiers["region"].unique()):
        off = sorted({d for cell in tiers[tiers.region == region]["agents_disabled"].fillna("")
                      for d in str(cell).split(",") if d})
        if off:
            disabled.append(f"{region}: {', '.join(off)}")
    if not disabled:
        return ""
    repeats = [f"{r} {m}" for r, m in sorted(_repeats(results))]
    same = ("; these tiers therefore equal the tier before them: "
            + "; ".join(repeats) if repeats else "")
    return (
        "- **Agents disabled by the region config** (`agents.<name>.enabled: "
        "false`, on evidence grounds): " + "; ".join(disabled) + ". A disabled "
        "agent is not run, and each tier fuses its enabled agents with the "
        "weights renormalised over them" + same + ". "
        "`eval/method_comparison_all_agents.csv` re-scores every tier with every "
        "agent enabled, as an appendix sensitivity, not the main result.\n"
    )


def _masked_note(results: pd.DataFrame) -> str:
    """How many test days were masked (no news published) and left unscored."""
    if "n_masked" not in results.columns or not results["n_masked"].fillna(0).any():
        return ""
    per_region = results.groupby("region")["n_masked"].max().astype(int)
    counts = sorted(set(per_region))
    shown = (f"{counts[0]} test days in every region" if len(counts) == 1
             else ", ".join(f"{r} {n}" for r, n in per_region.items()) + " test days")
    return (f"- **Days GDELT never published are masked:** {shown} are left out "
            "of every method's score (point estimates and intervals) and out of "
            "every training threshold. They are not filled.\n")


def _sensitivity_section(sensitivity: pd.DataFrame) -> list[str]:
    """Tier 1 and Tier 4 at each split date, with the paired Tier 4 - Tier 1."""
    def cell(row, tier: int) -> str:
        shown = _auc_cell(row)
        n = row.get("n_agents")
        return shown + (f" ({int(n)} agents)" if pd.notna(n) and int(n) < tier else "")

    out = ["\n### Split-date sensitivity\n",
           "Tier 1 and Tier 4 re-scored with the split moved; the end date, the "
           "masked days and the training-window threshold rule are unchanged. "
           "The main tables use the settings.yaml split.\n",
           "| region | split | scored test days (positives) | Tier 1 AUC | "
           "Tier 4 AUC | Tier 4 - Tier 1 (paired) |",
           "|---|---|---|---|---|---|"]
    ranges: list[str] = []
    for region in sorted(sensitivity["region"].unique()):
        rows = sensitivity[sensitivity.region == region]
        for split in sorted(rows["split_date"].unique()):
            at = rows[rows.split_date == split].set_index("method")
            t1 = at.loc["Tier 1"] if "Tier 1" in at.index else None
            t4 = at.loc["Tier 4"] if "Tier 4" in at.index else None
            ref = t1 if t1 is not None else t4
            cells = [(cell(r, k) if r is not None and pd.notna(r.get("auc")) else "—")
                     for r, k in ((t1, 1), (t4, 4))]
            delta = (_delta_cell(t4) if t4 is not None and _has_delta(sensitivity)
                     else "—")
            out.append(f"| {region} | {split} | {int(ref['n_test'])} "
                       f"({int(ref['test_positives'])}) | {cells[0]} | {cells[1]} | "
                       f"{delta} |")
        t1_auc = rows[rows.method == "Tier 1"]["auc"].dropna()
        if len(t1_auc) > 1:
            ranges.append(f"{region}: Tier 1 AUC ranges {t1_auc.min():.3f}-"
                          f"{t1_auc.max():.3f}")
    if ranges:
        out.append("\n" + "; ".join(ranges) + ".\n")
    if "n_agents" in sensitivity.columns and (
            sensitivity["n_agents"] < sensitivity["method"].map(_tier_k)).any():
        out.append("A tier with fewer agents than its number (in brackets) is one "
                   "whose region disables an agent it would add.\n")
    return out


def write_report(results: pd.DataFrame, sensitivity: pd.DataFrame | None = None) -> None:
    """Report the measured findings, including the unflattering ones."""
    scored = results.dropna(subset=["auc"])
    tiers = scored[scored.family == "tier"]
    ceiling = scored[_is_ceiling(scored)]
    repeats = _repeats(results)
    lines: list[str] = []

    lines.append("# Method comparison and agent ablation\n")
    if {"test_start", "test_end"} <= set(results.columns):
        window = (f"{results['test_start'].min()} to {results['test_end'].max()}, "
                  "the same fixed dates for every region (settings.yaml "
                  "`evaluation`); every method is fitted only on earlier days")
    else:
        window = "a fixed temporal split"
    lines.append(
        f"All figures are measured on a held-out window: {window}. Features "
        "are real connector output; labels are the level-shift rule. Nothing "
        "here is assumed.\n"
    )
    lines.append(
        "**The complete results table is [`eval/results_main.md`](results_main.md)**: "
        "every method per region in one fixed order, copied tiers and the models that "
        "read the geopolitical columns marked, the ORACLE as a separate ceiling row, "
        "and the best non-ceiling method per region. The tables below rank the same "
        "figures by AUC.\n"
    )

    lines.append("\n## Per-region results\n")
    for region in sorted(scored["region"].unique()):
        in_region = scored[scored.region == region]
        region_rows = in_region[~_is_ceiling(in_region)].sort_values("auc", ascending=False)
        region_ceiling = ceiling[ceiling.region == region]
        lines.append(f"\n### {region}\n")
        auc_head = f"AUC [{_ci_pct()} CI]" if _has_ci(results) else "AUC"
        lines.append(f"| method | kind | {auc_head} | F1 | alert rate |")
        lines.append("|---|---|---|---|---|")
        for _, row in region_rows.iterrows():
            lines.append(
                f"| {_label(row, repeats)} | {row['kind']} | {_auc_cell(row)} | "
                f"{row['f1']:.3f} | {row['alert_rate']:.2f} |"
            )
        # The ceiling sits below the ranked methods, not among them.
        for _, row in region_ceiling.iterrows():
            lines.append(
                f"| *{row['method']} (ceiling, not a method)* | {row['kind']} | "
                f"{_auc_cell(row)} | {row['f1']:.3f} | {row['alert_rate']:.2f} |"
            )
        for _, row in region_ceiling.iterrows():
            lines.append(
                f"\nCeiling: {row['method']} ({_auc_cell(row)}) reads yesterday's "
                "true label, which no detector has at inference. It bounds what "
                "one-day-lagged knowledge of the answer would score.\n"
            )
    lines.append("\n## What the numbers show\n")

    # Monotonicity: does adding agents help? Every claim below is computed.
    lines.append("\n### Does adding agents improve AUC?\n")
    peaks: list[str] = []
    tier3_drops: list[bool] = []
    no_geo: list[str] = []
    steps = apart = 0
    for region in sorted(tiers["region"].unique()):
        series = tiers[tiers.region == region].sort_values("method")
        values = series["auc"].tolist()
        names = series["method"].tolist()
        peak = names[int(np.argmax(values))]
        peaks.append(peak)
        if len(values) >= 3:
            # Only where Tier 3 really adds geopolitical (it is not disabled).
            agents = (series["agents"].fillna("").str.split(",").tolist()
                      if "agents" in series.columns else None)
            if agents is None or ("geopolitical" in agents[2]
                                  and "geopolitical" not in agents[1]):
                tier3_drops.append(values[2] < values[1])
            else:
                no_geo.append(region)
        if _has_ci(series):
            for (_, a), (_, b) in zip(series.iloc[:-1].iterrows(), series.iloc[1:].iterrows()):
                if (region, b["method"]) in repeats:
                    continue
                steps += 1
                apart += "do not overlap" in _separation(a, b)
        marked = [f"{v:.3f}" + (" (=)" if (region, m) in repeats else "")
                  for v, m in zip(values, names)]
        lines.append(f"- **{region}**: " + " -> ".join(marked) + f"  (best: {peak})")
    if any(r in {region for region, _ in repeats} for r in tiers["region"].unique()):
        lines.append("\n(=): the tier repeats the one before it; the agent it would "
                     "add is disabled in that region.")
    with_delta = (tiers.dropna(subset=["delta_auc"]) if _has_delta(tiers)
                  else tiers.iloc[0:0])
    is_copy = pd.Series([(r, m) in repeats for r, m in
                         zip(with_delta["region"], with_delta["method"])],
                        index=with_delta.index, dtype=bool)
    compared = with_delta[~is_copy]
    copies = int(is_copy.sum())
    left_out = (f" ({copies} tier{' that repeats the tier before it is' if copies == 1 else 's that repeat the tier before them are'} "
                "not counted.)" if copies else "")
    hedge = (f" These are point estimates: consecutive tiers' intervals are "
             f"clearly apart in {apart} of {steps} steps, and overlap alone is "
             "inconclusive." if steps else "")
    if not compared.empty:
        verdicts = compared.apply(_delta_verdict, axis=1)
        n = len(compared)
        below_n, above_n = int((verdicts == "below").sum()), int((verdicts == "above").sum())
        hedge = (" These are point estimates. Tested against Tier 1 with the paired "
                 "difference on the same resampled blocks, a tier is clearly below "
                 f"Tier 1 in {below_n} of {n} comparisons, clearly above in "
                 f"{above_n} of {n}, and inconclusive (interval includes 0) in "
                 f"{n - below_n - above_n} of {n}.")
    hedge += left_out if hedge else ""
    if peaks and all(p in ("Tier 1", "Tier 2") for p in peaks):
        lines.append(
            "\nIn every region the peak is Tier 1 or Tier 2"
            + (", and adding the geopolitical agent at Tier 3 lowers AUC each "
               "time" + (f" (it is disabled in {', '.join(no_geo)})" if no_geo else "")
               if tier3_drops and all(tier3_drops) else "")
            + "." + hedge + " This does not support the claim that each agent "
            "adds value.\n"
        )
    else:
        lines.append(
            f"\nPeaks by region: {', '.join(peaks)}. The tier progression is "
            "not uniform across regions." + hedge + "\n"
        )
    if not with_delta.empty:
        tier_names = sorted(with_delta["method"].unique())
        lines.append("\n#### Paired ΔAUC against Tier 1 (Tier k - Tier 1)\n")
        lines.append("Bold: the paired interval excludes 0.\n")
        lines.append("| region | " + " | ".join(tier_names) + " |")
        lines.append("|---|" + "---|" * len(tier_names))
        for region in sorted(with_delta["region"].unique()):
            at = with_delta[with_delta.region == region].set_index("method")
            lines.append(f"| {region} | " + " | ".join(
                (_delta_cell(at.loc[t])
                 + (f" (= Tier {_tier_k(t) - 1})" if (region, t) in repeats else ""))
                if t in at.index else "—" for t in tier_names) + " |")
    if sensitivity is not None and not sensitivity.empty:
        lines += _sensitivity_section(sensitivity)

    # Sub-chance tiers.
    below = tiers[tiers.auc < 0.5]
    if _has_ci(results):
        # Only an interval wholly below 0.5 rules out chance.
        chance = below[~(below["auc_ci_high"] < 0.5)]
        below = below[below["auc_ci_high"] < 0.5]
        if not chance.empty:
            lines.append("\n### Tiers scoring below 0.5\n")
            lines.append(
                "Below 0.5 but indistinguishable from chance (the interval "
                "includes 0.5): "
                + ", ".join(f"{r['region']} {_label(r, repeats)} ({_auc_cell(r)})"
                            for _, r in chance.iterrows()) + ".\n"
            )
    if not below.empty:
        lines.append("\n### Some multi-agent scores are anti-correlated with the label\n")
        for _, row in below.iterrows():
            lines.append(f"- {row['region']} {_label(row, repeats)}: AUC {_auc_cell(row)}")
        lines.append(
            "\nAn AUC below 0.5"
            + (" with its whole interval below 0.5" if _has_ci(results) else "")
            + " is not noise — it means the score runs opposite "
            "to the label. The likely cause is a mismatch of definitions: these "
            "agents are shock detectors (z-scores and isolation forests against "
            "rolling baselines), while the label marks *sustained* level shifts. "
            "Once traffic has settled at a lower level, a shock detector sees a "
            "stable series and reports calm. The two are measuring different "
            "things, and that is an architectural finding rather than a bug.\n"
        )

    # False positive harness.
    malacca = results[results.region == "malacca"].dropna(subset=["alert_rate"])
    if not malacca.empty:
        distinct = np.array([(r, m) not in repeats
                             for r, m in zip(malacca["region"], malacca["method"])], dtype=bool)
        tier_rate = malacca[(malacca.family == "tier") & distinct]["alert_rate"].mean()
        base_rate = _real_baselines(malacca)["alert_rate"].mean()
        lines.append("\n### False-positive harness (Malacca)\n")
        lines.append(
            f"Malacca has zero labelled disruptions in the evaluated data, so every "
            f"alert is a false alarm. Mean alert rate: **tiers {tier_rate:.0%}**, "
            f"**baselines {base_rate:.0%}** (detection methods only; the oracle "
            "and controls are left out"
            + ("; a tier that repeats the one before it counts once"
               if any(r == "malacca" for r, _ in repeats) else "")
            + "). "
            + ("The multi-agent tiers are the noisier of the two.\n"
               if tier_rate > base_rate else
               "The tiers are the quieter of the two.\n")
        )

    n_regions = results["region"].nunique()
    untrainable = (
        results[results["not_applicable"].notna()]["region"].nunique()
        if "not_applicable" in results.columns else 0
    )
    lines.append("\n## Limits of this evaluation\n")
    lines.append(
        f"- **Supervised baselines are untrainable in {untrainable} of "
        f"{n_regions} regions.** Most labelled disruptions are recent (Houthi "
        "2024, Gatun drought 2023-24), so the temporal split leaves no "
        "positives in training there. Those rows are reported as "
        "not-applicable rather than as a 0.5 score, which would have read as "
        "'no better than chance'.\n"
        + (_ci_note(results) if _has_ci(results) else
           "- **One split, no confidence intervals.** Treat differences under "
           "roughly 0.05 AUC as noise.\n")
        + _alert_rule_note(results)
        + _disabled_note(results)
        + _masked_note(results)
        + "- **Malacca cannot be evaluated for detection** — no positives at all.\n"
        "- **News features come from GDELT 1.0 event exports** (corridor box "
        "filter v2) in every region; volume is articles per 100k GDELT "
        "events that day. The news agent's tone means skip days without "
        "articles instead of counting them as neutral tone.\n"
        + _dating_note(results)
        + "- **The label is one definition among several.** The level-shift rule "
        "recovers two documented events, but agents built against a different "
        "definition of disruption will score poorly whatever their merit.\n"
    )

    (_EVAL / "COMPARISON_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    _G_METHOD.mkdir(parents=True, exist_ok=True)
    _G_TIER.mkdir(parents=True, exist_ok=True)
    results = _load()
    sensitivity_csv = _EVAL / "split_sensitivity.csv"
    sensitivity = pd.read_csv(sensitivity_csv) if sensitivity_csv.exists() else None
    graph_method_auc(results)
    graph_region_rankings(results)
    graph_heatmap(results)
    graph_f1_fpr(results)
    graph_alert_rate(results)
    graph_tiers(results)
    write_report(results, sensitivity=sensitivity)
    print(f"graphs -> {_G_METHOD}, {_G_TIER}")
    print(f"report -> {_EVAL / 'COMPARISON_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
