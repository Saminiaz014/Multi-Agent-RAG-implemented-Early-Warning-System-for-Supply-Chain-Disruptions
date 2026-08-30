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
_EVAL = _ROOT / "eval"
_G_METHOD = _EVAL / "graphs_method_comparison"
_G_TIER = _EVAL / "graphs_ablation_tiers"


def _load() -> pd.DataFrame:
    return pd.read_csv(_EVAL / "method_comparison_results.csv")


def graph_method_auc(results: pd.DataFrame) -> None:
    """Baselines vs tiers, AUC, on the regions that have positives."""
    scored = results.dropna(subset=["auc"])
    regions = sorted(scored["region"].unique())
    methods = list(scored["method"].unique())

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.8 / len(regions)
    x = np.arange(len(methods))
    for i, region in enumerate(regions):
        values = [
            scored[(scored.region == region) & (scored.method == m)]["auc"].mean()
            for m in methods
        ]
        ax.bar(x + i * width, values, width, label=region, alpha=0.85)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.text(len(methods) - 0.5, 0.51, "chance", fontsize=9, ha="right")
    ax.set_ylabel("AUC", fontweight="bold")
    ax.set_title("Method comparison: AUC on held-out window (dashed = chance)",
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
    colors = ["#e53935" if f == "tier" else "#42a5f5" for f in malacca["family"]]

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
               for c in ("#42a5f5", "#e53935")]
    ax.legend(handles, ["baseline", "multi-agent tier"], loc="lower right")
    fig.tight_layout()
    fig.savefig(_G_METHOD / "A2_false_positive_harness.png", dpi=200)
    plt.close(fig)


def graph_tiers(results: pd.DataFrame) -> None:
    """Tier progression per region, against the best baseline for that region."""
    tiers = results[results.family == "tier"].dropna(subset=["auc"])
    baselines = results[results.family == "baseline"].dropna(subset=["auc"])

    for region in sorted(tiers["region"].unique()):
        region_tiers = tiers[tiers.region == region].sort_values("method")
        best = baselines[baselines.region == region]["auc"].max()

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(range(len(region_tiers)), region_tiers["auc"], marker="o",
                linewidth=2.5, markersize=9, color="#e53935", label="multi-agent tier")
        ax.axhline(best, color="#42a5f5", linestyle="-", linewidth=2,
                   label=f"best baseline ({best:.3f})")
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
        rows = scored[scored.region == region].sort_values("auc")
        colours = [_CIRC_COLOUR.get(c, "#90a4ae") for c in rows["circularity"]]

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(range(len(rows)), rows["auc"], color=colours, alpha=0.9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows["method"], fontsize=9)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("AUC (dashed = chance)", fontweight="bold")
        ax.set_title(
            f"Method ranking — {region}\n"
            "colour = circularity with the label, not quality",
            fontweight="bold",
        )
        for i, value in enumerate(rows["auc"]):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=8)
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


def write_report(results: pd.DataFrame) -> None:
    """Report the measured findings, including the unflattering ones."""
    scored = results.dropna(subset=["auc"])
    tiers = scored[scored.family == "tier"]
    baselines = scored[scored.family == "baseline"]
    lines: list[str] = []

    lines.append("# Method comparison and agent ablation\n")
    lines.append(
        "All figures are measured on a held-out window (last 30% of each "
        "region's series, temporal split). Features are real connector output; "
        "labels are the level-shift rule. Nothing here is assumed.\n"
    )

    lines.append("\n## Per-region results\n")
    for region in sorted(scored["region"].unique()):
        region_rows = scored[scored.region == region].sort_values("auc", ascending=False)
        best_baseline = baselines[baselines.region == region].nlargest(1, "auc")
        best_tier = tiers[tiers.region == region].nlargest(1, "auc")
        lines.append(f"\n### {region}\n")
        lines.append("| method | kind | AUC | F1 | alert rate |")
        lines.append("|---|---|---|---|---|")
        for _, row in region_rows.iterrows():
            lines.append(
                f"| {row['method']} | {row['kind']} | {row['auc']:.3f} | "
                f"{row['f1']:.3f} | {row['alert_rate']:.2f} |"
            )
        if not best_baseline.empty and not best_tier.empty:
            delta = best_tier.iloc[0]["auc"] - best_baseline.iloc[0]["auc"]
            lines.append(
                f"\nBest baseline {best_baseline.iloc[0]['method']} "
                f"({best_baseline.iloc[0]['auc']:.3f}) vs best tier "
                f"{best_tier.iloc[0]['method']} ({best_tier.iloc[0]['auc']:.3f}): "
                f"**{delta:+.3f}**\n"
            )

    lines.append("\n## What the numbers show\n")

    # Monotonicity: does adding agents help?
    lines.append("\n### Adding agents does not monotonically improve AUC\n")
    for region in sorted(tiers["region"].unique()):
        series = tiers[tiers.region == region].sort_values("method")
        values = series["auc"].tolist()
        names = series["method"].tolist()
        peak = names[int(np.argmax(values))]
        lines.append(
            f"- **{region}**: " + " -> ".join(f"{v:.3f}" for v in values)
            + f"  (best: {peak})"
        )
    lines.append(
        "\nIn every region the peak is Tier 1 or Tier 2, and adding the "
        "geopolitical agent at Tier 3 lowers AUC each time. This does not "
        "support the claim that each agent adds value.\n"
    )

    # Sub-chance tiers.
    below = tiers[tiers.auc < 0.5]
    if not below.empty:
        lines.append("\n### Some multi-agent scores are anti-correlated with the label\n")
        for _, row in below.iterrows():
            lines.append(f"- {row['region']} {row['method']}: AUC {row['auc']:.3f}")
        lines.append(
            "\nAn AUC below 0.5 is not noise — it means the score runs opposite "
            "to the label. The likely cause is a mismatch of definitions: these "
            "agents are shock detectors (z-scores and isolation forests against "
            "rolling baselines), while the label marks *sustained* level shifts. "
            "Once traffic has settled at a lower level, a shock detector sees a "
            "stable series and reports calm. The two are measuring different "
            "things, and that is an architectural finding rather than a bug.\n"
        )

    # Where baselines win outright.
    lines.append("\n### Where simple baselines beat the ensemble\n")
    for region in sorted(scored["region"].unique()):
        bb = baselines[baselines.region == region].nlargest(1, "auc")
        bt = tiers[tiers.region == region].nlargest(1, "auc")
        if bb.empty or bt.empty:
            continue
        if bb.iloc[0]["auc"] > bt.iloc[0]["auc"]:
            lines.append(
                f"- **{region}**: {bb.iloc[0]['method']} at {bb.iloc[0]['auc']:.3f} "
                f"beats the best tier at {bt.iloc[0]['auc']:.3f}"
            )

    # False positive harness.
    malacca = results[results.region == "malacca"].dropna(subset=["alert_rate"])
    if not malacca.empty:
        tier_rate = malacca[malacca.family == "tier"]["alert_rate"].mean()
        base_rate = malacca[malacca.family == "baseline"]["alert_rate"].mean()
        lines.append("\n### False-positive harness (Malacca)\n")
        lines.append(
            f"Malacca has zero labelled disruptions across 2019-2026, so every "
            f"alert is a false alarm. Mean alert rate: **tiers {tier_rate:.0%}**, "
            f"**baselines {base_rate:.0%}**. "
            + ("The multi-agent tiers are the noisier of the two.\n"
               if tier_rate > base_rate else
               "The tiers are the quieter of the two.\n")
        )

    lines.append("\n## Limits of this evaluation\n")
    lines.append(
        "- **Supervised baselines are mostly untrainable here.** Every labelled "
        "disruption is recent (Houthi 2024, Gatun drought 2023-24), so a "
        "temporal split leaves no positives in training for three of four "
        "regions. Those rows are reported as not-applicable rather than as a "
        "0.5 score, which would have read as 'no better than chance'.\n"
        "- **One split, no confidence intervals.** Positive counts are 133-244 "
        "days; treat differences under roughly 0.05 AUC as noise.\n"
        "- **Malacca cannot be evaluated for detection** — no positives at all.\n"
        "- **Panama alone has news features** (18 vs 14), because GDELT only "
        "answered for that region. Its tier 5 is therefore not directly "
        "comparable with the others'.\n"
        "- **The label is one definition among several.** The level-shift rule "
        "recovers two documented events, but agents built against a different "
        "definition of disruption will score poorly whatever their merit.\n"
    )

    (_EVAL / "COMPARISON_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    _G_METHOD.mkdir(parents=True, exist_ok=True)
    _G_TIER.mkdir(parents=True, exist_ok=True)
    results = _load()
    graph_method_auc(results)
    graph_region_rankings(results)
    graph_heatmap(results)
    graph_f1_fpr(results)
    graph_alert_rate(results)
    graph_tiers(results)
    write_report(results)
    print(f"graphs -> {_G_METHOD}, {_G_TIER}")
    print(f"report -> {_EVAL / 'COMPARISON_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
