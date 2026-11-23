"""Thesis figure D1 as a full-page 2x2 grid: AUC by tier per region, alert rate for Malacca.

Drawn from ``eval/method_comparison_results.csv`` alone. Everything the figure
states comes from that file: the lines, the dashed reference, the test window in
the footer, and the headline (which says "lowers" only if Tier 5 is below Tier 1
in every evaluable region).

Writes ``thesis/figures/ch4/D1_tier_progression_2x2.{pdf,png}``;
``scripts/make_all_figures.py`` runs it.

Usage: ``python eval/ch4/fig_tiers_2x2.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
THESIS_FIGURES = ROOT / "thesis" / "figures" / "ch4"
RESULTS = ROOT / "eval" / "method_comparison_results.csv"
FIGURE_NAME = "D1_tier_progression_2x2"
REGIONS = ["hormuz", "bab_el_mandeb", "panama", "malacca"]
SHORT = {"hormuz": "Hormuz", "bab_el_mandeb": "Bab el-Mandeb", "panama": "Panama",
         "malacca": "Malacca"}
LINE = "#0072B2"   # shipping blue, as in the original figure
MUTED = "#777777"
TIERS = [f"Tier {i}" for i in range(1, 6)]
#: Rows that are not methods a tier is compared with.
NOT_OTHER_METHODS = ["multi-agent", "oracle", "control"]


def headline(csv: pd.DataFrame) -> str:
    """The title, claiming only what the results support.

    "Lowers discrimination in every evaluable region" needs Tier 5 below
    Tier 1 in each and, where the file has the paired Tier 5 - Tier 1
    intervals (``delta_auc_ci_high``), every interval below 0. If some
    include 0 it says AUC (point estimates) and how many are clear.
    """
    rows = csv[csv["method"].isin(["Tier 1", "Tier 5"]) & csv["auc"].notna()]
    wide = rows.pivot(index="region", columns="method", values="auc")
    if not len(wide) or not (wide["Tier 5"] < wide["Tier 1"]).all():
        return "Adding agents does not lift discrimination consistently"
    if "delta_auc_ci_high" not in csv.columns:
        return "Adding agents lowers AUC in every evaluable region"
    high = csv[(csv["method"] == "Tier 5") & csv["region"].isin(wide.index)].set_index("region")[
        "delta_auc_ci_high"]
    clear = int((high < 0).sum())
    if clear == len(wide):
        return "Adding agents lowers discrimination in every evaluable region"
    return (f"Adding agents lowers AUC in every evaluable region "
            f"(clearly in {clear} of {len(wide)})")


def footer(csv: pd.DataFrame) -> str:
    return (f"data: eval/method_comparison_results.csv · test window "
            f"{csv['test_start'].min()} to {csv['test_end'].max()}")


def draw(csv: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """Draw the figure from a results table; returns the files written."""
    out_dir = Path(out_dir) if out_dir is not None else THESIS_FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#E6E6E6", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 7.6))
    for ax, region in zip(axes.flat, REGIONS):
        sub = csv[csv.region == region].set_index("method")
        metric = "alert_rate" if region == "malacca" else "auc"
        vals = [sub.loc[t, metric] for t in TIERS]
        ax.plot(range(1, 6), vals, marker="o", color=LINE, lw=2.2, ms=7)
        base = sub[~sub.kind.isin(NOT_OTHER_METHODS)][metric].dropna()
        if region == "malacca":
            ax.axhline(base.min(), ls="--", color=MUTED, lw=1.1)
            ax.text(5, base.min() + 0.006, f"lowest other: {base.min():.2f}",
                    ha="right", va="bottom", fontsize=8.5, color=MUTED)
            ax.set_ylim(0, max(0.1, max(vals) * 1.2))        # tiers can exceed any fixed limit
            ax.set_ylabel("Alert rate (lower is better)", fontsize=10)
        else:
            best = base.idxmax()
            name = best.split(" ", 1)[1].replace("logistic regression", "logistic reg.")
            ax.axhline(base.max(), ls="--", color=MUTED, lw=1.1)
            ax.text(5, base.max() + 0.012, f"best other: {base.max():.2f} ({name})",
                    ha="right", va="bottom", fontsize=8.5, color=MUTED)
            ax.axhline(0.5, ls=":", color=MUTED, lw=1)
            ax.set_ylim(0.3, 1.08)
            ax.set_ylabel("ROC AUC", fontsize=10)
        for i, v in enumerate(vals, 1):
            if i in (1, 5):
                ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                            xytext=(6, 6) if i == 1 else (0, -15),
                            ha="left" if i == 1 else "center", fontsize=9.5)
        ax.set_xticks(range(1, 6), ["T1", "T2", "T3", "T4", "T5"], fontsize=9.5)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(SHORT[region], loc="left", fontsize=12, fontweight="bold")
    fig.suptitle(headline(csv), x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.005, footer(csv), fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=2.2, w_pad=2.0)
    paths = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{FIGURE_NAME}.{ext}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> int:
    for path in draw(pd.read_csv(RESULTS)):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
