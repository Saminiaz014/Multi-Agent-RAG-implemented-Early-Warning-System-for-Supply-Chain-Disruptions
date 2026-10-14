"""Rebuild every Chapter 4 explainability figure with one command.

By default this first regenerates the inputs — Alert Explanation Records
(``run_aer.py``), the scorecard (``run_scorecard.py``), the knowledge-base
backfill (``run_backfill.py``, from its raw-response cache) and precedent
grounding (``run_precedents.py``) — and then
draws each figure into ``thesis/figures/ch4/`` as PNG (300 dpi), PDF and a
``.caption.txt`` carrying the exact numbers shown.

Usage::

    python eval/explainability/make_figures.py            # rebuild inputs + figures
    python eval/explainability/make_figures.py --reuse    # figures from existing CSVs
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.viz import thesis_style as style  # noqa: E402  (sets the Agg backend)

import matplotlib.pyplot as plt  # noqa: E402

#: Regions in display order (positives first; Malacca is the FP harness).
REGION_ORDER = ("hormuz", "bab_el_mandeb", "panama", "malacca")


def _run(script: str) -> None:
    spec = importlib.util.spec_from_file_location(script, _HERE / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.main() != 0:
        raise SystemExit(f"{script} failed")


def tp_episode_note(composition: pd.DataFrame) -> str:
    """'TP episodes: 5 / 7 / 8 (Hormuz / Bab el-Mandeb / Panama)' from the data."""
    sub = composition[(composition.variant == "with_duration")
                      & (composition.episode_label == "TP")]
    counts = sub.groupby("region")["n_label_episodes"].first()
    regions = [r for r in REGION_ORDER if counts.get(r, 0) > 0]
    numbers = " / ".join(str(int(counts[r])) for r in regions)
    names = " / ".join(style.REGION_LABELS[r] for r in regions)
    return f"TP episodes: {numbers} ({names})"


def _leaders(data: pd.DataFrame) -> str:
    """One sentence per episode class naming its leading driver in each region."""
    parts = []
    for label in ("FP", "TP"):
        sub = data[(data.episode_label == label) & (data.n_label_episodes > 0)]
        named = []
        for region in REGION_ORDER:
            rows = sub[sub.region == region]
            if rows.empty:
                continue
            top = rows.sort_values("n_episodes", ascending=False).iloc[0]
            named.append(f"{style.REGION_LABELS[region]} "
                         f"{style.AGENT_LABELS[top.agent].lower()} "
                         f"({int(top.n_episodes)}/{int(top.n_label_episodes)})")
        parts.append(f"Leading driver of {label} episodes: " + "; ".join(named) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------- F14
def figure_f14(composition: pd.DataFrame) -> list[Path]:
    """Top exact driver of TP vs FP alert episodes, per region."""
    style.apply()
    data = composition[composition.variant == "with_duration"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.0), sharex=True)
    lines = []
    for ax, region in zip(axes.flat, REGION_ORDER):
        sub = data[data.region == region]
        ax.set_title(style.REGION_LABELS[region], loc="left")
        for y, label in enumerate(("FP", "TP")):
            rows = sub[sub.episode_label == label]
            total = int(rows["n_label_episodes"].iloc[0]) if len(rows) else 0
            if total == 0:
                ax.text(0.0, y, "  no TP episodes", va="center", fontsize=8, color=style.INK_MUTED)
                continue
            left = 0.0
            for agent in style.AGENT_ORDER:
                row = rows[rows.agent == agent]
                n = int(row["n_episodes"].iloc[0]) if len(row) else 0
                if n == 0:
                    continue
                width = n / total
                ax.barh(y, width, left=left, height=0.62, color=style.AGENT_COLORS[agent],
                        edgecolor="white", linewidth=1.5)
                if width >= 0.12:
                    ax.text(left + width / 2, y, str(n), ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
                left += width
            ax.text(1.02, y, f"n = {total}", va="center", fontsize=8, color=style.INK)
            breakdown = ", ".join(
                f"{style.AGENT_LABELS[a]} {int(r.n_episodes)}/{total} "
                f"({r.share:.0%}, 95% CI {r.share_ci_lo:.0%}–{r.share_ci_hi:.0%})"
                for a in style.AGENT_ORDER
                for r in rows[rows.agent == a].itertuples() if r.n_episodes > 0
            )
            lines.append(f"{style.REGION_LABELS[region]} {label}: {breakdown}")
        ax.set_yticks([0, 1], ["FP episodes", "TP episodes"])
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, 1.6)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.supxlabel("Share of episodes whose largest exact Shapley value belongs to the agent",
                  fontsize=9, color=style.INK)
    present = [a for a in style.AGENT_ORDER if a in set(data[data.n_episodes > 0].agent)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.AGENT_COLORS[a]) for a in present]
    fig.legend(handles, [style.AGENT_LABELS[a] for a in present], loc="upper center",
               ncol=len(present), bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    note = tp_episode_note(composition)
    style.add_provenance(fig, "real_eval", "D", f"Tier 5, train-p90 alerts · {note}")

    nodur = composition[(composition.variant == "without_duration")
                        & (composition.episode_label == "FP")
                        & (composition.agent == "geopolitical")]
    nodur_text = "; ".join(
        f"{style.REGION_LABELS[r.region]} {int(r.n_episodes)}/{int(r.n_label_episodes)}"
        for r in nodur.itertuples() if r.region in ("hormuz", "malacca"))
    caption = (
        "F14 — Which agent drives true- versus false-positive alert episodes. For every "
        "Tier 5 alert episode (consecutive train-p90 alert days, gaps of at most 3 days "
        "merged) the driver is the agent with the largest exact Shapley value summed over "
        "the episode's days (Path D composite, calm-day baselines). Bars give the share of "
        "TP and FP episodes per driver; segment numbers are episode counts. "
        + _leaders(data) + " Shares carry episode-level bootstrap 95% CIs (1,000 "
        "resamples).\n"
        + "\n".join(lines)
        + f"\nWithout the shipping duration branch, geopolitical-led FP episodes: {nodur_text}."
        + f"\n{note}. Data: real_eval (2023-08-31 to 2025-08-30 test window)."
    )
    return style.save(fig, "F14_fp_driver_composition", caption)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true",
                        help="draw from existing CSVs instead of regenerating them")
    args = parser.parse_args()
    if not args.reuse:
        _run("run_aer")
        _run("run_scorecard")
        _run("run_backfill")      # served from backfill_raw/ cache: no API quota spent
        _run("run_precedents")
        _run("run_triage_inputs")   # Phase 4 C0: bit-identity-checked export
        _run("run_triage")
        _run("run_metric8")         # Phase 5, synthetic seed 44
        _run("run_narratives")      # Phase 6: AER narratives + E9
    composition = pd.read_csv(_HERE / "fp_driver_composition.csv")
    written = figure_f14(composition)
    for path in written:
        print(f"wrote {path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
