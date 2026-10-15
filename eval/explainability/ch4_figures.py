"""Chapter 4 figures F1-F16 (F11 dropped), drawn from existing outputs only.

Called by ``make_figures.py``. Nothing here runs an agent, a retriever or an
API: every figure reads the CSV / JSON / JSONL artefacts of Phases 1-6 and
writes PNG (300 dpi), PDF and a ``.caption.txt`` with the exact numbers shown.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from src.viz import thesis_style as style

import matplotlib.pyplot as plt  # noqa: E402  (backend set by thesis_style)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

_OUT = Path(__file__).resolve().parent
REGIONS = ("hormuz", "bab_el_mandeb", "panama", "malacca")
LABELLED = ("hormuz", "bab_el_mandeb", "panama")
WEIGHTS = {"shipping": 0.25, "market": 0.15, "geopolitical": 0.25,
           "natural_disaster": 0.10, "news_sentiment": 0.10}
TP_NOTE = "TP episodes: 5 / 7 / 8 (Hormuz / Bab el-Mandeb / Panama); Malacca none"
METHOD_COLORS = {"exact": "#0072B2", "surrogate_prod": "#D55E00", "surrogate_refit": "#E69F00",
                 "random": "#8c8c8c"}
METHOD_LABELS = {"exact": "Exact Shapley", "surrogate_prod": "Surrogate (production)",
                 "surrogate_refit": "Surrogate (refit on real)", "random": "Random order"}


# ------------------------------------------------------------------ loaders
def _scorecard() -> pd.DataFrame:
    return pd.read_csv(_OUT / "scorecard.csv")


def _sc(sc: pd.DataFrame, region: str, metric: str, method: str = "exact",
        level: str = "episode", variant: str = "with_duration") -> pd.Series | None:
    rows = sc[(sc.region == region) & (sc.metric == metric) & (sc.method == method)
              & (sc.variant == variant) & (sc.level == level)]
    return None if rows.empty else rows.iloc[0]


def _days() -> tuple[pd.DataFrame, dict]:
    days = pd.read_csv(_OUT / "triage_inputs" / "days.csv", parse_dates=["date"])
    baselines = json.loads((_OUT / "triage_inputs" / "baselines.json").read_text(encoding="utf-8"))
    return days, baselines


def _agents(frame: pd.DataFrame) -> list[str]:
    return [c[6:] for c in frame.columns if c.startswith("score_") and frame[c].notna().all()]


def _phi(frame: pd.DataFrame, agents: list[str], base: dict) -> pd.DataFrame:
    mass = sum(WEIGHTS[a] for a in agents)
    return pd.DataFrame({a: WEIGHTS[a] * (frame[f"score_{a}"].to_numpy() - base[a]) / mass
                         for a in agents}, index=frame.index)


def _aers(region: str) -> list[dict]:
    return [json.loads(line) for line in (_OUT / "aer" / f"{region}.jsonl").open(encoding="utf-8")]


def _shade_truth(ax, dates: pd.Series, y: np.ndarray, color: str = "#e8e8e8") -> None:
    """Shade runs of labelled disruption days."""
    y = np.asarray(y).astype(bool)
    start = None
    for i, flag in enumerate(np.r_[y, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ax.axvspan(dates.iloc[start], dates.iloc[i - 1], color=color, lw=0, zorder=0)
            start = None


# ----------------------------------------------------------------------- F1
def f1_pipeline() -> list[Path]:
    style.apply()
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    boxes = [
        (0.1, "Five agent\nscores", "per day"),
        (1.75, "Exact Shapley\n(agents)", "32 coalitions,\ncalm baselines"),
        (3.4, "Within-agent\nbreakdown", "terms + branch,\nexact feature φ"),
        (5.05, "Precedents\n(wrapper)", "region, date < alert,\nde-duplicated"),
        (6.7, "Narrative", "template +\nclaim checker"),
        (8.35, "Action rule", "no action / monitor /\nreroute / escalate"),
    ]
    for x, title, sub in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.75), 1.45, 1.35, boxstyle="round,pad=0.03",
                                    fc="#f4f6fa", ec=style.INK_MUTED, lw=0.8))
        ax.text(x + 0.725, 2.68, title, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color=style.INK)
        ax.text(x + 0.725, 2.05, sub, ha="center", va="center", fontsize=5.9,
                color=style.INK_MUTED)
    for (x, *_), (nx, *_) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((x + 1.5, 2.42), (nx - 0.05, 2.42), arrowstyle="-|>",
                                     mutation_scale=9, color=style.INK_MUTED, lw=0.8))
    ax.add_patch(FancyBboxPatch((0.6, 0.2), 3.75, 0.95, boxstyle="round,pad=0.03",
                                fc="white", ec="#D55E00", lw=0.8, ls="--"))
    ax.text(2.475, 0.68, "SHAP surrogate (random forest, 17 features):\ncomparator only, never shown as the explanation",
            ha="center", va="center", fontsize=6.2, color="#D55E00")
    ax.add_patch(FancyArrowPatch((2.47, 1.72), (2.47, 1.18), arrowstyle="<|-|>",
                                 mutation_scale=8, color="#D55E00", lw=0.7, ls="--"))
    ax.text(4.9, 0.68, "Alert Explanation Record (AER): one per alert day,\n"
            "with uncertainty band and single-agent counterfactual",
            ha="left", va="center", fontsize=6.2, color=style.INK)
    caption = ("F1 — Explanation pipeline. Each alert day's composite (five agent scores) is "
               "decomposed exactly by Shapley values over the agents (efficiency residual "
               "≤ 1.1e-16), each agent's score is broken into its own terms, region- and "
               "date-filtered precedents are retrieved, and a deterministic narrative whose "
               "claims are checked against the record feeds the transparent action rule. The "
               "SHAP surrogate is retained only as a comparator.")
    return style.save(fig, "F1_explanation_pipeline", caption)


# ----------------------------------------------------------------------- F2
def f2_alert_anatomy() -> list[Path]:
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    lines = []
    for ax, region in zip(axes.flat, ("hormuz", "bab_el_mandeb", "panama", "malacca")):
        recs = _aers(region)
        want = "FP" if region == "malacca" else "TP"
        rec = max((r for r in recs if r["label"]["alert_correct"] == want), key=lambda r: r["margin"])
        order = sorted(rec["agent_shapley"], key=lambda a: -abs(rec["agent_shapley"][a]))
        level = rec["baseline_value"]
        ax.bar(0, level, color="#c8c8c8", width=0.6)
        ax.text(0, level + 0.01, f"{level:.3f}", ha="center", va="bottom", fontsize=7)
        for i, agent in enumerate(order, start=1):
            phi = rec["agent_shapley"][agent]
            ax.bar(i, phi, bottom=level, color=style.AGENT_COLORS[agent], width=0.6,
                   edgecolor="white", linewidth=1)
            ax.text(i, max(level, level + phi) + 0.01, f"{phi:+.3f}", ha="center",
                    va="bottom", fontsize=7)
            level += phi
        n = len(order) + 1
        ax.bar(n, rec["score"], color=style.INK, width=0.6)
        ax.text(n, rec["score"] + 0.01, f"{rec['score']:.3f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(rec["threshold"], color="#b00020", ls="--", lw=0.8)
        ax.text(n + 0.45, rec["threshold"], "threshold", color="#b00020", fontsize=6.5,
                va="bottom", ha="right")
        short = {"shipping": "Ship.", "market": "Mkt.", "geopolitical": "Geo.",
                 "natural_disaster": "Dis.", "news_sentiment": "News"}
        ax.set_xticks(range(n + 1), ["calm"] + [short[a] for a in order] + ["score"], fontsize=7)
        ax.set_ylim(0, max(rec["score"], rec["threshold"]) * 1.25)
        ax.set_title(f"{style.REGION_LABELS[region]} — {want}, {rec['date']}", loc="left",
                     fontsize=8.5)
        ax.grid(axis="x", visible=False)
        lines.append(f"{style.REGION_LABELS[region]} {rec['date']} ({want}): calm composite "
                     f"{rec['baseline_value']:.3f}, "
                     + ", ".join(f"{a} {rec['agent_shapley'][a]:+.3f}" for a in order)
                     + f" → score {rec['score']:.3f} vs threshold {rec['threshold']:.3f}")
    axes[0, 0].set_ylabel("Composite risk")
    axes[1, 0].set_ylabel("Composite risk")
    fig.tight_layout()
    style.add_provenance(fig, "real_eval", "D", "highest-margin TP alert day per region (Malacca: FP) · "
                         "Ship. shipping, Mkt. market, Geo. geopolitical, Dis. natural disaster")
    caption = ("F2 — Alert anatomy. Exact agent Shapley values move the composite from its "
               "calm-day value to the alert score (Path D, residual ≤ 1.1e-16); one TP alert "
               "per region, Malacca an FP (no TP exists).\n" + "\n".join(lines))
    return style.save(fig, "F2_alert_anatomy", caption)


# ----------------------------------------------------------------------- F3
def f3_attribution_timeline() -> list[Path]:
    style.apply()
    days, baselines = _days()
    episodes = pd.read_csv(_OUT / "alerts_episodes.csv", parse_dates=["start", "end"])
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 8.2), sharex=True)
    lines = []
    for ax, region in zip(axes, REGIONS):
        test = days[(days.region == region) & (days.variant == "with_duration")
                    & (days.window == "test")].reset_index(drop=True)
        agents = _agents(test)
        base = baselines["with_duration"][region]
        phi = _phi(test, agents, base)
        _shade_truth(ax, test.date, test.y_true.to_numpy())
        pos = np.zeros(len(test))
        neg = np.zeros(len(test))
        for agent in style.AGENT_ORDER:
            if agent not in phi:
                continue
            v = phi[agent].to_numpy()
            up, down = np.clip(v, 0, None), np.clip(v, None, 0)
            ax.fill_between(test.date, pos, pos + up, color=style.AGENT_COLORS[agent], lw=0,
                            label=style.AGENT_LABELS[agent], step="mid")
            ax.fill_between(test.date, neg, neg + down, color=style.AGENT_COLORS[agent], lw=0,
                            alpha=0.55, step="mid")
            pos, neg = pos + up, neg + down
        calm_value = sum(WEIGHTS[a] * base[a] for a in agents) / sum(WEIGHTS[a] for a in agents)
        line = float(test.threshold.iloc[0]) - calm_value
        ax.axhline(line, color="#b00020", lw=0.8, ls="--")
        ax.axhline(0, color=style.INK_MUTED, lw=0.5)
        ep = episodes[episodes.region == region]
        for row in ep.itertuples():
            ax.plot([row.start, row.end], [pos.max() * 1.05] * 2,
                    color="#b00020" if row.episode_label == "FP" else "#009E73", lw=2.2,
                    solid_capstyle="butt")
        ax.set_title(style.REGION_LABELS[region], loc="left", fontsize=8.5)
        ax.set_ylabel("φ (score − calm)", fontsize=7.5)
        lines.append(f"{style.REGION_LABELS[region]}: alert line at +{line:.3f} above the calm "
                     f"composite {calm_value:.3f}; {len(ep)} episodes "
                     f"({(ep.episode_label == 'TP').sum()} TP).")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    style.add_provenance(fig, "real_eval", "D", TP_NOTE)
    caption = ("F3 — Attribution timeline over the 2023-08-31 → 2025-08-30 test window. "
               "Stacked exact Shapley values per agent (above zero: pushes risk up; below: "
               "down, lighter), the alert line relative to the calm composite (red dashed), "
               "labelled disruption days (grey) and alert episodes (bars on top: green TP, "
               "red FP).\n" + "\n".join(lines) + f"\n{TP_NOTE}.")
    return style.save(fig, "F3_attribution_timeline", caption)


# ----------------------------------------------------------------------- F4
def _curves(scores: dict, base: dict, agents: list[str], order: list[str]):
    mass = sum(WEIGHTS[a] for a in agents)

    def value(present: frozenset) -> float:
        return sum(WEIGHTS[a] * (scores[a] if a in present else base[a]) for a in agents) / mass

    full, empty = value(frozenset(agents)), value(frozenset())
    gap = full - empty
    if gap < 0.01:
        return None
    dele = [(value(frozenset(order[k:])) - empty) / gap for k in range(len(order) + 1)]
    ins = [(value(frozenset(order[:k])) - empty) / gap for k in range(len(order) + 1)]
    return np.array(dele), np.array(ins)


def f4_deletion_insertion() -> list[Path]:
    style.apply()
    days, baselines = _days()
    sur = pd.read_csv(_OUT / "triage_inputs" / "surrogate.csv")
    sc = _scorecard()
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.2), sharey=True)
    lines = []
    for col, region in enumerate(REGIONS):
        test = days[(days.region == region) & (days.variant == "with_duration")
                    & (days.window == "test") & days.alert]
        agents = _agents(test)
        base = baselines["with_duration"][region]
        s_reg = sur[sur.region == region].set_index("date")
        acc = {m: ([], []) for m in ("exact", "surrogate_prod", "random")}
        for _, row in test.iterrows():
            scores = {a: float(row[f"score_{a}"]) for a in agents}
            phi = {a: WEIGHTS[a] * (scores[a] - base[a]) for a in agents}
            sp = {a: float(s_reg.loc[row.date.strftime("%Y-%m-%d"), f"phi_sur_{a}"]) for a in agents}
            orders = {"exact": sorted(agents, key=lambda a: -phi[a]),
                      "surrogate_prod": sorted(agents, key=lambda a: -sp[a])}
            for method, order in orders.items():
                c = _curves(scores, base, agents, order)
                if c is not None:
                    acc[method][0].append(c[0])
                    acc[method][1].append(c[1])
            rand = [_curves(scores, base, agents, list(p)) for p in permutations(agents)]
            rand = [r for r in rand if r is not None]
            if rand:
                acc["random"][0].append(np.mean([r[0] for r in rand], axis=0))
                acc["random"][1].append(np.mean([r[1] for r in rand], axis=0))
        x = np.linspace(0, 1, len(agents) + 1)
        for row_i, (kind, metric) in enumerate((("deletion", "agent deletion AUC (lower is better)"),
                                                ("insertion", "agent insertion AUC (higher is better)"))):
            ax = axes[row_i, col]
            for method in ("exact", "surrogate_prod", "random"):
                curves = acc[method][row_i]
                if not curves:
                    continue
                row = _sc(sc, region, metric, method)
                auc = f" ({row.value:.2f})" if row is not None else ""
                ax.plot(x, np.mean(curves, axis=0), marker="o", ms=3, lw=1.5,
                        color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            aucs = " · ".join(f"{k} {_sc(sc, region, metric, m).value:.2f}" for k, m in
                              (("E", "exact"), ("S", "surrogate_prod"), ("R", "random")))
            ax.set_title(f"{style.REGION_LABELS[region].replace('Strait of ', '')}, {kind}\nAUC {aucs}",
                         fontsize=7, loc="left")
            vals = {m: _sc(sc, region, metric, m) for m in ("exact", "surrogate_prod",
                                                            "surrogate_refit", "random")}
            lines.append(f"{style.REGION_LABELS[region]} {kind} AUC: " + ", ".join(
                f"{METHOD_LABELS[m]} {v.value:.3f} [{v.ci_lo:.3f}, {v.ci_hi:.3f}]"
                for m, v in vals.items() if v is not None))
    fig.supylabel("Share of the score departure kept", fontsize=8)
    fig.supxlabel("Share of agents removed (deletion) / restored (insertion)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.tight_layout()
    style.add_provenance(fig, "real_eval", "D", "mean curves over alert days; AUCs episode-level "
                         "(E exact, S production surrogate, R random)")
    caption = ("F4 — Deletion (top) and insertion (bottom) curves at agent level: agents are "
               "moved to (or restored from) their calm baseline in attribution order. Curves "
               "are means over alert days whose score sits ≥ 0.01 above the calm composite; "
               "legend AUCs are episode-level means (episode bootstrap 95% CIs below). The "
               "refitted surrogate is reported in the numbers only.\n" + "\n".join(lines)
               + f"\n{TP_NOTE}.")
    return style.save(fig, "F4_deletion_insertion", caption)


# ----------------------------------------------------------------------- F5
def f5_surrogate_honesty() -> list[Path]:
    style.apply()
    sc = _scorecard()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.1, 1.3, 1.2]})
    prod = {k: _sc(sc, "hormuz", k, "surrogate_prod", "point").value
            for k in ("r2_train", "r2_chrono_holdout", "r2_cv5")}
    labels = {"r2_train": "train", "r2_chrono_holdout": "out-of-time", "r2_cv5": "5-fold CV"}
    ax = axes[0]
    ax.bar(range(3), [prod[k] for k in labels], color="#D55E00", width=0.6)
    for i, k in enumerate(labels):
        ax.text(i, prod[k] + 0.02, f"{prod[k]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(3), list(labels.values()), fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title("Production surrogate,\nown synthetic year (R²)", loc="left", fontsize=8)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    width = 0.2
    real = [_sc(sc, r, "r2_on_real_test_window", "surrogate_prod", "point").value for r in REGIONS]
    refit = [_sc(sc, r, "r2_chrono_holdout", "surrogate_refit", "point").value for r in REGIONS]
    xs = np.arange(len(REGIONS))
    ax.bar(xs - width / 2, real, width, color="#D55E00", label="production, real test window")
    ax.bar(xs + width / 2, refit, width, color="#E69F00", label="refit, real test window")
    for x, v in zip(xs - width / 2, real):
        ax.text(x - 0.05, min(v, 0) - 0.3, f"{v:.2f}", ha="right", fontsize=6)
    for x, v in zip(xs + width / 2, refit):
        ax.text(x + 0.05, (v if v > 0 else v - 0.3) + 0.05, f"{v:.2f}", ha="left", fontsize=6)
    ax.axhline(0, color=style.INK_MUTED, lw=0.6)
    ax.set_xticks(xs, ["Hormuz", "B-el-M", "Panama", "Malacca"], fontsize=7)
    ax.set_title("Out-of-time R² on real data", loc="left", fontsize=8)
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    for j, method in enumerate(("surrogate_prod", "surrogate_refit")):
        rows = [_sc(sc, r, "local error |f(x) - g(x)| on alert days", method) for r in REGIONS]
        y = xs + (j - 0.5) * 0.25
        ax.errorbar([r.value for r in rows], y,
                    xerr=[[r.value - r.ci_lo for r in rows], [r.ci_hi - r.value for r in rows]],
                    fmt="o", ms=4, color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                    capsize=2, lw=1)
    ax.axvline(0, color=METHOD_COLORS["exact"], lw=1.2, label="Exact (residual ≤ 1e-16)")
    ax.set_yticks(xs, [style.REGION_LABELS[r].replace("Strait of ", "") for r in REGIONS], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("|f(x) − g(x)| on alert days", fontsize=7)
    ax.set_title("Local error vs the real score", loc="left", fontsize=8)
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    style.add_provenance(fig, "real_eval / synthetic_seed42 (left)", "D")
    errs = "; ".join(
        f"{style.REGION_LABELS[r]} prod {_sc(sc, r, 'local error |f(x) - g(x)| on alert days', 'surrogate_prod').value:.3f}"
        f" / refit {_sc(sc, r, 'local error |f(x) - g(x)| on alert days', 'surrogate_refit').value:.3f}"
        for r in REGIONS)
    caption = ("F5 — Surrogate honesty. Left: the production SHAP surrogate fits its own "
               f"synthetic training year (R² train {prod['r2_train']:.3f}) but not a held-out "
               f"later part of it ({prod['r2_chrono_holdout']:.3f}; CV {prod['r2_cv5']:.3f}). "
               "Middle: on the real test windows it is worse than a constant (R² "
               + ", ".join(f"{style.REGION_LABELS[r]} {v:.2f}" for r, v in zip(REGIONS, real))
               + "); refitting on real train data gives "
               + ", ".join(f"{v:.2f}" for v in refit) + ". Right: episode-mean local error "
               f"with 95% CIs ({errs}); the exact decomposition has none.\n{TP_NOTE}.")
    return style.save(fig, "F5_surrogate_honesty", caption)


# ----------------------------------------------------------------------- F6
def f6_compactness_stability() -> list[Path]:
    style.apply()
    sc = _scorecard()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    xs = np.arange(len(REGIONS))
    lines = []
    for ax, metric, title in ((axes[0], "k80 agents", "Agents for 80% of |φ|"),
                              (axes[1], "k80 features", "Features for 80% of |φ|")):
        for j, method in enumerate(("exact", "surrogate_prod", "surrogate_refit")):
            rows = [_sc(sc, r, metric, method) for r in REGIONS]
            y = xs + (j - 1) * 0.22
            ax.errorbar([r.value for r in rows], y,
                        xerr=[[r.value - r.ci_lo for r in rows], [r.ci_hi - r.value for r in rows]],
                        fmt="o", ms=4, capsize=2, lw=1, color=METHOD_COLORS[method],
                        label=METHOD_LABELS[method])
            lines.append(f"{title}, {METHOD_LABELS[method]}: "
                         + ", ".join(f"{style.REGION_LABELS[r]} {v.value:.2f}" for r, v in zip(REGIONS, rows)))
        ax.set_yticks(xs, [style.REGION_LABELS[r].replace("Strait of ", "") for r in REGIONS], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.36, 1.06), fontsize=6.5)
    ax = axes[2]
    sigmas = (0.01, 0.05, 0.1)
    for region in REGIONS:
        rows = [_sc(sc, region, f"Spearman rho under score noise sigma={s}") for s in sigmas]
        ax.errorbar(sigmas, [r.value for r in rows],
                    yerr=[[r.value - r.ci_lo for r in rows], [r.ci_hi - r.value for r in rows]],
                    marker="o", ms=3.5, capsize=2, lw=1.2, label=style.REGION_LABELS[region])
        lines.append(f"Stability ρ ({style.REGION_LABELS[region]}): "
                     + ", ".join(f"σ={s}: {r.value:.3f}" for s, r in zip(sigmas, rows)))
    ax.set_xlabel("Agent-score noise σ", fontsize=7)
    ax.set_ylabel("Spearman ρ of exact φ", fontsize=7)
    ax.set_title("Stability under noise (exact)", loc="left", fontsize=8)
    ax.legend(fontsize=5.8)
    fig.tight_layout()
    style.add_provenance(fig, "real_eval", "D")
    caption = ("F6 — Compactness and stability. Left/middle: episode-mean number of agents / "
               "features needed to cover 80% of the absolute attribution (95% CIs). Right: rank "
               "stability of the exact agent attribution when agent scores are perturbed.\n"
               + "\n".join(lines) + f"\n{TP_NOTE}.")
    return style.save(fig, "F6_compactness_stability", caption)


# ----------------------------------------------------------------------- F7
def f7_rag() -> list[Path]:
    style.apply()
    diag = pd.read_csv(_OUT / "rag_diagnostics.csv")
    configs = {"production": ("Production retriever", "#D55E00"),
               "wrapper_production": ("Wrapper, production KB", "#E69F00"),
               "wrapper_backfill": ("Wrapper + 2023-25 backfill", "#0072B2")}
    metrics = {"temporal_valid": "Dated before the alert", "region_match": "Same region",
               "duplicate_rate": "Duplicates in top 3"}
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    xs = np.arange(len(REGIONS))
    lines = []
    for ax, (metric, title) in zip(axes[:3], metrics.items()):
        for j, (cfg, (label, color)) in enumerate(configs.items()):
            vals = [diag[(diag.config == cfg) & (diag.region == r) & (diag.metric == metric)]
                    .value.iloc[0] for r in REGIONS]
            ax.bar(xs + (j - 1) * 0.26, vals, 0.26, color=color, label=label)
            lines.append(f"{title}, {label}: " + ", ".join(
                f"{style.REGION_LABELS[r]} {v:.2f}" for r, v in zip(REGIONS, vals)))
        ax.set_xticks(xs, ["Hormuz", "B-el-M", "Panama", "Malacca"], fontsize=6.3, rotation=30)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.4, 1.07), fontsize=6.5)
    ax = axes[3]
    p3 = _OUT / "rag_precision_at_3.csv"
    if p3.exists():
        tab = pd.read_csv(p3)
        tab = tab[(tab.episode_label == "all")]
        for j, measure in enumerate(("precision@3 (lenient)", "precision@3 (strict)")):
            rows = tab[tab.measure == measure].set_index("region")
            order = [r for r in ("all", *REGIONS) if r in rows.index]
            ax.errorbar(rows.loc[order, "value"], np.arange(len(order)) + j * 0.2,
                        xerr=[rows.loc[order, "value"] - rows.loc[order, "ci_lo"],
                              rows.loc[order, "ci_hi"] - rows.loc[order, "value"]],
                        fmt="o", capsize=2, label=measure)
        ax.set_yticks(np.arange(len(order)), order, fontsize=6.5)
        ax.legend(fontsize=5.8)
        p3_text = "Precision@3 from the author's annotation is shown in the right panel."
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Precision@3 (E8)\npending the author's\nrelevance annotation\n"
                "(60 pairs, 20 episodes)", ha="center", va="center", fontsize=8,
                color=style.INK_MUTED, transform=ax.transAxes)
        p3_text = ("Precision@3 (E8) is pending the author's annotation of 60 blind (episode, "
                   "document) pairs; the panel will fill when compute_precision_at_3.py has run.")
    fig.tight_layout()
    style.add_provenance(fig, "real_eval", "D", "one query per episode (first day)")
    caption = ("F7 — Precedent retrieval diagnostics, before (production) and after (eval "
               "wrapper; wrapper + backfill). Every configuration has a hit above the 0.55 "
               "similarity floor in 100% of episodes, so the floor is not selective. "
               + p3_text + "\n" + "\n".join(lines) + f"\n{TP_NOTE}.")
    return style.save(fig, "F7_rag_diagnostics", caption)


# ----------------------------------------------------------------------- F8
PRIMARY = {"P1_top_driver_geopolitical": "P1 top driver = geopolitical",
           "P2_agreement": "P2 agreement (low)", "P3_dispersion": "P3 dispersion (high)",
           "S0_score": "S0 score only (baseline)"}


def f8_triage_forest() -> list[Path]:
    style.apply()
    res = pd.read_csv(_OUT / "triage_results.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = 0
    ticks, labels, lines = [], [], []
    for region in LABELLED:
        for ind, label in PRIMARY.items():
            for k, variant in enumerate(("with_duration", "without_duration")):
                r = res[(res.region == region) & (res.indicator == ind) & (res.variant == variant)].iloc[0]
                color = "#6b6b6b" if ind == "S0_score" else "#0072B2"
                face = color if variant == "with_duration" else "white"
                ax.errorbar(r.auc_fp, y + k * 0.28, xerr=[[r.auc_fp - r.ci_lo], [r.ci_hi - r.auc_fp]],
                            fmt="s" if ind == "S0_score" else "o", ms=4.5, mfc=face, mec=color,
                            color=color, capsize=2, lw=1)
                if variant == "with_duration":
                    star = " *" if r.get("evidence") is True else ""
                    ph = f"Holm p={r.p_holm:.4f}" if pd.notna(r.get("p_holm")) else "baseline"
                    lines.append(f"{style.REGION_LABELS[region]} {label}: {r.auc_fp:.2f} "
                                 f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}] ({ph}){star}")
                    if star:
                        ax.text(r.ci_hi + 0.02, y, "*", fontsize=11, va="center", color="#b00020")
            ticks.append(y + 0.14)
            labels.append(f"{style.REGION_LABELS[region].replace('Strait of ', '')} · {label}")
            y += 1
        n = res[(res.region == region) & (res.variant == "with_duration")].iloc[0]
        flag = "" if n.interpretable_n else " (n < 5: not interpreted)"
        ax.text(1.02, y - 2.5, f"{int(n.n_tp_episodes)} TP / {int(n.n_fp_episodes)} FP{flag}",
                fontsize=6.5, color=style.INK_MUTED, transform=ax.get_yaxis_transform())
        y += 0.6
    ax.axvline(0.5, color=style.INK_MUTED, ls="--", lw=0.8)
    ax.set_yticks(ticks, labels, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC for false-positive episodes (registered direction; 0.5 = chance)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    style.add_provenance(fig, "real_eval", "D", "filled: with duration · hollow: without")
    caption = ("F8 — Explanation-informed triage. Forest plot of the three pre-registered "
               "primary indicators (first day of each episode) and the score-only baseline, "
               "AUC for separating FP from TP episodes with stratified episode-bootstrap 95% "
               "CIs; filled = with the shipping duration branch, hollow = without (circularity "
               "control). * = Holm-adjusted p < 0.05 and ≥ 5 TP and ≥ 5 FP episodes. Malacca "
               "has no TP episodes and is evaluated separately. Hormuz: small-n (5 TP episodes "
               "with duration, 7 without).\n" + "\n".join(lines) + f"\n{TP_NOTE}.")
    return style.save(fig, "F8_triage_forest", caption)


# ----------------------------------------------------------------------- F9
def f9_uncertainty_null() -> list[Path]:
    style.apply()
    ep = pd.read_csv(_OUT / "triage_inputs" / "episodes.csv")
    res = pd.read_csv(_OUT / "triage_results.csv")
    ep = ep[ep.variant == "with_duration"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    rng = np.random.default_rng(0)
    lines = []
    for ax, region in zip(axes, LABELLED):
        g = ep[ep.region == region]
        for x, (label, color) in enumerate((("TP", "#009E73"), ("FP", "#b00020"))):
            vals = g[g.fp == (label == "FP")].P3_dispersion.to_numpy()
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(vals)), vals, s=12, color=color,
                       alpha=0.8, lw=0)
            ax.hlines(np.median(vals), x - 0.25, x + 0.25, color=style.INK, lw=1.2)
        r = res[(res.region == region) & (res.indicator == "P3_dispersion")
                & (res.variant == "with_duration")].iloc[0]
        ax.set_xticks([0, 1], [f"TP (n={int(r.n_tp_episodes)})", f"FP (n={int(r.n_fp_episodes)})"])
        ax.set_title(f"{style.REGION_LABELS[region]}\nAUC {r.auc_fp:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]",
                     loc="left", fontsize=8)
        ax.grid(axis="x", visible=False)
        lines.append(f"{style.REGION_LABELS[region]}: AUC {r.auc_fp:.2f} [{r.ci_lo:.2f}, "
                     f"{r.ci_hi:.2f}], Holm p = {r.p_holm:.2f}")
    axes[0].set_ylabel("Agent dispersion, first day")
    fig.tight_layout()
    style.add_provenance(fig, "real_eval", "D", TP_NOTE)
    caption = ("F9 — Uncertainty vs correctness: a null result. First-day agent dispersion "
               "(registered primary P3, hypothesis: higher → false positive) for TP and FP "
               "episodes; lines are medians. No region separates the two classes; every "
               "CI spans 0.5 and no Holm-adjusted p is below 0.05.\n" + "\n".join(lines)
               + f"\n{TP_NOTE}.")
    return style.save(fig, "F9_uncertainty_null", caption)


# ---------------------------------------------------------------------- F10
def f10_decision() -> list[Path]:
    style.apply()
    m8 = json.loads((_OUT / "metric8_refresh.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    colors = {"P0-surrogate": "#D55E00", "P0-exact": "#0072B2"}
    lines = []
    for ax, (key, title) in zip(axes, (("macro_f1", "Macro-F1 (4 actions)"),
                                       ("reroute", "Reroute recall"))):
        for j, variant in enumerate(("P0-surrogate", "P0-exact")):
            vals = []
            for mode in ("hand_tuned", "optimized"):
                v = m8["modes"][mode]["variants"][variant]
                vals.append(v["macro_f1"] if key == "macro_f1" else v["recall"].get("reroute", 0.0))
            xs = np.arange(2) + (j - 0.5) * 0.35
            ax.bar(xs, vals, 0.35, color=colors[variant], label=variant)
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
            lines.append(f"{title}, {variant}: hand-tuned {vals[0]:.3f}, optimized {vals[1]:.3f}")
        ax.set_xticks(range(2), ["hand-tuned", "optimized"])
        ax.set_ylim(0, 1.12)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    style.add_provenance(fig, "synthetic_seed44", "C")
    cmp_ = {m: m8["modes"][m]["comparison_P0_exact_vs_P0_surrogate"] for m in ("hand_tuned", "optimized")}
    caption = ("F10 — Decision value (SYNTHETIC, seed 44). The day-level action rule fed the "
               "top agent from the production surrogate on the mismatched seed-42 row "
               "(P0-surrogate, what the dashboard did) or from exact Shapley values of the "
               "plotted day (P0-exact). The top agents differ on "
               f"{cmp_['hand_tuned']['days_top_agent_differs']}/365 and "
               f"{cmp_['optimized']['days_top_agent_differs']}/365 days, but the rule consults "
               f"them on only {cmp_['hand_tuned']['high_non_sustained_days']} / "
               f"{cmp_['optimized']['high_non_sustained_days']} days; McNemar p = "
               f"{cmp_['hand_tuned']['p_two_sided']:.2f} / {cmp_['optimized']['p_two_sided']:.3f}. "
               "The surrogate never yields 'reroute'. Correct actions are rule-derived from "
               "injected scenarios, not observed decisions.\n" + "\n".join(lines))
    return style.save(fig, "F10_decision_p0", caption)


# ---------------------------------------------------------------------- F12
_HEAT = [  # (column, metric, method, level, mapping to "goodness" in [0, 1], format)
    ("E1 residual", "max |efficiency residual| (agent level)", "exact", "point",
     lambda v: 1.0 if v < 1e-9 else 0.0, "{:.0e}"),
    ("E2 deletion AUC", "agent deletion AUC (lower is better)", "exact", "episode",
     lambda v: np.clip(1 - 2 * v, 0, 1), "{:.2f}"),
    ("E3 k80 agents", "k80 agents", "exact", "episode",
     lambda v: np.clip((5 - v) / 4, 0, 1), "{:.2f}"),
    ("E4 ρ (σ=0.05)", "Spearman rho under score noise sigma=0.05", "exact", "episode",
     lambda v: np.clip(v, 0, 1), "{:.2f}"),
    ("E5 surrogate = exact", "top agent agrees with exact", "surrogate_prod", "episode",
     lambda v: np.clip(v, 0, 1), "{:.2f}"),
    ("E6 feasible CF", "share of alert days with a feasible single-agent counterfactual", "exact",
     "episode", lambda v: np.clip(v, 0, 1), "{:.2f}"),
    ("E7 AUC (band width)", "AUC of band_width for FP vs TP episodes", "exact", "episode",
     lambda v: np.clip(2 * (v - 0.5), 0, 1), "{:.2f}"),
    ("E8 precision@3", "precision@3 (lenient / strict)", "exact", "episode",
     lambda v: np.clip(v, 0, 1), "{:.2f}"),
    ("E9 faithfulness", "narrative faithfulness (supported claims / claims)", "exact", "episode",
     lambda v: np.clip(v, 0, 1), "{:.2f}"),
]


def f12_scorecard_heatmap() -> list[Path]:
    style.apply()
    sc = _scorecard()
    raw = np.full((len(REGIONS), len(_HEAT)), np.nan)
    good = np.full_like(raw, np.nan)
    for i, region in enumerate(REGIONS):
        for j, (_, metric, method, level, fn, _) in enumerate(_HEAT):
            row = _sc(sc, region, metric, method, level)
            if row is not None and pd.notna(row.value):
                raw[i, j], good[i, j] = row.value, fn(row.value)
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    ax.imshow(np.ma.masked_invalid(good), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(REGIONS)):
        for j, (*_, fmt) in enumerate(_HEAT):
            text = "pending" if np.isnan(raw[i, j]) and j == 7 else (
                "n/a" if np.isnan(raw[i, j]) else fmt.format(raw[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=6.8,
                    color="white" if (not np.isnan(good[i, j]) and good[i, j] > 0.6) else style.INK)
    ax.set_xticks(range(len(_HEAT)), [h[0] for h in _HEAT], rotation=30, ha="right", fontsize=6.8)
    ax.set_yticks(range(len(REGIONS)), [style.REGION_LABELS[r] for r in REGIONS], fontsize=7.5)
    ax.grid(False)
    fig.tight_layout()
    style.add_provenance(fig, "real_eval", "D", TP_NOTE)
    caption = ("F12 — METRIC 9 explainability scorecard, exact attribution, per region "
               "(episode-level values; raw value printed, colour = goodness on a fixed scale: "
               "E1 1 if residual < 1e-9; E2 1 − 2·AUC; E3 (5 − k80)/4; E4 ρ; E5 share; E6 share; "
               "E7 2·(AUC − 0.5); E8 precision@3; E9 faithfulness). E5 measures how often the "
               "production surrogate names the exact top agent. E7 is undefined for Malacca "
               "(no TP episodes). E8 is pending the author's annotation.\n"
               + "\n".join(f"{style.REGION_LABELS[r]}: " + ", ".join(
                   f"{h[0]} {('pending' if np.isnan(raw[i, j]) else h[5].format(raw[i, j]))}"
                   for j, h in enumerate(_HEAT)) for i, r in enumerate(REGIONS))
               + f"\n{TP_NOTE}.")
    return style.save(fig, "F12_scorecard_heatmap", caption)


# ---------------------------------------------------------------------- F13
def f13_hormuz_case() -> list[Path]:
    style.apply()
    days, baselines = _days()
    with_d = days[(days.region == "hormuz") & (days.variant == "with_duration")
                  & (days.window == "test")].reset_index(drop=True)
    no_d = days[(days.region == "hormuz") & (days.variant == "without_duration")
                & (days.window == "test")].reset_index(drop=True)
    combined = with_d.score_shipping.to_numpy()
    shock = no_d.score_shipping.to_numpy()
    duration_won = combined > shock + 1e-12
    agents = _agents(with_d)
    phi = _phi(with_d, agents, baselines["with_duration"]["hormuz"])["shipping"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for ax in axes:
        _shade_truth(ax, with_d.date, with_d.y_true.to_numpy())
    ax = axes[0]
    ax.plot(with_d.date, shock, color="#56B4E9", lw=1, label="shock branch (isolation forest + max |z|)")
    ax.plot(with_d.date, np.where(duration_won, combined, np.nan), color="#0072B2", lw=2.2,
            label="duration branch, where it wins the max()")
    ax.set_ylabel("Shipping score")
    ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    ax.fill_between(with_d.date, 0, phi, where=phi >= 0, color="#0072B2", lw=0, step="mid")
    ax.fill_between(with_d.date, 0, phi, where=phi < 0, color="#0072B2", alpha=0.45, lw=0, step="mid")
    alert = with_d.alert.to_numpy()
    ax.scatter(with_d.date[alert], np.full(alert.sum(), phi.max() * 1.1), marker="|", s=30,
               color=np.where(with_d.y_true[alert] == 1, "#009E73", "#b00020"))
    ax.axhline(0, color=style.INK_MUTED, lw=0.5)
    ax.set_ylabel("Exact shipping φ")
    fig.tight_layout()
    style.add_provenance(fig, "real_eval", "D", "grey = labelled disruption days · ticks: alert days (green TP, red FP)")
    y = with_d.y_true.to_numpy() == 1
    first_label = with_d.date[np.flatnonzero(y)[0]] if y.any() else None
    first_dur = with_d.date[np.flatnonzero(duration_won)[0]] if duration_won.any() else None
    phi_all = _phi(with_d, agents, baselines["with_duration"]["hormuz"])
    means = {a: (float(phi_all[a][y].mean()), float(phi_all[a][~y].mean())) for a in agents}
    caption = (
        "F13 — The Hormuz case. Top: the shipping agent's two branches over the test window "
        "(its score is max(shock, duration); the duration branch is the sustained 30-day "
        "shortfall against the trailing 365-day median, the label's own statistic). Bottom: "
        "the exact Shapley value of shipping in the Tier 5 composite; ticks mark alert days.\n"
        f"The duration branch wins on {int((duration_won & y).sum())} of {int(y.sum())} labelled "
        f"disruption days ({int(duration_won.sum())} test days in all, {int((duration_won & alert).sum())} "
        f"of {int(alert.sum())} alert days); it first wins on {first_dur:%Y-%m-%d}, after the first "
        f"labelled day {first_label:%Y-%m-%d}. Even so, shipping's exact contribution is no larger "
        f"on labelled days (mean φ {means['shipping'][0]:.3f}) than on unlabelled ones "
        f"({means['shipping'][1]:.3f}); geopolitical contributes {means['geopolitical'][0]:.3f} vs "
        f"{means['geopolitical'][1]:.3f} and market {means['market'][0]:.3f} vs "
        f"{means['market'][1]:.3f}. Labelled days are alerted on {alert[y].mean():.1%} of "
        f"occasions, unlabelled days on {alert[~y].mean():.1%}: the decomposition shows that "
        "the composite does not track the label, because the signal that does (duration) is "
        "no stronger than the day-to-day contributions of the other agents.\n"
        f"{TP_NOTE}.")
    return style.save(fig, "F13_hormuz_case", caption)


# ---------------------------------------------------------------------- F15
def f15_market_saturation() -> list[Path]:
    style.apply()
    days, baselines = _days()
    sat = pd.read_csv(_OUT / "market_saturation.csv")
    sat = sat[sat.variant == "with_duration"].set_index("region")
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.4), sharey=True)
    lines = []
    for ax, region in zip(axes, REGIONS):
        test = days[(days.region == region) & (days.variant == "with_duration") & (days.window == "test")]
        ax.hist(test.score_market, bins=np.linspace(0, 1, 21), color=style.AGENT_COLORS["market"],
                edgecolor="white", lw=0.8)
        calm = baselines["with_duration"][region]["market"]
        ax.axvline(calm, color=style.INK, ls="--", lw=0.9)
        share = sat.loc[region, "share_test_days_ge_0.95 (day-level)"]
        ax.text(calm - 0.02, ax.get_ylim()[1] * 0.92, f"calm {calm:.2f}", ha="right", fontsize=6.5)
        ax.set_title(f"{style.REGION_LABELS[region]}\n≥ 0.95 on {share:.0%} of days", loc="left",
                     fontsize=7.5)
        ax.set_xlabel("Market score", fontsize=7)
        lines.append(f"{style.REGION_LABELS[region]}: calm baseline {calm:.3f}; test days ≥ 0.95 "
                     f"{share:.1%}; clipped at 1.0 {sat.loc[region, 'share_test_days_clipped_at_1 (day-level)']:.1%}; "
                     f"mean |φ_market| on alerts {sat.loc[region, 'mean_abs_phi_market_on_alerts']:.3f}")
    axes[0].set_ylabel("Test days")
    fig.tight_layout()
    style.add_provenance(fig, "real_eval", "D", "market inputs are global FRED series")
    caption = ("F15 — Market saturation. Distribution of the market agent's daily score on the "
               "test window; dashed line = its calm-day median on the train window. The agent "
               "sits at or near its ceiling on about a quarter to a third of days, calm ones "
               "included, so measured against its calm level it can add little to an alert; "
               "it was never the top driver of an episode. Report only; the agent is unchanged.\n"
               + "\n".join(lines))
    return style.save(fig, "F15_market_saturation", caption)


# ---------------------------------------------------------------------- F16
def f16_fp_lift() -> list[Path]:
    style.apply()
    primary = pd.read_csv(_OUT / "fp_lift_primary.csv")
    strict = pd.read_csv(_OUT / "fp_strict_lift.csv")
    rows = list(REGIONS) + ["pooled"]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    lines = []
    for k, (table, label, color, face) in enumerate((
            (primary, "Pre-registered rule (primary)", "#0072B2", "#0072B2"),
            (strict, "Stricter rule (EXPLORATORY, post hoc)", "#D55E00", "white"))):
        for i, region in enumerate(rows):
            r = table[table.region == region].iloc[0]
            y = i + (k - 0.5) * 0.3
            ax.errorbar(r.lift_fp_vs_calm, y, xerr=[[r.lift_fp_vs_calm - r.ci_lo], [r.ci_hi - r.lift_fp_vs_calm]],
                        fmt="o", ms=4.5, mfc=face, mec=color, color=color, capsize=2, lw=1,
                        label=label if i == 0 else None)
            lines.append(f"{label}, {region}: FP {r.fp_rate:.2f} vs calm {r.calm_rate:.2f}, lift "
                         f"{r.lift_fp_vs_calm:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]")
    ax.axvline(1.0, color=style.INK, lw=0.9)
    ax.set_yticks(range(len(rows)), [style.REGION_LABELS.get(r, "Pooled") for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Lift = share of FP-episode windows with conflict evidence ÷ share of calm-day windows")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    style.add_provenance(fig, "real_eval", "D", "8-day windows; production + 2023-25 backfill")
    caption = ("F16 — Do false-positive alerts coincide with conflict evidence more than ordinary "
               "days do? No. Lift of the share of 8-day windows containing a geopolitical / "
               "conflict document, FP-episode windows vs calm-day windows, with 1,000-resample "
               "bootstrap 95% CIs; 1.0 = no difference. The pre-registered rule is the primary "
               "outcome; the stricter rule was defined after seeing it and run once.\n"
               + "\n".join(lines))
    return style.save(fig, "F16_fp_lift", caption)


ALL = {
    "F1": f1_pipeline, "F2": f2_alert_anatomy, "F3": f3_attribution_timeline,
    "F4": f4_deletion_insertion, "F5": f5_surrogate_honesty, "F6": f6_compactness_stability,
    "F7": f7_rag, "F8": f8_triage_forest, "F9": f9_uncertainty_null, "F10": f10_decision,
    "F12": f12_scorecard_heatmap, "F13": f13_hormuz_case, "F15": f15_market_saturation,
    "F16": f16_fp_lift,
}
