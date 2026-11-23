"""Chapter 4 lead-time analysis and figure (thesis Fig. 11 replacement; not for the paper).

Every alert here is the harness's (``scripts/run_method_comparison.py``): the
tier composites from ``_compose``, the threshold and ``>=`` rule from
``alert_flags``, the masked news days left unscored, and the controls whose
alerts are defined directly taken from ``_DIRECT_ALERTS``.

Methods: Tiers 1-5 and the M0 random control, at each method's own operating
point (training-window 90th percentile).

Lead time (post hoc, not pre-registered): for each run of labelled test days
that starts after the first test day, the onset ``t0`` is its first day. The
lead is ``t0 - a`` where ``a`` is the first alert day in ``[t0 - 30, t0 + 30]``.
Positive = the alert came before the label's onset; +30 = an alert was already
active 30 days before; negative = after; no alert in the window = missed.

**Every method leads, the random control included.** The label is a lagged
30-day mean held for 14 days, so its onset comes after the physical drop. And
if a method's alerts fell independently across days, one that alerts on a
fraction ``p`` of days would have an alert in the 30 days before an onset with
probability ``1 - (1 - p)^30``. Real alerts are autocorrelated, so that is an
upper reference, not their probability; the tiers alert on far more days than
the random control, up to about three in four, where a lead says nothing. Lead
time here is not evidence of early warning, and the figure's caption says so.

Two steps, so the figure is drawn from a committed file:

* compute (needs the git-ignored frames in ``data/eval/``): writes
  ``eval/lead_time.csv``;
* plot (``--plot-only``, and ``scripts/make_all_figures.py``): reads that CSV
  and writes ``thesis/figures/ch4/D2_lead_time_all_tiers.{pdf,png}`` and its
  caption.

Usage::

    python eval/ch4/detection_figures.py              # compute, then plot
    python eval/ch4/detection_figures.py --plot-only  # plot from eval/lead_time.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LEAD_CSV = ROOT / "eval" / "lead_time.csv"
THESIS_FIGURES = ROOT / "thesis" / "figures" / "ch4"
FIGURE_NAME = "D2_lead_time_all_tiers"
WINDOW = 30
#: Malacca has no labelled onset, so no lead time.
REGIONS = ("hormuz", "bab_el_mandeb", "panama")
METHODS = tuple(f"Tier {k}" for k in range(1, 6)) + ("M0 random",)
SHORT = {"hormuz": "Hormuz", "bab_el_mandeb": "Bab el-Mandeb", "panama": "Panama",
         "malacca": "Malacca"}
#: Tiers are ordered, so one hue, light to dark (the dataviz ordinal ramp,
#: validated); the random control is a neutral gray with a hatch.
TIER_COLORS = {"Tier 1": "#86b6ef", "Tier 2": "#5598e7", "Tier 3": "#2a78d6",
               "Tier 4": "#1c5cab", "Tier 5": "#104281"}
CONTROL_COLOR = "#898781"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


# ------------------------------------------------------------------ lead time
def onsets(y) -> list[int]:
    """Start index of each labelled run, excluding a run already open at day 0."""
    starts = np.flatnonzero(np.diff(np.r_[0, np.asarray(y).astype(int)]) == 1)
    return [int(s) for s in starts if s > 0]


def lead_for_onset(alert: np.ndarray, t0: int, window: int = WINDOW) -> tuple[float, bool, bool]:
    """``(lead_days, censored_at_window, missed)`` for one onset.

    ``lead_days`` is ``t0`` minus the first alert day within ``window`` days of
    ``t0`` (NaN if there is none); ``censored_at_window`` is True when that
    alert was already active at the start of the window.
    """
    lo, hi = max(0, t0 - window), min(len(alert) - 1, t0 + window)
    hits = np.flatnonzero(np.asarray(alert, dtype=bool)[lo:hi + 1])
    if not len(hits):
        return float("nan"), False, True
    return float(t0 - (lo + int(hits[0]))), bool(lo == t0 - window and hits[0] == 0), False


# --------------------------------------------------- alerts, from the harness
def harness():
    """The comparison harness, loaded by path."""
    spec = importlib.util.spec_from_file_location(
        "run_method_comparison", ROOT / "scripts" / "run_method_comparison.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_method_comparison"] = module     # @dataclass resolves its module by name
    spec.loader.exec_module(module)
    return module


@dataclass
class RegionContext:
    """One region's split, masks and agent scores, as the harness builds them."""

    region: str
    train: pd.DataFrame
    test: pd.DataFrame
    keep: np.ndarray            # scored test days (masked news days are False)
    train_keep: np.ndarray
    y: np.ndarray
    config: dict
    on_test: dict
    on_train: dict
    features: list[str]


def region_context(region: str, rmc) -> RegionContext:
    """Load a region's frame and score every agent once (needs ``data/eval``)."""
    from src.core.config_manager import load_config_for_region

    frame, manifest = rmc._load(region)
    config = load_config_for_region(region)
    train, test, keep, train_keep, y, _ = rmc._main_split(frame, manifest)
    return RegionContext(
        region=region, train=train, test=test, keep=keep, train_keep=train_keep, y=y,
        config=config, on_test=rmc._all_agent_scores(config, train, test),
        on_train=rmc._all_agent_scores(config, train, train),
        features=[c for c in frame.columns if "__" in c])


def method_alerts(name: str, ctx: RegionContext, rmc) -> np.ndarray | None:
    """A method's alert on each test day, exactly as the harness scores it.

    Tiers fuse the enabled agents (``_compose``); other methods are the
    harness's baselines, with the controls whose alerts are defined directly
    (``_DIRECT_ALERTS``) taken as defined. Masked days never alert.

    Returns:
        A boolean array over the test days, or ``None`` if the method has no
        scores (no enabled agent, or a supervised model without positives).
    """
    n = len(ctx.test)
    if name.startswith("Tier"):
        scores, _ = rmc._compose(int(name.split()[1]), ctx.on_test, ctx.config)
        if scores is None:
            return None
        ref, _ = rmc._compose(int(name.split()[1]), ctx.on_train, ctx.config)
        # As _tier_rows: without training scores the threshold comes from the test scores.
        flags = rmc.alert_flags(scores, ref if ref is not None else scores,
                                train_keep=ctx.train_keep if ref is not None else ctx.keep)
    else:
        fn, _kind = rmc.BASELINES[name]
        row_wise = name in rmc._ROW_WISE
        fit_train = ctx.train[ctx.train_keep] if row_wise else ctx.train
        try:
            scores = np.asarray(fn(fit_train, ctx.test, ctx.features), dtype=float)
        except rmc._NotTrainable:
            return None
        direct = rmc._DIRECT_ALERTS[name][1](ctx.train, ctx.test) if name in rmc._DIRECT_ALERTS else None
        flags = rmc.alert_flags(scores[-n:], scores[:len(fit_train)],
                                train_keep=None if row_wise else ctx.train_keep, direct=direct)
    return flags & ctx.keep


def lead_times(regions: tuple[str, ...] = REGIONS, methods: tuple[str, ...] = METHODS,
               rmc=None, contexts: dict[str, RegionContext] | None = None) -> pd.DataFrame:
    """One row per region, method and labelled onset."""
    rmc = rmc or harness()
    rows = []
    for region in regions:
        ctx = (contexts or {}).get(region) or region_context(region, rmc)
        dates = pd.to_datetime(ctx.test["timestamp"]).dt.strftime("%Y-%m-%d").to_numpy()
        for name in methods:
            flags = method_alerts(name, ctx, rmc)
            if flags is None:
                continue
            agents = ""
            if name.startswith("Tier"):
                _, used = rmc._compose(int(name.split()[1]), ctx.on_test, ctx.config)
                agents = ",".join(used)
            rule = (rmc._DIRECT_ALERTS[name][0] if name in rmc._DIRECT_ALERTS
                    else rmc._THRESHOLD_RULE)
            for t0 in onsets(ctx.y):
                lead, censored, missed = lead_for_onset(flags, t0)
                rows.append({"region": region, "method": name, "agents": agents,
                             "onset": dates[t0], "lead_days": lead,
                             "censored_at_window": censored, "missed": missed,
                             "alert_rate": float(flags[ctx.keep].mean()), "alert_rule": rule})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- figure
def headline(lt: pd.DataFrame) -> str:
    """The figure's title, true of the data: do all methods lead every onset?"""
    if (lt["lead_days"] > 0).all():
        return "Every method 'leads' the label, including the random control"
    n_lead = int((lt["lead_days"] > 0).sum())
    return (f"Most methods 'lead' the label, including the random control "
            f"({n_lead} of {len(lt)} method-onsets)")


def _copies(lt: pd.DataFrame) -> list[str]:
    """Tiers (per region) whose agent set equals the tier before it."""
    out = []
    tiers = lt[lt["method"].str.startswith("Tier")].drop_duplicates(["region", "method"])
    for region, rows in tiers.groupby("region"):
        rows = rows.iloc[np.argsort([int(m.split()[1]) for m in rows["method"]], kind="stable")]
        previous = None
        for _, row in rows.iterrows():
            if previous is not None and row["agents"] == previous["agents"]:
                out.append(f"{SHORT[region]} {row['method']} (= Tier {int(row['method'].split()[1]) - 1})")
            previous = row
    return out


def _rates_lines(lt: pd.DataFrame) -> tuple[list[str], str]:
    """Each method's alert rate per region, and a sentence naming the highest one."""
    rates = lt.drop_duplicates(["region", "method"])
    lines = []
    for region in REGIONS:
        sub = rates[rates["region"] == region]
        parts = [f"{m} {sub.loc[sub['method'] == m, 'alert_rate'].iloc[0]:.0%}"
                 for m in METHODS if (sub["method"] == m).any()]
        if parts:
            lines.append(f"{SHORT[region]}: " + ", ".join(parts))
    top = rates.loc[rates["alert_rate"].idxmax()]
    return lines, (f"The highest alert rate is {top['alert_rate']:.0%} "
                   f"({SHORT[top['region']]}, {top['method']}); at that rate an alert before any "
                   "onset is close to certain")


def _leads_sentence(lt: pd.DataFrame) -> str:
    """Whether every method leads, said only as far as the table supports it."""
    leads = lt["lead_days"] > 0
    if leads.all():
        return "Every method, including the random control, 'leads' the label."
    rand = lt["method"] == "M0 random"
    return (f"{int(leads.sum())} of {len(lt)} method-onsets 'lead' the label; the random control "
            f"leads {int(leads[rand].sum())} of {int(rand.sum())}.")


def caption(lt: pd.DataFrame) -> str:
    """The figure caption, with every bar's value and every method's alert rate."""
    random_rate = lt.loc[lt["method"] == "M0 random", "alert_rate"].mean()
    chance = 1 - (1 - random_rate) ** WINDOW if pd.notna(random_rate) else float("nan")
    rate_lines, highest = _rates_lines(lt)
    lines = []
    for region in REGIONS:
        sub = lt[lt["region"] == region]
        for method in METHODS:
            rows = sub[sub["method"] == method].sort_values("onset")
            if rows.empty:
                continue
            lines.append(f"{SHORT[region]}, {method}: " + ", ".join(
                f"{r.onset} {'missed' if r.missed else f'{r.lead_days:+.0f} d'}"
                for r in rows.itertuples()))
    copies = _copies(lt)
    return (
        "Lead time per labelled onset (the first day of a labelled run; a run already open on "
        "the first test day is excluded). Lead = onset minus the first alert within ±30 days, "
        "at each method's own operating point (training-window 90th percentile, alert when the "
        "score is at or above it); +30 means an alert was already active 30 days before onset, "
        "× means no alert in the window.\n"
        f"{_leads_sentence(lt)} The label is a lagged 30-day mean (its onset comes after the "
        "physical drop), and a method that alerts on many days has an alert before any onset "
        f"almost regardless of signal: if its alerts fell independently across days, one "
        f"alerting on about {random_rate:.0%} of days, as the random control does, would have an "
        f"alert in the 30 days before an onset with probability {chance:.0%}. Real alerts are "
        "autocorrelated, so that is an upper reference, not the tiers' probability. Lead time "
        "here is not evidence of early warning.\n"
        "Alert rate (share of test days):\n" + "\n".join(rate_lines) + f".\n{highest}.\n"
        + (f"Tiers that copy the tier before them (a disabled agent) show identical bars: "
           f"{', '.join(copies)}.\n" if copies else "")
        + "Post-hoc analysis, not pre-registered. For the thesis; not for the paper.\n"
        + "\n".join(lines))


def fig_lead(lt: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """Draw the figure from a lead-time table; returns the files written."""
    out_dir = Path(out_dir) if out_dir is not None else THESIS_FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = [r for r in REGIONS if (lt["region"] == r).any()]
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "hatch.linewidth": 0.6})
    fig, axes = plt.subplots(len(regions), 1, figsize=(6.3, 2.6 * len(regions) + 1.0), squeeze=False)
    values = lt["lead_days"].dropna()
    lo = min(-10.0, float(np.floor(values.min())) - 3) if len(values) else -10.0
    hi = max(34.0, float(np.ceil(values.max())) + 3) if len(values) else 34.0
    methods = [m for m in METHODS if (lt["method"] == m).any()]
    width = 0.8 / len(methods)
    for ax, region in zip(axes[:, 0], regions):
        sub = lt[lt["region"] == region]
        dates = sorted(sub["onset"].unique())
        for j, method in enumerate(methods):
            rows = sub[sub["method"] == method].set_index("onset").reindex(dates)
            x = np.arange(len(dates)) + (j - (len(methods) - 1) / 2) * width
            lead = rows["lead_days"].to_numpy(dtype=float)
            control = method == "M0 random"
            ax.bar(x, np.nan_to_num(lead), width * 0.92, label=method,
                   color=CONTROL_COLOR if control else TIER_COLORS[method],
                   hatch="////" if control else None,
                   edgecolor=MUTED if control else "none", linewidth=0.4)
            for xi, miss in zip(x, rows["missed"].fillna(False).to_numpy(dtype=bool)):
                if miss:
                    ax.text(xi, 0.8, "×", ha="center", va="bottom", fontsize=9, color=INK)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(dates)), dates, fontsize=8)
        ax.set_xlim(-0.6, len(dates) - 0.4)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("Lead before label onset (days)", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(SHORT[region], loc="left", fontsize=10.5, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.955), handlelength=1.4, columnspacing=1.2)
    fig.suptitle(headline(lt), x=0.02, ha="left", fontsize=10.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=1.8)
    paths = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{FIGURE_NAME}.{ext}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    text = out_dir / f"{FIGURE_NAME}.caption.txt"
    text.write_text(caption(lt), encoding="utf-8")
    return paths + [text]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-only", action="store_true",
                        help="draw from eval/lead_time.csv; no data needed")
    args = parser.parse_args(argv)
    if args.plot_only:
        lt = pd.read_csv(LEAD_CSV)
    else:
        lt = lead_times()
        lt.to_csv(LEAD_CSV, index=False)
        print(f"wrote {LEAD_CSV} ({len(lt)} rows)")
        print(lt.groupby(["region", "method"]).agg(
            onsets=("onset", "count"), missed=("missed", "sum"),
            median_lead=("lead_days", "median"), alert_rate=("alert_rate", "first")
        ).round(3).to_string())
    for path in fig_lead(lt):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
