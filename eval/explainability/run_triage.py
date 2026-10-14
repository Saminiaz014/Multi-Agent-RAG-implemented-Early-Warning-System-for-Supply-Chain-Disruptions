"""Phase 4 — does the explanation tell you when to trust an alert?

Implements ``PREREGISTERED_RULES.md`` §C (commit ``f89f445``) on the C0 export
(``triage_inputs/``), the AERs' precedents and the Phase 1 episode table. No
agent, API or retrieval call.

Unit: the alert episode; FP = 1. Primary indicators (first day): P1 top exact
driver = geopolitical, P2 agreement, P3 dispersion. S0 = first-day composite
(score-only baseline). Secondary indicators are EXPLORATORY; C1 and episode
length are retrospective and reported separately.

Writes ``triage_results.csv``, ``triage_retrospective.csv``, ``triage_loro.csv``,
``triage_malacca.csv``.

Usage::

    python eval/explainability/run_triage.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.explainability_metrics import BOOT_SEED, N_BOOT, k_cover  # noqa: E402
from src.evaluation.path_d import CSV_WEIGHTS, REGIONS  # noqa: E402
from src.explainability.alert_explanation import (  # noqa: E402
    episodes,
    noise_scales,
    score_uncertainty,
)
from src.explainability.exact_shapley import value_fn_path_d  # noqa: E402

_OUT = _ROOT / "eval" / "explainability"
_IN = _OUT / "triage_inputs"
VARIANTS = ("with_duration", "without_duration")
MIN_N = 5

PRIMARY = ("P1_top_driver_geopolitical", "P2_agreement", "P3_dispersion")
SECONDARY = ("X1_k80_agent", "X2_surrogate_disagrees", "X3_max_precedent_similarity",
             "X4_noise_robustness")
RETROSPECTIVE = ("C1_top_driver_geopolitical_episode", "R2_episode_length")
BASELINE = ("S0_score",)
#: +1: higher value -> FP (registered direction); -1: lower value -> FP.
DIRECTION = {
    "P1_top_driver_geopolitical": 1, "P2_agreement": -1, "P3_dispersion": 1,
    "S0_score": -1, "X1_k80_agent": -1, "X2_surrogate_disagrees": 1,
    "X3_max_precedent_similarity": -1, "X4_noise_robustness": -1,
    "C1_top_driver_geopolitical_episode": 1, "R2_episode_length": -1,
}


# ------------------------------------------------------------------ stats
def auc_boot(values: np.ndarray, fp: np.ndarray, rng: np.random.Generator,
             n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """AUC for FP (ties count half) with a stratified, vectorised bootstrap CI."""
    pos, neg = values[fp == 1], values[fp == 0]
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan, np.nan

    def auc(p: np.ndarray, n: np.ndarray) -> np.ndarray:
        diff = p[..., :, None] - n[..., None, :]
        return ((diff > 0) + 0.5 * (diff == 0)).mean(axis=(-1, -2))

    point = float(auc(pos, neg))
    draws = auc(pos[rng.integers(0, len(pos), (n_boot, len(pos)))],
                neg[rng.integers(0, len(neg), (n_boot, len(neg)))])
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def mwu_p(values: np.ndarray, fp: np.ndarray) -> float:
    """One-sided Mann-Whitney U p-value: FP episodes higher (registered direction)."""
    from scipy.stats import mannwhitneyu

    pos, neg = values[fp == 1], values[fp == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(mannwhitneyu(pos, neg, alternative="greater").pvalue)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaNs left as NaN, excluded from m)."""
    p = np.asarray(pvalues, dtype=float)
    ok = np.flatnonzero(np.isfinite(p))
    adjusted = np.full(len(p), np.nan)
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(order) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def rate_ci(flags: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    flags = np.asarray(flags, dtype=float)
    if len(flags) == 0:
        return np.nan, np.nan, np.nan
    draws = flags[rng.integers(0, len(flags), (N_BOOT, len(flags)))].mean(axis=1)
    return float(flags.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------- features
def episode_table(days: pd.DataFrame, baselines: dict, surrogate: pd.DataFrame,
                  precedent_sim: dict) -> tuple[pd.DataFrame, dict]:
    """One row per episode x variant x region, with every registered indicator."""
    rows, cutoffs = [], {}
    for variant in VARIANTS:
        for region in REGIONS:
            sub = days[(days.variant == variant) & (days.region == region)]
            test = sub[sub.window == "test"].reset_index(drop=True)
            train = sub[sub.window == "train"].reset_index(drop=True)
            agents = [c[6:] for c in sub.columns if c.startswith("score_") and sub[c].notna().all()]
            base = baselines[variant][region]
            mass = sum(CSV_WEIGHTS[a] for a in agents)
            value_fn = value_fn_path_d(CSV_WEIGHTS, agents)
            threshold = float(test.threshold.iloc[0])

            def phi_of(frame: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({a: CSV_WEIGHTS[a] * (frame[f"score_{a}"] - base[a]) / mass
                                     for a in agents})

            # Cut-offs from the train window (§C.2).
            calm = train[train.y_true == 0]
            q90 = {a: float(calm[f"score_{a}"].quantile(0.90)) for a in agents}
            train_alert = train[train.alert]
            disp_q75 = float(train_alert[[f"score_{a}" for a in agents]].std(axis=1, ddof=0)
                             .quantile(0.75))
            k80_q25 = float(phi_of(train_alert).apply(lambda r: k_cover(r.to_dict()), axis=1)
                            .quantile(0.25))
            train_eps = episodes(train.date.tolist(), train.alert.to_numpy())
            len_q50 = float(np.median([len(g) for g in train_eps]))
            sigma = noise_scales({a: train[f"score_{a}"].to_numpy() for a in agents},
                                 (train.y_true == 0).to_numpy())
            cutoffs[(variant, region)] = {"agreement_q90": q90, "dispersion_q75": disp_q75,
                                          "k80_q25": k80_q25, "episode_length_q50": len_q50}

            sur = surrogate[surrogate.region == region].set_index("date")
            phi_test = phi_of(test)
            for k, members in enumerate(episodes(test.date.tolist(), test.alert.to_numpy())):
                first = members[0]
                row = test.iloc[first]
                scores = {a: float(row[f"score_{a}"]) for a in agents}
                phi = phi_test.iloc[first].to_dict()
                top = max(phi, key=phi.get)
                sur_phi = {a: float(sur.loc[row.date, f"phi_sur_{a}"]) for a in agents}
                summed = phi_test.iloc[members].sum().to_dict()
                unc = score_uncertainty(scores, value_fn, threshold, sigma,
                                        seed=int(first) + 1, n_draws=200)
                label_fp = int(not test.y_true.iloc[members[0]:members[-1] + 1].any())
                rows.append({
                    "data_origin": "real_eval", "variant": variant, "region": region,
                    "episode": f"{region}_E{k + 1:03d}", "start": row.date, "fp": label_fp,
                    "S0_score": float(row.composite),
                    "P1_top_driver_geopolitical": float(top == "geopolitical"),
                    "P2_agreement": float(sum(scores[a] > q90[a] for a in agents)),
                    "P3_dispersion": float(np.std(list(scores.values()))),
                    "X1_k80_agent": k_cover(phi),
                    "X2_surrogate_disagrees": float(max(sur_phi, key=sur_phi.get) != top),
                    "X3_max_precedent_similarity": precedent_sim.get((region, row.date), np.nan),
                    "X4_noise_robustness": unc["frac_draws_above_threshold"],
                    "C1_top_driver_geopolitical_episode": float(
                        max(summed, key=summed.get) == "geopolitical"),
                    "R2_episode_length": float(len(members)),
                })
    return pd.DataFrame(rows), cutoffs


def flags(ep: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    """Registered low-trust flags (§C.2) per episode."""
    out = pd.DataFrame(index=ep.index)
    c = ep.apply(lambda r: cutoffs[(r.variant, r.region)], axis=1)
    out["P1_top_driver_geopolitical"] = ep.P1_top_driver_geopolitical == 1
    out["P2_agreement"] = ep.P2_agreement <= 1
    out["P3_dispersion"] = ep.P3_dispersion > c.map(lambda d: d["dispersion_q75"])
    out["X1_k80_agent"] = ep.X1_k80_agent <= c.map(lambda d: d["k80_q25"])
    out["X2_surrogate_disagrees"] = ep.X2_surrogate_disagrees == 1
    out["X4_noise_robustness"] = ep.X4_noise_robustness < 0.95
    out["C1_top_driver_geopolitical_episode"] = ep.C1_top_driver_geopolitical_episode == 1
    out["R2_episode_length"] = ep.R2_episode_length < c.map(lambda d: d["episode_length_q50"])
    return out


# ---------------------------------------------------------------- analyses
def single_indicator(ep: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, retro = [], []
    for (variant, region), g in ep.groupby(["variant", "region"], sort=False):
        fp = g.fp.to_numpy()
        n_fp, n_tp = int(fp.sum()), int((1 - fp).sum())
        interpretable = n_tp >= MIN_N and n_fp >= MIN_N
        note = "small-n (5 TP episodes with duration, 7 without)" if region == "hormuz" else ""
        block = []
        for name in (*BASELINE, *PRIMARY, *SECONDARY, *RETROSPECTIVE):
            values = DIRECTION[name] * g[name].to_numpy(dtype=float)
            auc, lo, hi = auc_boot(values, fp, rng)
            tier = ("baseline" if name in BASELINE else "primary" if name in PRIMARY
                    else "secondary (EXPLORATORY)" if name in SECONDARY else "retrospective")
            record = {"data_origin": "real_eval", "variant": variant, "region": region,
                      "indicator": name, "tier": tier, "auc_fp": auc, "ci_lo": lo, "ci_hi": hi,
                      "n_tp_episodes": n_tp, "n_fp_episodes": n_fp,
                      "n_with_value": int(np.isfinite(g[name].to_numpy(dtype=float)).sum()),
                      "interpretable_n": interpretable, "note": note,
                      "p_one_sided": mwu_p(values, fp) if name in PRIMARY else np.nan}
            (retro if tier == "retrospective" else block).append(record)
        adjusted = holm([r["p_one_sided"] for r in block if r["tier"] == "primary"])
        for r, p in zip([r for r in block if r["tier"] == "primary"], adjusted):
            r["p_holm"] = p
            r["evidence"] = bool(r["interpretable_n"] and np.isfinite(p) and p < 0.05)
        rows += block
    return pd.DataFrame(rows), pd.DataFrame(retro)


def loro(ep: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-region-out logistic: primaries vs score-only (§C.4)."""
    from sklearn.linear_model import LogisticRegression

    models = {"primaries": list(PRIMARY), "score_only": ["S0_score"]}
    rows, preds = [], []
    for variant in VARIANTS:
        data = ep[ep.variant == variant]
        held = {}
        for region in REGIONS:
            train, test = data[data.region != region], data[data.region == region]
            p = {}
            for name, cols in models.items():
                X, Xt = train[cols].to_numpy(float), test[cols].to_numpy(float)
                mu, sd = X.mean(axis=0), X.std(axis=0)
                sd[sd == 0] = 1.0
                model = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                                           solver="lbfgs", max_iter=1000)
                model.fit((X - mu) / sd, train.fp.to_numpy())
                p[name] = model.predict_proba((Xt - mu) / sd)[:, 1]
            held[region] = (test, p)
            preds.append(test.assign(**{f"p_fp_{k}": v for k, v in p.items()}))
            fp = test.fp.to_numpy()
            if fp.min() == fp.max():
                continue  # Malacca: no TP episodes -> evaluated in the Malacca table
            full, lo_f, hi_f = auc_boot(p["primaries"], fp, rng)
            base, lo_b, hi_b = auc_boot(p["score_only"], fp, rng)
            # Paired bootstrap of the difference on identical resamples.
            pos, neg = np.flatnonzero(fp == 1), np.flatnonzero(fp == 0)
            ip = pos[rng.integers(0, len(pos), (N_BOOT, len(pos)))]
            ineg = neg[rng.integers(0, len(neg), (N_BOOT, len(neg)))]

            def auc_draw(s: np.ndarray) -> np.ndarray:
                d = s[ip][:, :, None] - s[ineg][:, None, :]
                return ((d > 0) + 0.5 * (d == 0)).mean(axis=(1, 2))

            delta = auc_draw(p["primaries"]) - auc_draw(p["score_only"])
            rows.append({"data_origin": "real_eval", "variant": variant, "held_out": region,
                         "auc_primaries": full, "ci_lo_primaries": lo_f, "ci_hi_primaries": hi_f,
                         "auc_score_only": base, "ci_lo_score_only": lo_b,
                         "ci_hi_score_only": hi_b, "delta_auc": full - base,
                         "delta_ci_lo": float(np.percentile(delta, 2.5)),
                         "delta_ci_hi": float(np.percentile(delta, 97.5)),
                         "n_tp_episodes": int((1 - fp).sum()), "n_fp_episodes": int(fp.sum()),
                         "interpretable_n": bool((1 - fp).sum() >= MIN_N and fp.sum() >= MIN_N)})
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def malacca(ep: pd.DataFrame, flag_frame: pd.DataFrame, held: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        sel = ep.variant == variant
        mal = sel & (ep.region == "malacca")
        tp_other = sel & (ep.region != "malacca") & (ep.fp == 0)
        rules = {name: flag_frame[name] for name in
                 (*PRIMARY, "X1_k80_agent", "X2_surrogate_disagrees", "X4_noise_robustness")}
        hv = held[held.variant == variant].set_index(["region", "episode"])
        for model in ("primaries", "score_only"):
            col = hv[f"p_fp_{model}"] >= 0.5
            rules[f"LORO_{model} (P(FP) >= 0.5)"] = pd.Series(
                [bool(col.get((r, e), False)) for r, e in zip(ep.region, ep.episode)],
                index=ep.index) & sel
        for name, flag in rules.items():
            m, m_lo, m_hi = rate_ci(flag[mal].to_numpy(), rng)
            t, t_lo, t_hi = rate_ci(flag[tp_other].to_numpy(), rng)
            rows.append({"data_origin": "real_eval", "variant": variant, "rule": name,
                         "status": ("EXPLORATORY" if name.startswith("X") else "registered"),
                         "malacca_fp_flagged": m, "ci_lo": m_lo, "ci_hi": m_hi,
                         "n_malacca_fp": int(mal.sum()),
                         "cost_tp_flagged_other_regions": t, "cost_ci_lo": t_lo,
                         "cost_ci_hi": t_hi, "n_tp_other": int(tp_other.sum())})
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    days = pd.read_csv(_IN / "days.csv")
    baselines = json.loads((_IN / "baselines.json").read_text(encoding="utf-8"))
    surrogate = pd.read_csv(_IN / "surrogate.csv")
    precedent_sim = {}
    for region in REGIONS:
        for line in (_OUT / "aer" / f"{region}.jsonl").open(encoding="utf-8"):
            rec = json.loads(line)
            sims = [p["similarity"] for p in rec["precedents"]]
            precedent_sim[(region, rec["date"])] = max(sims) if sims else 0.0

    ep, cutoffs = episode_table(days, baselines, surrogate, precedent_sim)
    phase1 = pd.read_csv(_OUT / "alerts_episodes.csv")
    check = ep[ep.variant == "with_duration"].groupby("region").fp.agg(["count", "sum"])
    ref = phase1.groupby("region").episode_label.agg(
        count="count", sum=lambda s: int((s == "FP").sum()))
    if not check.sort_index().equals(ref.sort_index().astype(check.dtypes)):
        raise SystemExit(f"episode table differs from Phase 1:\n{check}\n{ref}")

    rng = np.random.default_rng(BOOT_SEED)
    results, retro = single_indicator(ep, rng)
    loro_rows, held = loro(ep, rng)
    mal = malacca(ep, flags(ep, cutoffs), held, rng)

    results.to_csv(_OUT / "triage_results.csv", index=False)
    retro.to_csv(_OUT / "triage_retrospective.csv", index=False)
    loro_rows.to_csv(_OUT / "triage_loro.csv", index=False)
    mal.to_csv(_OUT / "triage_malacca.csv", index=False)
    ep.to_csv(_OUT / "triage_inputs" / "episodes.csv", index=False)
    (_OUT / "triage_inputs" / "cutoffs.json").write_text(json.dumps(
        {f"{v}|{r}": c for (v, r), c in cutoffs.items()}, indent=2), encoding="utf-8")
    print(f"Phase 4 analysis done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
