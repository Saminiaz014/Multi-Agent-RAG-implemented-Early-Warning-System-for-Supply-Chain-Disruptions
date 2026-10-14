"""METRIC 9 — explainability scorecard, exact attribution vs surrogate.

For every region and both circularity variants (shipping duration branch
included / excluded) this explains every Tier 5 alert day three ways — exact
agent Shapley (primary), the production SHAP surrogate, and the same surrogate
recipe refitted on the region's real train window — and scores the
explanations with :mod:`src.evaluation.explainability_metrics`.

Unit of evidence: the alert *episode*. Day values are averaged within each
episode; the reported value is the mean over episodes with a 1,000-resample
episode bootstrap 95% CI. Day-level means are written as secondary rows
(``level == "day"``), never as the headline.

Writes (under ``eval/explainability/``):

* ``scorecard.csv`` — region x metric x method x variant.
* ``fp_driver_composition.csv`` — top exact driver of TP vs FP episodes.
* ``market_saturation.csv`` — how often the market agent sits at its ceiling.
* ``fp_characterisation.csv`` — knowledge-base evidence near each FP episode
  (read from a temporary *copy* of ChromaDB).

Usage::

    python eval/explainability/run_scorecard.py
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import sys
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import path_d  # noqa: E402
from src.evaluation.explainability_metrics import (  # noqa: E402
    bootstrap_auc,
    deletion_insertion,
    episode_bootstrap,
    k_cover,
    order_by,
    random_order_auc,
    spearman,
    top1,
    top1_agreement,
)
from src.explainability.agent_breakdown import present_columns  # noqa: E402
from src.explainability.alert_explanation import score_uncertainty  # noqa: E402
from src.explainability.alert_sets import VARIANTS, AlertSet, _composite, build_alert_set  # noqa: E402
from src.explainability.exact_shapley import (  # noqa: E402
    efficiency_residual,
    minimal_drop,
    shapley_agents,
    value_fn_path_d,
)
from src.explainability.feature_shapley import (  # noqa: E402
    calm_inputs,
    coalition_scores,
    feature_shapley,
    inputs,
    players_of,
)
from src.explainability.shap_explainer import FEATURE_AGENT_MAP  # noqa: E402
from src.explainability.surrogate_comparator import (  # noqa: E402
    agent_attribution,
    feature_attribution,
    production_surrogate,
    real_features,
    refit_surrogate,
)

_OUT = _ROOT / "eval" / "explainability"
_KB = _ROOT / "data" / "knowledge_base" / ".chromadb"
TIER = 5
NOISE_SIGMAS = (0.01, 0.05, 0.10)
NOISE_DRAWS = 20
SURROGATES = ("surrogate_prod", "surrogate_refit")
#: Agents whose within-agent feature attribution is scored for E2 (per brief).
E2_FEATURE_AGENTS = ("shipping", "market")

#: Words marking a news item as geopolitical / conflict (FP characterisation).
GEO_PATTERN = re.compile(
    r"\b(attack\w*|strike\w*|missile\w*|drone\w*|houthi\w*|militar\w*|nav(y|al)|warship\w*|"
    r"seiz\w*|hijack\w*|pira\w*|sanction\w*|war|conflict\w*|tension\w*|threat\w*|"
    r"escalat\w*|clash\w*|blockade\w*|militant\w*|terror\w*|hostil\w*)\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------- per day
def _surrogate_feature_phi(shap_feat: dict[str, float], agent: str, players: list[str]) -> dict:
    """Surrogate SHAP for an agent's players (a player with no surrogate feature gets 0)."""
    return {p: (shap_feat.get(p, 0.0) if FEATURE_AGENT_MAP.get(p) == agent else 0.0)
            for p in players}


def explain_days(aset: AlertSet, surrogates: dict) -> list[dict]:
    """Every per-day quantity the scorecard needs, one dict per alert day."""
    data = aset.data
    agents, weights, mass = aset.agents, aset.weights, aset.mass
    value_fn = value_fn_path_d(weights, agents)
    calm_mask = data.y_train == 0
    present = {a: present_columns(data.agents[a].test_frame) for a in agents}
    feat_calm = {
        a: calm_inputs(a, data.agents[a].agent, data.agents[a].train_frame, calm_mask,
                       present[a], aset.include_duration)
        for a in agents
    }
    test_X = real_features(data.test)
    alert_idx = aset.alert_idx
    sur_shap = {name: s.shap(test_X.iloc[alert_idx]) for name, s in surrogates.items()}
    sur_pred = {name: s.predict(test_X.iloc[alert_idx]) for name, s in surrogates.items()}
    episode_of = {i: k for k, g in enumerate(aset.episodes) for i in g}

    days = []
    for j, idx in enumerate(alert_idx):
        scores = {a: float(aset.test_scores[a][idx]) for a in agents}
        base = aset.baselines

        def v_agents(coalition: frozenset, scores=scores) -> float:
            return value_fn({a: (scores[a] if a in coalition else base[a]) for a in agents})

        phi = shapley_agents(scores, base, value_fn)
        day = {
            "idx": int(idx), "episode": episode_of[int(idx)], "y": int(aset.y[idx]),
            "f": float(aset.composite[idx]), "phi_exact": phi,
            "residual": efficiency_residual(phi, scores, base, value_fn),
        }

        # Feature level, exact: within-agent Shapley scaled into the composite.
        feat_exact: dict[str, float] = {}
        feat_games: dict[str, tuple] = {}
        feat_residual = 0.0
        for a in agents:
            run = data.agents[a]
            row = run.test_frame.iloc[idx]
            res = feature_shapley(a, run.agent, row, feat_calm[a], present[a], aset.include_duration)
            if res is None:
                continue
            feat_residual = max(feat_residual, abs(res["residual"]))
            for p, value in res["phi"].items():
                feat_exact[f"{a}:{p}"] = weights[a] / mass * value
            if a in E2_FEATURE_AGENTS:
                full = inputs(a, run.agent, row, present[a], aset.include_duration)
                table = coalition_scores(a, run.agent, full, feat_calm[a])
                feat_games[a] = (players_of(a, full), table, res["phi"])
        day["phi_feature_exact"] = feat_exact
        day["feature_residual"] = feat_residual

        # Surrogates.
        day["phi_sur"], day["outside_share"], day["local_error"] = {}, {}, {}
        day["phi_feature_sur"] = {}
        for name in surrogates:
            phi_s, outside = agent_attribution(sur_shap[name][j], agents)
            day["phi_sur"][name] = phi_s
            day["outside_share"][name] = outside
            day["local_error"][name] = abs(day["f"] - float(sur_pred[name][j]))
            feats = feature_attribution(sur_shap[name][j])
            day["phi_feature_sur"][name] = {
                f"{FEATURE_AGENT_MAP[f]}:{f}": v for f, v in feats.items()
                if FEATURE_AGENT_MAP[f] in agents
            }

        # E2 agent level.
        e2 = {"exact": deletion_insertion(order_by(phi, agents), v_agents),
              "random": random_order_auc(agents, v_agents)}
        for name in surrogates:
            e2[name] = deletion_insertion(order_by(day["phi_sur"][name], agents), v_agents)
        day["e2_agent"] = e2

        # E2 feature level inside shipping and market.
        day["e2_feature"] = {}
        for a, (players, table, phi_f) in feat_games.items():
            v = table.__getitem__
            res = {"exact": deletion_insertion(order_by(phi_f, players), v),
                   "random": random_order_auc(players, v)}
            for name in surrogates:
                sur = _surrogate_feature_phi(feature_attribution(sur_shap[name][j]), a, players)
                res[name] = deletion_insertion(order_by(sur, players), v)
            day["e2_feature"][a] = res

        # E3 compactness.
        day["k80_agent"] = {"exact": k_cover(phi),
                            **{n: k_cover(day["phi_sur"][n]) for n in surrogates}}
        day["k80_feature"] = {"exact": k_cover(feat_exact),
                              **{n: k_cover(day["phi_feature_sur"][n]) for n in surrogates}}

        # E4b noise stability (exact; agent-score noise, fixed seed per day).
        rng = np.random.default_rng(int(idx) * 7919 + len(aset.region))
        day["noise"] = {}
        for sigma in NOISE_SIGMAS:
            rhos, tops = [], []
            for _ in range(NOISE_DRAWS):
                noisy = {a: float(np.clip(scores[a] + rng.normal(0, sigma), 0, 1)) for a in agents}
                phi_n = {a: weights[a] * (noisy[a] - base[a]) / mass for a in agents}
                rhos.append(spearman(phi, phi_n))
                tops.append(top1_agreement(phi, phi_n))
            day["noise"][sigma] = (float(np.nanmean(rhos)) if np.isfinite(rhos).any()
                                   else float("nan"), float(np.mean(tops)))

        # E5 coherence.
        day["coherent"] = {n: top1_agreement(phi, day["phi_sur"][n]) for n in surrogates}

        # E6 contrastivity.
        lead = top1(phi)
        required = minimal_drop(lead, scores, value_fn, aset.threshold)
        day["cf_feasible"] = float(required is not None and required >= base[lead])
        day["cf_drop"] = float("nan") if required is None else scores[lead] - required

        # E7 uncertainty.
        unc = score_uncertainty(scores, value_fn, aset.threshold, aset.sigma,
                                seed=int(idx) + 1, n_draws=200)
        day["dispersion"] = unc["agent_dispersion"]
        day["band_width"] = unc["bootstrap_hi"] - unc["bootstrap_lo"]
        day["frac_below"] = 1.0 - unc["frac_draws_above_threshold"]
        days.append(day)
    return days


# ----------------------------------------------------------------- rows
class Rows:
    """Accumulates scorecard rows for one region x variant."""

    def __init__(self, aset: AlertSet, days: list[dict]) -> None:
        self.aset, self.days, self.rows = aset, days, []
        self.n_episodes = len(aset.episodes)
        self.labels = aset.episode_labels()
        self.n_tp = sum(1 for lab in self.labels if lab == "TP")

    def _base(self) -> dict:
        return {"data_origin": path_d.DATA_ORIGIN, "path": "D", "weight_mode": "hand_tuned",
                "tier": TIER, "variant": self.aset.variant, "region": self.aset.region,
                "n_episodes": self.n_episodes, "n_tp_episodes": self.n_tp,
                "n_alert_days": len(self.days)}

    def add_point(self, mid: str, metric: str, method: str, value: float,
                  level: str = "point", note: str = "", lo=np.nan, hi=np.nan,
                  n_days_used=np.nan) -> None:
        self.rows.append({**self._base(), "metric_id": mid, "metric": metric, "method": method,
                          "level": level, "value": value, "ci_lo": lo, "ci_hi": hi,
                          "n_days_used": n_days_used, "note": note})

    def add(self, mid: str, metric: str, method: str, get, note: str = "") -> None:
        """Episode-level mean + bootstrap CI, and the day-level mean as secondary."""
        per_episode: dict[int, list[float]] = defaultdict(list)
        day_values = []
        for d in self.days:
            value = get(d)
            if value is not None and np.isfinite(value):
                per_episode[d["episode"]].append(value)
                day_values.append(value)
        episode_means = [np.mean(v) for v in per_episode.values()]
        mean, lo, hi = episode_bootstrap(episode_means)
        self.add_point(mid, metric, method, mean, "episode", note, lo, hi, len(day_values))
        self.add_point(mid, metric, method,
                       float(np.mean(day_values)) if day_values else float("nan"),
                       "day", "secondary: alert days are not independent",
                       n_days_used=len(day_values))


def stability_pairs(days: list[dict], key) -> dict[int, list[tuple[float, float]]]:
    """(Spearman, top-1 agreement) for consecutive alert days within each episode."""
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for d in days:
        by_episode[d["episode"]].append(d)
    out = {}
    for ep, members in by_episode.items():
        pairs = [(spearman(key(a), key(b)), top1_agreement(key(a), key(b)))
                 for a, b in zip(members, members[1:])]
        if pairs:
            out[ep] = pairs
    return out


def scorecard_rows(aset: AlertSet, days: list[dict], surrogates: dict) -> list[dict]:
    r = Rows(aset, days)
    if not days:
        return r.rows
    # E1 faithfulness.
    r.add_point("E1", "max |efficiency residual| (agent level)", "exact",
                max(abs(d["residual"]) for d in days))
    r.add_point("E1", "max |efficiency residual| (feature level)", "exact",
                max(d["feature_residual"] for d in days))
    for name, s in surrogates.items():
        for key, value in s.r2.items():
            note = ("on its own synthetic seed-42 year" if name == "surrogate_prod"
                    else "fitted on this region's real train window; hold-out = test window")
            r.add_point("E1", key, name, value, note=note)
        r.add("E1", "local error |f(x) - g(x)| on alert days", name,
              lambda d, n=name: d["local_error"][n])
        r.add("E1", "share of |SHAP| on agents outside the composite", name,
              lambda d, n=name: d["outside_share"][n])

    # E2 deletion / insertion (days where the score is >= MIN_GAP above baseline).
    for method in ("exact", *surrogates, "random"):
        r.add("E2", "agent deletion AUC (lower is better)", method,
              lambda d, m=method: d["e2_agent"][m][0])
        r.add("E2", "agent insertion AUC (higher is better)", method,
              lambda d, m=method: d["e2_agent"][m][1])
        for a in E2_FEATURE_AGENTS:
            if a not in aset.agents:
                continue
            r.add("E2", f"{a} feature deletion AUC (lower is better)", method,
                  lambda d, m=method, a=a: d["e2_feature"].get(a, {}).get(m, (np.nan,))[0])
            r.add("E2", f"{a} feature insertion AUC (higher is better)", method,
                  lambda d, m=method, a=a: d["e2_feature"].get(a, {}).get(m, (np.nan, np.nan))[1])

    # E3 compactness.
    for method in ("exact", *surrogates):
        r.add("E3", "k80 agents", method, lambda d, m=method: d["k80_agent"][m])
        r.add("E3", "k80 features", method, lambda d, m=method: d["k80_feature"][m])

    # E4a stability across consecutive days within an episode.
    keys = {"exact": lambda d: d["phi_exact"],
            **{n: (lambda d, n=n: d["phi_sur"][n]) for n in surrogates}}
    for method, key in keys.items():
        pairs = stability_pairs(days, key)
        rho = [np.nanmean([p[0] for p in v]) if np.isfinite([p[0] for p in v]).any() else np.nan
               for v in pairs.values()]
        agree = [np.mean([p[1] for p in v]) for v in pairs.values()]
        for metric, values in (("day-to-day Spearman rho within episode", rho),
                               ("day-to-day top-1 agreement within episode", agree)):
            mean, lo, hi = episode_bootstrap(values)
            r.add_point("E4", metric, method, mean, "episode",
                        f"{len(values)} multi-day episodes", lo, hi)
    # E4b stability under agent-score noise (exact only).
    for sigma in NOISE_SIGMAS:
        r.add("E4", f"Spearman rho under score noise sigma={sigma}", "exact",
              lambda d, s=sigma: d["noise"][s][0], note="agent-score space; no surrogate analogue")
        r.add("E4", f"top-1 agreement under score noise sigma={sigma}", "exact",
              lambda d, s=sigma: d["noise"][s][1])

    # E5 coherence.
    for name in surrogates:
        r.add("E5", "top agent agrees with exact", name, lambda d, n=name: d["coherent"][n])
    r.add_point("E5", "top agent agrees with precedents' primary_agents", "exact", np.nan,
                note="pending Phase 3 (precedent retrieval)")

    # E6 contrastivity.
    r.add("E6", "share of alert days with a feasible single-agent counterfactual", "exact",
          lambda d: d["cf_feasible"])
    r.add("E6", "required score drop of the top agent", "exact", lambda d: d["cf_drop"])

    # E7 uncertainty usefulness (FP = positive class), episode level.
    per_episode = defaultdict(list)
    for d in days:
        per_episode[d["episode"]].append(d)
    labels = [1 if r.labels[k] == "FP" else 0 for k in sorted(per_episode)]
    for feature in ("dispersion", "band_width", "frac_below"):
        values = [np.mean([d[feature] for d in per_episode[k]]) for k in sorted(per_episode)]
        point, lo, hi = bootstrap_auc(values, labels)
        r.add_point("E7", f"AUC of {feature} for FP vs TP episodes", "exact", point, "episode",
                    "undefined: no TP episodes" if not np.isfinite(point) else "", lo, hi)
    r.add_point("E7", "calibration", "exact", np.nan,
                note="not estimable: 0 train-window positives in 3 of 4 regions; label is "
                     "the shipping level-shift statistic")

    # E8, E9 come later.
    r.add_point("E8", "precedent grounding", "exact", np.nan, note="pending Phase 3")
    r.add_point("E9", "narrative faithfulness", "exact", np.nan, note="pending Phase 6")
    return r.rows


# ------------------------------------------------------------ extra outputs
def driver_composition(aset: AlertSet, days: list[dict]) -> list[dict]:
    """Top exact driver per episode (argmax of the episode-summed Shapley value)."""
    sums: dict[int, Counter] = defaultdict(Counter)
    for d in days:
        sums[d["episode"]].update(d["phi_exact"])
    labels = aset.episode_labels()
    top = {k: max(c, key=c.get) for k, c in sums.items()}
    rows = []
    for label in ("TP", "FP"):
        eps = [k for k in top if labels[k] == label]
        for agent in aset.agents:
            hits = np.array([1.0 if top[k] == agent else 0.0 for k in eps])
            share, lo, hi = episode_bootstrap(hits) if len(hits) else (np.nan,) * 3
            rows.append({
                "data_origin": path_d.DATA_ORIGIN, "path": "D", "variant": aset.variant,
                "region": aset.region, "episode_label": label, "agent": agent,
                "n_episodes": int(hits.sum()) if len(hits) else 0,
                "n_label_episodes": len(eps), "share": share, "share_ci_lo": lo,
                "share_ci_hi": hi,
            })
    return rows


def market_saturation(aset: AlertSet, days: list[dict]) -> dict:
    data = aset.data
    run = data.agents["market"]
    test = run.test_scores
    calm_train = run.train_scores[data.y_train == 0]
    per_episode: dict[int, list] = defaultdict(list)
    for d in days:
        phi = d["phi_exact"]
        total = sum(abs(v) for v in phi.values()) or np.nan
        per_episode[d["episode"]].append((abs(phi["market"]), abs(phi["market"]) / total))
    mag = [np.mean([p[0] for p in v]) for v in per_episode.values()]
    share = [np.mean([p[1] for p in v]) for v in per_episode.values()]
    m, m_lo, m_hi = episode_bootstrap(mag)
    s, s_lo, s_hi = episode_bootstrap(share)
    return {
        "data_origin": path_d.DATA_ORIGIN, "path": "D", "variant": aset.variant,
        "region": aset.region,
        "market_calm_baseline": aset.baselines["market"],
        "share_test_days_ge_0.95 (day-level)": float(np.mean(test >= 0.95)),
        "share_test_days_clipped_at_1 (day-level)": float(np.mean(test >= 1.0 - 1e-12)),
        "share_train_calm_days_ge_0.95 (day-level)": float(np.mean(calm_train >= 0.95)),
        "mean_abs_phi_market_on_alerts": m, "ci_lo": m_lo, "ci_hi": m_hi,
        "mean_share_abs_phi_market": s, "share_ci_lo": s_lo, "share_ci_hi": s_hi,
        "mean_abs_phi_market_day_level": float(np.mean([abs(d["phi_exact"]["market"]) for d in days])),
        "n_episodes": len(aset.episodes),
    }


def _kb_documents() -> pd.DataFrame:
    """All live knowledge-base documents, read from a temporary copy of ChromaDB."""
    import chromadb

    tmp = Path(tempfile.mkdtemp(prefix="xai_kb_"))
    try:
        shutil.copytree(_KB, tmp / "kb")
        client = chromadb.PersistentClient(path=str(tmp / "kb"))
        got = client.get_collection("live_extracted_context").get(include=["metadatas", "documents"])
        frame = pd.DataFrame([{**m, "doc_id": i, "text": t}
                              for i, m, t in zip(got["ids"], got["metadatas"], got["documents"])])
        del client
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame["text_hash"] = frame["text"].map(lambda t: hashlib.sha1(str(t).encode("utf-8")).hexdigest())
    return frame


def fp_characterisation(asets: list[AlertSet], days_by_region: dict[str, list[dict]]) -> list[dict]:
    """Knowledge-base evidence of geopolitical risk around each FP episode."""
    kb = _kb_documents()
    news = kb[kb["source_api"].isin(["newsapi", "serpapi"])].copy()
    news["geo"] = (news.get("title", "").fillna("") + " " + news["text"]).str.contains(GEO_PATTERN)
    acled_daily = kb[(kb["source_api"] == "acled") & (kb["event_type"] != "conflict_summary")
                     & (kb["event_type"] != "historical_scenario")]
    acled_annual = kb[(kb["source_api"] == "acled") & (kb["event_type"] == "conflict_summary")]

    rows = []
    for aset in asets:
        days = days_by_region[aset.region]
        sums: dict[int, Counter] = defaultdict(Counter)
        for d in days:
            sums[d["episode"]].update(d["phi_exact"])
        labels, ids = aset.episode_labels(), aset.episode_ids()
        test_start = pd.Timestamp(aset.dates[0])
        test_end = pd.Timestamp(aset.dates[-1])
        region_news = news[news["region"] == aset.region]
        coverage = int(((region_news["event_date"] >= test_start)
                        & (region_news["event_date"] <= test_end)).sum())
        for k, members in enumerate(aset.episodes):
            if labels[k] != "FP":
                continue
            start = pd.Timestamp(aset.dates[members[0]])
            end = pd.Timestamp(aset.dates[members[-1]])
            lo = start - pd.Timedelta(days=7)
            in_window = region_news[(region_news["event_date"] >= lo)
                                    & (region_news["event_date"] <= end)]
            in_window = in_window.drop_duplicates("text_hash")
            geo_news = in_window[in_window["geo"]]
            daily = acled_daily[(acled_daily["region"] == aset.region)
                                & (acled_daily["event_date"] >= lo)
                                & (acled_daily["event_date"] <= end)]
            annual = acled_annual[(acled_annual["region"] == aset.region)
                                  & (acled_annual["event_date"].dt.year == start.year)
                                  & (acled_annual["event_date"] <= end)]
            matched = len(geo_news) + len(daily)
            # How far the nearest region news item of any date is: separates
            # "no event in the window" from "the knowledge base has no
            # coverage near this date at all".
            gaps = [(lo - d).days if d < lo else max((d - end).days, 0)
                    for d in region_news["event_date"].dropna()]
            nearest = int(min(gaps)) if gaps else None
            rows.append({
                "data_origin": path_d.DATA_ORIGIN, "path": "D", "variant": aset.variant,
                "region": aset.region, "episode_id": ids[k],
                "start": start.date().isoformat(), "end": end.date().isoformat(),
                "n_alert_days": len(members),
                "top_exact_driver": max(sums[k], key=sums[k].get),
                "window_start": lo.date().isoformat(), "window_end": end.date().isoformat(),
                "n_news_any": len(in_window), "n_news_geopolitical": len(geo_news),
                "n_acled_daily": len(daily),
                "acled_annual_military_events": float(pd.to_numeric(
                    annual.get("military_events"), errors="coerce").sum()) if len(annual) else np.nan,
                "kb_news_docs_in_region_test_window": coverage,
                "nearest_region_news_days": nearest,
                "matched_titles": " | ".join(geo_news.get("title", pd.Series(dtype=str))
                                             .fillna("").head(3)),
                "tag": "risk_without_traffic_drop" if matched else "unexplained",
            })
    return rows


# --------------------------------------------------------------------- main
def _check_against_aer(aset: AlertSet, days: list[dict]) -> None:
    """The scorecard explains the same alerts, with the same values, as the AERs."""
    path = _OUT / "aer" / f"{aset.region}.jsonl"
    if aset.variant != "with_duration" or not path.exists():
        return
    import json
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    assert len(records) == len(days), (aset.region, len(records), len(days))
    for rec, d in zip(records, days):
        for a, v in rec["agent_shapley"].items():
            assert abs(v - d["phi_exact"][a]) < 1e-12, (rec["alert_id"], a)


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    prod = production_surrogate()
    print(f"surrogate_prod R²: {prod.r2}")

    scorecard, composition, saturation = [], [], []
    fp_inputs: tuple[list, dict] = ([], {})
    for variant in VARIANTS:
        for region in path_d.REGIONS:
            aset = build_alert_set(region, variant)
            train_composite = _composite(aset.train_scores, aset.agents, aset.weights)
            refit = refit_surrogate(real_features(aset.data.train), train_composite,
                                    real_features(aset.data.test), aset.composite)
            # The production surrogate's fit on this region's real test window.
            prod_real = float(_r2_real(prod, aset))
            surrogates = {"surrogate_prod": prod, "surrogate_refit": refit}
            days = explain_days(aset, surrogates)
            _check_against_aer(aset, days)
            rows = scorecard_rows(aset, days, surrogates)
            for row in rows:
                if row["metric"] == "r2_chrono_holdout" and row["method"] == "surrogate_prod":
                    rows.append({**row, "metric": "r2_on_real_test_window", "value": prod_real,
                                 "note": "synthetic-trained surrogate vs this region's Path D "
                                         "composite, all test days"})
                    break
            scorecard.extend(rows)
            composition.extend(driver_composition(aset, days))
            saturation.append(market_saturation(aset, days))
            if variant == "with_duration":
                fp_inputs[0].append(aset)
                fp_inputs[1][region] = days
            print(f"[{variant}] {region}: {len(days)} alert days, {len(aset.episodes)} episodes")

    pd.DataFrame(scorecard).to_csv(_OUT / "scorecard.csv", index=False)
    pd.DataFrame(composition).to_csv(_OUT / "fp_driver_composition.csv", index=False)
    pd.DataFrame(saturation).to_csv(_OUT / "market_saturation.csv", index=False)
    pd.DataFrame(fp_characterisation(*fp_inputs)).to_csv(_OUT / "fp_characterisation.csv",
                                                         index=False)
    print("wrote scorecard.csv, fp_driver_composition.csv, market_saturation.csv, "
          "fp_characterisation.csv")
    return 0


def _r2_real(surrogate, aset: AlertSet) -> float:
    from sklearn.metrics import r2_score

    return r2_score(aset.composite, surrogate.predict(real_features(aset.data.test)))


if __name__ == "__main__":
    sys.exit(main())
