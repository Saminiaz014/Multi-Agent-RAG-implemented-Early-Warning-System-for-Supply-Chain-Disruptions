"""Build Alert Explanation Records for every Tier 5 alert on the real test window.

Reads Path D (``src/evaluation/path_d.py``), which reproduces
``eval/method_comparison_results.csv`` exactly; never runs the method
comparison itself. An alert is a test day whose Tier 5 composite exceeds the
tier's train-p90 operating point — the CSV's own definition.

Writes (all under ``eval/explainability/``):

* ``baselines.json`` — calm baselines per region x agent (real) and for the
  synthetic seed-42 world.
* ``aer/{region}.jsonl`` — one AER per alert day.
* ``alerts_episodes.csv`` — one row per alert episode.

Usage::

    python eval/explainability/run_aer.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.config_manager import load_config_for_region  # noqa: E402
from src.evaluation import path_d  # noqa: E402
from src.explainability.agent_breakdown import breakdown, present_columns  # noqa: E402
from src.explainability.alert_explanation import (  # noqa: E402
    build_aer,
    episodes,
    noise_scales,
)
from src.explainability.baselines import real_baselines, synthetic_baselines  # noqa: E402

_OUT = _ROOT / "eval" / "explainability"
TIER = 5
WEIGHT_MODE = "hand_tuned"  # Path D: the CSV's weight vector (path_d.CSV_WEIGHTS)


def _composite(per_agent: dict[str, np.ndarray], agents: list[str]) -> np.ndarray:
    """Path D composite in the script's accumulation order."""
    total = np.zeros(len(next(iter(per_agent.values()))))
    mass = 0.0
    for a in agents:
        w = path_d.CSV_WEIGHTS[a]
        total += w * per_agent[a]
        mass += w
    return total / mass


def _no_duration(data: path_d.RegionEval, agents: list[str]) -> tuple[np.ndarray, float]:
    """Tier composite with shipping reduced to its shock score, and its own p90.

    The label is the shipping level-shift statistic; this is the same alert
    rule with the duration branch removed, re-thresholded on train so the
    operating point is recalibrated rather than inherited.
    """
    def series(window: str) -> dict[str, np.ndarray]:
        out = {}
        for a in agents:
            run = data.agents[a]
            if a == "shipping":
                frame = run.test_frame if window == "test" else run.train_frame
                out[a] = np.nan_to_num(frame["shock_score"].to_numpy(dtype=float), nan=0.0)
            else:
                out[a] = run.test_scores if window == "test" else run.train_scores
        return out

    test = _composite(series("test"), agents)
    threshold = float(np.quantile(np.nan_to_num(_composite(series("train"), agents)), 0.90))
    return test, threshold


def episode_counts(dates: list[str], alert: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """Alert-day and episode TP/FP counts for one alert rule."""
    groups = episodes(dates, alert)
    tp_episodes = sum(1 for g in groups if y[g[0]:g[-1] + 1].any())
    return {
        "alert_days": int(alert.sum()),
        "tp_days": int((alert & (y == 1)).sum()),
        "fp_days": int((alert & (y == 0)).sum()),
        "episodes": len(groups),
        "tp_episodes": tp_episodes,
        "fp_episodes": len(groups) - tp_episodes,
    }


def region_summary(region: str) -> dict:
    """Episode counts with and without the shipping duration branch."""
    data = path_d.load_region(region)
    agents = data.tier_agents(TIER)
    scores = data.tier_scores(TIER)
    threshold = data.tier_threshold(TIER)
    dates = pd.to_datetime(data.test["timestamp"]).dt.strftime("%Y-%m-%d").tolist()
    y = data.y_test
    nd_scores, nd_threshold = _no_duration(data, agents)
    return {
        "agents": agents,
        "threshold": threshold,
        "with_duration": episode_counts(dates, scores > threshold, y),
        "threshold_without_duration": nd_threshold,
        "without_duration": episode_counts(dates, nd_scores > nd_threshold, y),
    }


def explain_region(region: str, baselines: dict[str, float]) -> tuple[list[dict], list[dict]]:
    data = path_d.load_region(region)
    agents = data.tier_agents(TIER)
    scores = data.tier_scores(TIER)
    threshold = data.tier_threshold(TIER)
    alert = scores > threshold
    y = data.y_test
    dates = pd.to_datetime(data.test["timestamp"]).dt.strftime("%Y-%m-%d").tolist()

    sigma = noise_scales({a: data.agents[a].train_scores for a in agents}, data.y_train == 0)
    present = {a: present_columns(data.agents[a].test_frame) for a in agents}
    nd_scores, nd_threshold = _no_duration(data, agents) if "shipping" in agents else (scores, threshold)

    groups = episodes(dates, alert)
    records: list[dict] = []
    episode_rows: list[dict] = []
    for k, members in enumerate(groups, start=1):
        episode_id = f"{region}_E{k:03d}"
        episode_records = []
        for idx in members:
            day_scores = {a: float(data.agents[a].test_scores[idx]) for a in agents}
            breakdowns = {
                a: breakdown(a, data.agents[a].agent, data.agents[a].test_frame.iloc[idx], present[a])
                for a in agents
            }
            aer = build_aer(
                alert_id=f"{region}_{dates[idx]}",
                episode_id=episode_id,
                region=region,
                date=dates[idx],
                path="D",
                weight_mode=WEIGHT_MODE,
                data_origin=path_d.DATA_ORIGIN,
                tier=TIER,
                agents=agents,
                weights=path_d.CSV_WEIGHTS,
                scores=day_scores,
                baselines=baselines,
                threshold=threshold,
                breakdowns=breakdowns,
                noise_sigma=sigma,
                y_true=int(y[idx]),
                circularity_control={
                    "shipping_branch": breakdowns.get("shipping", {}).get("branch_winner"),
                    "score_without_duration": float(nd_scores[idx]),
                    "threshold_without_duration": nd_threshold,
                    "alert_without_duration": bool(nd_scores[idx] > nd_threshold),
                },
            ).to_dict()
            # Guard: the record explains exactly the composite the CSV scored.
            assert abs(aer["score"] - scores[idx]) < 1e-12
            episode_records.append(aer)

        start, end = members[0], members[-1]
        span_truth = y[start:end + 1]
        top_agents = Counter(r["top_drivers"][0]["agent"] for r in episode_records)
        episode_rows.append({
            "data_origin": path_d.DATA_ORIGIN,
            "path": "D",
            "weight_mode": WEIGHT_MODE,
            "tier": TIER,
            "region": region,
            "episode_id": episode_id,
            "start": dates[start],
            "end": dates[end],
            "n_alert_days": len(members),
            "n_tp_days": int(sum(y[i] for i in members)),
            "n_fp_days": int(sum(1 - y[i] for i in members)),
            "episode_label": "TP" if span_truth.any() else "FP",
            "max_score": float(max(r["score"] for r in episode_records)),
            "max_margin": float(max(r["margin"] for r in episode_records)),
            "threshold": threshold,
            "top_agent_mode": top_agents.most_common(1)[0][0],
            "shipping_duration_won_days": sum(
                1 for r in episode_records
                if r["circularity_control"].get("shipping_branch") == "duration"
            ),
            "n_alert_days_without_duration": sum(
                1 for r in episode_records if r["circularity_control"]["alert_without_duration"]
            ),
        })
        records.extend(episode_records)
    return records, episode_rows


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger().setLevel(logging.ERROR)
    (_OUT / "aer").mkdir(parents=True, exist_ok=True)

    real = {r: real_baselines(path_d.load_region(r)) for r in path_d.REGIONS}
    synthetic = synthetic_baselines(load_config_for_region("hormuz"))
    (_OUT / "baselines.json").write_text(json.dumps({
        "definition": "median agent score on calm days of the window the agent was fitted on",
        "real_eval": {
            "data_origin": "real_eval",
            "window": "train (first 70%), y_true == 0, fitted on train and scored on train",
            "weights": path_d.CSV_WEIGHTS,
            "regions": real,
        },
        "synthetic_seed42": {
            "data_origin": "synthetic_seed42",
            "window": "seed-42 train realisation, is_disruption == False",
            "weight_mode": "hand_tuned",
            "agents": synthetic,
        },
    }, indent=2), encoding="utf-8")

    all_episodes: list[dict] = []
    worst_residual = 0.0
    for region in path_d.REGIONS:
        records, rows = explain_region(region, real[region])
        with (_OUT / "aer" / f"{region}.jsonl").open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        all_episodes.extend(rows)
        if records:
            worst_residual = max(worst_residual, max(abs(r["efficiency_residual"]) for r in records))
        print(f"[{region}] {len(records)} AERs, {len(rows)} episodes")

    pd.DataFrame(all_episodes).to_csv(_OUT / "alerts_episodes.csv", index=False)
    summary = {
        "data_origin": path_d.DATA_ORIGIN,
        "path": "D",
        "tier": TIER,
        "episode_gap_days": 3,
        "max_abs_efficiency_residual": worst_residual,
        "regions": {r: region_summary(r) for r in path_d.REGIONS},
    }
    (_OUT / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"max |efficiency residual| = {worst_residual:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
