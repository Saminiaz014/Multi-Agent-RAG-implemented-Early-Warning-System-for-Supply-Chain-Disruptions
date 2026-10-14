"""The Tier 5 alert set of one region, with and without the shipping duration branch.

Both the scorecard and the triage analysis work on the same object: the
region's per-agent test scores, the composite and its train-p90 operating
point, the alert days grouped into episodes, and the calm baselines and noise
scales the explanations are measured against.

Two variants, because the real label is the shipping level-shift statistic:

* ``with_duration`` — the composite the CSV scored (bit-identical to
  :meth:`~src.evaluation.path_d.RegionEval.tier_scores`).
* ``without_duration`` — shipping reduced to its shock score
  (isolation forest + max |z|); the operating point, baselines and noise
  scales are all recomputed on the train window, so nothing is inherited from
  the variant that contains the label's statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation import path_d
from src.explainability.alert_explanation import episodes, noise_scales

TIER = 5
VARIANTS: tuple[str, ...] = ("with_duration", "without_duration")


@dataclass
class AlertSet:
    """Everything needed to explain one region's alerts under one variant."""

    region: str
    variant: str
    data: path_d.RegionEval
    agents: list[str]
    weights: dict[str, float]
    test_scores: dict[str, np.ndarray]
    train_scores: dict[str, np.ndarray]
    composite: np.ndarray
    threshold: float
    dates: list[str]
    y: np.ndarray
    baselines: dict[str, float]
    sigma: dict[str, float]
    episodes: list[list[int]] = field(default_factory=list)

    @property
    def mass(self) -> float:
        return sum(self.weights[a] for a in self.agents)

    @property
    def alert_idx(self) -> np.ndarray:
        return np.flatnonzero(self.composite > self.threshold)

    @property
    def include_duration(self) -> bool:
        return self.variant == "with_duration"

    def episode_ids(self) -> list[str]:
        return [f"{self.region}_E{k:03d}" for k in range(1, len(self.episodes) + 1)]

    def episode_labels(self) -> list[str]:
        """TP if any labelled-disruption day falls inside the episode's span."""
        return ["TP" if self.y[g[0]:g[-1] + 1].any() else "FP" for g in self.episodes]


def _composite(per_agent: dict[str, np.ndarray], agents: list[str],
               weights: dict[str, float]) -> np.ndarray:
    """Path D composite in the script's accumulation order."""
    total = np.zeros(len(per_agent[agents[0]]))
    mass = 0.0
    for a in agents:
        total += weights[a] * per_agent[a]
        mass += weights[a]
    return total / mass


def build_alert_set(region: str, variant: str = "with_duration") -> AlertSet:
    """Assemble the Tier 5 alert set for ``region`` under ``variant``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    data = path_d.load_region(region)
    agents = data.tier_agents(TIER)
    weights = dict(path_d.CSV_WEIGHTS)

    test = {a: data.agents[a].test_scores for a in agents}
    train = {a: data.agents[a].train_scores for a in agents}
    if variant == "without_duration" and "shipping" in agents:
        run = data.agents["shipping"]
        test["shipping"] = np.nan_to_num(run.test_frame["shock_score"].to_numpy(float), nan=0.0)
        train["shipping"] = np.nan_to_num(run.train_frame["shock_score"].to_numpy(float), nan=0.0)

    composite = _composite(test, agents, weights)
    threshold = float(np.quantile(np.nan_to_num(_composite(train, agents, weights)), 0.90))
    calm = data.y_train == 0
    dates = pd.to_datetime(data.test["timestamp"]).dt.strftime("%Y-%m-%d").tolist()
    out = AlertSet(
        region=region,
        variant=variant,
        data=data,
        agents=agents,
        weights=weights,
        test_scores=test,
        train_scores=train,
        composite=composite,
        threshold=threshold,
        dates=dates,
        y=data.y_test,
        baselines={a: float(np.median(train[a][calm])) for a in agents},
        sigma=noise_scales(train, calm),
    )
    out.episodes = episodes(dates, composite > threshold)
    return out
