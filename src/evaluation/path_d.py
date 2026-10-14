"""Path D — the real-data evaluation scores, rebuilt for explanation.

``scripts/run_method_comparison.py`` produces ``eval/method_comparison_results.csv``
by fitting each agent on the first 70% of a region's series and scoring the
held-out 30%; a tier's composite is the weighted mean of its agents' scores.
Explaining those numbers means reproducing them exactly, so this module does
not re-implement any of it. It loads that script as a module and calls its own
``_load``, ``_split``, ``_agent_frame`` and ``_score_metrics``.

What it adds is the agents' *intermediate* columns. The script keeps only each
agent's final ``anomaly_score``; the explanation layer also needs the
sub-scores the agents already compute on the way (shipping's ``shock_score``
and ``duration_score``, the disaster agent's ``raw_composite``, ...). They are
read from the same ``detect`` output, never recomputed.

**Weights.** The CSV was generated with the pre-routing-removal weight vector
(:data:`CSV_WEIGHTS`). ``settings.yaml`` now carries a 3-decimal
renormalisation of it, which moves Tier >= 2 AUC by about 1e-4
(``experiments/ablation_v2/ABLATION_V2_REPORT.md``). Every Path D explanation
uses :data:`CSV_WEIGHTS` so it explains the numbers the thesis reports.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _PROJECT_ROOT / "scripts" / "run_method_comparison.py"

#: The inter-agent weights ``eval/method_comparison_results.csv`` was built
#: with. Tier scores renormalise over the tier's agents, so only the ratios
#: matter; the current ``settings.yaml`` vector is these ratios rounded.
CSV_WEIGHTS: dict[str, float] = {
    "shipping": 0.25,
    "market": 0.15,
    "geopolitical": 0.25,
    "natural_disaster": 0.10,
    "news_sentiment": 0.10,
}

#: Regions in the evaluation set, in the order the CSV lists them.
REGIONS: tuple[str, ...] = ("bab_el_mandeb", "hormuz", "malacca", "panama")

#: Every Path D artefact carries this provenance tag.
DATA_ORIGIN = "real_eval"


@lru_cache(maxsize=1)
def comparison_module() -> ModuleType:
    """Import ``scripts/run_method_comparison.py`` without running it.

    Its ``main()`` is guarded by ``__name__ == "__main__"``, so importing it
    never touches the CSV.
    """
    spec = importlib.util.spec_from_file_location("_run_method_comparison", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_run_method_comparison", module)
    spec.loader.exec_module(module)
    return module


def csv_config(region: str) -> dict:
    """The region's merged config with the inter-agent weights the CSV used."""
    from src.core.config_manager import load_config_for_region

    config = load_config_for_region(region)
    return {**config, "weights": dict(CSV_WEIGHTS)}


@dataclass
class AgentRun:
    """One agent fitted on a region's train window.

    Attributes:
        agent: The fitted agent (read for its weights; never mutated).
        test_scores: Per-day ``anomaly_score`` on the test window, aligned to
            its timestamps exactly as the script aligns them (missing -> 0.0).
        train_scores: The same, fitted on train and scored on train — what the
            script uses to set each tier's train-p90 operating point.
        test_frame: The agent's full ``detect`` output on the test window,
            reindexed to the test timestamps (intermediate columns included).
        train_frame: The same for the train window.
    """

    agent: object
    test_scores: np.ndarray
    train_scores: np.ndarray
    test_frame: pd.DataFrame
    train_frame: pd.DataFrame


def _score_window(domain: str, config: dict, fit_on: pd.DataFrame, score_on: pd.DataFrame):
    """Mirror of the script's ``_agent_scores`` that also keeps the frame.

    Returns ``(agent, aligned_scores, aligned_frame)`` or ``None`` when the
    script would have skipped the agent.
    """
    from src.agents.disaster_agent import DisasterAgent
    from src.agents.geopolitical_agent import GeopoliticalAgent
    from src.agents.market_agent import MarketAgent
    from src.agents.news_agent import NewsAgent
    from src.agents.shipping_agent import ShippingAgent

    classes = {
        "shipping": ShippingAgent, "market": MarketAgent,
        "geopolitical": GeopoliticalAgent, "natural_disaster": DisasterAgent,
        "news_sentiment": NewsAgent,
    }
    mod = comparison_module()
    fit_frame = mod._agent_frame(fit_on, domain)
    score_frame = mod._agent_frame(score_on, domain)
    if fit_frame is None or score_frame is None:
        return None
    try:
        agent = classes[domain](config=dict(config.get("agents", {}).get(domain, {})))
        agent.fit(fit_frame)
        scored = agent.detect(agent.preprocess(score_frame))
    except Exception as exc:  # same tolerance as the script
        logger.warning("agent %s unavailable: %s", domain, str(exc)[:120])
        return None
    index = pd.to_datetime(score_frame["timestamp"])
    frame = scored.set_index(pd.to_datetime(scored["timestamp"])).drop(columns="timestamp")
    frame = frame.reindex(index)
    scores = np.nan_to_num(frame["anomaly_score"].to_numpy(dtype=float), nan=0.0)
    return agent, scores, frame


@dataclass
class RegionEval:
    """Everything Path D knows about one region.

    Attributes:
        region: Region key.
        train: Train-window rows of ``data/eval/{region}.csv``.
        test: Test-window rows.
        agents: ``{domain: AgentRun}`` for every domain the script could build,
            in :data:`_TIER_ORDER` order.
    """

    region: str
    train: pd.DataFrame
    test: pd.DataFrame
    agents: dict[str, AgentRun] = field(default_factory=dict)

    @property
    def y_test(self) -> np.ndarray:
        return self.test["y_true"].astype(int).to_numpy()

    @property
    def y_train(self) -> np.ndarray:
        return self.train["y_true"].astype(int).to_numpy()

    def tier_agents(self, tier: int) -> list[str]:
        """The agents the CSV's Tier ``tier`` actually used."""
        order = comparison_module()._TIER_ORDER[:tier]
        return [d for d in order if d in self.agents]

    def tier_scores(self, tier: int, window: str = "test",
                    weights: dict[str, float] | None = None) -> np.ndarray:
        """Tier composite, accumulated in the script's order and arithmetic."""
        weights = weights or CSV_WEIGHTS
        n = len(self.test) if window == "test" else len(self.train)
        total = np.zeros(n)
        mass = 0.0
        for domain in self.tier_agents(tier):
            run = self.agents[domain]
            scores = run.test_scores if window == "test" else run.train_scores
            weight = float(weights.get(domain, 0.1))
            total += weight * scores
            mass += weight
        return total / mass

    def tier_threshold(self, tier: int) -> float:
        """The tier's operating point: 90th percentile of its train scores."""
        train = np.nan_to_num(self.tier_scores(tier, "train"))
        return float(np.quantile(train, 0.90))


@lru_cache(maxsize=8)
def load_region(region: str) -> RegionEval:
    """Fit every agent the script fits for ``region`` and keep their frames.

    Cached per process: the agents are deterministic (the isolation forest is
    seeded), so one fit serves every caller.
    """
    mod = comparison_module()
    frame, _ = mod._load(region)
    train, test = mod._split(frame)
    config = csv_config(region)
    out = RegionEval(region=region, train=train, test=test)
    for domain in mod._TIER_ORDER:
        on_test = _score_window(domain, config, train, test)
        on_train = _score_window(domain, config, train, train)
        if on_test is None or on_train is None:
            continue
        agent, test_scores, test_frame = on_test
        _, train_scores, train_frame = on_train
        out.agents[domain] = AgentRun(
            agent=agent,
            test_scores=test_scores,
            train_scores=train_scores,
            test_frame=test_frame,
            train_frame=train_frame,
        )
    return out


def eval_data_available() -> bool:
    """Whether the gitignored ``data/eval`` cache exists on this machine."""
    return all((_PROJECT_ROOT / "data" / "eval" / f"{r}.csv").exists() for r in REGIONS)
