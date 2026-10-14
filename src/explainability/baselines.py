"""Calm-day baselines: the reference point every Shapley value is measured from.

An agent "absent" from a coalition is set to its calm baseline — the median of
its own score on days the label calls normal, in the window it was fitted on.
A Shapley value is then "how much this agent's departure from an ordinary day
moved the composite". Zero would be the wrong reference: several agents never
sit at zero on a calm day (the geopolitical sigmoid floors near 0.05, the
shipping isolation score is spread around the middle of its training range).

Two sets are built:

* ``real_eval`` — per region, from each agent's train-window scores (fitted on
  train, scored on train, as the method-comparison script does to set its
  operating point) on days with ``y_true == 0``.
* ``synthetic_seed42`` — the dashboard's world: agents fitted on the seed-42
  train realisation and scored on it, days with ``is_disruption == False``.
  Weights are the hand-tuned layout, as the dashboard uses by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.path_d import RegionEval


def real_baselines(data: RegionEval) -> dict[str, float]:
    """Median train-window score per agent on ``y_true == 0`` days."""
    calm = data.y_train == 0
    return {
        domain: float(np.median(run.train_scores[calm]))
        for domain, run in data.agents.items()
    }


def _hand_tuned_params(config: dict) -> dict:
    """The dashboard's hand-tuned parameter layout, without importing Streamlit."""
    from src.optimization.weight_config import resolve_active_weights

    layout = resolve_active_weights({**config, "weight_mode": "hand_tuned"})
    return {
        "inter_weights": {k: float(v) for k, v in layout["inter_agent_weights"].items()},
        "intra": {k: dict(v) for k, v in layout["intra_agent_weights"].items()},
        "thresholds": {k: float(v) for k, v in layout["thresholds"].items()},
    }


def synthetic_calibration(config: dict, params: dict | None = None,
                          manager=None) -> tuple[dict[str, float], dict[str, float]]:
    """Calm baselines and noise scales from the seed-42 train realisation.

    Agents are fitted on seed-42 train and scored on it; calm days are the
    non-disruption days. Returns ``(baselines, sigma)``: each agent's calm
    median, and its per-day noise scale (std of calm day-to-day changes / sqrt 2,
    as for the real-data AERs).

    Args:
        config: Application config (split seeds).
        params: Evaluator parameter layout; defaults to the hand-tuned one.
        manager: An existing ``DataSplitManager`` to reuse its splits.
    """
    from src.explainability.alert_explanation import noise_scales
    from src.optimization.data_split import DataSplitManager
    from src.optimization.pipeline_evaluator import PipelineEvaluator

    manager = manager or DataSplitManager(config)
    splits = manager.get_splits()
    truth = manager.get_ground_truth("train")
    calm_dates = set(truth.index[~truth.to_numpy()])
    agents = PipelineEvaluator.build_agents(params or _hand_tuned_params(config))

    baselines: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for name, agent in agents.items():
        train = splits["train"][name]
        agent.fit(train)
        validated = agent.run_dataframe(train)
        scores = validated["anomaly_score"].to_numpy(dtype=float)
        calm = pd.to_datetime(validated["timestamp"]).isin(calm_dates).to_numpy()
        baselines[name] = float(np.median(scores[calm]))
        sigma.update(noise_scales({name: scores}, calm))
    return baselines, sigma


def synthetic_baselines(config: dict, params: dict | None = None,
                        manager=None) -> dict[str, float]:
    """Median seed-42 train score per agent on non-disruption days."""
    return synthetic_calibration(config, params, manager)[0]
