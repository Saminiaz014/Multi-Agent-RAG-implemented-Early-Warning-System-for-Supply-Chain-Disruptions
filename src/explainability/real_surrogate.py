"""SHAP surrogate fitted on real corridor data instead of the synthetic series.

The deployed surrogate (:class:`~src.explainability.shap_explainer.SurrogateShapExplainer`)
learns the composite from one synthetic year. This module provides the pieces
to fit the same model on each region's real frame (``data/eval/``):

* :func:`canonical_features` — the 17 features of :data:`ALL_FEATURE_NAMES`;
  a feature the region does not supply is NaN, never zero;
* :func:`temporal_split` — first 70 % of rows train, last 30 % test;
* :func:`composite_from_agent_scores` — the target, from the existing
  :meth:`RiskEngine.compute_risk`, with its per-agent breakdown;
* :func:`fit_surrogate` and :func:`evaluate_surrogate` — the unchanged
  ``RandomForestRegressor(n_estimators=100, random_state=42)`` and its
  held-out fit and fidelity.

No scaler or imputer is used, so nothing is refitted on test rows; missing
values are handled by the forest's native NaN support.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.aggregation.risk_engine import RiskEngine
from src.agents.base_agent import DetectionResult
from src.explainability.shap_explainer import ALL_FEATURE_NAMES, FEATURE_AGENT_MAP

logger = logging.getLogger(__name__)

TRAIN_FRACTION: float = 0.70


def canonical_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The 17 canonical features from a ``domain__feature`` frame.

    Returns:
        ``(features, missing)``: one column per :data:`ALL_FEATURE_NAMES`
        entry, NaN throughout for a feature the frame does not contain, and
        the names of those missing features.
    """
    by_name = {c.split("__", 1)[1]: c for c in frame.columns if "__" in c}
    out = pd.DataFrame(index=frame.index)
    missing: list[str] = []
    for name in ALL_FEATURE_NAMES:
        if name in by_name:
            out[name] = pd.to_numeric(frame[by_name[name]], errors="coerce")
        else:
            out[name] = np.nan
            missing.append(name)
    return out, missing


def temporal_split(
    frame: pd.DataFrame, train_fraction: float = TRAIN_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First ``train_fraction`` of rows (by time) train, the rest test. No shuffling."""
    if "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp", kind="stable")
    cut = int(len(frame) * train_fraction)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def composite_from_agent_scores(
    agent_scores: dict[str, np.ndarray], config: dict
) -> pd.DataFrame:
    """Per-day composite from the existing aggregator.

    Args:
        agent_scores: Agent name → per-day anomaly scores (equal lengths).
            Agents that could not be built are simply absent, and
            :meth:`RiskEngine.compute_risk` renormalises over those present.
        config: Region config (weights and thresholds).

    Returns:
        One row per day: ``risk_score``, ``risk_level``, ``top_agent`` (largest
        pre-amplification contribution) and ``<agent>_contribution`` columns.
    """
    engine = RiskEngine(config)
    names = list(agent_scores)
    n = len(next(iter(agent_scores.values()))) if names else 0
    rows: list[dict[str, Any]] = []
    for i in range(n):
        outputs = [
            DetectionResult(
                agent_name=name,
                anomaly_scores=np.array([float(agent_scores[name][i])]),
                anomaly_flags=np.array([False]),
                feature_names=[],
            )
            for name in names
        ]
        result = engine.compute_risk(outputs)
        contributions = {a: v["contribution"] for a, v in result["contributing_agents"].items()}
        row = {
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "top_agent": max(contributions, key=contributions.get) if contributions else None,
        }
        row.update({f"{a}_contribution": v for a, v in contributions.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def fit_surrogate(X: pd.DataFrame, y: np.ndarray):
    """The deployed surrogate's model and hyperparameters, fitted on ``X``."""
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X.to_numpy(dtype=float), np.asarray(y, dtype=float))
    return model


def evaluate_surrogate(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    flagged_test: np.ndarray,
    top_agent_test: np.ndarray,
) -> dict:
    """In-sample and held-out fit, error on flagged days, and SHAP fidelity.

    Args:
        flagged_test: True on test days the aggregator rates medium or above.
        top_agent_test: The aggregator's largest-contribution agent per test day.
    """
    import shap
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    features = list(X_test.columns)
    pred_train = model.predict(X_train.to_numpy(dtype=float))
    pred_test = model.predict(X_test.to_numpy(dtype=float))
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    flagged = np.asarray(flagged_test, dtype=bool)

    metrics: dict[str, Any] = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "r2_train": float(r2_score(y_train, pred_train)),
        "r2_test": float(r2_score(y_test, pred_test)),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "spearman_test": float(spearmanr(pred_test, y_test).statistic),
        "share_test_medium_or_above": float(flagged.mean()) if len(flagged) else None,
        "n_test_medium_or_above": int(flagged.sum()),
        "mae_test_medium_or_above": (
            float(mean_absolute_error(y_test[flagged], pred_test[flagged])) if flagged.any() else None
        ),
        "bias_test_medium_or_above": (
            float(np.mean(pred_test[flagged] - y_test[flagged])) if flagged.any() else None
        ),
    }

    if flagged.any():
        sv = shap.TreeExplainer(model).shap_values(X_test.to_numpy(dtype=float)[flagged])
        order = np.argsort(-np.abs(sv), axis=1)
        top_agents = np.asarray(top_agent_test, dtype=object)[flagged]
        top1 = [FEATURE_AGENT_MAP[features[o[0]]] == a for o, a in zip(order, top_agents)]
        top3 = [a in {FEATURE_AGENT_MAP[features[j]] for j in o[:3]} for o, a in zip(order, top_agents)]
        metrics["top1_agent_agreement"] = float(np.mean(top1))
        metrics["top3_agent_agreement"] = float(np.mean(top3))
    else:
        metrics["top1_agent_agreement"] = None
        metrics["top3_agent_agreement"] = None
    return metrics
