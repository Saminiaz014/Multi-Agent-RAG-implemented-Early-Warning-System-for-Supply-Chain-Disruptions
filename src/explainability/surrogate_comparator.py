"""The SHAP surrogate, demoted to a comparator for the exact attribution.

Two surrogates are compared against the exact agent Shapley values:

* ``surrogate_prod`` — the surrogate exactly as the pipeline and dashboard
  build it (:func:`~src.explainability.shap_explainer.build_shap_training_data`:
  a 100-tree random forest fitted to one synthetic seed-42 year), applied to
  the real evaluation features. This is what a user of the system is shown.
  It is not retrained.
* ``surrogate_refit`` — the same recipe (random forest, 17 features, tree
  SHAP) fitted on the region's *real* train window to that window's Path D
  composite. It is the strongest version of the surrogate approach, so a gap
  between it and the exact attribution is a property of surrogates, not of a
  training-distribution mismatch.

Agent-level surrogate attributions sum feature SHAP values per producing
agent (:data:`~src.explainability.shap_explainer.FEATURE_AGENT_MAP`). Features
of agents outside the composite still receive SHAP mass (their inputs are
zero-filled, as the production explainer does); that mass is reported, not
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from src.explainability.shap_explainer import (
    ALL_FEATURE_NAMES,
    FEATURE_AGENT_MAP,
    SurrogateShapExplainer,
)

_SEED = 42


@dataclass
class Surrogate:
    """A fitted surrogate and its honesty numbers."""

    name: str
    rf: object
    explainer: object
    r2: dict[str, float]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.rf.predict(align(X))

    def shap(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values, shape (n, 17), in :data:`ALL_FEATURE_NAMES` order."""
        raw = self.explainer.shap_values(align(X))
        return np.asarray(raw[0] if isinstance(raw, list) else raw, dtype=float)


def align(X: pd.DataFrame) -> np.ndarray:
    """The production column alignment: missing features become 0.0."""
    return SurrogateShapExplainer._align_features(X)


def real_features(frame: pd.DataFrame) -> pd.DataFrame:
    """``domain__feature`` evaluation columns -> the surrogate's feature names."""
    renamed = {c: c.split("__", 1)[1] for c in frame.columns if "__" in c}
    out = frame.rename(columns=renamed)
    return out[[c for c in ALL_FEATURE_NAMES if c in out.columns]].reset_index(drop=True)


def _r2_triple(X: pd.DataFrame, y: np.ndarray, holdout: tuple | None = None) -> tuple[dict[str, float], object]:
    """Train R², out-of-time R², and 5-fold shuffled CV R² for the RF recipe."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_score

    rf = RandomForestRegressor(n_estimators=100, random_state=_SEED, n_jobs=-1)
    Xa = align(X)
    rf.fit(Xa, y)
    out = {"r2_train": float(r2_score(y, rf.predict(Xa)))}
    if holdout is None:  # chronological 70/30 inside the training data
        cut = int(len(y) * 0.7)
        oot = RandomForestRegressor(n_estimators=100, random_state=_SEED, n_jobs=-1)
        oot.fit(Xa[:cut], y[:cut])
        out["r2_chrono_holdout"] = float(r2_score(y[cut:], oot.predict(Xa[cut:])))
    else:  # the real test window
        Xh, yh = holdout
        out["r2_chrono_holdout"] = float(r2_score(yh, rf.predict(align(Xh))))
    cv = KFold(5, shuffle=True, random_state=0)
    out["r2_cv5"] = float(np.mean(cross_val_score(
        RandomForestRegressor(n_estimators=100, random_state=_SEED, n_jobs=-1),
        Xa, y, cv=cv, scoring="r2")))
    return out, rf


@lru_cache(maxsize=1)
def production_surrogate() -> Surrogate:
    """The surrogate as the pipeline builds it (Hormuz config, seed-42 synthetic)."""
    import shap

    from src.core.config_manager import load_config_for_region
    from src.explainability.shap_explainer import build_shap_training_data

    X, y = build_shap_training_data(load_config_for_region("hormuz"))
    r2, rf = _r2_triple(X, np.asarray(y, dtype=float))
    return Surrogate("surrogate_prod", rf, shap.TreeExplainer(rf), r2)


def refit_surrogate(train_X: pd.DataFrame, train_y: np.ndarray,
                    test_X: pd.DataFrame, test_y: np.ndarray) -> Surrogate:
    """The surrogate recipe fitted to one region's real train window."""
    import shap

    r2, rf = _r2_triple(train_X, np.asarray(train_y, dtype=float),
                        holdout=(test_X, np.asarray(test_y, dtype=float)))
    return Surrogate("surrogate_refit", rf, shap.TreeExplainer(rf), r2)


def agent_attribution(shap_row: np.ndarray, agents: list[str]) -> tuple[dict[str, float], float]:
    """Sum feature SHAP per agent in ``agents``; also the |SHAP| share outside them."""
    phi = dict.fromkeys(agents, 0.0)
    outside = 0.0
    for value, feature in zip(shap_row, ALL_FEATURE_NAMES):
        agent = FEATURE_AGENT_MAP[feature]
        if agent in phi:
            phi[agent] += float(value)
        else:
            outside += abs(float(value))
    total = float(np.sum(np.abs(shap_row)))
    return phi, (outside / total if total > 0 else 0.0)


def feature_attribution(shap_row: np.ndarray) -> dict[str, float]:
    """Feature name -> SHAP value."""
    return {f: float(v) for f, v in zip(ALL_FEATURE_NAMES, shap_row)}
