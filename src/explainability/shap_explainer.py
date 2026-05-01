"""SHAP-based explainability wrapper.

Wraps a fitted scikit-learn estimator and exposes per-feature SHAP
contribution scores, making model decisions interpretable for analysts.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ShapExplainer:
    """Generate SHAP feature-contribution explanations for a fitted model.

    Supports tree-based models (via :class:`shap.TreeExplainer`) and any
    other sklearn-compatible estimator (via :class:`shap.KernelExplainer`
    with a small background sample).

    Args:
        model: A fitted sklearn-compatible estimator.
        feature_names: Ordered list of feature column names.
        background_data: Background dataset for KernelExplainer (required
            when ``model`` is not tree-based). Ignored for tree models.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: pd.DataFrame | None = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self._explainer: Any | None = None
        self._background = background_data
        self._build_explainer()

    def _build_explainer(self) -> None:
        """Instantiate the appropriate SHAP explainer for the model type."""
        try:
            shap_module = importlib.import_module("shap")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "The 'shap' package is required for ShapExplainer. Install it with 'pip install shap'."
            ) from exc

        try:
            self._explainer = shap_module.TreeExplainer(self.model)
            logger.info("Using TreeExplainer for %s.", type(self.model).__name__)
        except Exception:
            if self._background is None:
                raise ValueError(
                    "background_data is required for non-tree models."
                )
            self._explainer = shap_module.KernelExplainer(
                self.model.decision_function,
                shap_module.sample(self._background, 50),
            )
            logger.info("Using KernelExplainer for %s.", type(self.model).__name__)

    def explain(self, df: pd.DataFrame) -> dict:
        """Compute SHAP values for each sample in ``df``.

        Args:
            df: Feature DataFrame with the same columns used during model
                training.

        Returns:
            Dictionary with keys:
                - ``shap_values`` (np.ndarray): shape (n_samples, n_features).
                - ``mean_abs_shap`` (dict[str, float]): mean |SHAP| per feature.
                - ``feature_names`` (list[str]): ordered feature labels.
        """
        if self._explainer is None:
            raise RuntimeError("Explainer has not been built — call _build_explainer.")

        X = df[self.feature_names].to_numpy()
        shap_values = self._explainer.shap_values(X)

        # IsolationForest TreeExplainer returns a list [normal, anomaly]; take anomaly.
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        mean_abs = {
            name: float(np.mean(np.abs(shap_values[:, i])))
            for i, name in enumerate(self.feature_names)
        }
        sorted_contributions = dict(
            sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)
        )
        return {
            "shap_values": shap_values,
            "mean_abs_shap": sorted_contributions,
            "feature_names": self.feature_names,
        }

    def top_features(self, df: pd.DataFrame, n: int = 5) -> list[tuple[str, float]]:
        """Return the top-n features ranked by mean absolute SHAP value.

        Args:
            df: Feature DataFrame.
            n: Number of top features to return.

        Returns:
            List of (feature_name, mean_abs_shap) tuples, descending order.
        """
        explanation = self.explain(df)
        items = list(explanation["mean_abs_shap"].items())
        return items[:n]
