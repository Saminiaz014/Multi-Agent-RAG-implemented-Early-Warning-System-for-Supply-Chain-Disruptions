"""Compute metrics D1-D15 for a baseline's anomaly scores vs. ground truth.

D1/D2 (volume-under-surface PR/ROC) require a time-series-anomaly library
(e.g. tslearn) that isn't in this project's dependencies yet; D11-D15
(event-based / range-based / affiliation / point-adjusted F1 variants)
require segment-level evaluation logic beyond a pointwise sklearn metric.
Both groups are left as declared NaN placeholders rather than silently
approximated — see the metric-spec doc for their intended definitions.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# D1/D2 (VUS-PR, VUS-ROC) and D11-D15 (event/range/affiliation/PA-F1/PA%K)
# are not yet implemented — see module docstring.
_UNIMPLEMENTED_METRICS: tuple[str, ...] = (
    "D1_vus_pr",
    "D2_vus_roc",
    "D11_event_f1",
    "D12_range_f1",
    "D13_affiliation_f1",
    "D14_pa_f1",
    "D15_pa_k_0_2",
)


class BaselineEvaluator:
    """Compute D1-D15 metrics on anomaly scores vs. ground truth."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        anomaly_scores: np.ndarray,
        scenario_id: str,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Compute metrics D1-D15.

        Args:
            y_true: Binary labels (1 = disruption, 0 = normal).
            anomaly_scores: Continuous scores in ``[0, 1]``.
            scenario_id: Used only for log messages.
            threshold: Operating point (tau) for the threshold-dependent
                metrics (D5, D7-D10); should be pre-declared from a
                validation split, not fit on this same data.

        Returns:
            Dict of D1-D15 metric values (native Python floats; NaN for
            not-yet-implemented metrics — see module docstring).
        """
        y_true = np.asarray(y_true)
        anomaly_scores = np.asarray(anomaly_scores)
        metrics: dict[str, float] = {name: float("nan") for name in _UNIMPLEMENTED_METRICS}

        # D4: AUC-ROC. Undefined (raises) when y_true has only one class,
        # which happens for every control scenario whose event window
        # doesn't overlap the split being scored (e.g. N_QUIET/N_DECOY).
        try:
            metrics["D4_auc_roc"] = float(roc_auc_score(y_true, anomaly_scores))
        except ValueError:
            logger.debug("[%s] D4_auc_roc undefined (single-class y_true).", scenario_id)
            metrics["D4_auc_roc"] = float("nan")

        # D3: AUC-PR.
        try:
            precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
            metrics["D3_auc_pr"] = float(auc(recall, precision))
        except ValueError:
            logger.debug("[%s] D3_auc_pr undefined (single-class y_true).", scenario_id)
            metrics["D3_auc_pr"] = float("nan")

        # Threshold-dependent metrics (D5, D7-D10) at the pre-declared tau.
        y_pred = (anomaly_scores >= threshold).astype(int)

        metrics["D5_f1_tau"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["D7_precision_tau"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["D8_recall_tau"] = float(recall_score(y_true, y_pred, zero_division=0))

        tn = int(((1 - y_true) * (1 - y_pred)).sum())
        fp = int(((1 - y_true) * y_pred).sum())
        metrics["D9_fpr_tau"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        metrics["D10_macro_f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

        # D6: best achievable F1 across a full threshold sweep (an oracle
        # upper bound, not the declared operating point). The sweep skips
        # 0.0: with ">=" a constant-zero score array (never_alarm) would
        # match at tau=0.0 and be credited with alarming on every day,
        # which is exactly the behavior this baseline is meant to rule out.
        best_f1 = 0.0
        for tau in np.linspace(0.01, 1.0, 100):
            y_p = (anomaly_scores >= tau).astype(int)
            best_f1 = max(best_f1, f1_score(y_true, y_p, zero_division=0))
        metrics["D6_best_f1"] = float(best_f1)

        return metrics
