"""Weight-set evaluator backing the Optuna objective.

Given one candidate weight set (Layer-1 intra-agent weights, Layer-2
inter-agent weights, Layer-3 thresholds), this module runs the full
six-agent pipeline on a pair of data splits and scores it:

    fit  on the *fit* split   (Isolation-Forest baselines learn here)
    score on the *eval* split (metrics are computed here)

The detection models are fitted on one realisation and evaluated on an
independent one, so a weight set is rewarded only when it generalises across
noise. The composite daily risk is aggregated exactly as
:class:`~src.aggregation.risk_engine.RiskEngine` does it — renormalised
weighted mean across active agents, then a non-linear agreement bonus — so
the number the optimizer maximises is the number the live pipeline produces.

Three metrics are combined into the scalar objective:

* **F1** — precision/recall balance of HIGH-risk alerts vs ground truth.
* **lead-time score** — how many days before each disruption the composite
  first crosses the MEDIUM alert line, normalised by a 5-day horizon.
* **FPR** — false-positive rate of HIGH-risk alerts (penalised).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.agents.disaster_agent import DisasterAgent
from src.agents.geopolitical_agent import GeopoliticalAgent
from src.agents.market_agent import MarketAgent
from src.agents.news_agent import NewsAgent
from src.agents.routing_agent import RoutingAgent
from src.agents.shipping_agent import ShippingAgent

logger = logging.getLogger(__name__)

# Maximum early-warning horizon, in days, used to normalise the lead-time
# score. A disruption flagged 5+ days early scores a full 1.0.
_MAX_LEAD_DAYS: int = 5
# Per-agent mean score above which an agent "agrees" a disruption is underway
# (mirrors RiskEngine._AGREEMENT_THRESHOLD).
_AGREEMENT_THRESHOLD: float = 0.5


@dataclass
class EvalMetrics:
    """Metrics produced by one pipeline evaluation."""

    f1: float
    precision: float
    recall: float
    fpr: float
    lead_time_days: float
    lead_time_score: float
    objective: float
    n_days: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "fpr": self.fpr,
            "lead_time_days": self.lead_time_days,
            "lead_time_score": self.lead_time_score,
            "objective": self.objective,
            "n_days": self.n_days,
        }


class PipelineEvaluator:
    """Run + score the six-agent pipeline for a candidate weight set.

    Args:
        data_manager: A :class:`~src.optimization.data_split.DataSplitManager`
            whose splits have been generated.
        objective_weights: ``{"f1", "lead_time", "fpr_penalty"}`` blend used
            to collapse the three metrics into the scalar objective.
    """

    def __init__(self, data_manager, objective_weights: dict | None = None) -> None:
        self.data_manager = data_manager
        ow = objective_weights or {}
        self.w_f1: float = float(ow.get("f1", 0.50))
        self.w_lead: float = float(ow.get("lead_time", 0.30))
        self.w_fpr: float = float(ow.get("fpr_penalty", 0.20))
        # Audit trail of which splits have been *evaluated* on — lets the
        # leakage test prove the test split is untouched during optimization.
        self.evaluated_splits: list[str] = []

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------
    @staticmethod
    def build_agents(params: dict) -> dict:
        """Instantiate the six agents with a trial's weights + thresholds.

        Args:
            params: Structured parameter dict with ``intra`` (per-agent
                weights) and ``thresholds`` sub-dicts.

        Returns:
            ``{connector_name: agent}`` for all six agents.
        """
        intra = params["intra"]
        thr = params["thresholds"]

        shipping = ShippingAgent(
            config={"contamination": 0.05, "z_threshold": 2.0}
        )
        shipping.set_weights(
            intra["shipping"]["isolation_forest"], intra["shipping"]["zscore"]
        )
        shipping.set_threshold(thr["shipping_threshold"])

        market = MarketAgent(config={"baseline_years": 5, "threshold": 0.50})
        market.set_weights(
            intra["market"]["oil"],
            intra["market"]["trade_volume"],
            intra["market"]["freight"],
        )
        market.set_z_threshold(thr["market_z_threshold"])

        geo = GeopoliticalAgent(config={})
        geo.set_weights(
            intra["geopolitical"]["sanctions"],
            intra["geopolitical"]["military"],
            intra["geopolitical"]["diplomatic"],
            intra["geopolitical"]["stability"],
        )
        geo.set_threshold(thr["geopolitical_threshold"])

        disaster = DisasterAgent(config={})
        disaster.set_weights(
            intra["natural_disaster"]["earthquake"],
            intra["natural_disaster"]["tsunami"],
            intra["natural_disaster"]["cyclone"],
            intra["natural_disaster"]["severe_weather"],
        )
        disaster.set_threshold(
            thr["disaster_threshold"], thr.get("disaster_single_event")
        )

        routing = RoutingAgent(config={})
        routing.set_weights(
            intra["routing"]["model_score"], intra["routing"]["transit_zscore"]
        )
        routing.set_threshold(thr["routing_threshold"])

        news = NewsAgent(config={})
        news.set_weights(
            intra["news_sentiment"]["sentiment"],
            intra["news_sentiment"]["consensus"],
            intra["news_sentiment"]["velocity"],
            intra["news_sentiment"]["volume"],
        )
        news.set_threshold(negative_threshold=thr["news_negative_threshold"])

        return {
            "shipping": shipping,
            "market": market,
            "geopolitical": geo,
            "natural_disaster": disaster,
            "routing": routing,
            "news_sentiment": news,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, params: dict, fit_split: str, eval_split: str) -> EvalMetrics:
        """Fit on ``fit_split``, score on ``eval_split``, return metrics.

        Args:
            params: Structured weight/threshold dict (see :meth:`build_agents`).
            fit_split: Split whose realisation calibrates the detectors.
            eval_split: Split whose realisation the metrics are computed on.

        Returns:
            :class:`EvalMetrics` for the evaluation split.
        """
        splits = self.data_manager.get_splits()
        self.evaluated_splits.append(eval_split)
        agents = self.build_agents(params)
        fit_frames = splits[fit_split]
        eval_frames = splits[eval_split]

        score_series: dict[str, pd.Series] = {}
        for name, agent in agents.items():
            try:
                agent.fit(fit_frames[name])
                validated = agent.run_dataframe(eval_frames[name])
                if "anomaly_score" not in validated or "timestamp" not in validated:
                    continue
                series = pd.Series(
                    validated["anomaly_score"].to_numpy(dtype=float),
                    index=pd.to_datetime(validated["timestamp"]),
                )
                score_series[name] = series[~series.index.duplicated(keep="first")]
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[PipelineEvaluator] agent %s failed: %s", name, exc)

        if not score_series:
            return EvalMetrics(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0)

        scores_df = pd.DataFrame(score_series).sort_index()
        risk = self._aggregate_daily(scores_df, params)
        y_true = self.data_manager.get_ground_truth(eval_split)
        return self._compute_metrics(risk, y_true, params["thresholds"])

    # ------------------------------------------------------------------
    # Internal: aggregation + metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_daily(scores_df: pd.DataFrame, params: dict) -> pd.Series:
        """Composite daily risk — mirrors ``RiskEngine.compute_risk`` vectorised."""
        inter = params["inter_weights"]
        thr = params["thresholds"]
        bonus_3 = float(thr.get("agreement_bonus_3", 1.15))
        bonus_5 = float(thr.get("agreement_bonus_5", 1.25))

        cols = [c for c in scores_df.columns if inter.get(c, 0.0) > 0]
        if not cols:
            return pd.Series(0.0, index=scores_df.index)
        weight_total = sum(float(inter[c]) for c in cols)
        filled = scores_df[cols].fillna(0.0)

        base = np.zeros(len(filled), dtype=float)
        for c in cols:
            base += (float(inter[c]) / weight_total) * filled[c].to_numpy()

        agreement = (filled.to_numpy() > _AGREEMENT_THRESHOLD).sum(axis=1)
        amplification = np.where(
            agreement >= 5, bonus_5, np.where(agreement >= 3, bonus_3, 1.0)
        )
        risk = np.minimum(base * amplification, 1.0)
        return pd.Series(risk, index=filled.index)

    def _compute_metrics(
        self, risk: pd.Series, y_true: pd.Series, thresholds: dict
    ) -> EvalMetrics:
        """Compute F1 / FPR / lead-time on aligned days."""
        risk_high = float(thresholds["risk_high"])
        risk_medium = float(thresholds["risk_medium"])

        idx = risk.index
        yt = y_true.reindex(idx).fillna(False).astype(bool).to_numpy()
        rk = risk.to_numpy()

        # HIGH-risk alert is the positive prediction for F1 / FPR.
        pred = rk >= risk_high
        tp = int((yt & pred).sum())
        fp = int((~yt & pred).sum())
        fn = int((yt & ~pred).sum())
        tn = int((~yt & ~pred).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Lead time: earliest MEDIUM-level alert in the 5-day run-up to each
        # ground-truth disruption window.
        alert_med = rk >= risk_medium
        lead_days = self._lead_times(yt, alert_med)
        lead_time_days = float(np.mean(lead_days)) if lead_days else 0.0
        lead_time_score = float(np.clip(lead_time_days / _MAX_LEAD_DAYS, 0.0, 1.0))

        objective = (
            self.w_f1 * f1
            + self.w_lead * lead_time_score
            - self.w_fpr * fpr
        )

        return EvalMetrics(
            f1=f1,
            precision=precision,
            recall=recall,
            fpr=fpr,
            lead_time_days=lead_time_days,
            lead_time_score=lead_time_score,
            objective=float(objective),
            n_days=len(idx),
            extra={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        )

    @staticmethod
    def _lead_times(y_true: np.ndarray, alert_med: np.ndarray) -> list[int]:
        """Per-window early-warning lead, clamped to ``[0, _MAX_LEAD_DAYS]``."""
        leads: list[int] = []
        n = len(y_true)
        i = 0
        while i < n:
            if y_true[i]:
                start = i
                while i < n and y_true[i]:
                    i += 1
                # Earliest MEDIUM alert within [start - MAX_LEAD, start].
                lead = 0
                for d in range(max(0, start - _MAX_LEAD_DAYS), start + 1):
                    if alert_med[d]:
                        lead = start - d
                        break
                leads.append(int(min(lead, _MAX_LEAD_DAYS)))
            else:
                i += 1
        return leads
