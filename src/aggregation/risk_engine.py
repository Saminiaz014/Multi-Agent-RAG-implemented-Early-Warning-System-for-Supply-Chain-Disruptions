"""Weighted risk aggregation engine.

Combines per-agent anomaly scores into a single composite risk score and
maps it to a human-readable risk level using configurable thresholds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd

from src.agents.base_agent import DetectionResult

logger = logging.getLogger(__name__)

# Per-agent mean score above which an agent is considered "in agreement"
# that a disruption is underway. Drives the non-linear agreement bonus.
_AGREEMENT_THRESHOLD: float = 0.5
# Multiplicative amplification applied to the base weighted risk when
# multiple independent agents corroborate a disruption.
_AGREEMENT_BONUS_3PLUS: float = 1.15
_AGREEMENT_BONUS_5PLUS: float = 1.25


class RiskLevel(str, Enum):
    """Categorical risk classification."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskEngine:
    """Aggregate agent outputs into a composite disruption risk score.

    The engine performs a weighted average of per-agent anomaly scores
    using weights defined in ``config["weights"]``. The composite score
    is then bucketed into :class:`RiskLevel` using ``config["thresholds"]``.

    Args:
        config: Top-level application config dict (reads ``weights`` and
            ``thresholds`` sub-keys).
    """

    def __init__(self, config: dict) -> None:
        self.weights: dict[str, float] = config.get("weights", {})
        self.threshold_critical: float = config["thresholds"]["risk_critical"]
        self.threshold_high: float = config["thresholds"]["risk_high"]
        self.threshold_medium: float = config["thresholds"]["risk_medium"]
        self._validate_weights()

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-3):
            logger.warning(
                "Agent weights sum to %.4f — expected 1.0. "
                "Scores will be renormalised.",
                total,
            )

    def aggregate(self, results: list[DetectionResult]) -> dict:
        """Compute a weighted composite risk score from agent results.

        Only agents whose names appear in ``self.weights`` contribute to
        the composite score. Agents with no configured weight are logged
        as warnings and skipped.

        Args:
            results: Detection results from all active agents.

        Returns:
            Dictionary with keys:
                - ``composite_score`` (float): weighted mean in [0, 1].
                - ``risk_level`` (str): one of CRITICAL / HIGH / MEDIUM / LOW.
                - ``agent_scores`` (dict[str, float]): mean score per agent.
        """
        if not results:
            logger.warning("No detection results provided — returning zero risk.")
            return {
                "composite_score": 0.0,
                "risk_level": RiskLevel.LOW,
                "agent_scores": {},
            }

        agent_scores: dict[str, float] = {}
        weighted_sum: float = 0.0
        weight_total: float = 0.0

        for result in results:
            agent_name = result.agent_name
            weight = self.weights.get(agent_name)
            if weight is None:
                logger.warning(
                    "Agent '%s' has no configured weight — skipping.", agent_name
                )
                continue
            mean_score = float(np.mean(result.anomaly_scores))
            agent_scores[agent_name] = mean_score
            weighted_sum += weight * mean_score
            weight_total += weight

        composite = weighted_sum / weight_total if weight_total > 0 else 0.0
        risk_level = self._classify(composite)

        logger.info(
            "Risk aggregation complete | composite=%.4f | level=%s",
            composite,
            risk_level,
        )
        return {
            "composite_score": composite,
            "risk_level": risk_level,
            "agent_scores": agent_scores,
        }

    def _classify(self, score: float) -> RiskLevel:
        """Map a numeric score to a :class:`RiskLevel`.

        Args:
            score: Composite risk score in [0, 1].

        Returns:
            Appropriate :class:`RiskLevel` enum member.
        """
        if score >= self.threshold_critical:
            return RiskLevel.CRITICAL
        if score >= self.threshold_high:
            return RiskLevel.HIGH
        if score >= self.threshold_medium:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    # ------------------------------------------------------------------
    # Spec API: non-linear, agreement-amplified risk with full breakdown
    # ------------------------------------------------------------------
    def classify_risk(self, score: float) -> str:
        """Map a numeric risk score to a three-level string label.

        Unlike :meth:`_classify` (which exposes the four-level
        :class:`RiskLevel` enum used by the legacy ``aggregate`` path), this
        returns the lowercase ``"high"`` / ``"medium"`` / ``"low"`` labels
        expected by the API and SHAP layers. Thresholds are read from
        ``config["thresholds"]`` to keep the engine fully config-driven.

        Args:
            score: Composite risk score in [0, 1].

        Returns:
            ``"high"`` if ``score >= risk_high``, ``"medium"`` if
            ``score >= risk_medium``, otherwise ``"low"``.
        """
        if score >= self.threshold_high:
            return "high"
        if score >= self.threshold_medium:
            return "medium"
        return "low"

    def compute_risk(self, agent_outputs: list[DetectionResult]) -> dict:
        """Aggregate agent outputs into an agreement-amplified risk score.

        This is the richer counterpart to :meth:`aggregate`. Beyond a plain
        weighted mean it (a) redistributes configured weights across only the
        active agents so they always sum to 1.0, (b) applies a non-linear
        agreement bonus when multiple independent agents corroborate a
        disruption, and (c) returns a full per-agent contribution breakdown
        for downstream SHAP and API consumption.

        Args:
            agent_outputs: Detection results from the active agents. Agents
                absent from this list — or absent from ``config["weights"]`` —
                are treated as disabled and have their weight redistributed.

        Returns:
            Dictionary with keys:
                - ``risk_score`` (float): amplified composite risk in [0, 1].
                - ``risk_level`` (str): ``"high"`` / ``"medium"`` / ``"low"``.
                - ``reason`` (str): human-readable explanation of what is
                  driving the score, for downstream API / dashboard consumers.
                - ``contributing_agents`` (dict): per-agent
                  ``{"score", "weight", "contribution"}`` (pre-amplification).
                - ``agent_agreement`` (int): agents scoring above the
                  agreement threshold.
                - ``timestamp`` (str): ISO-8601 UTC time of computation.
                - ``metadata`` (dict): ``active_agents`` count and the
                  redistributed ``weights_used``.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Keep only agents that are both present and have a configured weight.
        scores: dict[str, float] = {}
        raw_weights: dict[str, float] = {}
        for result in agent_outputs:
            weight = self.weights.get(result.agent_name)
            if weight is None:
                logger.warning(
                    "Agent '%s' has no configured weight — excluded from risk.",
                    result.agent_name,
                )
                continue
            if len(result.anomaly_scores) == 0:
                logger.warning(
                    "Agent '%s' produced no scores — excluded from risk.",
                    result.agent_name,
                )
                continue
            scores[result.agent_name] = float(np.mean(result.anomaly_scores))
            raw_weights[result.agent_name] = float(weight)

        active = len(scores)
        if active == 0:
            logger.warning("No active agents — returning zero risk.")
            return {
                "risk_score": 0.0,
                "risk_level": "low",
                "reason": "No active agents contributed signals; risk defaults to LOW.",
                "contributing_agents": {},
                "agent_agreement": 0,
                "timestamp": timestamp,
                "metadata": {"active_agents": 0, "weights_used": {}},
            }

        # Redistribute weight proportionally across active agents → sums to 1.0.
        weight_total = sum(raw_weights.values())
        norm_weights = {
            name: w / weight_total for name, w in raw_weights.items()
        }

        contributing: dict[str, dict[str, float]] = {}
        base_risk = 0.0
        for name, score in scores.items():
            weight = norm_weights[name]
            contribution = weight * score
            base_risk += contribution
            contributing[name] = {
                "score": round(score, 6),
                "weight": round(weight, 6),
                "contribution": round(contribution, 6),
            }

        # Pre-amplification weighted sum, kept for the explanation breakdown.
        base_sum = base_risk

        # Non-linear agreement bonus: corroboration across independent agents
        # raises confidence beyond the linear weighted sum.
        agreement = sum(1 for s in scores.values() if s > _AGREEMENT_THRESHOLD)
        amplification = 1.0
        if agreement >= 5:
            amplification = _AGREEMENT_BONUS_5PLUS
        elif agreement >= 3:
            amplification = _AGREEMENT_BONUS_3PLUS
        base_risk *= amplification

        risk_score = float(min(base_risk, 1.0))
        risk_level = self.classify_risk(risk_score)

        reason = self._build_reason(
            risk_level=risk_level,
            contributing=contributing,
            scores=scores,
            base_sum=base_sum,
            agreement=agreement,
            amplification=amplification,
        )

        logger.info(
            "compute_risk | score=%.4f | level=%s | active=%d | agreement=%d | %s",
            risk_score,
            risk_level,
            active,
            agreement,
            reason,
        )
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reason": reason,
            "contributing_agents": contributing,
            "agent_agreement": agreement,
            "timestamp": timestamp,
            "metadata": {
                "active_agents": active,
                "weights_used": {n: round(w, 6) for n, w in norm_weights.items()},
            },
        }

    @staticmethod
    def _build_reason(
        *,
        risk_level: str,
        contributing: dict[str, dict[str, float]],
        scores: dict[str, float],
        base_sum: float,
        agreement: int,
        amplification: float,
    ) -> str:
        """Build a human-readable explanation of what drives the risk score.

        Names the agents contributing most of the weighted risk, flags how
        many independent agents corroborate the disruption, and notes any
        agreement amplification — so a downstream analyst can see *why* the
        score landed where it did, not just the number.

        Args:
            risk_level: Final ``"high"`` / ``"medium"`` / ``"low"`` label.
            contributing: Per-agent ``{"score", "weight", "contribution"}``.
            scores: Per-agent mean anomaly score.
            base_sum: Pre-amplification weighted sum (sum of contributions).
            agreement: Count of agents above the agreement threshold.
            amplification: Agreement multiplier applied (1.0 / 1.15 / 1.25).

        Returns:
            One-sentence explanation string.
        """
        if not contributing:
            return "No active agents contributed signals; risk defaults to LOW."

        # Rank agents by their share of the weighted risk.
        ranked = sorted(
            contributing.items(),
            key=lambda kv: kv[1]["contribution"],
            reverse=True,
        )
        lead_name, lead_info = ranked[0]
        share = (lead_info["contribution"] / base_sum * 100.0) if base_sum > 0 else 0.0

        parts = [f"{risk_level.upper()} risk."]
        parts.append(
            f"Primary driver: {lead_name} "
            f"(mean anomaly {lead_info['score']:.2f}, "
            f"{share:.0f}% of weighted risk)."
        )

        # Name up to two supporting agents that are actually elevated.
        supporting = [
            f"{name} ({info['score']:.2f})"
            for name, info in ranked[1:4]
            if info["score"] > _AGREEMENT_THRESHOLD
        ]
        if supporting:
            parts.append(f"Supporting signals: {', '.join(supporting)}.")

        if agreement >= 3:
            bonus_pct = int(round((amplification - 1.0) * 100))
            parts.append(
                f"{agreement} agents corroborate "
                f"(+{bonus_pct}% agreement amplification)."
            )
        elif agreement > 0:
            parts.append(
                f"{agreement} agent(s) above the alert threshold "
                f"({_AGREEMENT_THRESHOLD:.2f})."
            )
        else:
            parts.append(
                "No agent exceeded the alert threshold; score reflects "
                "low-level background signals only."
            )
        return " ".join(parts)

    def compute_risk_timeseries(
        self, all_outputs_by_day: dict[str, list[DetectionResult]]
    ) -> pd.DataFrame:
        """Compute a daily risk series from per-day agent outputs.

        Each day's list of :class:`DetectionResult` objects is passed through
        :meth:`compute_risk`; the resulting scalar risk, level, and per-agent
        contributions are flattened into one row per day.

        Args:
            all_outputs_by_day: Mapping of day label (e.g. ``"2024-03-01"``)
                to that day's list of agent detection results.

        Returns:
            DataFrame sorted by ``timestamp`` with columns ``timestamp``,
            ``risk_score``, ``risk_level``, ``agent_agreement``, and one
            ``<agent>_contribution`` column per agent seen across all days
            (missing agents on a given day are filled with 0.0).
        """
        rows: list[dict] = []
        agent_names: set[str] = set()
        for day, outputs in all_outputs_by_day.items():
            result = self.compute_risk(outputs)
            row: dict = {
                "timestamp": day,
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "agent_agreement": result["agent_agreement"],
            }
            for name, info in result["contributing_agents"].items():
                col = f"{name}_contribution"
                row[col] = info["contribution"]
                agent_names.add(col)
            rows.append(row)

        if not rows:
            base_cols = ["timestamp", "risk_score", "risk_level", "agent_agreement"]
            return pd.DataFrame(columns=base_cols)

        df = pd.DataFrame(rows)
        for col in agent_names:
            if col not in df.columns:
                df[col] = 0.0
        df[list(agent_names)] = df[list(agent_names)].fillna(0.0)
        return df.sort_values("timestamp").reset_index(drop=True)
