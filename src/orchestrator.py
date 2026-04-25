"""Main pipeline orchestrator.

Wires together ingestion → detection → aggregation → explainability → RAG
into a single end-to-end run.  Each stage is driven by the configuration
loaded from ``config/settings.yaml``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinate all pipeline stages for a single disruption assessment run.

    The orchestrator is stateless between runs — it holds references to
    configured components and calls them in order for each incoming batch
    of signals.

    Args:
        config: Full application configuration dictionary.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._agents: list[Any] = []
        logger.info("Orchestrator initialised.")

    def register_agent(self, agent: Any) -> None:
        """Register a detection agent for use in the pipeline.

        Args:
            agent: A fitted :class:`~src.agents.base_agent.BaseAgent` instance.
        """
        self._agents.append(agent)
        logger.debug("Registered agent: %s", agent.name)

    def run(self, df: pd.DataFrame) -> dict:
        """Execute the full pipeline on a feature DataFrame.

        Stages:
            1. Detection — each registered agent scores ``df``.
            2. Aggregation — :class:`~src.aggregation.risk_engine.RiskEngine`
               computes a composite risk score.
            3. Explainability — SHAP contributions (wired in next phase).
            4. RAG context — historical precedent retrieval (wired in next phase).

        Args:
            df: Feature-ready DataFrame produced by the ingestion layer.

        Returns:
            Pipeline output dictionary with keys ``composite_score``,
            ``risk_level``, ``agent_scores``, ``shap``, and ``context``.
        """
        if not self._agents:
            logger.warning("No agents registered — returning empty result.")
            return {
                "composite_score": 0.0,
                "risk_level": "LOW",
                "agent_scores": {},
                "shap": {},
                "context": [],
            }

        from src.aggregation.risk_engine import RiskEngine

        results = [agent.detect(df) for agent in self._agents]
        engine = RiskEngine(self.config)
        aggregated = engine.aggregate(results)

        output = {**aggregated, "shap": {}, "context": []}
        logger.info(
            "Pipeline run complete | score=%.4f | level=%s",
            output["composite_score"],
            output["risk_level"],
        )
        return output
