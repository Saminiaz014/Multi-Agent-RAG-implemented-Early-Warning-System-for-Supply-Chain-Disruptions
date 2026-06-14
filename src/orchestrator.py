"""Main pipeline orchestrator.

Wires together ingestion → detection → aggregation → explainability → RAG
into a single end-to-end run. Each stage is driven by configuration loaded
from ``config/settings.yaml``.

The orchestrator owns the data connectors and is responsible for:

- Initialising :class:`~src.ingestion.ShippingConnector` and
  :class:`~src.ingestion.MarketConnector` in the source mode specified by
  ``config["ingestion"][...]["source_mode"]``.
- Fetching, aligning, and merging the two domain feeds into a unified
  daily-frequency DataFrame that downstream agents consume directly.
- Falling back to the synthetic generator on a per-connector basis when a
  CSV file is missing, so a partial dataset does not break the pipeline.
- Running registered agents over the combined frame and aggregating their
  outputs into a composite risk score (single-shot) or a per-day risk
  time series.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.ingestion import (
    DisasterConnector,
    GeopoliticalConnector,
    MarketConnector,
    NewsConnector,
    RoutingConnector,
    ShippingConnector,
)

logger = logging.getLogger(__name__)

# Config agent-name → connector class for the four event/domain agents that
# each consume their own single-domain frame (unlike shipping+market, which
# are merged onto a shared daily index).
_DOMAIN_CONNECTORS: dict[str, type] = {
    "geopolitical": GeopoliticalConnector,
    "natural_disaster": DisasterConnector,
    "routing": RoutingConnector,
    "news_sentiment": NewsConnector,
}


class Orchestrator:
    """Coordinate all pipeline stages for a single disruption assessment run.

    The orchestrator holds references to configured components (connectors,
    agents) and calls them in order. The legacy :meth:`run` entry point is
    preserved for callers that pass a pre-built feature frame; new code
    should prefer :meth:`run_full_pipeline` (single-shot risk) or
    :meth:`run_timeseries_analysis` (per-day risk series).

    Args:
        config: Full application configuration dictionary. The
            ``ingestion.shipping`` and ``ingestion.market`` sub-keys, when
            present, configure the connectors; ``weights`` and
            ``thresholds`` drive the aggregation engine.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._agents: list[Any] = []

        # Resolve the active weight source once. In "optimized" mode this loads
        # config/optimized_weights.yaml; a missing/invalid file logs a warning
        # and transparently falls back to the hand-tuned settings.yaml values.
        from src.optimization.weight_config import resolve_active_weights

        self._weight_mode: str = str(config.get("weight_mode", "hand_tuned")).lower()
        self._weight_layout: dict = resolve_active_weights(config)
        logger.info(
            "[Orchestrator] weight_mode='%s' (active source='%s')",
            self._weight_mode,
            self._weight_layout.get("source", "hand_tuned"),
        )

        ingestion_cfg = config.get("ingestion", {}) or {}
        shipping_cfg = dict(ingestion_cfg.get("shipping", {}) or {})
        market_cfg = dict(ingestion_cfg.get("market", {}) or {})

        self._shipping_mode: str = (
            shipping_cfg.get("source_mode") or "synthetic"
        ).lower()
        self._market_mode: str = (
            market_cfg.get("source_mode") or "synthetic"
        ).lower()

        self._shipping_connector: ShippingConnector = ShippingConnector(
            source_mode=self._shipping_mode, config=shipping_cfg
        )
        self._market_connector: MarketConnector = MarketConnector(
            source_mode=self._market_mode, config=market_cfg
        )

        # Own the four single-domain connectors (geopolitical, natural
        # disaster, routing, news). Each is config-driven via its
        # ``agents.<name>`` block (``data_mode`` / ``csv_path``) and only
        # built when that agent is enabled, so a disabled agent is never
        # ingested or detected.
        agents_cfg = config.get("agents", {}) or {}
        self._domain_connectors: dict[str, Any] = {}
        for name, connector_cls in _DOMAIN_CONNECTORS.items():
            agent_cfg = agents_cfg.get(name, {}) or {}
            if not agent_cfg.get("enabled", True):
                logger.info("Domain agent '%s' disabled — connector skipped.", name)
                continue
            self._domain_connectors[name] = connector_cls(config=agent_cfg)

        logger.info(
            "Orchestrator initialised | shipping_mode='%s' | market_mode='%s' "
            "| domain_connectors=%s",
            self._shipping_mode,
            self._market_mode,
            list(self._domain_connectors.keys()),
        )

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def register_agent(self, agent: Any) -> None:
        """Register a detection agent for use in the pipeline.

        Args:
            agent: A fitted :class:`~src.agents.base_agent.BaseAgent` instance.
        """
        self._agents.append(agent)
        # Inject the active weight set so registered agents honour weight_mode.
        try:
            from src.optimization.weight_config import apply_weights_to_agent

            if apply_weights_to_agent(agent, self._weight_layout):
                logger.debug(
                    "Applied '%s' weights to agent: %s",
                    self._weight_layout.get("source", "hand_tuned"),
                    agent.name,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to apply weights to %s: %s", agent.name, exc)
        logger.debug("Registered agent: %s", agent.name)

    # ------------------------------------------------------------------
    # Step a — Ingest + merge
    # ------------------------------------------------------------------

    def ingest(self) -> pd.DataFrame:
        """Fetch shipping + market data, align, and merge on timestamp.

        - Shipping data is fetched first; its date range defines the
          target index for the combined frame.
        - Market data is aligned to that index via
          :meth:`MarketConnector.align_with_shipping`, forward-filling
          weekends/holidays.
        - The aligned market frame is merged onto shipping using a left
          join on ``timestamp``. Market's ``is_disruption`` column is
          renamed to ``market_is_disruption`` so the primary shipping
          ground-truth label survives the merge intact.
        - ``oil_price_usd`` (NaN in the shipping CSV) is back-filled from
          the market connector's ``brent_crude_usd`` column.

        Returns:
            Unified daily-frequency feature frame combining shipping +
            market signals on a single ``timestamp`` index.
        """
        shipping_df = self._safe_fetch(self._shipping_connector, "shipping")
        market_df = self._safe_fetch(self._market_connector, "market")

        self._warn_if_market_coverage_short(shipping_df, market_df)
        aligned_market_df = self._market_connector.align_with_shipping(
            shipping_df, market_df
        )

        shipping_df = shipping_df.copy()
        shipping_df["timestamp"] = pd.to_datetime(shipping_df["timestamp"])

        market_for_merge = aligned_market_df.copy()
        market_for_merge["timestamp"] = pd.to_datetime(market_for_merge["timestamp"])
        if "is_disruption" in market_for_merge.columns:
            market_for_merge = market_for_merge.rename(
                columns={"is_disruption": "market_is_disruption"}
            )

        combined = shipping_df.merge(
            market_for_merge, on="timestamp", how="left", suffixes=("", "_market")
        )

        if (
            "brent_crude_usd" in combined.columns
            and "oil_price_usd" in combined.columns
        ):
            combined["oil_price_usd"] = combined["oil_price_usd"].fillna(
                combined["brent_crude_usd"]
            )

        logger.info(
            "[Orchestrator.ingest] Loaded %d shipping days, %d market days, "
            "%d aligned days",
            len(shipping_df),
            len(market_df),
            len(combined),
        )
        return combined

    # ------------------------------------------------------------------
    # Step b-f — Full pipeline (single-shot risk)
    # ------------------------------------------------------------------

    def run_full_pipeline(self) -> dict:
        """End-to-end ingest → detect → aggregate.

        Each registered agent is run on the combined frame using
        ``agent.run_dataframe`` + ``agent.to_detection_result`` when those
        methods are available (the shipping and market agents both expose
        them); otherwise the agent's plain ``detect`` is invoked as a
        fallback for legacy agents.

        Returns:
            Dictionary with keys ``composite_score``, ``risk_level``,
            ``agent_scores``, ``shap``, ``context``, and ``data`` (the
            ingest summary).
        """
        combined = self.ingest()
        results = self._run_agents(combined)

        from src.aggregation.risk_engine import RiskEngine, RiskLevel

        if not results:
            logger.warning(
                "[Orchestrator.run_full_pipeline] no agent results — "
                "returning zero risk."
            )
            return {
                "composite_score": 0.0,
                "risk_level": RiskLevel.LOW,
                "agent_scores": {},
                "shap": {},
                "context": [],
                "data": self._summarise_combined(combined),
            }

        engine = RiskEngine(self.config)
        aggregated = engine.aggregate(results)
        output = {
            **aggregated,
            "shap": {},
            "context": [],
            "data": self._summarise_combined(combined),
        }
        logger.info(
            "[Orchestrator.run_full_pipeline] complete | score=%.4f | level=%s",
            output["composite_score"],
            output["risk_level"],
        )
        return output

    # ------------------------------------------------------------------
    # Per-day risk time series
    # ------------------------------------------------------------------

    def run_timeseries_analysis(self) -> pd.DataFrame:
        """Run every registered agent and emit a per-day composite-risk series.

        For each agent that exposes ``run_dataframe``, the per-row
        ``anomaly_score`` is captured under the column
        ``<agent_name>_score``. The per-day composite is the weighted mean
        of the agent scores using ``config["weights"]``; the resulting
        score is then bucketed into a categorical ``risk_level`` using
        ``config["thresholds"]``.

        Returns:
            DataFrame indexed by date with columns ``timestamp``,
            ``<agent>_score`` for every active agent, ``composite_score``,
            and ``risk_level``. Empty DataFrame when no agents are
            registered or none can produce per-row scores.
        """
        combined = self.ingest()
        if not self._agents:
            logger.warning(
                "[Orchestrator.run_timeseries_analysis] no agents — "
                "returning empty frame."
            )
            return pd.DataFrame()

        score_frames: list[pd.DataFrame] = []
        for agent in self._agents:
            if not hasattr(agent, "run_dataframe"):
                logger.warning(
                    "[Orchestrator] agent %s lacks run_dataframe; skipping "
                    "from time-series analysis.",
                    agent.name,
                )
                continue
            try:
                validated = agent.run_dataframe(combined)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[Orchestrator] agent %s failed during run_dataframe: %s",
                    agent.name,
                    exc,
                )
                continue
            if "timestamp" not in validated.columns:
                continue
            frame = (
                validated[["timestamp", "anomaly_score"]]
                .rename(columns={"anomaly_score": f"{agent.name}_score"})
                .copy()
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            score_frames.append(frame)

        if not score_frames:
            return pd.DataFrame()

        merged = score_frames[0]
        for frame in score_frames[1:]:
            merged = merged.merge(frame, on="timestamp", how="outer")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        weights = self.config.get("weights", {}) or {}
        composite = pd.Series(0.0, index=merged.index)
        weight_total = 0.0
        for col in [c for c in merged.columns if c.endswith("_score")]:
            agent_name = col[: -len("_score")]
            w = float(weights.get(agent_name, 0.0))
            if w <= 0:
                continue
            composite = composite + merged[col].fillna(0.0) * w
            weight_total += w
        if weight_total > 0:
            merged["composite_score"] = composite / weight_total
        else:
            merged["composite_score"] = 0.0

        thresholds = self.config.get("thresholds", {}) or {}
        merged["risk_level"] = merged["composite_score"].apply(
            lambda s: _classify_score(float(s), thresholds)
        )
        logger.info(
            "[Orchestrator.run_timeseries_analysis] produced %d days of "
            "composite risk scores over %d agents",
            len(merged),
            len(score_frames),
        )
        return merged

    # ------------------------------------------------------------------
    # Legacy entry-point (kept for backwards compatibility)
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> dict:
        """Execute the detection + aggregation stages on a pre-built frame.

        This is the historical entry point used by scenario tests that
        already have a feature-ready DataFrame. New code should prefer
        :meth:`run_full_pipeline` (handles ingestion internally) or
        :meth:`run_timeseries_analysis` (per-day series).

        Args:
            df: Feature-ready DataFrame produced by the ingestion layer
                or hand-built by a caller.

        Returns:
            Pipeline output dictionary with keys ``composite_score``,
            ``risk_level``, ``agent_scores``, ``shap``, and ``context``.
        """
        from src.aggregation.risk_engine import RiskEngine, RiskLevel

        if not self._agents:
            logger.warning("No agents registered — returning empty result.")
            return {
                "composite_score": 0.0,
                "risk_level": RiskLevel.LOW,
                "agent_scores": {},
                "shap": {},
                "context": [],
            }

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_fetch(self, connector: Any, name: str) -> pd.DataFrame:
        """Fetch from a connector, falling back to synthetic on CSV failure."""
        try:
            return connector.fetch()
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "[Orchestrator] %s connector failed in source_mode='%s' "
                "(%s) — falling back to synthetic.",
                name,
                connector.source_mode,
                exc,
            )
            connector.source_mode = "synthetic"
            return connector.fetch()

    def fetch_domain(self, name: str) -> pd.DataFrame:
        """Fetch a single-domain frame, falling back to synthetic on failure.

        Mirrors :meth:`_safe_fetch` for the four event/domain connectors,
        which select their mode via ``data_mode`` (rather than ``source_mode``)
        and each return their own single-domain DataFrame.

        Args:
            name: Config agent name — one of ``geopolitical``,
                ``natural_disaster``, ``routing``, ``news_sentiment``.

        Returns:
            The connector's fetched DataFrame.

        Raises:
            KeyError: If ``name`` is not an owned (enabled) domain connector.
        """
        connector = self._domain_connectors[name]
        try:
            return connector.fetch()
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "[Orchestrator] %s connector failed in data_mode='%s' (%s) — "
                "falling back to synthetic.",
                name,
                getattr(connector, "data_mode", "?"),
                exc,
            )
            connector.data_mode = "synthetic"
            return connector.fetch()

    def _warn_if_market_coverage_short(
        self, shipping_df: pd.DataFrame, market_df: pd.DataFrame
    ) -> None:
        """Warn when market data doesn't fully span the shipping date range."""
        if shipping_df.empty or market_df.empty:
            return
        ship_min = pd.Timestamp(shipping_df["timestamp"].min())
        ship_max = pd.Timestamp(shipping_df["timestamp"].max())
        mk_min = pd.Timestamp(market_df["timestamp"].min())
        mk_max = pd.Timestamp(market_df["timestamp"].max())
        if mk_min > ship_min or mk_max < ship_max:
            logger.warning(
                "[Orchestrator] market data range [%s..%s] does not fully "
                "cover shipping range [%s..%s]; uncovered days will be "
                "forward-filled or remain NaN.",
                mk_min.date(), mk_max.date(), ship_min.date(), ship_max.date(),
            )

    def _run_agents(self, combined: pd.DataFrame) -> list:
        """Run every registered agent and collect DetectionResults."""
        results = []
        for agent in self._agents:
            try:
                if hasattr(agent, "run_dataframe") and hasattr(
                    agent, "to_detection_result"
                ):
                    validated = agent.run_dataframe(combined)
                    results.append(agent.to_detection_result(validated))
                else:
                    results.append(agent.detect(combined))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[Orchestrator] agent %s failed: %s", agent.name, exc
                )
        return results

    @staticmethod
    def _summarise_combined(combined: pd.DataFrame) -> dict:
        """One-line summary of the combined ingest frame."""
        if combined.empty:
            return {"rows": 0}
        return {
            "rows": len(combined),
            "start": str(pd.Timestamp(combined["timestamp"].min()).date()),
            "end": str(pd.Timestamp(combined["timestamp"].max()).date()),
            "columns": list(combined.columns),
        }


def _classify_score(score: float, thresholds: dict) -> str:
    """Bucket a composite risk score into a categorical risk level.

    Args:
        score: Composite risk score in ``[0, 1]``.
        thresholds: Dict carrying ``risk_critical`` / ``risk_high`` /
            ``risk_medium`` / ``risk_low`` boundaries.

    Returns:
        ``"CRITICAL"`` / ``"HIGH"`` / ``"MEDIUM"`` / ``"LOW"``.
    """
    crit = float(thresholds.get("risk_critical", 0.8))
    high = float(thresholds.get("risk_high", 0.6))
    med = float(thresholds.get("risk_medium", 0.4))
    if score >= crit:
        return "CRITICAL"
    if score >= high:
        return "HIGH"
    if score >= med:
        return "MEDIUM"
    return "LOW"
