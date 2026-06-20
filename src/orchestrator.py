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

        self._shap_explainer: Any | None = None
        self._last_agent_frames: dict[str, pd.DataFrame] = {}

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
        """End-to-end ingest → detect → aggregate across all six agents.

        When no agents have been registered explicitly, every agent enabled
        in ``config["agents"]`` is auto-built and registered (honouring the
        active ``weight_mode``). Shipping + market run on the merged daily
        frame; the four single-domain agents (geopolitical, natural disaster,
        routing, news) each run on their own connector's frame. Any agent or
        connector that fails is logged and skipped — one failure never aborts
        the run.

        Collected :class:`DetectionResult` outputs are passed to both
        :meth:`RiskEngine.aggregate` (legacy ``composite_score`` /
        ``agent_scores`` keys) and :meth:`RiskEngine.compute_risk` (the richer
        agreement-amplified ``risk_score`` / ``contributing_agents`` breakdown),
        so downstream consumers get the full six-agent composite either way.

        Returns:
            Dictionary with the legacy keys (``composite_score``,
            ``risk_level``, ``agent_scores``), the rich risk keys
            (``risk_score``, ``contributing_agents``, ``agent_agreement``,
            ``reason``), ``shap``, ``context``, ``data`` (ingest summary), and
            a ``metadata`` block reporting ``agents_active``, ``data_modes``,
            and ``weight_mode``.
        """
        if not self._agents:
            self._build_enabled_agents()

        combined = self.ingest()
        results, agents_active, data_modes = self._run_agents(combined)

        from src.aggregation.risk_engine import RiskEngine, RiskLevel

        metadata: dict[str, Any] = {
            "agents_active": agents_active,
            "data_modes": data_modes,
            "weight_mode": self._weight_mode,
        }

        if not results:
            logger.warning(
                "[Orchestrator.run_full_pipeline] no agent results — "
                "returning zero risk."
            )
            return {
                "composite_score": 0.0,
                "risk_level": RiskLevel.LOW,
                "agent_scores": {},
                "risk_score": 0.0,
                "risk_level_label": "low",
                "contributing_agents": {},
                "agent_agreement": 0,
                "reason": "No active agents contributed signals; risk defaults to LOW.",
                "shap": {},
                "context": [],
                "data": self._summarise_combined(combined),
                "metadata": metadata,
            }

        engine = RiskEngine(self.config)
        aggregated = engine.aggregate(results)
        risk = engine.compute_risk(results)
        metadata.update(risk.get("metadata", {}))

        output = {
            **aggregated,
            "risk_score": risk["risk_score"],
            "risk_level_label": risk["risk_level"],
            "contributing_agents": risk["contributing_agents"],
            "agent_agreement": risk["agent_agreement"],
            "reason": risk["reason"],
            "shap": {},
            "explanation": {},
            "context": [],
            "data": self._summarise_combined(combined),
            "metadata": metadata,
        }

        # SHAP explainability — lazy-train surrogate then explain current state.
        # Wrapped in try/except so a SHAP failure never aborts the pipeline.
        try:
            from src.explainability.shap_explainer import (
                SurrogateShapExplainer,
                build_shap_training_data,
            )

            if self._shap_explainer is None:
                self._shap_explainer = SurrogateShapExplainer()
            if not self._shap_explainer.is_trained:
                features_df, risk_scores = build_shap_training_data(self.config)
                self._shap_explainer.train_surrogate(
                    features_df, risk_scores, weight_mode=self._weight_mode
                )
            current_features = self._build_shap_features_row(combined)
            shap_result = self._shap_explainer.explain(current_features)
            explanation_text = self._shap_explainer.generate_explanation_text(
                risk_score=output["risk_score"],
                risk_level=output.get("risk_level_label", "unknown"),
                weight_mode=self._weight_mode,
                shap_result=shap_result,
            )
            output["explanation"] = {
                "top_drivers": shap_result["top_drivers"],
                "expected_value": shap_result["expected_value"],
                "text": explanation_text,
                "surrogate_r2": self._shap_explainer.r2,
            }
        except Exception as exc:
            logger.warning("[Orchestrator] SHAP explainability failed: %s", exc)

        # RAG: retrieve historical precedents matching the current signal profile.
        # Failures are logged and never abort the pipeline.
        output["historical_context"] = []
        try:
            from src.rag.context_retriever import ContextRetriever

            rag_cfg = self.config.get("rag", {}) or {}
            if rag_cfg:
                _rag = ContextRetriever(rag_cfg)
                _rag.build_index("data/knowledge_base/disruption_cases.json")
                agent_scores: dict[str, float] = {
                    name: float(v) if isinstance(v, (int, float)) else float(v.get("score", 0.0))
                    for name, v in (output.get("agent_scores") or {}).items()
                }
                ctx_results = _rag.query(
                    agent_scores, top_k=rag_cfg.get("top_k", 3)
                )
                output["historical_context"] = ctx_results
                logger.info(
                    "[Orchestrator] RAG retrieved %d historical case(s).",
                    len(ctx_results),
                )
        except Exception as exc:
            logger.warning("[Orchestrator] RAG context retrieval failed: %s", exc)

        logger.info(
            "[Orchestrator.run_full_pipeline] complete | score=%.4f | level=%s "
            "| agents_active=%s | weight_mode=%s",
            output["composite_score"],
            output["risk_level"],
            agents_active,
            self._weight_mode,
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
        if not self._agents:
            self._build_enabled_agents()

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
            # Domain agents (geopolitical/disaster/routing/news) consume their
            # own single-domain frame; shipping + market share the merged frame.
            frame = self._frame_for_agent(agent.name, combined)
            if frame is None:
                continue
            try:
                validated = agent.run_dataframe(frame)
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

    def _run_agents(
        self, combined: pd.DataFrame
    ) -> tuple[list, list[str], dict[str, str]]:
        """Run every registered agent and collect DetectionResults.

        Shipping + market run on the merged daily frame; the four single-domain
        agents each run on their own connector's frame. Failures are logged and
        skipped (graceful degradation).

        Returns:
            Tuple of ``(results, agents_active, data_modes)`` where
            ``agents_active`` lists the agents that ran successfully and
            ``data_modes`` maps each to the source mode it ingested from.
        """
        results = []
        agents_active: list[str] = []
        data_modes: dict[str, str] = {}
        for agent in self._agents:
            frame = self._frame_for_agent(agent.name, combined)
            if frame is None:
                continue
            self._last_agent_frames[agent.name] = frame
            try:
                if hasattr(agent, "run_dataframe") and hasattr(
                    agent, "to_detection_result"
                ):
                    validated = agent.run_dataframe(frame)
                    results.append(agent.to_detection_result(validated))
                else:
                    results.append(agent.detect(frame))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[Orchestrator] agent %s failed: %s", agent.name, exc
                )
                continue
            agents_active.append(agent.name)
            data_modes[agent.name] = self._agent_data_mode(agent.name)
        return results, agents_active, data_modes

    def _frame_for_agent(
        self, name: str, combined: pd.DataFrame
    ) -> pd.DataFrame | None:
        """Resolve the input frame for an agent (domain-aware, fail-safe).

        Returns the merged shipping+market frame for the shipping/market
        agents, the agent's own single-domain frame for the four domain
        agents, or ``None`` when a domain agent's connector is absent (disabled)
        or its fetch fails — so the agent is simply skipped, never crashing the
        run.
        """
        if name not in _DOMAIN_CONNECTORS:
            return combined
        if name not in self._domain_connectors:
            logger.warning(
                "[Orchestrator] domain agent '%s' has no connector — skipping.",
                name,
            )
            return None
        try:
            return self.fetch_domain(name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[Orchestrator] domain connector '%s' failed: %s — skipping.",
                name,
                exc,
            )
            return None

    def _agent_data_mode(self, name: str) -> str:
        """Report the source/data mode an agent's connector ingested from."""
        if name == "shipping":
            return self._shipping_mode
        if name == "market":
            return self._market_mode
        connector = self._domain_connectors.get(name)
        return str(getattr(connector, "data_mode", "unknown")) if connector else "unknown"

    def _build_enabled_agents(self) -> None:
        """Construct + register every agent enabled in ``config["agents"]``.

        Honours each agent's ``enabled`` flag (default ``True``) so a disabled
        agent is never built, detected, or weighted. Active weights are applied
        on registration via :meth:`register_agent`, so the roster respects the
        configured ``weight_mode``. A domain agent whose connector was skipped
        (disabled) is not registered.
        """
        from src.agents.disaster_agent import DisasterAgent
        from src.agents.geopolitical_agent import GeopoliticalAgent
        from src.agents.market_agent import MarketAgent
        from src.agents.news_agent import NewsAgent
        from src.agents.routing_agent import RoutingAgent
        from src.agents.shipping_agent import ShippingAgent

        agents_cfg = self.config.get("agents", {}) or {}

        def _enabled(name: str) -> bool:
            return bool((agents_cfg.get(name, {}) or {}).get("enabled", True))

        # Shipping + market run on the merged daily frame.
        if _enabled("shipping"):
            self.register_agent(
                ShippingAgent(config=dict(agents_cfg.get("shipping", {}) or {}))
            )
        else:
            logger.info("Agent 'shipping' disabled — not registered.")
        if _enabled("market"):
            self.register_agent(
                MarketAgent(config=dict(agents_cfg.get("market", {}) or {}))
            )
        else:
            logger.info("Agent 'market' disabled — not registered.")

        # The four single-domain agents — only when both enabled and the
        # matching connector was built.
        domain_classes: dict[str, type] = {
            "geopolitical": GeopoliticalAgent,
            "natural_disaster": DisasterAgent,
            "routing": RoutingAgent,
            "news_sentiment": NewsAgent,
        }
        for name, agent_cls in domain_classes.items():
            if not _enabled(name):
                logger.info("Agent '%s' disabled — not registered.", name)
                continue
            if name not in self._domain_connectors:
                logger.warning(
                    "Agent '%s' enabled but its connector is absent — skipping.",
                    name,
                )
                continue
            self.register_agent(agent_cls(config=dict(agents_cfg.get(name, {}) or {})))

        logger.info(
            "[Orchestrator] built %d enabled agent(s): %s",
            len(self._agents),
            [a.name for a in self._agents],
        )

    def _build_shap_features_row(self, combined: pd.DataFrame) -> pd.DataFrame:
        """Build a single-row 20-feature DataFrame from current pipeline state.

        Shipping and market features are read from the last row of ``combined``.
        Domain agent features (geopolitical, natural_disaster, routing,
        news_sentiment) are read from the last row of each agent's cached raw
        frame in ``self._last_agent_frames``.  Absent columns are filled 0.0.
        """
        from src.explainability.shap_explainer import ALL_FEATURE_NAMES, FEATURE_AGENT_MAP

        last_combined = combined.iloc[[-1]] if len(combined) > 0 else pd.DataFrame()
        row: dict[str, float] = {}
        for feat in ALL_FEATURE_NAMES:
            agent_name = FEATURE_AGENT_MAP.get(feat, "")
            if agent_name in ("shipping", "market"):
                if not last_combined.empty and feat in last_combined.columns:
                    val = last_combined.iloc[-1][feat]
                    row[feat] = float(val) if pd.notna(val) else 0.0
                else:
                    row[feat] = 0.0
            else:
                frame = self._last_agent_frames.get(agent_name)
                if frame is not None and not frame.empty and feat in frame.columns:
                    val = frame.iloc[-1][feat]
                    row[feat] = float(val) if pd.notna(val) else 0.0
                else:
                    row[feat] = 0.0
        return pd.DataFrame([row], columns=ALL_FEATURE_NAMES)

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
