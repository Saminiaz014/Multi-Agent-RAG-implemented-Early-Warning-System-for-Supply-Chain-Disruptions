"""SHAP-based explainability wrappers.

Two public classes:

* :class:`ShapExplainer` — low-level wrapper for any fitted sklearn model.
  Kept for backward compatibility with the existing API / RAG layer.

* :class:`SurrogateShapExplainer` — surrogate-based 20-feature explainer for
  the full 6-agent pipeline.  A RandomForestRegressor is trained to reproduce
  the pipeline's per-day composite risk scores; a SHAP TreeExplainer then
  produces fast, exact Shapley values with per-feature agent attribution.

Module-level helpers:

* :data:`ALL_FEATURE_NAMES` — ordered list of the 20 canonical features.
* :data:`FEATURE_AGENT_MAP` — maps every feature name to its source agent.
* :func:`build_shap_training_data` — generate (features_df, risk_scores)
  from synthetic 6-agent data for surrogate training.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

#: Ordered 20-feature list spanning all 6 agents used by the surrogate.
#: (The objective specification names "19 features"; the correct enumeration
#: below yields 20 — 3+3+4+4+3+3.  All downstream code uses this constant.)
ALL_FEATURE_NAMES: list[str] = [
    # Shipping (3)
    "vessel_count",
    "avg_delay_hours",
    "congestion_index",
    # Market (3)
    "brent_crude_usd",
    "trade_volume_index",
    "freight_rate_index",
    # Geopolitical (4)
    "sanctions_severity",
    "military_activity_index",
    "diplomatic_incident_score",
    "regime_stability_index",
    # Natural Disaster (4)
    "earthquake_severity",
    "tsunami_risk",
    "cyclone_severity",
    "severe_weather_index",
    # Routing (3)
    "rerouting_percentage",
    "avg_route_deviation_km",
    "transit_volume_ratio",
    # News Sentiment (3)
    "sentiment_score",
    "source_consensus",
    "article_volume",
]

#: Maps each feature name to the agent domain that produces it.
FEATURE_AGENT_MAP: dict[str, str] = {
    "vessel_count": "shipping",
    "avg_delay_hours": "shipping",
    "congestion_index": "shipping",
    "brent_crude_usd": "market",
    "trade_volume_index": "market",
    "freight_rate_index": "market",
    "sanctions_severity": "geopolitical",
    "military_activity_index": "geopolitical",
    "diplomatic_incident_score": "geopolitical",
    "regime_stability_index": "geopolitical",
    "earthquake_severity": "natural_disaster",
    "tsunami_risk": "natural_disaster",
    "cyclone_severity": "natural_disaster",
    "severe_weather_index": "natural_disaster",
    "rerouting_percentage": "routing",
    "avg_route_deviation_km": "routing",
    "transit_volume_ratio": "routing",
    "sentiment_score": "news_sentiment",
    "source_consensus": "news_sentiment",
    "article_volume": "news_sentiment",
}

# Default inter-agent weights when not provided via config.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "shipping": 0.25,
    "market": 0.15,
    "geopolitical": 0.25,
    "natural_disaster": 0.10,
    "routing": 0.15,
    "news_sentiment": 0.10,
}


# ---------------------------------------------------------------------------
# Legacy wrapper (unchanged — kept for backward compatibility)
# ---------------------------------------------------------------------------


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
                "The 'shap' package is required for ShapExplainer. "
                "Install it with 'pip install shap'."
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
            raise RuntimeError(
                "Explainer has not been built — call _build_explainer."
            )

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


# ---------------------------------------------------------------------------
# Six-agent surrogate explainer
# ---------------------------------------------------------------------------


class SurrogateShapExplainer:
    """Random-Forest surrogate + SHAP explainer for the 6-agent risk pipeline.

    A :class:`~sklearn.ensemble.RandomForestRegressor` is trained to reproduce
    the pipeline's per-day composite risk scores from the 20 canonical input
    features.  A SHAP ``TreeExplainer`` is then built on that surrogate,
    providing fast, exact Shapley values with per-feature agent attribution.

    Workflow::

        explainer = SurrogateShapExplainer()
        features_df, risk_scores = build_shap_training_data(config)
        explainer.train_surrogate(features_df, risk_scores)

        row = features_df.iloc[[155]]  # peak-disruption day
        result = explainer.explain(row)
        text = explainer.generate_explanation_text(0.82, "high", "hand_tuned", result)
    """

    def __init__(self) -> None:
        self._rf: Any = None
        self._shap_explainer: Any = None
        self._trained: bool = False
        self.r2: float = 0.0

    @property
    def is_trained(self) -> bool:
        """True after :meth:`train_surrogate` has completed successfully."""
        return self._trained

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_surrogate(
        self,
        features_df: pd.DataFrame,
        risk_scores: np.ndarray,
        *,
        weight_mode: str = "hand_tuned",
    ) -> float:
        """Fit the RF surrogate and build the SHAP tree explainer.

        Missing columns (e.g. from a disabled agent) are filled with 0.0 so
        the call never crashes regardless of which agents are active.

        Args:
            features_df: DataFrame with :data:`ALL_FEATURE_NAMES` columns.
                Columns absent from the frame are treated as 0.0.
            risk_scores: Per-row composite risk scores in ``[0, 1]``.
            weight_mode: Active weight source label, used only for logging.

        Returns:
            Train-set R² of the surrogate (should be > 0.85).
        """
        import shap
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score

        X = self._align_features(features_df)
        y = np.asarray(risk_scores, dtype=float)

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        r2 = float(r2_score(y, rf.predict(X)))
        self.r2 = r2
        logger.info(
            "[SurrogateShapExplainer.train_surrogate] R²=%.4f | n=%d | weight_mode=%s",
            r2,
            len(y),
            weight_mode,
        )
        if r2 < 0.85:
            logger.warning(
                "[SurrogateShapExplainer] R² %.4f < 0.85 — surrogate may not "
                "accurately reproduce the pipeline risk scores.",
                r2,
            )

        self._rf = rf
        self._shap_explainer = shap.TreeExplainer(rf)
        self._trained = True
        return r2

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain(self, features_row: pd.DataFrame) -> dict:
        """Compute SHAP values for a single row.

        Args:
            features_row: One-row DataFrame.  Missing columns are filled with
                0.0 so disabled-agent features do not crash the call.

        Returns:
            Dictionary with keys:

            - ``shap_values`` (dict[str, float]): per-feature SHAP value.
            - ``top_drivers`` (list[dict]): top 3 features by ``|SHAP|``,
              each carrying ``feature``, ``agent``, and ``shap_value``.
            - ``feature_names`` (list[str]): ordered feature list.
            - ``expected_value`` (float): SHAP base value (mean prediction).
        """
        if not self._trained:
            raise RuntimeError(
                "SurrogateShapExplainer not trained — call train_surrogate() first."
            )

        X = self._align_features(features_row)  # shape (1, 20)
        raw = self._shap_explainer.shap_values(X)

        # RF regressor returns ndarray (n_samples, n_features);
        # older shap versions may return a list — handle both.
        if isinstance(raw, list):
            shap_arr = np.mean(raw, axis=0)[0]
        else:
            shap_arr = raw[0]  # shape (20,)

        shap_dict = {
            feat: round(float(shap_arr[i]), 6)
            for i, feat in enumerate(ALL_FEATURE_NAMES)
        }

        sorted_feats = sorted(
            shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        top_drivers = [
            {
                "feature": feat,
                "agent": FEATURE_AGENT_MAP.get(feat, "unknown"),
                "shap_value": round(val, 4),
            }
            for feat, val in sorted_feats[:3]
        ]

        ev = self._shap_explainer.expected_value
        expected_value = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

        return {
            "shap_values": shap_dict,
            "top_drivers": top_drivers,
            "feature_names": ALL_FEATURE_NAMES,
            "expected_value": round(expected_value, 6),
        }

    # ------------------------------------------------------------------
    # Text explanation
    # ------------------------------------------------------------------

    def generate_explanation_text(
        self,
        risk_score: float,
        risk_level: str,
        weight_mode: str,
        shap_result: dict,
    ) -> str:
        """Build a natural-language explanation with agent attribution.

        Example output::

            "Risk is HIGH (0.82) [optimized weights]. Primary drivers:
            vessel rerouting increased (+0.28, routing agent),
            sanctions severity elevated (+0.22, geopolitical agent),
            negative news sentiment (+0.18, news_sentiment agent)."

        Args:
            risk_score: Numeric composite risk score in ``[0, 1]``.
            risk_level: ``"high"`` / ``"medium"`` / ``"low"`` (or uppercase).
            weight_mode: ``"hand_tuned"`` or ``"optimized"``.
            shap_result: Output of :meth:`explain`.

        Returns:
            One-paragraph explanation string.
        """
        drivers = shap_result.get("top_drivers", [])
        level_str = str(risk_level).upper()
        header = (
            f"Risk is {level_str} ({float(risk_score):.2f}) [{weight_mode} weights]."
        )
        if not drivers:
            return header

        driver_texts = []
        for d in drivers:
            feat = d["feature"].replace("_", " ")
            agent = d["agent"].replace("_", " ")
            val = float(d["shap_value"])
            sign = "+" if val >= 0 else ""
            driver_texts.append(f"{feat} ({sign}{val:.2f}, {agent} agent)")

        return f"{header} Primary drivers: {', '.join(driver_texts)}."

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def generate_shap_plot(
        self,
        features_df: pd.DataFrame,
        risk_scores: np.ndarray,
    ) -> list[str]:
        """Save beeswarm and waterfall SHAP plots to ``data/processed/``.

        Args:
            features_df: Full training DataFrame (n_samples × features).
            risk_scores: Per-row composite risk scores used for training.

        Returns:
            Paths of written PNG files (only those that succeeded).
        """
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path

        if not self._trained:
            raise RuntimeError(
                "train_surrogate() must be called before generate_shap_plot()."
            )

        Path("data/processed").mkdir(parents=True, exist_ok=True)
        X = self._align_features(features_df)
        X_pd = pd.DataFrame(X, columns=ALL_FEATURE_NAMES)
        shap_values = self._shap_explainer.shap_values(X)

        paths: list[str] = []

        # --- Beeswarm (feature importance across all samples) ---
        try:
            shap.summary_plot(
                shap_values,
                X_pd,
                feature_names=ALL_FEATURE_NAMES,
                show=False,
                plot_type="dot",
            )
            plt.tight_layout()
            beeswarm_path = "data/processed/shap_beeswarm_6agent.png"
            plt.savefig(beeswarm_path, dpi=100, bbox_inches="tight")
            plt.close("all")
            paths.append(beeswarm_path)
            logger.info("[SurrogateShapExplainer] wrote %s", beeswarm_path)
        except Exception as exc:
            logger.warning(
                "[SurrogateShapExplainer] beeswarm plot failed: %s", exc
            )

        # --- Waterfall for the peak-risk instance ---
        try:
            peak_idx = int(np.argmax(risk_scores))
            ev = self._shap_explainer.expected_value
            base_val = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

            peak_shap = shap_values[peak_idx] if not isinstance(shap_values, list) \
                else np.mean(shap_values, axis=0)[peak_idx]

            shap_exp = shap.Explanation(
                values=peak_shap,
                base_values=base_val,
                data=X[peak_idx],
                feature_names=ALL_FEATURE_NAMES,
            )
            shap.waterfall_plot(shap_exp, max_display=15, show=False)
            plt.tight_layout()
            waterfall_path = "data/processed/shap_waterfall_6agent.png"
            plt.savefig(waterfall_path, dpi=100, bbox_inches="tight")
            plt.close("all")
            paths.append(waterfall_path)
            logger.info("[SurrogateShapExplainer] wrote %s", waterfall_path)
        except Exception as exc:
            logger.warning(
                "[SurrogateShapExplainer] waterfall plot failed: %s", exc
            )

        return paths

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _align_features(df: pd.DataFrame) -> np.ndarray:
        """Build a (n, 20) float array from a DataFrame.

        Columns in :data:`ALL_FEATURE_NAMES` that are absent from ``df``
        are filled with 0.0 (used when an agent is disabled).
        """
        out = np.zeros((len(df), len(ALL_FEATURE_NAMES)), dtype=float)
        for i, col in enumerate(ALL_FEATURE_NAMES):
            if col in df.columns:
                out[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        return out


# ---------------------------------------------------------------------------
# Training-data builder
# ---------------------------------------------------------------------------


def build_shap_training_data(
    config: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate per-day synthetic data for SHAP surrogate training.

    Fetches 365-day synthetic frames from all enabled connectors, fits and
    runs each agent, then returns the 20-column feature matrix and per-day
    composite risk scores.  Disabled agents have their columns zeroed.

    Args:
        config: Full application config (reads ``agents`` block for enabled
            flags + per-agent weights).

    Returns:
        ``(features_df, risk_scores)`` where ``features_df`` has shape
        ``(N, 20)`` and ``risk_scores`` has shape ``(N,)`` in ``[0, 1]``.
    """
    from src.agents.disaster_agent import DisasterAgent
    from src.agents.geopolitical_agent import GeopoliticalAgent
    from src.agents.market_agent import MarketAgent
    from src.agents.news_agent import NewsAgent
    from src.agents.routing_agent import RoutingAgent
    from src.agents.shipping_agent import ShippingAgent
    from src.ingestion import (
        DisasterConnector,
        GeopoliticalConnector,
        MarketConnector,
        NewsConnector,
        RoutingConnector,
        ShippingConnector,
    )

    agents_cfg = config.get("agents", {}) or {}
    weights = config.get("weights", {}) or {}

    def _enabled(name: str) -> bool:
        return bool((agents_cfg.get(name, {}) or {}).get("enabled", True))

    # --- Fetch raw frames + run agents (all in synthetic mode) ---
    raw_frames: dict[str, pd.DataFrame | None] = {}
    scored_frames: dict[str, pd.DataFrame | None] = {}

    if _enabled("shipping"):
        conn = ShippingConnector(source_mode="synthetic", config={})
        df = conn.fetch()
        agent = ShippingAgent(config=dict(agents_cfg.get("shipping", {}) or {}))
        raw_frames["shipping"] = df
        scored_frames["shipping"] = agent.run_dataframe(df)
    else:
        raw_frames["shipping"] = None
        scored_frames["shipping"] = None

    if _enabled("market"):
        conn = MarketConnector(source_mode="synthetic", config={})
        df = conn.fetch()
        agent = MarketAgent(config=dict(agents_cfg.get("market", {}) or {}))
        raw_frames["market"] = df
        scored_frames["market"] = agent.run_dataframe(df)
    else:
        raw_frames["market"] = None
        scored_frames["market"] = None

    _domain_specs: list[tuple[str, Any, Any]] = [
        ("geopolitical", GeopoliticalConnector, GeopoliticalAgent),
        ("natural_disaster", DisasterConnector, DisasterAgent),
        ("routing", RoutingConnector, RoutingAgent),
        ("news_sentiment", NewsConnector, NewsAgent),
    ]
    for name, connector_cls, agent_cls in _domain_specs:
        if not _enabled(name):
            raw_frames[name] = None
            scored_frames[name] = None
            continue
        conn = connector_cls(config=dict(agents_cfg.get(name, {}) or {}))
        df = conn.fetch()
        agent = agent_cls(config=dict(agents_cfg.get(name, {}) or {}))
        raw_frames[name] = df
        scored_frames[name] = agent.run_dataframe(df)

    # --- Align to shortest scored output ---
    active_lengths = [
        len(sf)
        for sf in scored_frames.values()
        if sf is not None and "anomaly_score" in sf.columns
    ]
    if not active_lengths:
        raise ValueError("build_shap_training_data: no agent produced scored output.")
    n = min(active_lengths)

    # --- Extract 20 feature columns from raw frames ---
    # Market agent drops its first row (rolling window), so we take the last n
    # rows of the market raw frame to stay aligned with scored_frames["market"].
    _ship_cols = ["vessel_count", "avg_delay_hours", "congestion_index"]
    _mkt_cols = ["brent_crude_usd", "trade_volume_index", "freight_rate_index"]
    _domain_cols: dict[str, list[str]] = {
        "geopolitical": [
            "sanctions_severity",
            "military_activity_index",
            "diplomatic_incident_score",
            "regime_stability_index",
        ],
        "natural_disaster": [
            "earthquake_severity",
            "tsunami_risk",
            "cyclone_severity",
            "severe_weather_index",
        ],
        "routing": [
            "rerouting_percentage",
            "avg_route_deviation_km",
            "transit_volume_ratio",
        ],
        "news_sentiment": ["sentiment_score", "source_consensus", "article_volume"],
    }

    feature_data: dict[str, np.ndarray] = {}

    for col in _ship_cols:
        rf = raw_frames.get("shipping")
        feature_data[col] = (
            rf[col].iloc[:n].to_numpy(dtype=float)
            if rf is not None and col in rf.columns
            else np.zeros(n)
        )

    for col in _mkt_cols:
        rf = raw_frames.get("market")
        feature_data[col] = (
            rf[col].iloc[-n:].to_numpy(dtype=float)  # aligned with scored output
            if rf is not None and col in rf.columns
            else np.zeros(n)
        )

    for domain_name, cols in _domain_cols.items():
        rf = raw_frames.get(domain_name)
        for col in cols:
            feature_data[col] = (
                rf[col].iloc[:n].to_numpy(dtype=float)
                if rf is not None and col in rf.columns
                else np.zeros(n)
            )

    features_df = pd.DataFrame(feature_data, columns=ALL_FEATURE_NAMES)

    # --- Compute per-day composite risk scores ---
    risk_scores = np.zeros(n, dtype=float)
    total_weight = 0.0
    for agent_name, scored in scored_frames.items():
        if scored is None or "anomaly_score" not in scored.columns:
            continue
        w = float(weights.get(agent_name, _DEFAULT_WEIGHTS.get(agent_name, 1 / 6)))
        scores = scored["anomaly_score"].to_numpy(dtype=float)
        # Trim to n (handles market's dropped row and any edge cases)
        scores = scores[:n]
        if len(scores) < n:
            padded = np.zeros(n)
            padded[: len(scores)] = scores
            scores = padded
        risk_scores += w * scores
        total_weight += w
    if total_weight > 0:
        risk_scores /= total_weight

    logger.info(
        "[build_shap_training_data] n=%d rows | features=%d | "
        "risk_range=[%.3f, %.3f] | active_agents=%s",
        n,
        len(ALL_FEATURE_NAMES),
        float(risk_scores.min()),
        float(risk_scores.max()),
        [k for k, v in scored_frames.items() if v is not None],
    )
    return features_df, risk_scores


# ---------------------------------------------------------------------------
# Proxy risk-score helper (used by compare_explanations)
# ---------------------------------------------------------------------------


def _proxy_risk_scores(
    features_df: pd.DataFrame,
    weights: dict[str, float],
) -> np.ndarray:
    """Derive per-row composite risk from raw features and inter-agent weights.

    Each agent's contribution is the direction-corrected normalised mean of its
    columns (higher = more anomalous). The weighted sum is re-normalised so
    the output stays in ``[0, 1]``.

    Args:
        features_df: n × 20 feature DataFrame.
        weights: Mapping of agent name → inter-agent weight.

    Returns:
        Risk-score array of shape ``(n,)``.
    """
    n = len(features_df)

    def _norm(col: str, *, invert: bool = False) -> np.ndarray:
        v = features_df[col].to_numpy(dtype=float) if col in features_df.columns else np.zeros(n)
        lo, hi = v.min(), v.max()
        v = (v - lo) / (hi - lo) if hi > lo else np.zeros(n)
        return 1.0 - v if invert else v

    per_agent: dict[str, np.ndarray] = {
        "shipping": np.column_stack([
            _norm("vessel_count", invert=True),
            _norm("avg_delay_hours"),
            _norm("congestion_index"),
        ]).mean(axis=1),
        "market": np.column_stack([
            _norm("brent_crude_usd"),
            _norm("trade_volume_index", invert=True),
            _norm("freight_rate_index"),
        ]).mean(axis=1),
        "geopolitical": np.column_stack([
            _norm("sanctions_severity"),
            _norm("military_activity_index"),
            _norm("diplomatic_incident_score"),
            _norm("regime_stability_index", invert=True),
        ]).mean(axis=1),
        "natural_disaster": np.column_stack([
            _norm("earthquake_severity"),
            _norm("tsunami_risk"),
            _norm("cyclone_severity"),
            _norm("severe_weather_index"),
        ]).mean(axis=1),
        "routing": np.column_stack([
            _norm("rerouting_percentage"),
            _norm("avg_route_deviation_km"),
            _norm("transit_volume_ratio", invert=True),
        ]).mean(axis=1),
        "news_sentiment": np.column_stack([
            _norm("sentiment_score", invert=True),
            _norm("source_consensus"),
            _norm("article_volume"),
        ]).mean(axis=1),
    }

    risk = np.zeros(n)
    total_w = sum(float(weights.get(a, 0.0)) for a in per_agent)
    if total_w <= 0:
        return risk
    for agent, scores in per_agent.items():
        risk += float(weights.get(agent, 0.0)) * scores
    return np.clip(risk / total_w, 0.0, 1.0)


# ---------------------------------------------------------------------------
# compare_explanations
# ---------------------------------------------------------------------------


def compare_explanations(
    features_df: pd.DataFrame,
    risk_engine_ht: Any,
    risk_engine_opt: Any,
    features_row: pd.DataFrame | None = None,
) -> dict:
    """Compare SHAP explanations between hand-tuned and optimized RiskEngines.

    Trains two :class:`SurrogateShapExplainer` instances on the same
    ``features_df`` but with risk scores derived from each engine's
    inter-agent weights, then compares the per-feature SHAP attributions on a
    single row.

    Args:
        features_df: n × 20 feature DataFrame (same format as
            :func:`build_shap_training_data`).
        risk_engine_ht: :class:`~src.aggregation.risk_engine.RiskEngine`
            initialised with hand-tuned weights.
        risk_engine_opt: :class:`~src.aggregation.risk_engine.RiskEngine`
            initialised with optimized weights.
        features_row: Single-row DataFrame to explain.  Defaults to the
            row with the highest hand-tuned risk score.

    Returns:
        Dict with:

        - ``"hand_tuned"`` — ``{"top_drivers", "shap_values", "r2"}``.
        - ``"optimized"``  — ``{"top_drivers", "shap_values", "r2"}``.
        - ``"driver_rank_changes"`` — list of ``{"feature", "ht_rank",
          "opt_rank", "direction"}`` sorted by magnitude of rank shift.
        - ``"weight_mode_impact"`` — 1-2 sentence natural-language summary.
    """
    ht_scores = _proxy_risk_scores(features_df, risk_engine_ht.weights)
    opt_scores = _proxy_risk_scores(features_df, risk_engine_opt.weights)

    ht_explainer = SurrogateShapExplainer()
    ht_r2 = ht_explainer.train_surrogate(features_df, ht_scores, weight_mode="hand_tuned")

    opt_explainer = SurrogateShapExplainer()
    opt_r2 = opt_explainer.train_surrogate(features_df, opt_scores, weight_mode="optimized")

    if features_row is None:
        peak_idx = int(np.argmax(ht_scores))
        features_row = features_df.iloc[[peak_idx]]

    ht_result = ht_explainer.explain(features_row)
    opt_result = opt_explainer.explain(features_row)

    # Build absolute-SHAP rank maps (rank 1 = most important).
    def _ranks(shap_dict: dict[str, float]) -> dict[str, int]:
        ordered = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return {feat: idx + 1 for idx, (feat, _) in enumerate(ordered)}

    ht_ranks = _ranks(ht_result["shap_values"])
    opt_ranks = _ranks(opt_result["shap_values"])

    rank_changes = []
    for feat in ALL_FEATURE_NAMES:
        ht_r = ht_ranks.get(feat, len(ALL_FEATURE_NAMES))
        opt_r = opt_ranks.get(feat, len(ALL_FEATURE_NAMES))
        if ht_r != opt_r:
            rank_changes.append({
                "feature": feat,
                "ht_rank": ht_r,
                "opt_rank": opt_r,
                "direction": "up" if ht_r > opt_r else "down",
            })
    rank_changes.sort(key=lambda x: abs(x["ht_rank"] - x["opt_rank"]), reverse=True)

    # Natural-language impact summary.
    ht_w = risk_engine_ht.weights
    opt_w = risk_engine_opt.weights
    all_agents = set(ht_w) | set(opt_w)
    deltas = {a: float(opt_w.get(a, 0.0)) - float(ht_w.get(a, 0.0)) for a in all_agents}
    gained = max(deltas, key=lambda k: deltas[k], default="shipping")
    lost = min(deltas, key=lambda k: deltas[k], default="geopolitical")

    if rank_changes:
        top = rank_changes[0]
        feat_str = top["feature"].replace("_", " ")
        agent_str = FEATURE_AGENT_MAP.get(top["feature"], "unknown").replace("_", " ")
        impact = (
            f"Optimization raised the inter-agent weight for '{gained}' "
            f"(Δ{deltas[gained]:+.3f}) and reduced '{lost}' "
            f"(Δ{deltas[lost]:+.3f}), causing '{feat_str}' "
            f"({agent_str} agent) to move {top['direction']} in SHAP importance rankings."
        )
    else:
        impact = (
            "Optimization produced no significant change in SHAP feature rankings "
            "for this sample."
        )

    return {
        "hand_tuned": {
            "top_drivers": ht_result["top_drivers"],
            "shap_values": ht_result["shap_values"],
            "r2": ht_r2,
        },
        "optimized": {
            "top_drivers": opt_result["top_drivers"],
            "shap_values": opt_result["shap_values"],
            "r2": opt_r2,
        },
        "driver_rank_changes": rank_changes,
        "weight_mode_impact": impact,
    }


# ---------------------------------------------------------------------------
# generate_comparison_plot
# ---------------------------------------------------------------------------


def generate_comparison_plot(
    features_df: pd.DataFrame,
    save_dir: str = "data/processed/",
    risk_engine_ht: Any = None,
    risk_engine_opt: Any = None,
) -> list[str]:
    """Generate side-by-side SHAP comparison plots for hand-tuned vs optimized.

    Saves two files:

    - ``shap_comparison_waterfall.png`` — horizontal-bar waterfall showing the
      top-10 SHAP values for both weight modes on the peak-disruption row.
    - ``shap_comparison_importance.png`` — grouped bar chart comparing mean
      |SHAP| across all features for both weight modes.

    Args:
        features_df: n × 20 feature DataFrame.
        save_dir: Output directory (created if absent).
        risk_engine_ht: Optional pre-built hand-tuned
            :class:`~src.aggregation.risk_engine.RiskEngine`.  When ``None``
            the engine is built from ``config/settings.yaml``.
        risk_engine_opt: Optional pre-built optimized
            :class:`~src.aggregation.risk_engine.RiskEngine`.  When ``None``
            the engine is built from ``config/optimized_weights.yaml``.

    Returns:
        List of paths of the saved PNG files (only those that succeeded).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    _Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Build RiskEngines from config when not supplied.
    if risk_engine_ht is None or risk_engine_opt is None:
        from src.aggregation.risk_engine import RiskEngine

        ht_weights = dict(_DEFAULT_WEIGHTS)
        opt_weights = dict(_DEFAULT_WEIGHTS)
        try:
            import yaml
            cfg_path = _Path("config/settings.yaml")
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                ht_weights = {k: float(v) for k, v in cfg.get("weights", _DEFAULT_WEIGHTS).items()}
        except Exception as exc:
            logger.warning("[generate_comparison_plot] settings.yaml load failed: %s", exc)

        try:
            import yaml
            opt_path = _Path("config/optimized_weights.yaml")
            if opt_path.exists():
                opt_data = yaml.safe_load(opt_path.read_text(encoding="utf-8")) or {}
                inter = opt_data.get("inter_agent_weights", {})
                if inter:
                    opt_weights = {k: float(v) for k, v in inter.items()}
        except Exception as exc:
            logger.warning("[generate_comparison_plot] optimized_weights.yaml load failed: %s", exc)

        _thr = {"risk_critical": 0.8, "risk_high": 0.6, "risk_medium": 0.4}
        if risk_engine_ht is None:
            risk_engine_ht = RiskEngine({"weights": ht_weights, "thresholds": _thr})
        if risk_engine_opt is None:
            risk_engine_opt = RiskEngine({"weights": opt_weights, "thresholds": _thr})

    comparison = compare_explanations(features_df, risk_engine_ht, risk_engine_opt)
    ht_shap = comparison["hand_tuned"]["shap_values"]
    opt_shap = comparison["optimized"]["shap_values"]

    ht_vals = np.array([ht_shap.get(f, 0.0) for f in ALL_FEATURE_NAMES])
    opt_vals = np.array([opt_shap.get(f, 0.0) for f in ALL_FEATURE_NAMES])
    feat_labels = [f.replace("_", " ") for f in ALL_FEATURE_NAMES]

    paths: list[str] = []

    # ---- Side-by-side waterfall ----
    try:
        top_n = 10
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, vals, title in zip(
            axes,
            [ht_vals, opt_vals],
            ["Hand-Tuned Weights", "Optimized Weights"],
        ):
            sorted_idx = np.argsort(np.abs(vals))[::-1][:top_n]
            s_feats = [feat_labels[i] for i in sorted_idx][::-1]
            s_vals = vals[sorted_idx][::-1]
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in s_vals]
            ax.barh(range(len(s_feats)), s_vals, color=colors, edgecolor="white", linewidth=0.4)
            ax.set_yticks(range(len(s_feats)))
            ax.set_yticklabels(s_feats, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("SHAP value", fontsize=9)
        fig.suptitle(
            "SHAP Waterfall: Hand-Tuned vs Optimized Weights (peak-disruption day)",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        wf_path = str(_Path(save_dir) / "shap_comparison_waterfall.png")
        plt.savefig(wf_path, dpi=120, bbox_inches="tight")
        plt.close("all")
        paths.append(wf_path)
        logger.info("[generate_comparison_plot] saved %s", wf_path)
    except Exception as exc:
        logger.warning("[generate_comparison_plot] waterfall plot failed: %s", exc)

    # ---- Feature importance bar chart ----
    try:
        ht_abs = np.abs(ht_vals)
        opt_abs = np.abs(opt_vals)
        avg_imp = (ht_abs + opt_abs) / 2.0
        top_idx = np.argsort(avg_imp)[::-1][:15]
        x_labels = [feat_labels[i] for i in top_idx]

        x = np.arange(len(x_labels))
        w = 0.38
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - w / 2, ht_abs[top_idx], w, label="Hand-Tuned", color="#1f77b4", alpha=0.85)
        ax.bar(x + w / 2, opt_abs[top_idx], w, label="Optimized", color="#ff7f0e", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("|SHAP value|", fontsize=10)
        ax.set_title(
            "Feature Importance: Hand-Tuned vs Optimized Weights",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10)
        plt.tight_layout()
        imp_path = str(_Path(save_dir) / "shap_comparison_importance.png")
        plt.savefig(imp_path, dpi=120, bbox_inches="tight")
        plt.close("all")
        paths.append(imp_path)
        logger.info("[generate_comparison_plot] saved %s", imp_path)
    except Exception as exc:
        logger.warning("[generate_comparison_plot] importance plot failed: %s", exc)

    return paths


# ---------------------------------------------------------------------------
# compute_faithfulness
# ---------------------------------------------------------------------------


def compute_faithfulness(
    features_df: pd.DataFrame,
    risk_scores: np.ndarray,
    ground_truth_disruptions: "np.ndarray | list[bool]",
) -> float:
    """Measure how faithfully SHAP top-3 features identify genuine anomalies.

    For each ground-truth disruption day, the three features with the largest
    absolute SHAP values are compared against a per-feature baseline built from
    non-disruption days.  A feature is *genuinely anomalous* when it deviates
    more than 1.5 standard deviations from that baseline mean.  Faithfulness
    is the fraction of top-3 SHAP features that passed this check, averaged
    over all disruption days.

    Args:
        features_df: n × 20 feature DataFrame.
        risk_scores: Per-row composite risk scores of shape ``(n,)``.
        ground_truth_disruptions: Boolean mask of shape ``(n,)`` — ``True``
            for confirmed disruption days.

    Returns:
        Faithfulness score in ``[0, 1]``.  Target: > 0.8.
    """
    flags = np.asarray(ground_truth_disruptions, dtype=bool)
    disruption_idx = np.where(flags)[0]
    baseline_mask = ~flags

    if baseline_mask.sum() == 0:
        logger.warning("[compute_faithfulness] no baseline days available.")
        return 0.0
    if len(disruption_idx) == 0:
        logger.warning("[compute_faithfulness] no disruption days in ground truth.")
        return 0.0

    baseline_df = features_df.iloc[baseline_mask]
    b_mean = baseline_df.mean(axis=0)
    b_std = baseline_df.std(axis=0)
    b_std = b_std.where(b_std > 1e-8, 1e-8)  # prevent div-by-zero

    explainer = SurrogateShapExplainer()
    explainer.train_surrogate(features_df, risk_scores)

    faithful = 0
    total = 0
    for idx in disruption_idx:
        row = features_df.iloc[[idx]]
        result = explainer.explain(row)
        top3 = result["top_drivers"][:3]
        for driver in top3:
            feat = driver["feature"]
            if feat not in features_df.columns:
                total += 1
                continue
            deviation = abs(float(features_df.iloc[idx][feat]) - float(b_mean[feat])) / float(b_std[feat])
            if deviation > 1.5:
                faithful += 1
            total += 1

    score = faithful / total if total > 0 else 0.0
    logger.info(
        "[compute_faithfulness] faithful=%d / %d = %.3f",
        faithful,
        total,
        score,
    )
    return score
