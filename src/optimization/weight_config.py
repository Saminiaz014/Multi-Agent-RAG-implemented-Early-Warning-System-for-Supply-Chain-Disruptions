"""Active-weight resolution for the ``weight_mode`` switch.

A single source of truth for *which* weights the pipeline should use:

- ``weight_mode: "hand_tuned"`` — weights come from ``config/settings.yaml``
  (the ``weights`` / ``thresholds`` blocks and each ``agents.*.weights``
  sub-block), exactly as the pipeline behaved before optimization existed.
- ``weight_mode: "optimized"`` — weights come from
  ``config/optimized_weights.yaml``, the file Optuna regenerates after a
  successful optimization run.

The optimized file is laid out in three layers mirroring the pipeline:

    inter_agent_weights   → Layer 2 (RiskEngine aggregation weights)
    intra_agent_weights   → Layer 1 (per-agent feature weights)
    thresholds            → Layer 3 (risk + per-agent detection thresholds)

Loading is defensive: a missing or malformed optimized file logs a warning
and yields ``{}`` so callers can transparently fall back to hand-tuned
weights rather than crash.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_OPTIMIZED_PATH = Path("config/optimized_weights.yaml")


def optimized_weights_path(config: dict | None = None) -> Path:
    """Return the path to the optimized-weights YAML file.

    Args:
        config: Optional full config; an ``optimization.optimized_weights_path``
            key overrides the default location when present.

    Returns:
        Path to ``config/optimized_weights.yaml`` (or the configured override).
    """
    if config:
        opt = config.get("optimization", {}) or {}
        override = opt.get("optimized_weights_path")
        if override:
            return Path(override)
    return _DEFAULT_OPTIMIZED_PATH


def load_optimized_weights(config: dict | None = None) -> dict:
    """Load the Optuna-generated weight file.

    Args:
        config: Optional full config (used only to resolve the file path).

    Returns:
        Parsed weight dict with ``inter_agent_weights`` /
        ``intra_agent_weights`` / ``thresholds`` keys, or ``{}`` when the
        file is absent, empty, or unparseable.
    """
    path = optimized_weights_path(config)
    if not path.exists():
        logger.warning(
            "Optimized-weights file not found at %s — callers should fall "
            "back to hand-tuned weights.",
            path,
        )
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse %s (%s) — ignoring.", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Optimized-weights file %s is empty — ignoring.", path)
        return {}
    return data


def resolve_active_weights(config: dict) -> dict:
    """Resolve the weights the pipeline should currently use.

    Honours ``config["weight_mode"]``. In ``"optimized"`` mode the optimized
    file is loaded and, on any failure, the function logs and falls back to
    the hand-tuned layout assembled from ``config``.

    Args:
        config: Full application config dict.

    Returns:
        Dict with ``inter_agent_weights``, ``intra_agent_weights``,
        ``thresholds``, and a ``source`` field (``"optimized"`` /
        ``"hand_tuned"``) recording which path was actually taken.
    """
    mode = str(config.get("weight_mode", "hand_tuned")).lower()
    if mode == "optimized":
        optimized = load_optimized_weights(config)
        if optimized:
            optimized.setdefault("source", "optimized")
            return optimized
        logger.warning(
            "weight_mode='optimized' but optimized weights unavailable — "
            "falling back to hand-tuned weights."
        )
    return _hand_tuned_layout(config)


def apply_weights_to_agent(agent, layout: dict) -> bool:
    """Inject a resolved weight ``layout`` into a single agent in place.

    Dispatches on ``agent.name`` to the agent's ``set_weights`` /
    ``set_threshold`` methods. A no-op (returning ``False``) for unknown
    agents so callers can stay agnostic about the agent roster.

    Args:
        agent: A detection agent exposing ``set_weights`` / ``set_threshold``.
        layout: Output of :func:`resolve_active_weights` (``intra_agent_weights``
            + ``thresholds`` blocks).

    Returns:
        ``True`` if weights were applied, ``False`` for an unrecognised agent.
    """
    intra = layout.get("intra_agent_weights", {}) or {}
    thr = layout.get("thresholds", {}) or {}
    name = getattr(agent, "name", None)

    if name == "shipping":
        w = intra.get("shipping", {})
        agent.set_weights(w.get("isolation_forest", 0.70), w.get("zscore", 0.30))
        if "shipping_threshold" in thr:
            agent.set_threshold(thr["shipping_threshold"])
    elif name == "market":
        w = intra.get("market", {})
        agent.set_weights(
            w.get("oil", 0.40), w.get("trade_volume", 0.35), w.get("freight", 0.25)
        )
        if "market_z_threshold" in thr:
            agent.set_z_threshold(thr["market_z_threshold"])
    elif name == "geopolitical":
        w = intra.get("geopolitical", {})
        agent.set_weights(
            w.get("sanctions", 0.35), w.get("military", 0.25),
            w.get("diplomatic", 0.25), w.get("stability", 0.15),
        )
        if "geopolitical_threshold" in thr:
            agent.set_threshold(thr["geopolitical_threshold"])
    elif name == "natural_disaster":
        w = intra.get("natural_disaster", {})
        agent.set_weights(
            w.get("earthquake", 0.35), w.get("tsunami", 0.30),
            w.get("cyclone", 0.20), w.get("severe_weather", 0.15),
        )
        if "disaster_threshold" in thr:
            agent.set_threshold(
                thr["disaster_threshold"], thr.get("disaster_single_event")
            )
    elif name == "routing":
        w = intra.get("routing", {})
        agent.set_weights(w.get("model_score", 0.60), w.get("transit_zscore", 0.40))
        if "routing_threshold" in thr:
            agent.set_threshold(thr["routing_threshold"])
    elif name == "news_sentiment":
        w = intra.get("news_sentiment", {})
        agent.set_weights(
            w.get("sentiment", 0.40), w.get("consensus", 0.25),
            w.get("velocity", 0.20), w.get("volume", 0.15),
        )
        agent.set_threshold(
            negative_threshold=thr.get("news_negative_threshold"),
            consensus_threshold=thr.get("news_consensus_threshold"),
        )
    else:
        return False
    return True


def _hand_tuned_layout(config: dict) -> dict:
    """Assemble the optimized-file layout from the hand-tuned ``settings.yaml``."""
    agents = config.get("agents", {}) or {}

    def agent_weights(name: str) -> dict:
        return (agents.get(name, {}) or {}).get("weights", {}) or {}

    shipping_cfg = agents.get("shipping", {}) or {}
    market_cfg = agents.get("market", {}) or {}
    routing_w = agent_weights("routing")
    thresholds_cfg = config.get("thresholds", {}) or {}

    return {
        "source": "hand_tuned",
        "inter_agent_weights": dict(config.get("weights", {}) or {}),
        "intra_agent_weights": {
            "shipping": {
                # Synthetic-mode IF/zscore blend; hand-tuned default 0.70/0.30.
                "isolation_forest": 0.70,
                "zscore": 0.30,
            },
            "market": {"oil": 0.40, "trade_volume": 0.35, "freight": 0.25},
            "geopolitical": agent_weights("geopolitical")
            or {"sanctions": 0.35, "military": 0.25, "diplomatic": 0.25, "stability": 0.15},
            "natural_disaster": agent_weights("natural_disaster")
            or {"earthquake": 0.35, "tsunami": 0.30, "cyclone": 0.20, "severe_weather": 0.15},
            "routing": {
                "model_score": float(routing_w.get("model_score", 0.60)),
                "transit_zscore": float(routing_w.get("transit_zscore", 0.40)),
            },
            "news_sentiment": agent_weights("news_sentiment")
            or {"sentiment": 0.40, "consensus": 0.25, "velocity": 0.20, "volume": 0.15},
        },
        "thresholds": {
            "risk_high": float(thresholds_cfg.get("risk_high", 0.60)),
            "risk_medium": float(thresholds_cfg.get("risk_medium", 0.40)),
            "agreement_bonus_3": 1.15,
            "agreement_bonus_5": 1.25,
            "shipping_threshold": float(shipping_cfg.get("threshold", 0.65)),
            "market_z_threshold": float(market_cfg.get("z_threshold", 2.5)),
            "geopolitical_threshold": float(
                (agents.get("geopolitical", {}) or {}).get("threshold", 0.5)
            ),
            "disaster_threshold": float(
                (agents.get("natural_disaster", {}) or {}).get("threshold", 0.30)
            ),
            "disaster_single_event": float(
                (agents.get("natural_disaster", {}) or {}).get(
                    "single_event_threshold", 0.40
                )
            ),
            "routing_threshold": float(
                (agents.get("routing", {}) or {}).get("threshold", 0.55)
            ),
            "news_negative_threshold": float(
                (agents.get("news_sentiment", {}) or {}).get("negative_threshold", -0.30)
            ),
            "news_consensus_threshold": float(
                (agents.get("news_sentiment", {}) or {}).get("consensus_threshold", 0.40)
            ),
        },
    }
