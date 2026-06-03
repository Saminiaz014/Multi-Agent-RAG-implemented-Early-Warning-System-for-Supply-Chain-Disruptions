"""Entrypoint for the Supply Chain Decision Support System.

Two modes:

- ``python main.py [--mode csv|synthetic]`` runs the full hybrid pipeline
  end-to-end (ingest → detect → aggregate → print summary).
- ``python main.py --serve`` starts the FastAPI service defined in
  :mod:`src.api.endpoints` on the host/port configured in
  ``settings.yaml`` (``api.host`` / ``api.port``).

The pipeline path is defensive: missing CSV files trigger an automatic
per-connector fallback to synthetic, agent failures are logged and the
risk summary is still printed with whatever data is available.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        path: Relative path to the settings file.

    Returns:
        Parsed configuration as a nested dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    """Configure root logger for console and file output.

    Args:
        config: Full application config dictionary; reads config["logging"].
    """
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "logs/pipeline.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Supply Chain Disruption Detection System — "
        "hybrid CSV / synthetic ingestion with multi-agent detection.",
    )
    parser.add_argument(
        "--mode",
        choices=["csv", "synthetic"],
        default=None,
        help="Override source_mode for both connectors "
        "(default: read from config/settings.yaml).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI HTTP service instead of running the pipeline.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(mode: str | None = None) -> dict:
    """Run the full hybrid pipeline and print a risk summary.

    Args:
        mode: Optional override for both connectors' ``source_mode``.
            When provided, it takes precedence over the value in
            ``config/settings.yaml``.

    Returns:
        Aggregated output dictionary from
        :class:`~src.aggregation.risk_engine.RiskEngine` — always
        populated (with ``LOW`` / 0.0 in degraded scenarios) so callers
        and the printed summary stay consistent.
    """
    from src.agents.market_agent import MarketAgent
    from src.agents.shipping_agent import ShippingAgent
    from src.aggregation.risk_engine import RiskEngine, RiskLevel
    from src.orchestrator import Orchestrator

    logger = logging.getLogger(__name__)
    config = load_config()
    if mode is not None:
        ingestion = config.setdefault("ingestion", {})
        ingestion.setdefault("shipping", {})["source_mode"] = mode
        ingestion.setdefault("market", {})["source_mode"] = mode
        logger.info("[main] --mode override applied: both connectors → '%s'", mode)

    orchestrator = Orchestrator(config=config)

    # -- a. INGEST -------------------------------------------------------
    try:
        shipping_df = orchestrator._safe_fetch(
            orchestrator._shipping_connector, "shipping"
        )
        market_df = orchestrator._safe_fetch(
            orchestrator._market_connector, "market"
        )
        logger.info(
            "[main.ingest] shipping rows=%d range=[%s..%s]",
            len(shipping_df),
            shipping_df["timestamp"].min(),
            shipping_df["timestamp"].max(),
        )
        logger.info(
            "[main.ingest] market rows=%d range=[%s..%s]",
            len(market_df),
            market_df["timestamp"].min(),
            market_df["timestamp"].max(),
        )
        orchestrator._warn_if_market_coverage_short(shipping_df, market_df)
        aligned_market = orchestrator._market_connector.align_with_shipping(
            shipping_df, market_df
        )
        combined_df = _merge_shipping_market(shipping_df, aligned_market)
        logger.info(
            "[main.ingest] combined rows=%d (shipping ⨝ aligned-market)",
            len(combined_df),
        )
    except Exception as exc:
        logger.exception("[main.ingest] failed: %s", exc)
        _print_summary(
            composite_score=0.0,
            risk_level=RiskLevel.LOW,
            agent_scores={},
            weights=config.get("weights", {}),
            shipping_windows=0,
            market_windows=0,
            note=f"INGEST FAILED: {exc}",
        )
        return {
            "composite_score": 0.0,
            "risk_level": RiskLevel.LOW,
            "agent_scores": {},
        }

    # -- b. DETECT -------------------------------------------------------
    shipping_agent = ShippingAgent(
        config={"contamination": 0.05, "threshold": 0.55, "z_threshold": 2.0}
    )
    market_agent = MarketAgent(
        config={"z_threshold": 1.5, "threshold": 0.50, "baseline_years": 5}
    )

    shipping_windows, shipping_result = _run_agent_safe(
        shipping_agent, combined_df, "shipping", logger
    )
    market_windows, market_result = _run_agent_safe(
        market_agent, combined_df, "market", logger
    )

    detection_results = [r for r in (shipping_result, market_result) if r is not None]

    # -- c. AGGREGATE ----------------------------------------------------
    try:
        engine = RiskEngine(config)
        aggregated = engine.aggregate(detection_results)
        logger.info(
            "[main.aggregate] composite=%.4f level=%s agents=%s",
            aggregated["composite_score"],
            aggregated["risk_level"],
            list(aggregated["agent_scores"].keys()),
        )
    except Exception as exc:
        logger.exception("[main.aggregate] failed: %s", exc)
        aggregated = {
            "composite_score": 0.0,
            "risk_level": RiskLevel.LOW,
            "agent_scores": {},
        }

    # -- d. SUMMARY ------------------------------------------------------
    _print_summary(
        composite_score=float(aggregated["composite_score"]),
        risk_level=aggregated["risk_level"],
        agent_scores=aggregated["agent_scores"],
        weights=config.get("weights", {}),
        shipping_windows=shipping_windows,
        market_windows=market_windows,
    )
    return aggregated


def start_api_server() -> None:
    """Start the FastAPI HTTP service via uvicorn."""
    import uvicorn

    logger = logging.getLogger(__name__)
    config = load_config()
    api_cfg = config.get("api", {}) or {}
    host = str(api_cfg.get("host", "0.0.0.0"))
    port = int(api_cfg.get("port", 8000))
    logger.info("[main.serve] starting FastAPI on %s:%d", host, port)
    print(f"Starting FastAPI server on http://{host}:{port} (Ctrl+C to stop)")
    uvicorn.run("src.api.endpoints:app", host=host, port=port, log_level="info")


def main() -> None:
    """Top-level entrypoint."""
    # Windows terminals default to cp1252, which can't encode the
    # box-drawing characters used in the summary. Force UTF-8 if the
    # platform supports it; harmless on POSIX.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:  # pragma: no cover - best-effort
                pass

    args = parse_args()
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Pipeline initialized")
    print("Pipeline initialized")

    if args.serve:
        start_api_server()
    else:
        run_pipeline(mode=args.mode)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_shipping_market(shipping_df, aligned_market_df):
    """Left-join shipping + aligned market on timestamp; backfill oil price."""
    import pandas as pd

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
    return combined


def _run_agent_safe(agent, df, name: str, logger: logging.Logger):
    """Run an agent, returning (window_count, DetectionResult|None) on failure."""
    try:
        windows = agent.run(df)
        validated = agent.run_dataframe(df)
        result = agent.to_detection_result(validated)
        logger.info(
            "[main.detect/%s] %d anomaly windows produced", name, len(windows)
        )
        return len(windows), result
    except Exception as exc:
        logger.exception("[main.detect/%s] agent failed: %s", name, exc)
        return 0, None


def _print_summary(
    *,
    composite_score: float,
    risk_level,
    agent_scores: dict,
    weights: dict,
    shipping_windows: int,
    market_windows: int,
    note: str | None = None,
) -> None:
    """Render the formatted risk summary box to stdout."""
    width = 38

    def row(text: str) -> str:
        return "║" + text.ljust(width) + "║"

    def center(text: str) -> str:
        return "║" + text.center(width) + "║"

    level_str = (
        risk_level.value if hasattr(risk_level, "value") else str(risk_level)
    )
    ship_score = float(agent_scores.get("shipping", 0.0))
    mkt_score = float(agent_scores.get("market", 0.0))
    ship_w = float(weights.get("shipping", 0.0))
    mkt_w = float(weights.get("market", 0.0))
    n_agents = len(agent_scores)

    lines = [
        "╔" + "═" * width + "╗",
        center("SUPPLY CHAIN DSS — RISK SUMMARY"),
        "╠" + "═" * width + "╣",
        row(f"  Risk Score : {composite_score:.2f}"),
        row(f"  Risk Level : {level_str}"),
        row(f"  Shipping   : {ship_score:.2f}  (w={ship_w:.2f})"),
        row(f"  Market     : {mkt_score:.2f}  (w={mkt_w:.2f})"),
        row(f"  Agreement  : {n_agents} agents"),
        row(f"  Windows    : ship={shipping_windows} mkt={market_windows}"),
    ]
    if note:
        lines.append("╠" + "═" * width + "╣")
        for chunk in _wrap(note, width - 4):
            lines.append(row(f"  {chunk}"))
    lines.append("╚" + "═" * width + "╝")
    print("\n".join(lines))


def _wrap(text: str, width: int) -> list[str]:
    """Trivial whitespace wrapper for summary notes."""
    words = text.split()
    out: list[str] = []
    current = ""
    for w in words:
        candidate = (current + " " + w).strip() if current else w
        if len(candidate) > width:
            if current:
                out.append(current)
            current = w
        else:
            current = candidate
    if current:
        out.append(current)
    return out


if __name__ == "__main__":
    main()
