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
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna weight optimization (train/val/test split) and write "
        "config/optimized_weights.yaml + data/processed/optimization_results.json.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override optimization.n_trials for this --optimize run.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(mode: str | None = None) -> dict:
    """Run the full six-agent pipeline and print a JSON risk assessment.

    Delegates ingestion → detection → aggregation to
    :meth:`~src.orchestrator.Orchestrator.run_full_pipeline`, which wires every
    agent enabled in ``config["agents"]`` into the pipeline (shipping + market
    on the merged daily frame; geopolitical, disaster, routing, news each on
    their own connector frame), honours the active ``weight_mode``, and
    degrades gracefully when an individual agent or connector fails.

    Args:
        mode: Optional override for both connectors' ``source_mode``.
            When provided, it takes precedence over the value in
            ``config/settings.yaml``.

    Returns:
        The full pipeline output dictionary from ``run_full_pipeline`` —
        always populated (with ``LOW`` / 0.0 in degraded scenarios) so callers
        and the printed summary stay consistent.
    """
    import json

    from src.aggregation.risk_engine import RiskLevel
    from src.orchestrator import Orchestrator

    logger = logging.getLogger(__name__)
    config = load_config()

    if mode is not None:
        ingestion = config.setdefault("ingestion", {})
        ingestion.setdefault("shipping", {})["source_mode"] = mode
        ingestion.setdefault("market", {})["source_mode"] = mode
        logger.info("[main] --mode override applied: both connectors → '%s'", mode)

    try:
        orchestrator = Orchestrator(config=config)
        result = orchestrator.run_full_pipeline()
    except Exception as exc:
        logger.exception("[main.run_pipeline] failed: %s", exc)
        _print_summary(
            composite_score=0.0,
            risk_level=RiskLevel.LOW,
            agent_scores={},
            weights=config.get("weights", {}),
            shipping_windows=0,
            market_windows=0,
            note=f"PIPELINE FAILED: {exc}",
        )
        return {
            "composite_score": 0.0,
            "risk_level": RiskLevel.LOW,
            "agent_scores": {},
        }

    # -- JSON risk assessment (machine-readable, all six agents) ---------
    print(json.dumps(_jsonable(_assessment_view(result)), indent=2))

    # -- Human-readable summary box -------------------------------------
    metadata = result.get("metadata", {}) or {}
    _print_summary(
        composite_score=float(result.get("composite_score", 0.0)),
        risk_level=result.get("risk_level", RiskLevel.LOW),
        agent_scores=result.get("agent_scores", {}),
        weights=config.get("weights", {}),
        shipping_windows=0,
        market_windows=0,
        note=(
            f"weight_mode={metadata.get('weight_mode', 'hand_tuned')} | "
            f"agents_active={len(metadata.get('agents_active', []))}/6"
        ),
    )
    return result


def run_optimization(n_trials: int | None = None) -> dict:
    """Run Optuna weight optimization end-to-end.

    Generates the train/val/test splits, tunes all three weight layers on
    train→validation, evaluates the best weights once on the held-out test
    split, and persists ``config/optimized_weights.yaml`` plus
    ``data/processed/optimization_results.json``. Flip ``weight_mode`` to
    ``"optimized"`` in ``config/settings.yaml`` afterwards to use the result.

    Args:
        n_trials: Optional override for ``optimization.n_trials``.

    Returns:
        The results dict from
        :meth:`~src.optimization.weight_optimizer.WeightOptimizer.optimize`.
    """
    import optuna

    from src.optimization.data_split import DataSplitManager
    from src.optimization.weight_optimizer import WeightOptimizer

    logger = logging.getLogger(__name__)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = load_config()

    print("Generating train/validation/test splits (seeds 42/43/44)…")
    data_manager = DataSplitManager(config)
    data_manager.generate_splits()
    data_manager.validate_splits()

    optimizer = WeightOptimizer(config, data_manager=data_manager)
    trials = n_trials if n_trials is not None else optimizer.n_trials
    print(f"Running Optuna optimization ({trials} trials)…")
    results = optimizer.optimize(n_trials=trials)

    # Render analysis figures (best-effort — never fail the run on a plot error).
    try:
        from src.optimization.optimization_analysis import generate_optimization_report

        figures = generate_optimization_report(optimizer, results)
        print(f"Wrote {len(figures)} analysis figure(s) to data/processed/.")
    except Exception as exc:  # pragma: no cover - visualisation is optional
        logger.warning("[main.optimize] figure generation failed: %s", exc)

    logger.info(
        "[main.optimize] best trial %d | val objective %.4f | test F1 %.3f",
        results["best_trial"],
        results["best_objective_value"],
        results["test_metrics"]["f1"],
    )
    print(
        "\nOptimization complete. Set weight_mode: \"optimized\" in "
        "config/settings.yaml to use the tuned weights."
    )
    return results


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

    if args.optimize:
        run_optimization(n_trials=args.trials)
    elif args.serve:
        start_api_server()
    else:
        run_pipeline(mode=args.mode)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assessment_view(result: dict) -> dict:
    """Project the pipeline output into a compact JSON risk assessment.

    Pulls out the six-agent composite, the per-agent contribution breakdown,
    and the run metadata (``agents_active`` / ``data_modes`` / ``weight_mode``)
    so the printed JSON is the canonical machine-readable assessment.
    """
    metadata = result.get("metadata", {}) or {}
    risk_level = result.get("risk_level")
    return {
        "risk_score": result.get("risk_score", result.get("composite_score", 0.0)),
        "risk_level": risk_level.value if hasattr(risk_level, "value") else risk_level,
        "composite_score": result.get("composite_score", 0.0),
        "reason": result.get("reason", ""),
        "agent_agreement": result.get("agent_agreement", 0),
        "contributing_agents": result.get("contributing_agents", {}),
        "metadata": {
            "agents_active": metadata.get("agents_active", []),
            "data_modes": metadata.get("data_modes", {}),
            "weight_mode": metadata.get("weight_mode", "hand_tuned"),
            "active_agents": metadata.get("active_agents"),
            "weights_used": metadata.get("weights_used", {}),
        },
        "data": result.get("data", {}),
    }


def _jsonable(obj):
    """Recursively coerce numpy/enum values so ``json.dumps`` never fails."""
    import enum

    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


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
    # Canonical agent order + compact display labels for the summary box.
    display = [
        ("shipping", "Shipping"),
        ("market", "Market"),
        ("geopolitical", "Geopolitical"),
        ("natural_disaster", "Disaster"),
        ("routing", "Routing"),
        ("news_sentiment", "News"),
    ]
    n_agents = len(agent_scores)

    lines = [
        "╔" + "═" * width + "╗",
        center("SUPPLY CHAIN DSS — RISK SUMMARY"),
        "╠" + "═" * width + "╣",
        row(f"  Risk Score : {composite_score:.2f}"),
        row(f"  Risk Level : {level_str}"),
        "╟" + "─" * width + "╢",
    ]
    for key, label in display:
        if key not in agent_scores:
            continue
        score = float(agent_scores[key])
        w = float(weights.get(key, 0.0))
        lines.append(row(f"  {label:<12}: {score:.2f}  (w={w:.2f})"))
    lines.append("╟" + "─" * width + "╢")
    lines.append(row(f"  Agents     : {n_agents}/6 contributing"))
    lines.append(row(f"  Windows    : ship={shipping_windows} mkt={market_windows}"))
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
