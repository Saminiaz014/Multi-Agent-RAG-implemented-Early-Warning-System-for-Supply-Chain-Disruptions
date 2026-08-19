"""FastAPI endpoints for the Supply Chain DSS.

Exposes the full 6-agent pipeline over HTTP:
  - GET  /health              — liveness + system status
  - POST /predict             — run 6-agent detection, return risk assessment
  - POST /explain             — SHAP feature contributions + RAG historical context
  - GET  /agents              — list all 6 agents with metadata
  - POST /agents/toggle       — enable / disable an agent at runtime
  - GET  /weights             — current weight mode and inter-agent weights
  - POST /weights/switch      — switch weight mode at runtime (no file write)
  - GET  /optimization/results — Optuna optimization output JSON (or 404)
  - POST /populate            — trigger knowledge-base population (background)
  - GET  /status              — per-connector health and KB doc counts
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Supply Chain DSS API",
    description="Multi-agent disruption detection with SHAP explainability.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Region management (Phase 12.3) — /api/regions/{list,current,info,switch}.
# Imported after `app` exists because the router reads this module's active
# region back out of it.
from src.api.region_endpoints import router as _region_router  # noqa: E402

app.include_router(_region_router)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path("config/settings.yaml")
_OPT_RESULTS_PATH = Path("data/processed/optimization_results.json")

_config: dict = {}
_orchestrator: Any = None
_last_run_timestamp: str | None = None

#: Region the service scores (Phase 12.3). Process-global, not per-client —
#: see src/api/region_endpoints.py. Resolved lazily so an invalid
#: SUPPLY_CHAIN_REGION surfaces on first use rather than at import.
_active_region: str | None = None

ALL_AGENTS = [
    "shipping",
    "market",
    "geopolitical",
    "natural_disaster",
    "routing",
    "news_sentiment",
]


def get_active_region() -> str:
    """Return the region the service is currently scoring.

    Resolved on first call from ``SUPPLY_CHAIN_REGION``, falling back to
    ``hormuz`` — the same precedence the CLI and dashboard use.
    """
    global _active_region
    if _active_region is None:
        from src.core.config_manager import resolve_active_region

        _active_region = resolve_active_region()
    return _active_region


def set_active_region(region: str) -> str:
    """Switch the service to ``region`` and invalidate the cached state.

    Args:
        region: Canonical region key — validate it before calling; this
            function assumes it is already known to the registry.

    Returns:
        The newly active region key.
    """
    global _active_region, _config
    _active_region = region
    _config = {}
    _reset_orchestrator()
    return _active_region


def _load_config() -> dict:
    """Return the active region's merged config, cached until a region switch.

    Phase 12.3: this reads ``config/settings.yaml`` *merged with* the active
    region's overlay, so every endpoint below scores the selected chokepoint.
    A missing settings.yaml degrades to an empty config rather than failing
    the whole service at import.
    """
    global _config
    if _config:
        return _config
    if _CONFIG_PATH.exists():
        from src.core.config_manager import load_config_for_region

        _config = load_config_for_region(get_active_region()) or {}
    else:
        logger.warning("config/settings.yaml not found — using empty config.")
        _config = {}
    return _config


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from src.orchestrator import Orchestrator
        _orchestrator = Orchestrator(_load_config())
    return _orchestrator


def _reset_orchestrator() -> None:
    global _orchestrator
    _orchestrator = None


def _get_doc_counts() -> dict[str, int]:
    try:
        import chromadb
        client = chromadb.PersistentClient(path="data/knowledge_base/.chromadb")
        cfg = _load_config()
        rag_cfg = cfg.get("rag", {}) or {}
        cols_cfg = rag_cfg.get("collections", {}) or {}
        static_name = cols_cfg.get("static_cases", "disruption_cases")
        live_name = cols_cfg.get("live_context", "live_extracted_context")
        counts: dict[str, int] = {}
        for name in (static_name, live_name):
            try:
                col = client.get_or_create_collection(name=name)
                counts[name] = col.count()
            except Exception:
                counts[name] = 0
        return counts
    except Exception as exc:
        logger.debug("ChromaDB unavailable for doc count: %s", exc)
        return {}


def _total_doc_count() -> int:
    return sum(_get_doc_counts().values())


def _agent_data_mode(name: str, cfg: dict) -> str:
    ingestion_cfg = cfg.get("ingestion", {}) or {}
    agents_cfg = cfg.get("agents", {}) or {}
    if name == "shipping":
        return str((ingestion_cfg.get("shipping", {}) or {}).get("source_mode", "synthetic"))
    if name == "market":
        return str((ingestion_cfg.get("market", {}) or {}).get("source_mode", "synthetic"))
    return str((agents_cfg.get(name, {}) or {}).get("data_mode", "synthetic"))


def _active_inter_weights(cfg: dict) -> dict[str, float]:
    """Return the currently active inter-agent weights (respects weight_mode)."""
    weights = {k: float(v) for k, v in (cfg.get("weights", {}) or {}).items()}
    if cfg.get("weight_mode", "hand_tuned") == "optimized":
        try:
            from src.optimization.weight_config import load_optimized_weights
            opt = load_optimized_weights(cfg)
            inter = opt.get("inter_agent_weights", {})
            if inter:
                weights = {k: float(v) for k, v in inter.items()}
        except Exception as exc:
            logger.warning("Could not load optimized weights: %s", exc)
    return weights


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Optional feature overrides (pipeline fetches its own data).",
    )
    agent: str = Field(
        default="shipping",
        description="Target agent hint (pipeline always runs all enabled agents).",
    )


class PredictResponse(BaseModel):
    composite_score: float
    risk_level: str
    agent_scores: dict[str, float]
    risk_score: float = 0.0
    contributing_agents: dict[str, Any] = Field(default_factory=dict)
    agent_agreement: int = 0
    reason: str = ""
    weight_mode: str = ""


class ExplainRequest(BaseModel):
    features: dict[str, float] = Field(default_factory=dict)
    agent: str = Field(default="shipping")


class ExplainResponse(BaseModel):
    top_features: list[dict[str, Any]]
    context: list[dict[str, Any]]
    explanation: dict[str, Any] = Field(default_factory=dict)
    historical_context: dict[str, Any] | None = None


class AgentInfo(BaseModel):
    name: str
    enabled: bool
    data_mode: str
    detection_method: str
    current_weight: float


class ToggleRequest(BaseModel):
    agent: str
    enabled: bool


class ToggleResponse(BaseModel):
    agent: str
    enabled: bool
    agents_active: list[str]
    message: str


class WeightsResponse(BaseModel):
    weight_mode: str
    inter_agent_weights: dict[str, float]
    thresholds: dict[str, Any]


class WeightSwitchRequest(BaseModel):
    mode: str = Field(..., description="'optimized' or 'hand_tuned'")


class WeightSwitchResponse(BaseModel):
    weight_mode: str
    message: str


class PopulateResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health() -> dict:
    """Return service liveness status and system summary."""
    cfg = _load_config()
    agents_cfg = cfg.get("agents", {}) or {}
    active = [
        n for n in ALL_AGENTS
        if (agents_cfg.get(n, {}) or {}).get("enabled", True)
    ]
    return {
        "status": "ok",
        "agents_active": active,
        "total_agents": len(active),
        "weight_mode": cfg.get("weight_mode", "hand_tuned"),
        "knowledge_base_doc_count": _total_doc_count(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, summary="Run risk prediction")
async def predict(request: PredictRequest) -> PredictResponse:
    """Run the full 6-agent pipeline and return a composite risk assessment."""
    try:
        global _last_run_timestamp
        orch = _get_orchestrator()
        result = orch.run_full_pipeline()
        _last_run_timestamp = datetime.now(timezone.utc).isoformat()

        agent_scores: dict[str, float] = dict(result.get("agent_scores", {}))
        contributing: dict[str, Any] = dict(result.get("contributing_agents", {}))

        # If aggregate() agent_scores is empty but compute_risk() contributing has data,
        # derive float scores from the richer structure.
        if not agent_scores and contributing:
            agent_scores = {
                name: float(info.get("score", 0.0))
                for name, info in contributing.items()
            }

        metadata = result.get("metadata", {}) or {}
        return PredictResponse(
            composite_score=float(result.get("composite_score", 0.0)),
            risk_level=str(result.get("risk_level", "LOW")),
            agent_scores=agent_scores,
            risk_score=float(result.get("risk_score", 0.0)),
            contributing_agents=contributing,
            agent_agreement=int(result.get("agent_agreement", 0)),
            reason=str(result.get("reason", "")),
            weight_mode=str(metadata.get("weight_mode", "")),
        )
    except Exception as exc:
        logger.exception("POST /predict failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------

@app.post("/explain", response_model=ExplainResponse, summary="Explain a prediction")
async def explain(request: ExplainRequest) -> ExplainResponse:
    """Return SHAP feature contributions and RAG historical context."""
    try:
        orch = _get_orchestrator()
        result = orch.run_full_pipeline()

        explanation: dict[str, Any] = dict(result.get("explanation", {}) or {})
        historical_context = result.get("historical_context")

        top_drivers = explanation.get("top_drivers", [])
        if isinstance(top_drivers, list):
            top_features: list[dict[str, Any]] = list(top_drivers)
        elif isinstance(top_drivers, dict):
            top_features = [{"feature": k, "value": v} for k, v in top_drivers.items()]
        else:
            top_features = []

        context: list[dict[str, Any]] = []
        if historical_context and isinstance(historical_context, dict):
            context = list(historical_context.get("matches", []))

        return ExplainResponse(
            top_features=top_features,
            context=context,
            explanation=explanation,
            historical_context=historical_context,
        )
    except Exception as exc:
        logger.exception("POST /explain failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /agents
# ---------------------------------------------------------------------------

@app.get("/agents", response_model=list[AgentInfo], summary="List all agents")
async def list_agents() -> list[AgentInfo]:
    """List all 6 agents with their current enabled state, data mode, and weight."""
    try:
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {}) or {}
        weights = _active_inter_weights(cfg)

        return [
            AgentInfo(
                name=name,
                enabled=bool((agents_cfg.get(name, {}) or {}).get("enabled", True)),
                data_mode=_agent_data_mode(name, cfg),
                detection_method=str(
                    (agents_cfg.get(name, {}) or {}).get("detection_method", "unknown")
                ),
                current_weight=float(weights.get(name, 0.0)),
            )
            for name in ALL_AGENTS
        ]
    except Exception as exc:
        logger.exception("GET /agents failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /agents/toggle
# ---------------------------------------------------------------------------

@app.post(
    "/agents/toggle",
    response_model=ToggleResponse,
    summary="Enable or disable an agent",
)
async def toggle_agent(request: ToggleRequest) -> ToggleResponse:
    """Enable or disable a named agent. RiskEngine renormalises weights automatically."""
    try:
        if request.agent not in ALL_AGENTS:
            raise HTTPException(
                status_code=422, detail=f"Unknown agent: {request.agent!r}. "
                f"Valid agents: {ALL_AGENTS}"
            )

        cfg = _load_config()
        agents_cfg = cfg.setdefault("agents", {})
        if request.agent not in agents_cfg or agents_cfg[request.agent] is None:
            agents_cfg[request.agent] = {}
        agents_cfg[request.agent]["enabled"] = request.enabled

        # Invalidate cached orchestrator so it rebuilds with the new agent roster.
        _reset_orchestrator()

        active = [
            n for n in ALL_AGENTS
            if (agents_cfg.get(n, {}) or {}).get("enabled", True)
        ]
        return ToggleResponse(
            agent=request.agent,
            enabled=request.enabled,
            agents_active=active,
            message=f"Agent '{request.agent}' {'enabled' if request.enabled else 'disabled'}.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /agents/toggle failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /weights
# ---------------------------------------------------------------------------

@app.get("/weights", response_model=WeightsResponse, summary="Return current weights")
async def get_weights() -> WeightsResponse:
    """Return the active weight mode, inter-agent weights, and risk thresholds."""
    try:
        cfg = _load_config()
        weight_mode = cfg.get("weight_mode", "hand_tuned")
        thresholds: dict[str, Any] = dict(cfg.get("thresholds", {}) or {})
        weights = _active_inter_weights(cfg)

        if weight_mode == "optimized":
            try:
                from src.optimization.weight_config import load_optimized_weights
                opt = load_optimized_weights(cfg)
                opt_thr = opt.get("thresholds", {})
                if opt_thr:
                    thresholds = {**thresholds, **opt_thr}
            except Exception as exc:
                logger.warning("Could not load optimized thresholds: %s", exc)

        return WeightsResponse(
            weight_mode=weight_mode,
            inter_agent_weights=weights,
            thresholds={
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in thresholds.items()
            },
        )
    except Exception as exc:
        logger.exception("GET /weights failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /weights/switch
# ---------------------------------------------------------------------------

@app.post(
    "/weights/switch",
    response_model=WeightSwitchResponse,
    summary="Switch weight mode at runtime",
)
async def switch_weights(request: WeightSwitchRequest) -> WeightSwitchResponse:
    """Switch between 'hand_tuned' and 'optimized' weight modes without writing files."""
    try:
        if request.mode not in ("optimized", "hand_tuned"):
            raise HTTPException(
                status_code=422,
                detail="mode must be 'optimized' or 'hand_tuned'",
            )

        cfg = _load_config()
        cfg["weight_mode"] = request.mode
        _reset_orchestrator()

        return WeightSwitchResponse(
            weight_mode=request.mode,
            message=(
                f"Weight mode switched to '{request.mode}'. "
                "Orchestrator will reinitialize on next request."
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /weights/switch failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /optimization/results
# ---------------------------------------------------------------------------

@app.get("/optimization/results", summary="Return Optuna optimization results")
async def optimization_results() -> dict:
    """Return contents of data/processed/optimization_results.json, or 404."""
    if not _OPT_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Optimization results not found at "
                f"{_OPT_RESULTS_PATH}. "
                "Run `python main.py --optimize` first."
            ),
        )
    try:
        return json.loads(_OPT_RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.exception("GET /optimization/results failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /populate
# ---------------------------------------------------------------------------

def _run_populate_background() -> None:
    script = Path("scripts/populate_knowledge_base.py")
    if not script.exists():
        logger.error("populate_knowledge_base.py not found at %s", script)
        return
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if proc.returncode == 0:
            logger.info("Knowledge base population complete.")
        else:
            logger.error("Knowledge base population failed:\n%s", proc.stderr)
    except Exception as exc:
        logger.exception("Background populate task failed: %s", exc)


@app.post(
    "/populate",
    response_model=PopulateResponse,
    summary="Trigger knowledge base population (async)",
)
async def populate(background_tasks: BackgroundTasks) -> PopulateResponse:
    """Trigger scripts/populate_knowledge_base.py in the background. Returns immediately."""
    background_tasks.add_task(_run_populate_background)
    return PopulateResponse(
        status="started",
        message="Knowledge base population started in background.",
    )


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@app.get("/status", summary="System status and agent health")
async def status() -> dict:
    """Return agent health, ChromaDB doc counts per collection, and last pipeline run."""
    cfg = _load_config()
    agents_cfg = cfg.get("agents", {}) or {}
    doc_counts = _get_doc_counts()

    agent_status = {
        name: {
            "enabled": bool((agents_cfg.get(name, {}) or {}).get("enabled", True)),
            "data_mode": _agent_data_mode(name, cfg),
            "last_fetch": None,
        }
        for name in ALL_AGENTS
    }

    return {
        "agents": agent_status,
        "knowledge_base": doc_counts,
        "last_pipeline_run": _last_run_timestamp,
        "weight_mode": cfg.get("weight_mode", "hand_tuned"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
