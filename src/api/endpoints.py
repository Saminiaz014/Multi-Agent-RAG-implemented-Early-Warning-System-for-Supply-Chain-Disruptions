"""FastAPI endpoints for prediction and explanation.

Exposes the DSS pipeline over HTTP with two routes:
  - POST /predict  — run detection and return a risk assessment.
  - POST /explain  — return SHAP feature contributions for a prediction.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Supply Chain DSS API",
    description="Multi-agent disruption detection with SHAP explainability.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    """Input payload for the /predict endpoint.

    Attributes:
        features: Mapping of feature name → numeric value for a single
            observation.
        agent: Which agent domain to invoke (e.g. 'shipping', 'market').
    """

    features: dict[str, float] = Field(..., description="Feature name-value pairs.")
    agent: str = Field(default="shipping", description="Target detection agent.")


class PredictResponse(BaseModel):
    """Output from the /predict endpoint.

    Attributes:
        composite_score: Aggregated risk score in [0, 1].
        risk_level: Categorical level — HIGH, MEDIUM, or LOW.
        agent_scores: Per-agent mean anomaly scores.
    """

    composite_score: float
    risk_level: str
    agent_scores: dict[str, float]


class ExplainRequest(BaseModel):
    """Input payload for the /explain endpoint.

    Attributes:
        features: Mapping of feature name → numeric value.
        agent: Which agent domain to explain.
    """

    features: dict[str, float]
    agent: str = Field(default="shipping")


class ExplainResponse(BaseModel):
    """Output from the /explain endpoint.

    Attributes:
        top_features: Top contributing features with their mean |SHAP| scores.
        context: Retrieved historical precedents from the RAG layer.
    """

    top_features: list[dict[str, Any]]
    context: list[dict[str, Any]]


@app.post("/predict", response_model=PredictResponse, summary="Run risk prediction")
async def predict(request: PredictRequest) -> PredictResponse:
    """Run the detection pipeline and return a composite risk assessment.

    Args:
        request: Feature vector and target agent identifier.

    Returns:
        :class:`PredictResponse` with composite score and risk level.

    Raises:
        HTTPException: 422 if the feature vector is empty.
    """
    if not request.features:
        raise HTTPException(status_code=422, detail="Feature vector must not be empty.")

    # Placeholder — orchestrator integration wired in next phase.
    logger.info("POST /predict | agent=%s | features=%s", request.agent, request.features)
    return PredictResponse(
        composite_score=0.0,
        risk_level="LOW",
        agent_scores={request.agent: 0.0},
    )


@app.post("/explain", response_model=ExplainResponse, summary="Explain a prediction")
async def explain(request: ExplainRequest) -> ExplainResponse:
    """Return SHAP feature contributions and RAG context for a prediction.

    Args:
        request: Feature vector and target agent identifier.

    Returns:
        :class:`ExplainResponse` with top SHAP features and historical cases.

    Raises:
        HTTPException: 422 if the feature vector is empty.
    """
    if not request.features:
        raise HTTPException(status_code=422, detail="Feature vector must not be empty.")

    logger.info("POST /explain | agent=%s", request.agent)
    return ExplainResponse(
        top_features=[],
        context=[],
    )


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    """Return service liveness status.

    Returns:
        JSON object with ``status: ok``.
    """
    return {"status": "ok"}
