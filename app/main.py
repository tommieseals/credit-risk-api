"""FastAPI Credit Risk API."""
import uuid
import time
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (CreditApplication, RiskPrediction, ExplainabilityResult,
                         HealthResponse, BatchPredictionRequest, BatchPredictionResponse, RiskLevel)
from app.policy import policy_engine
from app.model_loader import model_loader, ModelNotFoundError
from app.explain import CreditExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_ORDER = ["age", "income", "employment_length", "credit_score",
                 "num_credit_lines", "credit_utilization", "loan_amount",
                 "debt_to_income", "num_delinquencies"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Credit Risk API...")
    try:
        model_loader.load()
    except ModelNotFoundError:
        logger.warning("Model not found")
    yield


app = FastAPI(title="Credit Risk API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def to_features(a: CreditApplication) -> np.ndarray:
    return np.array([a.age, a.income, a.employment_length, a.credit_score,
                     a.num_credit_lines, a.credit_utilization, a.loan_amount,
                     a.debt_to_income, a.num_delinquencies]).reshape(1, -1)


def risk_level(p: float) -> RiskLevel:
    if p < 0.15: return RiskLevel.LOW
    if p < 0.35: return RiskLevel.MEDIUM
    if p < 0.55: return RiskLevel.HIGH
    return RiskLevel.VERY_HIGH


def action(p: float, ok: bool) -> str:
    if not ok: return "DECLINE - Policy"
    if p < 0.15: return "APPROVE"
    if p < 0.35: return "APPROVE - Monitor"
    if p < 0.55: return "MANUAL_REVIEW"
    return "DECLINE - Risk"


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", model_loaded=model_loader.is_loaded, version=model_loader.version)


@app.post("/predict", response_model=RiskPrediction)
async def predict(application: CreditApplication):
    if not model_loader.is_loaded:
        raise HTTPException(503, "Model not loaded")
    pol = policy_engine.evaluate(application)
    prob = model_loader.get_model().predict_proba(to_features(application))[0][1]
    score = max(300, min(850, int(850 - prob * 550)))
    return RiskPrediction(probability_of_default=round(prob, 4), risk_level=risk_level(prob),
                          credit_score_assigned=score, recommended_action=action(prob, pol.approved),
                          policy_approved=pol.approved, policy_violations=pol.violations,
                          model_version=model_loader.version, prediction_id=str(uuid.uuid4()))


@app.post("/predict/explain", response_model=ExplainabilityResult)
async def explain(application: CreditApplication):
    if not model_loader.is_loaded:
        raise HTTPException(503, "Model not loaded")
    exp = CreditExplainer(model_loader.get_model(), FEATURE_ORDER)
    r = exp.explain_prediction(to_features(application))
    return ExplainabilityResult(prediction_id=str(uuid.uuid4()), **r)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch(req: BatchPredictionRequest):
    if not model_loader.is_loaded:
        raise HTTPException(503, "Model not loaded")
    t0 = time.time()
    preds = []
    for a in req.applications:
        pol = policy_engine.evaluate(a)
        prob = model_loader.get_model().predict_proba(to_features(a))[0][1]
        score = max(300, min(850, int(850 - prob * 550)))
        preds.append(RiskPrediction(probability_of_default=round(prob, 4), risk_level=risk_level(prob),
                                    credit_score_assigned=score, recommended_action=action(prob, pol.approved),
                                    policy_approved=pol.approved, policy_violations=pol.violations,
                                    model_version=model_loader.version, prediction_id=str(uuid.uuid4())))
    return BatchPredictionResponse(predictions=preds, processing_time_ms=round((time.time() - t0) * 1000, 2))


@app.get("/model/info")
async def info():
    return {"version": model_loader.version, "features": FEATURE_ORDER, "health": model_loader.health_check()}
