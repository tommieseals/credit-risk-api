"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CreditApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    employment_length: int = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    num_credit_lines: int = Field(..., ge=0)
    credit_utilization: float = Field(..., ge=0, le=1)
    loan_amount: float = Field(..., gt=0)
    loan_purpose: str
    debt_to_income: float = Field(..., ge=0, le=1)
    num_delinquencies: int = Field(..., ge=0)

    @field_validator("loan_purpose")
    @classmethod
    def validate_purpose(cls, v):
        valid = ["debt_consolidation", "credit_card", "home_improvement",
                 "major_purchase", "medical", "car", "vacation", "business", "other"]
        if v.lower() not in valid:
            raise ValueError(f"must be one of: {valid}")
        return v.lower()


class RiskPrediction(BaseModel):
    probability_of_default: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    credit_score_assigned: int = Field(..., ge=300, le=850)
    recommended_action: str
    policy_approved: bool
    policy_violations: List[str] = Field(default_factory=list)
    model_version: str
    prediction_id: str


class ExplainabilityResult(BaseModel):
    prediction_id: str
    feature_contributions: Dict[str, float]
    top_positive_factors: List[str]
    top_negative_factors: List[str]
    base_value: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class BatchPredictionRequest(BaseModel):
    applications: List[CreditApplication] = Field(..., max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: List[RiskPrediction]
    processing_time_ms: float
