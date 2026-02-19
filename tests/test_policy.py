"""Policy engine tests."""
import pytest
from app.policy import policy_engine
from app.schemas import CreditApplication


def make_app(**kw):
    d = {"age": 35, "income": 75000, "employment_length": 5, "credit_score": 720,
         "num_credit_lines": 4, "credit_utilization": 0.3, "loan_amount": 25000,
         "loan_purpose": "debt_consolidation", "debt_to_income": 0.25, "num_delinquencies": 0}
    d.update(kw)
    return CreditApplication(**d)


def test_good_app_passes():
    r = policy_engine.evaluate(make_app())
    assert r.approved and len(r.violations) == 0


def test_low_credit_score_fails():
    r = policy_engine.evaluate(make_app(credit_score=500))
    assert not r.approved
    assert any("credit score" in v.lower() for v in r.violations)


def test_high_dti_fails():
    r = policy_engine.evaluate(make_app(debt_to_income=0.50))
    assert not r.approved


def test_high_loan_to_income_fails():
    r = policy_engine.evaluate(make_app(income=30000, loan_amount=200000))
    assert not r.approved


def test_delinquencies_fails():
    r = policy_engine.evaluate(make_app(num_delinquencies=5))
    assert not r.approved


def test_utilization_fails():
    r = policy_engine.evaluate(make_app(credit_utilization=0.95))
    assert not r.approved


def test_low_income_fails():
    r = policy_engine.evaluate(make_app(income=15000))
    assert not r.approved
