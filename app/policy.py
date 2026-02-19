"""Policy engine for regulatory compliance."""
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PolicyResult:
    approved: bool
    violations: List[str]
    warnings: List[str]


class PolicyEngine:
    MIN_CREDIT_SCORE = 580
    MAX_DTI_RATIO = 0.43
    MIN_INCOME = 20000
    MAX_LOAN_TO_INCOME = 5.0
    MAX_DELINQUENCIES = 3
    MAX_CREDIT_UTILIZATION = 0.90

    def __init__(self):
        self.policies = [
            self._check_credit_score,
            self._check_debt_to_income,
            self._check_loan_to_income,
            self._check_delinquencies,
            self._check_utilization,
            self._check_income,
        ]

    def evaluate(self, app) -> PolicyResult:
        violations, warnings = [], []
        for policy in self.policies:
            passed, msg, is_warn = policy(app)
            if not passed:
                (warnings if is_warn else violations).append(msg)
        return PolicyResult(len(violations) == 0, violations, warnings)

    def _check_credit_score(self, app) -> Tuple[bool, str, bool]:
        if app.credit_score < self.MIN_CREDIT_SCORE:
            return False, f"Credit score {app.credit_score} below {self.MIN_CREDIT_SCORE}", False
        if app.credit_score < 620:
            return True, "Subprime credit score", True
        return True, "", False

    def _check_debt_to_income(self, app) -> Tuple[bool, str, bool]:
        if app.debt_to_income > self.MAX_DTI_RATIO:
            return False, f"Debt-to-income {app.debt_to_income:.0%} exceeds {self.MAX_DTI_RATIO:.0%}", False
        if app.debt_to_income > 0.36:
            return True, "Elevated DTI", True
        return True, "", False

    def _check_loan_to_income(self, app) -> Tuple[bool, str, bool]:
        ratio = app.loan_amount / app.income
        if ratio > self.MAX_LOAN_TO_INCOME:
            return False, f"Loan-to-income {ratio:.1f}x exceeds {self.MAX_LOAN_TO_INCOME}x", False
        return True, "", False

    def _check_delinquencies(self, app) -> Tuple[bool, str, bool]:
        if app.num_delinquencies > self.MAX_DELINQUENCIES:
            return False, f"Delinquencies {app.num_delinquencies} exceed {self.MAX_DELINQUENCIES}", False
        return True, "", False

    def _check_utilization(self, app) -> Tuple[bool, str, bool]:
        if app.credit_utilization > self.MAX_CREDIT_UTILIZATION:
            return False, f"Utilization {app.credit_utilization:.0%} exceeds {self.MAX_CREDIT_UTILIZATION:.0%}", False
        return True, "", False

    def _check_income(self, app) -> Tuple[bool, str, bool]:
        if app.income < self.MIN_INCOME:
            return False, f"Income ${app.income:,.0f} below ${self.MIN_INCOME:,}", False
        return True, "", False


policy_engine = PolicyEngine()
