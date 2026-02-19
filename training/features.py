"""Feature definitions."""
import numpy as np
import pandas as pd

FEATURE_ORDER = ["age", "income", "employment_length", "credit_score",
                 "num_credit_lines", "credit_utilization", "loan_amount",
                 "debt_to_income", "num_delinquencies"]


def generate_synthetic_data(n=10000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "age": np.random.normal(40, 12, n).clip(18, 80).astype(int),
        "income": np.random.lognormal(10.8, 0.7, n).clip(15000, 500000),
        "employment_length": np.random.exponential(5, n).clip(0, 40),
        "credit_score": np.random.normal(680, 80, n).clip(300, 850).astype(int),
        "num_credit_lines": np.random.poisson(5, n).clip(0, 30),
        "credit_utilization": np.random.beta(2, 5, n).clip(0, 1),
        "loan_amount": np.random.lognormal(9.5, 0.8, n).clip(1000, 200000),
        "debt_to_income": np.random.beta(2, 6, n).clip(0, 0.8),
        "num_delinquencies": np.random.poisson(0.5, n).clip(0, 10),
    })
    risk = (-0.005 * df.credit_score + 0.3 * df.debt_to_income + 0.15 * df.credit_utilization
            + 0.1 * df.num_delinquencies - 0.001 * df.income / 10000)
    prob = 1 / (1 + np.exp(-risk))
    prob = (prob + np.random.uniform(-0.1, 0.1, n)).clip(0.01, 0.99)
    df["default"] = (np.random.random(n) < prob).astype(int)
    return df
