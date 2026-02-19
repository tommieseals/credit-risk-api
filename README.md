# Credit Risk Assessment API

[![CI](https://github.com/tommieseals/credit-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/tommieseals/credit-risk-api/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready ML API for credit risk assessment, featuring **real-time predictions**, **SHAP explainability**, and **regulatory compliance** through a policy engine.

## 🎯 Key Features

- **FastAPI Backend**: High-performance async API with OpenAPI documentation
- **ML Pipeline**: Gradient Boosting model trained on credit application data
- **Explainability**: SHAP-based feature attribution for every prediction
- **Policy Engine**: Regulatory compliance checks (ATR rule, FCRA-ready)
- **Production Ready**: Docker support, CI/CD, comprehensive tests

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
├─────────────────────────────────────────────────────────────┤
│  /predict          │  /predict/explain  │  /predict/batch   │
├────────────────────┴───────────────────┴────────────────────┤
│                     Policy Engine                           │
│  • Credit Score Check  • DTI Validation  • Loan Limits     │
├─────────────────────────────────────────────────────────────┤
│                  ML Model (GradientBoosting)                │
├─────────────────────────────────────────────────────────────┤
│                   SHAP Explainer                            │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/tommieseals/credit-risk-api.git
cd credit-risk-api

# Install dependencies
pip install -r requirements.txt

# Train the model
make train

# Run the API
make run
```

### Using Docker

```bash
# Build and run with Docker
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

## 📊 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "employment_length": 5,
    "credit_score": 720,
    "num_credit_lines": 4,
    "credit_utilization": 0.30,
    "loan_amount": 25000,
    "loan_purpose": "debt_consolidation",
    "debt_to_income": 0.25,
    "num_delinquencies": 0
  }'
```

### Response
```json
{
  "probability_of_default": 0.1234,
  "risk_level": "low",
  "credit_score_assigned": 782,
  "recommended_action": "APPROVE - Standard terms",
  "policy_approved": true,
  "policy_violations": [],
  "model_version": "1.0.0",
  "prediction_id": "uuid-here"
}
```

### Get Explanation
```bash
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{ ... application data ... }'
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 📁 Project Structure

```
credit-risk-api/
├── app/
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Pydantic models
│   ├── policy.py         # Policy engine
│   ├── model_loader.py   # Model loading utilities
│   └── explain.py        # SHAP explainer
├── training/
│   ├── features.py       # Feature definitions
│   └── train.py          # Training pipeline
├── models/               # Saved models
├── tests/                # Test suite
├── docs/                 # Documentation
│   └── MODEL_CARD.md     # Model documentation
├── .github/workflows/    # CI/CD
├── Dockerfile
├── Makefile
└── requirements.txt
```

## 🔒 Policy Engine Rules

The policy engine enforces regulatory and business rules:

| Rule | Threshold | Regulation |
|------|-----------|------------|
| Minimum Credit Score | 580 | Business Rule |
| Max DTI Ratio | 43% | ATR Rule (CFPB) |
| Max Loan-to-Income | 5x | Business Rule |
| Max Delinquencies | 3 | Business Rule |
| Min Income | $20,000 | Business Rule |

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~82% |
| ROC AUC | ~87% |
| F1 Score | ~76% |

See [MODEL_CARD.md](docs/MODEL_CARD.md) for detailed model documentation.

## 🛠️ Development

```bash
# Format code
black app/ training/ tests/

# Sort imports
isort app/ training/ tests/

# Type checking
mypy app/
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

Built with ❤️ for ML Engineering portfolios

