# Model Card: Credit Risk Assessment Model

## Model Details

### Basic Information
- **Model Name**: Credit Risk Classifier v1.0
- **Model Type**: Gradient Boosting Classifier
- **Framework**: scikit-learn
- **Task**: Binary Classification (Default Prediction)
- **Version**: 1.0.0
- **Last Updated**: 2024

### Developers
- Developed as a portfolio demonstration project
- For production use, additional validation required

## Intended Use

### Primary Use Case
- Credit risk assessment for consumer loan applications
- Probability of default estimation
- Risk-based pricing recommendations

### Intended Users
- Loan underwriters
- Credit risk analysts
- Automated decision systems (with human oversight)

### Out-of-Scope Uses
- Not intended for: employment decisions, housing decisions, or any use prohibited by fair lending laws
- Should not be used as the sole basis for credit decisions

## Training Data

### Data Description
- **Source**: Synthetic data generated to match typical credit application distributions
- **Size**: 10,000 samples
- **Features**: 9 numerical features related to creditworthiness

### Feature List
| Feature | Description | Range |
|---------|-------------|-------|
| age | Applicant age | 18-100 |
| income | Annual income (USD) | 15,000+ |
| employment_length | Years at current employer | 0-50 |
| credit_score | FICO credit score | 300-850 |
| num_credit_lines | Open credit accounts | 0-30 |
| credit_utilization | Credit usage ratio | 0-1 |
| loan_amount | Requested amount (USD) | 1,000-200,000 |
| debt_to_income | DTI ratio | 0-0.8 |
| num_delinquencies | Past delinquencies | 0-10 |

### Data Preprocessing
- No missing value imputation required
- Features used in raw form (no transformations)
- Target: Binary (0 = No Default, 1 = Default)

## Evaluation

### Metrics
| Metric | Value |
|--------|-------|
| Accuracy | ~82% |
| Precision | ~78% |
| Recall | ~75% |
| F1 Score | ~76% |
| ROC AUC | ~87% |

### Evaluation Data
- 20% holdout test set (stratified split)
- 5-fold cross-validation for validation

## Ethical Considerations

### Fairness
- Model uses only financial factors
- No demographic features (age is used only as a credit history proxy)
- Regular fairness audits recommended for production use

### Privacy
- No PII stored in model
- Feature values not recoverable from model weights

### Limitations
- Trained on synthetic data
- May not capture real-world distributional shifts
- Requires periodic retraining

## Regulatory Compliance

### Applicable Regulations
- Fair Credit Reporting Act (FCRA)
- Equal Credit Opportunity Act (ECOA)
- Ability-to-Repay Rule

### Compliance Features
- SHAP-based explainability for adverse action notices
- Policy engine for regulatory rule enforcement
- Audit logging capability

## Caveats and Recommendations

### Known Limitations
1. Synthetic training data may not reflect real credit populations
2. No temporal validation (model has not seen actual defaults)
3. Feature distributions may differ from production data

### Recommendations
1. Validate on real historical data before production use
2. Implement monitoring for feature drift
3. Regular model retraining (quarterly recommended)
4. Human review for edge cases

## Maintenance

### Update Schedule
- Quarterly performance review
- Annual model refresh
- Immediate updates for detected bias

### Monitoring
- Feature drift detection
- Prediction distribution monitoring
- False positive/negative rate tracking
