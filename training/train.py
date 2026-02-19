"""Training pipeline."""
import sys
import logging
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.features import generate_synthetic_data, FEATURE_ORDER

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train(n=10000, test_size=0.2, seed=42):
    log.info("=" * 50)
    log.info("CREDIT RISK MODEL TRAINING")
    log.info("=" * 50)
    
    df = generate_synthetic_data(n, seed)
    log.info(f"Generated {n} samples, default rate: {df.default.mean():.2%}")
    
    X, y = df[FEATURE_ORDER], df["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    
    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=seed)
    model.fit(X_train, y_train)
    
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    log.info(f"CV AUC: {cv.mean():.4f} (+/- {cv.std():.4f})")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    log.info(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    log.info(f"Test ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    log.info("\n" + classification_report(y_test, y_pred))
    
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "version": "1.0.0", "feature_names": FEATURE_ORDER,
                 "trained_at": datetime.now().isoformat()}, "models/credit_risk_model.joblib")
    log.info("Model saved to models/credit_risk_model.joblib")


if __name__ == "__main__":
    train()
