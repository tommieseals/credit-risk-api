"""SHAP explainability."""
import shap
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CreditExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            try:
                self._explainer = shap.TreeExplainer(self.model)
            except:
                self._explainer = shap.Explainer(self.model)
        return self._explainer

    def explain_prediction(self, features, top_k=5):
        if features.ndim == 1:
            features = features.reshape(1, -1)
        shap_values = self.explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        vals = shap_values[0] if shap_values.ndim > 1 else shap_values
        contribs = dict(zip(self.feature_names, vals.tolist()))
        sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        pos = [f"{n}: +{v:.3f}" for n, v in sorted_c if v > 0][:top_k]
        neg = [f"{n}: {v:.3f}" for n, v in sorted_c if v < 0][:top_k]
        base = float(self.explainer.expected_value)
        if isinstance(base, (list, np.ndarray)):
            base = float(base[1])
        return {"feature_contributions": contribs, "top_positive_factors": pos,
                "top_negative_factors": neg, "base_value": base}
