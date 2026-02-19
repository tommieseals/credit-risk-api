"""Model loading with caching."""
import joblib
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    pass


class ModelLoader:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self._model = None
        self._version = None
        self._features = None

    @property
    def model_path(self):
        return self.models_dir / "credit_risk_model.joblib"

    @property
    def is_loaded(self):
        return self._model is not None

    @property
    def version(self):
        return self._version or "unknown"

    @property
    def feature_names(self):
        return self._features or []

    def load(self):
        if not self.model_path.exists():
            raise ModelNotFoundError(f"Model not found at {self.model_path}")
        data = joblib.load(self.model_path)
        if isinstance(data, dict):
            self._model = data["model"]
            self._version = data.get("version", "1.0.0")
            self._features = data.get("feature_names", [])
        else:
            self._model = data
            self._version = "1.0.0"
        return self._model

    def get_model(self):
        if not self.is_loaded:
            self.load()
        return self._model

    def health_check(self):
        try:
            if not self.model_path.exists():
                return {"healthy": False, "error": "Model not found"}
            if not self.is_loaded:
                self.load()
            return {"healthy": True, "version": self.version}
        except Exception as e:
            return {"healthy": False, "error": str(e)}


model_loader = ModelLoader()


@lru_cache(maxsize=1)
def get_model():
    return model_loader.get_model()
