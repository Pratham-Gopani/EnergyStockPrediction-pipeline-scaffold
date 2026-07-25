"""Loads the pretrained Open/Close regression models (cached via ModelRegistry)
and runs inference on a preprocessed feature vector.

NOTE: this single-model scaler->PCA->model path is the original scaffold's generic
plumbing example (paired with scripts/generate_dummy_models.py). The pipeline's
actual production prediction path -- covering the multiple uploaded Ridge/
RandomForest/XGBoost/PyTorch-LSTM/Keras-LSTM models -- is
prediction.model_registry.ModelRunner, driven by config/models.yaml. This module
is kept as a minimal single-model reference/fallback, not wired into
scheduler.tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from prediction.preprocessing import Preprocessor
from utils.cache import ModelRegistry
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger("prediction.inference")

_OPEN_MODEL_REGISTRY_KEY = "prediction_open_model"
_CLOSE_MODEL_REGISTRY_KEY = "prediction_close_model"


@dataclass(frozen=True)
class Prediction:
    ticker: str
    predicted_open: float
    predicted_close: float


class PredictionEngine:
    def __init__(self, config=None, preprocessor: Preprocessor | None = None):
        self.config = config or get_config()
        self.preprocessor = preprocessor or Preprocessor(self.config)

    def _load_open_model(self):
        import joblib  # lazy: heavy dependency

        return joblib.load(self.config.model_file("open_model_file"))

    def _load_close_model(self):
        import joblib  # lazy: heavy dependency

        return joblib.load(self.config.model_file("close_model_file"))

    def _open_model(self):
        registry = ModelRegistry()
        return registry.get_or_create(_OPEN_MODEL_REGISTRY_KEY, self._load_open_model)

    def _close_model(self):
        registry = ModelRegistry()
        return registry.get_or_create(_CLOSE_MODEL_REGISTRY_KEY, self._load_close_model)

    def predict(self, daily_row: dict) -> Prediction:
        features = self.preprocessor.transform(daily_row)
        predicted_open = float(self._open_model().predict(features)[0])
        predicted_close = float(self._close_model().predict(features)[0])
        return Prediction(
            ticker=daily_row.get("Ticker"),
            predicted_open=predicted_open,
            predicted_close=predicted_close,
        )
