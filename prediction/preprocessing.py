"""Pre-inference feature pipeline: build a feature vector from a Dataset-2 row in
the exact order recorded in models/metadata.json's `feature_columns`, then
nan_to_num -> scaler.transform -> pca.transform. The scaler/PCA objects are loaded
once via ModelRegistry and reused across calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.cache import ModelRegistry
from utils.config_loader import get_config
from utils.constants import TOPIC_COLUMNS
from utils.logger import get_logger

logger = get_logger("prediction.preprocessing")

DEFAULT_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj_Close",
    "Volume",
    "Sentiment_Score",
    "Sentiment_Positive",
    "Sentiment_Neutral",
    "Sentiment_Negative",
    *TOPIC_COLUMNS,
    "Article_Count",
    "Volatility",
]

_SCALER_REGISTRY_KEY = "prediction_scaler"
_PCA_REGISTRY_KEY = "prediction_pca"


class Preprocessor:
    def __init__(self, config=None):
        self.config = config or get_config()
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        metadata_path: Path = self.config.model_file("metadata_file")
        if not metadata_path.exists():
            logger.warning("No metadata.json found at %s; using default feature columns", metadata_path)
            return {"feature_columns": DEFAULT_FEATURE_COLUMNS}
        with open(metadata_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def feature_columns(self) -> list[str]:
        return self._metadata.get("feature_columns") or DEFAULT_FEATURE_COLUMNS

    def model_version(self) -> str:
        return str(self._metadata.get("model_version", "unknown"))

    def _load_scaler(self):
        import joblib  # lazy: heavy dependency

        path = self.config.model_file("scaler_file")
        return joblib.load(path)

    def _load_pca(self):
        import joblib  # lazy: heavy dependency

        path = self.config.model_file("pca_file")
        return joblib.load(path)

    def _scaler(self):
        registry = ModelRegistry()
        return registry.get_or_create(_SCALER_REGISTRY_KEY, self._load_scaler)

    def _pca(self):
        registry = ModelRegistry()
        return registry.get_or_create(_PCA_REGISTRY_KEY, self._load_pca)

    def build_feature_vector(self, daily_row: dict) -> np.ndarray:
        columns = self.feature_columns()
        raw = np.array([[daily_row.get(col) for col in columns]], dtype=float)
        return np.nan_to_num(raw)

    def transform(self, daily_row: dict) -> np.ndarray:
        raw_vector = self.build_feature_vector(daily_row)
        scaled = self._scaler().transform(raw_vector)
        reduced = self._pca().transform(scaled)
        return reduced
