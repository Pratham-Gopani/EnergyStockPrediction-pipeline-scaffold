"""Multi-backend model registry: loads every model declared in config/models.yaml
-- sklearn joblib artifacts (Ridge, RandomForest), an XGBoost joblib artifact, a
hand-reconstructed PyTorch LSTM (the uploaded file is a raw state_dict, not a full
model -- see _build_torch_lstm_module), and Keras 3 `.keras` LSTM models -- and
runs each one's prediction through a common interface. This is what lets the
orchestration layer get, sanitize, and store a prediction from EVERY uploaded
model per company/day, not just one.

All backend-specific imports (joblib, xgboost, torch, tensorflow) are lazy,
inside the loader functions, so importing this module doesn't require any of
them to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prediction.feature_engineering import FEATURE_SETS, feature_sequence, latest_feature_row
from utils.cache import ModelRegistry
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger("prediction.model_registry")


@dataclass(frozen=True)
class ModelPrediction:
    ticker: str
    model_name: str
    backend: str
    target: str  # "Open" | "Close"
    value: float
    unverified: bool


class ModelSpec:
    def __init__(self, raw: dict):
        self.name = raw["name"]
        self.target = raw["target"].capitalize()  # "open"/"close" (yaml) -> "Open"/"Close"
        self.backend = raw["backend"]
        self.file = raw["file"]
        self.feature_set = raw.get("feature_set")
        self.sequence_length = raw.get("sequence_length")
        self.architecture = raw.get("architecture") or {}
        self.target_scaler_file = raw.get("target_scaler_file")
        self.unverified = bool(raw.get("unverified", False))

    @property
    def registry_key(self) -> str:
        return f"model:{self.backend}:{self.name}:{self.target}:{self.file}"

    @property
    def columns(self) -> list[str]:
        if self.feature_set not in FEATURE_SETS:
            raise KeyError(f"Unknown feature_set '{self.feature_set}' for model {self.name}/{self.target}")
        return FEATURE_SETS[self.feature_set]


def load_model_specs(config=None) -> list[ModelSpec]:
    config = config or get_config()
    raw_list = config.get("models.models") or []
    return [ModelSpec(raw) for raw in raw_list]


# --- backend loaders -------------------------------------------------------

def _load_sklearn(path):
    import joblib  # lazy: heavy dependency

    return joblib.load(path)


def _load_xgboost(path):
    import joblib  # lazy: heavy dependency -- XGBRegressor pickles fine via joblib

    return joblib.load(path)


def _build_torch_lstm_module(architecture: dict):
    """Reconstructs the nn.Module matching the uploaded state_dict's parameter
    shapes (lstm.weight_ih_l0/weight_hh_l0/bias_ih_l0/bias_hh_l0, head.0.*,
    head.2.*): a single-layer LSTM feeding a 2-layer MLP head. The head's
    activation function (default ReLU) is NOT recoverable from the state_dict
    (it has no parameters) -- override architecture.head_activation in
    config/models.yaml if it turns out to be something else.
    """
    import torch.nn as nn

    input_size = architecture.get("input_size", 19)
    hidden_size = architecture.get("hidden_size", 48)
    head_hidden = architecture.get("head_hidden", 16)
    activation_name = architecture.get("head_activation", "relu")
    activation_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}.get(activation_name, nn.ReLU)

    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden),
                activation_cls(),
                nn.Linear(head_hidden, 1),
            )

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.head(h_n[-1])

    return LSTMRegressor()


def _load_torch(path, architecture: dict):
    import joblib  # lazy: the uploaded file is a plain-pickled state_dict, not torch.save format

    module = _build_torch_lstm_module(architecture)
    state_dict = joblib.load(path)
    module.load_state_dict(state_dict)
    module.eval()
    return module


def _load_keras(path):
    import tensorflow as tf  # lazy: heavy dependency

    return tf.keras.models.load_model(path)


def _load_target_scaler(path):
    import joblib  # lazy: heavy dependency

    return joblib.load(path)


class ModelRunner:
    """Loads (and caches, via ModelRegistry) every model in config/models.yaml
    and runs predictions for a single ticker's daily history.
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.specs = load_model_specs(self.config)
        self._registry = ModelRegistry()

    def _model_path(self, relative: str):
        return self.config.path("models_dir") / relative

    def _get_model(self, spec: ModelSpec):
        def factory():
            path = self._model_path(spec.file)
            if spec.backend == "sklearn":
                return _load_sklearn(path)
            if spec.backend == "xgboost":
                return _load_xgboost(path)
            if spec.backend == "torch":
                return _load_torch(path, spec.architecture)
            if spec.backend == "keras":
                return _load_keras(path)
            raise ValueError(f"Unknown model backend: {spec.backend}")

        return self._registry.get_or_create(spec.registry_key, factory)

    def _get_target_scaler(self, spec: ModelSpec):
        if not spec.target_scaler_file:
            return None
        path = self._model_path(spec.target_scaler_file)
        return self._registry.get_or_create(f"scaler:{spec.target_scaler_file}", lambda: _load_target_scaler(path))

    def predict_one(
        self, spec: ModelSpec, daily_history: pd.DataFrame, ticker: str, ticker_codes: dict[str, int]
    ) -> ModelPrediction | None:
        try:
            model = self._get_model(spec)
            if spec.unverified:
                logger.warning(
                    "Model %s/%s (%s) uses an UNVERIFIED feature reconstruction "
                    "-- see prediction/feature_engineering.py module docstring",
                    spec.name,
                    spec.target,
                    spec.backend,
                )

            columns = spec.columns

            if spec.backend in ("sklearn", "xgboost"):
                row = latest_feature_row(daily_history, ticker, ticker_codes)
                vector = np.array([[row[col] for col in columns]], dtype=float)
                value = float(model.predict(vector)[0])

            elif spec.backend == "torch":
                import torch  # lazy: heavy dependency

                sequence = feature_sequence(daily_history, ticker, ticker_codes, spec.sequence_length, columns)
                tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    value = float(model(tensor).reshape(-1)[0].item())

            elif spec.backend == "keras":
                sequence = feature_sequence(daily_history, ticker, ticker_codes, spec.sequence_length, columns)
                batch = sequence[np.newaxis, ...]
                raw_output = model.predict(batch, verbose=0)
                value = float(np.asarray(raw_output).reshape(-1)[0])
                scaler = self._get_target_scaler(spec)
                if scaler is not None:
                    value = float(scaler.inverse_transform([[value]])[0][0])

            else:
                raise ValueError(f"Unknown model backend: {spec.backend}")

            return ModelPrediction(
                ticker=ticker,
                model_name=spec.name,
                backend=spec.backend,
                target=spec.target,
                value=value,
                unverified=spec.unverified,
            )
        except Exception:  # noqa: BLE001 - one model failing must never block the others
            logger.exception("Prediction failed for model %s/%s (%s) on %s", spec.name, spec.target, spec.backend, ticker)
            return None

    def predict_all(
        self, daily_history: pd.DataFrame, ticker: str, ticker_codes: dict[str, int]
    ) -> list[ModelPrediction]:
        """Run every configured model for this ticker. A single model's failure
        (missing file, backend not installed, shape mismatch) is isolated and
        logged -- it never prevents the other models from producing a result.
        """
        predictions: list[ModelPrediction] = []
        for spec in self.specs:
            prediction = self.predict_one(spec, daily_history, ticker, ticker_codes)
            if prediction is not None:
                predictions.append(prediction)
        return predictions
