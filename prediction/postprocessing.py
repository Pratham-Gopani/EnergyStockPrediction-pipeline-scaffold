"""Post-inference safety net + output shaping for the multi-model prediction
registry (prediction.model_registry.ModelPrediction).

`sanitize_value` clamps a predicted price to +-clamp_pct around the last known
close. This is a real safety net, not decoration: an unverified deep-learning
reconstruction (see prediction/feature_engineering.py) -- or any model fed a
malformed feature vector -- can otherwise emit an absurd price that would
silently poison prediction_history.csv.
"""

from __future__ import annotations

from datetime import datetime

from prediction.model_registry import ModelPrediction
from utils.config_loader import get_config
from utils.constants import PREDICTION_COLUMNS
from utils.helpers import clamp


def sanitize_value(value: float, last_close: float, clamp_pct: float | None = None) -> float:
    if last_close is None or last_close != last_close or last_close <= 0:
        return value

    config = get_config()
    pct = clamp_pct if clamp_pct is not None else config.get("prediction.clamp_pct", 0.20)
    lo = last_close * (1 - pct)
    hi = last_close * (1 + pct)
    return clamp(value, lo, hi)


def sanitize_prediction(prediction: ModelPrediction, last_close: float, clamp_pct: float | None = None) -> ModelPrediction:
    sanitized_value = sanitize_value(prediction.value, last_close, clamp_pct)
    return ModelPrediction(
        ticker=prediction.ticker,
        model_name=prediction.model_name,
        backend=prediction.backend,
        target=prediction.target,
        value=sanitized_value,
        unverified=prediction.unverified,
    )


def to_prediction_row(
    prediction: ModelPrediction,
    company: str,
    news_date: str,
    last_close: float,
    predicted_at: datetime,
) -> dict:
    row = {col: None for col in PREDICTION_COLUMNS}
    row.update(
        {
            "Ticker": prediction.ticker,
            "Company": company,
            "News_Date": news_date,
            "Model_Name": prediction.model_name,
            "Backend": prediction.backend,
            "Target": prediction.target,
            "Predicted_Value": prediction.value,
            "Last_Close": last_close,
            "Unverified": prediction.unverified,
            "Predicted_At": predicted_at.isoformat(),
        }
    )
    return row
