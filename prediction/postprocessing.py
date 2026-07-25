"""Post-inference safety net + output shaping for the multi-model prediction
registry (prediction.model_registry.ModelPrediction).

`sanitize_value` clamps a predicted price to +-clamp_pct around the last known
close. This is a real safety net, not decoration: an unverified deep-learning
reconstruction (see prediction/feature_engineering.py) -- or any model fed a
malformed feature vector, or a model trained on a transformed target (e.g. a %
return or z-score) rather than an absolute price -- can otherwise emit an absurd
price that would silently poison prediction_history.csv. When the clamp actually
changes a value, that's a real signal something's off (garbage prediction, or a
raw output whose units don't match "price"), so it's logged loudly and the raw
pre-clamp value is preserved in Raw_Predicted_Value rather than being discarded.
"""

from __future__ import annotations

from datetime import datetime

from prediction.model_registry import ModelPrediction
from utils.config_loader import get_config
from utils.constants import PREDICTION_COLUMNS
from utils.helpers import clamp
from utils.logger import get_logger

logger = get_logger("prediction.postprocessing")


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
    if sanitized_value != prediction.value:
        logger.warning(
            "Clamped %s/%s (%s) prediction for %s: raw=%.4f is outside the sanitize band around "
            "last_close=%.4f -> stored as %.4f. A raw value this far off usually means the model's "
            "output isn't in the same units as price (e.g. a % return or scaled target) -- see "
            "Raw_Predicted_Value in prediction_history.csv.",
            prediction.model_name,
            prediction.target,
            prediction.backend,
            prediction.ticker,
            prediction.value,
            last_close,
            sanitized_value,
        )
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
    raw_value: float | None = None,
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
            "Raw_Predicted_Value": raw_value if raw_value is not None else prediction.value,
            "Last_Close": last_close,
            "Unverified": prediction.unverified,
            "Predicted_At": predicted_at.isoformat(),
        }
    )
    return row
