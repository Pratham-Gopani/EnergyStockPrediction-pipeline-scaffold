"""Post-inference safety net + output shaping.

`sanitize_prediction` clamps predicted Open/Close to +-clamp_pct around the last
known close price. This is a real safety net, not decoration: a dummy or broken
model can otherwise emit an absurd price (negative, or orders of magnitude off)
that would silently poison prediction_history.csv.
"""

from __future__ import annotations

from datetime import datetime

from prediction.inference import Prediction
from utils.config_loader import get_config
from utils.constants import PREDICTION_COLUMNS
from utils.helpers import clamp


def sanitize_prediction(prediction: Prediction, last_close: float, clamp_pct: float | None = None) -> Prediction:
    if last_close is None or last_close != last_close or last_close <= 0:
        return prediction

    config = get_config()
    pct = clamp_pct if clamp_pct is not None else config.get("prediction.clamp_pct", 0.20)
    lo = last_close * (1 - pct)
    hi = last_close * (1 + pct)

    return Prediction(
        ticker=prediction.ticker,
        predicted_open=clamp(prediction.predicted_open, lo, hi),
        predicted_close=clamp(prediction.predicted_close, lo, hi),
    )


def to_prediction_row(
    prediction: Prediction,
    company: str,
    news_date: str,
    last_close: float,
    model_version: str,
    predicted_at: datetime,
) -> dict:
    row = {col: None for col in PREDICTION_COLUMNS}
    row.update(
        {
            "Ticker": prediction.ticker,
            "Company": company,
            "News_Date": news_date,
            "Predicted_Open": prediction.predicted_open,
            "Predicted_Close": prediction.predicted_close,
            "Last_Close": last_close,
            "Model_Version": model_version,
            "Predicted_At": predicted_at.isoformat(),
        }
    )
    return row
