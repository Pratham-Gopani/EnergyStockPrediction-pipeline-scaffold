"""Pre-inference validation for a single Dataset-2 (daily aggregated) row. Returns
a list of human-readable problem strings; an empty list means the row is valid.
The caller (scheduler.tasks) is expected to skip+log rows with any problems rather
than feeding them to the prediction engine.
"""

from __future__ import annotations

from utils.config_loader import get_config
from utils.constants import TOPIC_COLUMNS
from utils.helpers import safe_float


def validate_daily_row(row: dict, prob_sum_tolerance: float | None = None) -> list[str]:
    problems: list[str] = []

    config = get_config()
    tolerance = prob_sum_tolerance if prob_sum_tolerance is not None else config.get(
        "prediction.prob_sum_tolerance", 0.05
    )

    ticker = row.get("Ticker")
    if not ticker or not str(ticker).strip():
        problems.append("Ticker is missing or blank")

    topic_sum = sum(safe_float(row.get(col)) for col in TOPIC_COLUMNS)
    if abs(topic_sum - 1.0) > tolerance:
        problems.append(f"Topic probabilities sum to {topic_sum:.4f}, expected ~1.0")

    sentiment_keys = ["Sentiment_Positive", "Sentiment_Neutral", "Sentiment_Negative"]
    if all(row.get(k) is not None for k in sentiment_keys):
        sentiment_sum = sum(safe_float(row.get(k)) for k in sentiment_keys)
        if abs(sentiment_sum - 1.0) > tolerance:
            problems.append(f"Sentiment probabilities sum to {sentiment_sum:.4f}, expected ~1.0")
    else:
        problems.append("Sentiment probability columns are missing")

    score = row.get("Sentiment_Score")
    if score is None:
        problems.append("Sentiment_Score is missing")
    else:
        score_val = safe_float(score, default=float("nan"))
        if score_val != score_val or not (-1.0 <= score_val <= 1.0):
            problems.append(f"Sentiment_Score {score} is out of range [-1, 1]")

    for ohlc_col in ("Open", "High", "Low", "Close"):
        value = row.get(ohlc_col)
        if value is None or value != value:  # None or NaN
            problems.append(f"{ohlc_col} is missing")

    return problems
