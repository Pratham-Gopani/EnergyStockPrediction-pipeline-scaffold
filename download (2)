"""Aggregate a company/day's article-level rows (Dataset 1) into a single
Dataset-2 row: weighted average of sentiment score/probs and topic probs
(weighted by News_Impact_Weight, which already sums to 1 for the batch), OHLCV +
News_Date taken via first(), headlines joined with " || ", Predicted_Topic as the
argmax of the WEIGHTED topic probabilities (not the per-article mode), and the
sentiment label derived from the same adaptive threshold used elsewhere. Pure
function -- no I/O, no config reads beyond what's passed in.
"""

from __future__ import annotations

import pandas as pd

from nlp.adaptive_threshold import ThresholdParams, adaptive_threshold, label_from_score
from utils.constants import DAILY_COLUMNS, OHLCV_COLUMNS, TOPIC_COLUMNS

HEADLINE_SEPARATOR = " || "


def aggregate_company_day(
    article_df: pd.DataFrame,
    ticker: str,
    company: str,
    news_date: str,
    ohlcv: dict,
    volatility: float | None,
    threshold_params: ThresholdParams,
) -> dict:
    """`article_df` must have at least News_Impact_Weight, Sentiment_Score,
    Sentiment_Positive/Neutral/Negative, TOPIC_COLUMNS, and Headline columns, with
    News_Impact_Weight already normalized to sum to 1 across the rows.
    """
    row: dict = {col: None for col in DAILY_COLUMNS}
    row["Ticker"] = ticker
    row["Company"] = company
    row["News_Date"] = news_date

    for col in OHLCV_COLUMNS:
        row[col] = ohlcv.get(col)

    if article_df.empty:
        row["Sentiment_Score"] = 0.0
        row["Sentiment_Positive"] = 0.0
        row["Sentiment_Neutral"] = 1.0
        row["Sentiment_Negative"] = 0.0
        for col in TOPIC_COLUMNS:
            row[col] = 1.0 / len(TOPIC_COLUMNS)
        row["Predicted_Topic"] = TOPIC_COLUMNS[0]
        row["Headlines"] = ""
        row["Article_Count"] = 0
    else:
        weights = article_df["News_Impact_Weight"].astype(float)

        row["Sentiment_Score"] = float((article_df["Sentiment_Score"].astype(float) * weights).sum())
        row["Sentiment_Positive"] = float((article_df["Sentiment_Positive"].astype(float) * weights).sum())
        row["Sentiment_Neutral"] = float((article_df["Sentiment_Neutral"].astype(float) * weights).sum())
        row["Sentiment_Negative"] = float((article_df["Sentiment_Negative"].astype(float) * weights).sum())

        weighted_topic_probs = {}
        for col in TOPIC_COLUMNS:
            weighted_topic_probs[col] = float((article_df[col].astype(float) * weights).sum())
            row[col] = weighted_topic_probs[col]
        row["Predicted_Topic"] = max(weighted_topic_probs, key=weighted_topic_probs.get)

        row["Headlines"] = HEADLINE_SEPARATOR.join(article_df["Headline"].fillna("").astype(str))
        row["Article_Count"] = int(len(article_df))

    row["Volatility"] = volatility
    threshold = adaptive_threshold(volatility, threshold_params)
    row["Sentiment_Label"] = label_from_score(row["Sentiment_Score"], threshold)

    return row
