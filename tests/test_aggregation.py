"""Tests for datasets.aggregation.aggregate_company_day: weighted sentiment/topic
averages, argmax-of-weighted-probs Predicted_Topic (not per-article mode),
" || "-joined headlines, and first()-style OHLCV pass-through.
"""

from __future__ import annotations

import pandas as pd

from datasets.aggregation import aggregate_company_day
from nlp.adaptive_threshold import ThresholdParams
from utils.constants import TOPIC_COLUMNS

PARAMS = ThresholdParams(base=0.15, k=0.8, min_threshold=0.08, max_threshold=0.45, v_ref=0.02)

OHLCV = {"Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 102.0, "Adj_Close": 102.0, "Volume": 1_000_000}


def _article_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_weighted_sentiment_average():
    rows = [
        {
            "Headline": "Headline A",
            "News_Impact_Weight": 0.75,
            "Sentiment_Score": 0.8,
            "Sentiment_Positive": 0.9,
            "Sentiment_Neutral": 0.05,
            "Sentiment_Negative": 0.05,
            **{col: (1.0 if i == 0 else 0.0) for i, col in enumerate(TOPIC_COLUMNS)},
        },
        {
            "Headline": "Headline B",
            "News_Impact_Weight": 0.25,
            "Sentiment_Score": -0.4,
            "Sentiment_Positive": 0.1,
            "Sentiment_Neutral": 0.2,
            "Sentiment_Negative": 0.7,
            **{col: (1.0 if i == 1 else 0.0) for i, col in enumerate(TOPIC_COLUMNS)},
        },
    ]
    df = _article_df(rows)

    result = aggregate_company_day(
        df, ticker="TEST.NS", company="Test Co", news_date="2025-08-13", ohlcv=OHLCV, volatility=0.02, threshold_params=PARAMS
    )

    expected_score = 0.75 * 0.8 + 0.25 * (-0.4)
    assert abs(result["Sentiment_Score"] - expected_score) < 1e-9


def test_predicted_topic_is_argmax_of_weighted_probs_not_mode():
    # Article A (higher weight) votes topic index 1; article B (lower weight, but
    # would win a simple majority vote if there were 2 more like it) votes index 0.
    rows = [
        {
            "Headline": "A",
            "News_Impact_Weight": 0.7,
            "Sentiment_Score": 0.0,
            "Sentiment_Positive": 0.3,
            "Sentiment_Neutral": 0.4,
            "Sentiment_Negative": 0.3,
            **{col: (1.0 if i == 1 else 0.0) for i, col in enumerate(TOPIC_COLUMNS)},
        },
        {
            "Headline": "B",
            "News_Impact_Weight": 0.3,
            "Sentiment_Score": 0.0,
            "Sentiment_Positive": 0.3,
            "Sentiment_Neutral": 0.4,
            "Sentiment_Negative": 0.3,
            **{col: (1.0 if i == 0 else 0.0) for i, col in enumerate(TOPIC_COLUMNS)},
        },
    ]
    df = _article_df(rows)

    result = aggregate_company_day(
        df, ticker="TEST.NS", company="Test Co", news_date="2025-08-13", ohlcv=OHLCV, volatility=0.02, threshold_params=PARAMS
    )

    assert result["Predicted_Topic"] == TOPIC_COLUMNS[1]


def test_headlines_joined_with_double_pipe():
    rows = [
        {
            "Headline": "First headline",
            "News_Impact_Weight": 0.5,
            "Sentiment_Score": 0.1,
            "Sentiment_Positive": 0.4,
            "Sentiment_Neutral": 0.4,
            "Sentiment_Negative": 0.2,
            **{col: 1.0 / len(TOPIC_COLUMNS) for col in TOPIC_COLUMNS},
        },
        {
            "Headline": "Second headline",
            "News_Impact_Weight": 0.5,
            "Sentiment_Score": 0.1,
            "Sentiment_Positive": 0.4,
            "Sentiment_Neutral": 0.4,
            "Sentiment_Negative": 0.2,
            **{col: 1.0 / len(TOPIC_COLUMNS) for col in TOPIC_COLUMNS},
        },
    ]
    df = _article_df(rows)

    result = aggregate_company_day(
        df, ticker="TEST.NS", company="Test Co", news_date="2025-08-13", ohlcv=OHLCV, volatility=0.02, threshold_params=PARAMS
    )

    assert result["Headlines"] == "First headline || Second headline"


def test_ohlcv_and_news_date_pass_through():
    rows = [
        {
            "Headline": "Only headline",
            "News_Impact_Weight": 1.0,
            "Sentiment_Score": 0.05,
            "Sentiment_Positive": 0.4,
            "Sentiment_Neutral": 0.4,
            "Sentiment_Negative": 0.2,
            **{col: 1.0 / len(TOPIC_COLUMNS) for col in TOPIC_COLUMNS},
        }
    ]
    df = _article_df(rows)

    result = aggregate_company_day(
        df, ticker="TEST.NS", company="Test Co", news_date="2025-08-13", ohlcv=OHLCV, volatility=0.02, threshold_params=PARAMS
    )

    assert result["News_Date"] == "2025-08-13"
    for key, value in OHLCV.items():
        assert result[key] == value
    assert result["Article_Count"] == 1


def test_topic_probabilities_preserved_in_output():
    rows = [
        {
            "Headline": "Only headline",
            "News_Impact_Weight": 1.0,
            "Sentiment_Score": 0.0,
            "Sentiment_Positive": 0.3,
            "Sentiment_Neutral": 0.4,
            "Sentiment_Negative": 0.3,
            **{col: (0.6 if i == 2 else 0.08) for i, col in enumerate(TOPIC_COLUMNS)},
        }
    ]
    df = _article_df(rows)

    result = aggregate_company_day(
        df, ticker="TEST.NS", company="Test Co", news_date="2025-08-13", ohlcv=OHLCV, volatility=0.02, threshold_params=PARAMS
    )

    for i, col in enumerate(TOPIC_COLUMNS):
        expected = 0.6 if i == 2 else 0.08
        assert abs(result[col] - expected) < 1e-9


def test_empty_article_df_produces_uniform_neutral_fallback():
    empty_df = pd.DataFrame(columns=["Headline", "News_Impact_Weight", "Sentiment_Score"])

    result = aggregate_company_day(
        empty_df, ticker="TEST.NS", company="Test Co", news_date="2025-08-13", ohlcv=OHLCV, volatility=0.02, threshold_params=PARAMS
    )

    assert result["Article_Count"] == 0
    assert result["Sentiment_Label"] == "Neutral"
    assert result["Headlines"] == ""
