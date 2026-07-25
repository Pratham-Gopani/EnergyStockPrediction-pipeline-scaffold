"""Tests for datasets.validator.validate_daily_row."""

from __future__ import annotations

from utils.constants import TOPIC_COLUMNS


def _valid_row() -> dict:
    row = {
        "Ticker": "TEST.NS",
        "Sentiment_Positive": 0.3,
        "Sentiment_Neutral": 0.4,
        "Sentiment_Negative": 0.3,
        "Sentiment_Score": 0.0,
        "Open": 100.0,
        "High": 105.0,
        "Low": 99.0,
        "Close": 102.0,
    }
    for i, col in enumerate(TOPIC_COLUMNS):
        row[col] = 1.0 / len(TOPIC_COLUMNS)
    return row


def test_valid_row_has_no_problems():
    from datasets.validator import validate_daily_row

    assert validate_daily_row(_valid_row()) == []


def test_missing_ticker_is_flagged():
    from datasets.validator import validate_daily_row

    row = _valid_row()
    row["Ticker"] = ""
    problems = validate_daily_row(row)
    assert any("Ticker" in p for p in problems)


def test_topic_probabilities_not_summing_to_one_is_flagged():
    from datasets.validator import validate_daily_row

    row = _valid_row()
    row[TOPIC_COLUMNS[0]] = 0.9  # now sums way over 1
    problems = validate_daily_row(row)
    assert any("Topic probabilities" in p for p in problems)


def test_sentiment_probabilities_not_summing_to_one_is_flagged():
    from datasets.validator import validate_daily_row

    row = _valid_row()
    row["Sentiment_Positive"] = 0.9
    problems = validate_daily_row(row)
    assert any("Sentiment probabilities" in p for p in problems)


def test_out_of_range_score_is_flagged():
    from datasets.validator import validate_daily_row

    row = _valid_row()
    row["Sentiment_Score"] = 5.0
    problems = validate_daily_row(row)
    assert any("out of range" in p for p in problems)


def test_missing_ohlc_is_flagged():
    from datasets.validator import validate_daily_row

    row = _valid_row()
    row["Close"] = None
    problems = validate_daily_row(row)
    assert any("Close is missing" in p for p in problems)
