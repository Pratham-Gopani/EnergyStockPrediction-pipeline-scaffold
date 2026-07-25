"""Tests for prediction.feature_engineering: pure pandas/numpy computations, no
heavy ML backends involved, so this runs with only the light dependency set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prediction.feature_engineering import (
    ENGINEERED_COLUMNS,
    FEATURE_SET_FULL,
    FEATURE_SET_KERAS_CLOSE,
    FEATURE_SET_KERAS_OPEN,
    FEATURE_SET_NO_TICKER,
    compute_daily_features,
    feature_sequence,
    latest_feature_row,
    ticker_code_map,
)
from utils.constants import DAILY_COLUMNS, TOPIC_COLUMNS


def _history(n_days: int, ticker: str = "TEST.NS") -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    rows = []
    price = 100.0
    for i, day in enumerate(dates):
        close = price + i * 1.5
        row = {col: None for col in DAILY_COLUMNS}
        row.update(
            {
                "Ticker": ticker,
                "Company": "Test Co",
                "News_Date": day.date().isoformat(),
                "Open": price + i,
                "High": close + 2,
                "Low": close - 2,
                "Close": close,
                "Adj_Close": close,
                "Volume": 1_000_000 + i * 1000,
                "Sentiment_Score": 0.1 * (i % 3),
            }
        )
        for j, col in enumerate(TOPIC_COLUMNS):
            row[col] = 1.0 / len(TOPIC_COLUMNS)
        rows.append(row)
    return pd.DataFrame(rows, columns=DAILY_COLUMNS)


def test_ticker_code_map_is_alphabetical():
    companies = [{"ticker": "ZZZ.NS"}, {"ticker": "AAA.NS"}, {"ticker": "MMM.NS"}]
    codes = ticker_code_map(companies)
    assert codes == {"AAA.NS": 0, "MMM.NS": 1, "ZZZ.NS": 2}


def test_compute_daily_features_has_all_engineered_columns():
    history = _history(10)
    features = compute_daily_features(history, "TEST.NS", {"TEST.NS": 5})
    assert list(features.columns) == ENGINEERED_COLUMNS
    assert len(features) == 10


def test_ticker_code_is_constant_across_rows():
    history = _history(5)
    features = compute_daily_features(history, "TEST.NS", {"TEST.NS": 7})
    assert (features["Ticker_Code"] == 7).all()


def test_log_volume_is_nonnegative():
    history = _history(5)
    features = compute_daily_features(history, "TEST.NS", {"TEST.NS": 0})
    assert (features["Log_Volume"] >= 0).all()


def test_cyclical_encodings_are_bounded():
    history = _history(30)
    features = compute_daily_features(history, "TEST.NS", {"TEST.NS": 0})
    for col in ["Month_sin", "Month_cos", "DOW_sin", "DOW_cos"]:
        assert (features[col] >= -1.0001).all()
        assert (features[col] <= 1.0001).all()


def test_empty_history_returns_empty_frame():
    empty = pd.DataFrame(columns=DAILY_COLUMNS)
    features = compute_daily_features(empty, "TEST.NS", {})
    assert features.empty
    assert list(features.columns) == ENGINEERED_COLUMNS


def test_latest_feature_row_is_the_last_row():
    history = _history(10)
    row = latest_feature_row(history, "TEST.NS", {"TEST.NS": 3})
    full = compute_daily_features(history, "TEST.NS", {"TEST.NS": 3})
    assert row["Sentiment_Score"] == full.iloc[-1]["Sentiment_Score"]


def test_feature_sequence_shape_matches_request():
    history = _history(20)
    sequence = feature_sequence(history, "TEST.NS", {"TEST.NS": 0}, sequence_length=15, columns=FEATURE_SET_NO_TICKER)
    assert sequence.shape == (15, len(FEATURE_SET_NO_TICKER))


def test_feature_sequence_pads_short_history_by_repeating_first_row():
    history = _history(5)
    sequence = feature_sequence(history, "TEST.NS", {"TEST.NS": 0}, sequence_length=15, columns=FEATURE_SET_NO_TICKER)
    assert sequence.shape == (15, len(FEATURE_SET_NO_TICKER))
    # The first 10 padded rows should all equal the earliest real row.
    assert np.allclose(sequence[0], sequence[9])


def test_feature_set_sizes_match_declared_model_input_widths():
    # These sizes are load-bearing: they must match the shapes recorded in
    # config/models.yaml (input_size=19 for the torch LSTM; 16/17 for the two
    # Keras LSTMs), or the demo script will fail with a shape mismatch.
    assert len(FEATURE_SET_FULL) == 20
    assert len(FEATURE_SET_NO_TICKER) == 19
    assert len(FEATURE_SET_KERAS_OPEN) == 16
    assert len(FEATURE_SET_KERAS_CLOSE) == 17
    assert "Ticker_Code" not in FEATURE_SET_NO_TICKER
