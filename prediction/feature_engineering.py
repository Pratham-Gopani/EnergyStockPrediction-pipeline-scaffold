"""Reconstructs the engineered feature set the uploaded Ridge/RandomForest/XGBoost
models expect (confirmed from each model's own `feature_names_in_`):

    Sentiment_Score, Prob_Topic_1..6, NewsDay_Intraday_Pct, NewsDay_Range_Pct,
    Log_Volume, Month_sin, Month_cos, DOW_sin, DOW_cos, Sentiment_RollMean_3,
    Sentiment_RollMean_7, Momentum_RollMean_3, Momentum_RollMean_7,
    Volatility_Roll7, Ticker_Code

*** IMPORTANT CAVEAT ***
The uploaded model files confirm the feature NAMES and ORDER, but not the exact
formula behind each derived column (e.g. whether NewsDay_Range_Pct divides by
Close or Open) or the Ticker_Code -> integer mapping used at training time. This
module uses standard, defensible definitions (documented per-function below) and
a deterministic alphabetical Ticker_Code assignment. If your original training
notebook used different formulas or a different ticker ordering, tree/boosting
model predictions (which split on Ticker_Code) will disagree from the original
training distribution; Ridge is affected least since Ticker_Code is just one
linear coefficient. Compare against your training script and adjust the formulas
here if they differ -- everything is a small, isolated function so a fix is a
one-line change.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from utils.constants import TOPIC_COLUMNS
from utils.helpers import safe_float

ENGINEERED_COLUMNS = [
    "Sentiment_Score",
    *TOPIC_COLUMNS,
    "NewsDay_Intraday_Pct",
    "NewsDay_Range_Pct",
    "Log_Volume",
    "Month_sin",
    "Month_cos",
    "DOW_sin",
    "DOW_cos",
    "Sentiment_RollMean_3",
    "Sentiment_RollMean_7",
    "Momentum_RollMean_3",
    "Momentum_RollMean_7",
    "Volatility_Roll7",
    "Ticker_Code",
]


# Named feature-set slices referenced by config/models.yaml. The three
# sklearn-family models (Ridge/RandomForest/XGBoost) confirmed FEATURE_SET_FULL
# themselves via feature_names_in_ -- that one is verified, not guessed. The
# other three are PLACEHOLDERS sized to match each deep-learning model's declared
# input width (19 for the PyTorch LSTM, 16/17 for the two Keras LSTMs) so the
# pipeline runs end-to-end; see the module docstring caveat before trusting their
# output numerically.
FEATURE_SET_FULL = ENGINEERED_COLUMNS
FEATURE_SET_NO_TICKER = [c for c in ENGINEERED_COLUMNS if c != "Ticker_Code"]
FEATURE_SET_KERAS_OPEN = FEATURE_SET_NO_TICKER[:16]
FEATURE_SET_KERAS_CLOSE = FEATURE_SET_NO_TICKER[:17]

FEATURE_SETS = {
    "engineered_v1": FEATURE_SET_FULL,
    "engineered_v1_no_ticker": FEATURE_SET_NO_TICKER,
    "engineered_v1_keras_open": FEATURE_SET_KERAS_OPEN,
    "engineered_v1_keras_close": FEATURE_SET_KERAS_CLOSE,
}


def ticker_code_map(companies: list[dict]) -> dict[str, int]:
    """Deterministic Ticker -> integer code: alphabetical rank of all configured
    tickers. This is a BEST-EFFORT default (see module docstring) -- override by
    passing an explicit map if you know the training-time LabelEncoder order.
    """
    tickers = sorted(company["ticker"] for company in companies)
    return {ticker: index for index, ticker in enumerate(tickers)}


def _intraday_pct(open_price: float, close_price: float) -> float:
    """(Close - Open) / Open * 100 -- the day's own directional move."""
    if not open_price:
        return 0.0
    return (close_price - open_price) / open_price * 100.0


def _range_pct(high_price: float, low_price: float, close_price: float) -> float:
    """(High - Low) / Close * 100 -- the day's own trading range, normalized by close."""
    if not close_price:
        return 0.0
    return (high_price - low_price) / close_price * 100.0


def _log_volume(volume: float) -> float:
    return math.log1p(max(0.0, volume or 0.0))


def _cyclical(value: int, period: int) -> tuple[float, float]:
    angle = 2 * math.pi * (value / period)
    return math.sin(angle), math.cos(angle)


def compute_daily_features(daily_history: pd.DataFrame, ticker: str, ticker_codes: dict[str, int]) -> pd.DataFrame:
    """Compute ENGINEERED_COLUMNS for every row of `daily_history` (must be a
    single ticker's DAILY_COLUMNS-shaped rows, sorted ascending by News_Date --
    the caller is responsible for that ordering). Rolling windows use
    min_periods=1 so early history isn't dropped, just noisier.

    Returns a new DataFrame with the same index as `daily_history`, containing
    only ENGINEERED_COLUMNS (the caller concats/selects as needed).
    """
    if daily_history.empty:
        return pd.DataFrame(columns=ENGINEERED_COLUMNS)

    history = daily_history.reset_index(drop=True)
    dates = pd.to_datetime(history["News_Date"])

    result = pd.DataFrame(index=history.index)
    result["Sentiment_Score"] = history["Sentiment_Score"].astype(float)
    for col in TOPIC_COLUMNS:
        result[col] = history[col].astype(float)

    open_ = history["Open"].astype(float)
    high = history["High"].astype(float)
    low = history["Low"].astype(float)
    close = history["Close"].astype(float)
    volume = history["Volume"].astype(float)

    result["NewsDay_Intraday_Pct"] = [
        _intraday_pct(o, c) for o, c in zip(open_, close)
    ]
    result["NewsDay_Range_Pct"] = [
        _range_pct(h, l, c) for h, l, c in zip(high, low, close)
    ]
    result["Log_Volume"] = volume.apply(_log_volume)

    months = dates.dt.month
    dows = dates.dt.dayofweek  # Monday=0
    month_sin, month_cos = zip(*(_cyclical(m, 12) for m in months))
    dow_sin, dow_cos = zip(*(_cyclical(d, 7) for d in dows))
    result["Month_sin"] = month_sin
    result["Month_cos"] = month_cos
    result["DOW_sin"] = dow_sin
    result["DOW_cos"] = dow_cos

    result["Sentiment_RollMean_3"] = result["Sentiment_Score"].rolling(3, min_periods=1).mean()
    result["Sentiment_RollMean_7"] = result["Sentiment_Score"].rolling(7, min_periods=1).mean()

    # "Momentum" = day-over-day close return, matching the intuitive meaning of
    # price momentum; rolled to smooth it over 3/7 days.
    daily_return = close.pct_change().fillna(0.0) * 100.0
    result["Momentum_RollMean_3"] = daily_return.rolling(3, min_periods=1).mean()
    result["Momentum_RollMean_7"] = daily_return.rolling(7, min_periods=1).mean()

    # Same return-volatility definition used elsewhere in the pipeline
    # (market.indicators.rolling_return_volatility), just windowed to 7 here.
    log_returns = np.log(close / close.shift(1))
    result["Volatility_Roll7"] = log_returns.rolling(7, min_periods=2).std().fillna(0.0)

    ticker_code = ticker_codes.get(ticker, -1)
    result["Ticker_Code"] = ticker_code

    return result.fillna(0.0)[ENGINEERED_COLUMNS]


def latest_feature_row(daily_history: pd.DataFrame, ticker: str, ticker_codes: dict[str, int]) -> dict:
    """The single most recent row of engineered features, as a plain dict --
    what the sklearn/XGBoost models consume.
    """
    features = compute_daily_features(daily_history, ticker, ticker_codes)
    if features.empty:
        return {col: 0.0 for col in ENGINEERED_COLUMNS}
    return {col: safe_float(features.iloc[-1][col]) for col in ENGINEERED_COLUMNS}


def feature_sequence(
    daily_history: pd.DataFrame,
    ticker: str,
    ticker_codes: dict[str, int],
    sequence_length: int,
    columns: list[str],
) -> np.ndarray:
    """The last `sequence_length` rows of the given `columns`, shaped
    (sequence_length, len(columns)) for a recurrent model. If fewer rows of
    history exist than `sequence_length`, the earliest available row is repeated
    at the front (a documented cold-start pad, not a training-time behavior).
    """
    features = compute_daily_features(daily_history, ticker, ticker_codes)
    if features.empty:
        return np.zeros((sequence_length, len(columns)), dtype=float)

    features = features[columns]
    if len(features) < sequence_length:
        pad_count = sequence_length - len(features)
        first_row = features.iloc[[0]]
        padding = pd.concat([first_row] * pad_count, ignore_index=True)
        features = pd.concat([padding, features], ignore_index=True)

    return features.iloc[-sequence_length:].to_numpy(dtype=float)
