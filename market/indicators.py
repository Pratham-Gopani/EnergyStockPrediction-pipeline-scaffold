"""Pure numpy/pandas volatility indicators used by the adaptive sentiment threshold
and by the pre-inference feature pipeline. No I/O, no config reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_return_volatility(close: pd.Series, window: int = 90) -> pd.Series:
    """Rolling standard deviation of log returns over `window` periods.

    Returns a Series aligned to `close`'s index; the first `window` entries are NaN
    until enough history has accumulated.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=window, min_periods=max(2, window // 3)).std()


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """ATR normalized by close price (ATR/Close), so it's comparable across tickers."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=max(2, window // 3)).mean()
    return atr / close
