"""Tests for market.indicators: rolling return volatility and ATR/Close, pure
numpy/pandas computations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market.indicators import average_true_range, rolling_return_volatility


def test_rolling_return_volatility_is_nonnegative_and_zero_for_flat_prices():
    flat_prices = pd.Series([100.0] * 50)
    vol = rolling_return_volatility(flat_prices, window=10)
    valid = vol.dropna()
    assert (valid >= 0).all()
    assert np.allclose(valid, 0.0, atol=1e-9)


def test_rolling_return_volatility_increases_with_noisier_prices():
    rng = np.random.default_rng(0)
    calm = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, 200)))
    volatile = pd.Series(100 + np.cumsum(rng.normal(0, 5.0, 200)))

    calm_vol = rolling_return_volatility(calm, window=90).dropna().iloc[-1]
    volatile_vol = rolling_return_volatility(volatile, window=90).dropna().iloc[-1]

    assert volatile_vol > calm_vol


def test_average_true_range_is_nonnegative():
    high = pd.Series([102, 103, 104, 103, 105] * 10, dtype=float)
    low = pd.Series([98, 99, 100, 99, 101] * 10, dtype=float)
    close = pd.Series([100, 101, 102, 101, 103] * 10, dtype=float)

    atr_over_close = average_true_range(high, low, close, window=14)
    valid = atr_over_close.dropna()
    assert (valid >= 0).all()
