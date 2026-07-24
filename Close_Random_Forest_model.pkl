"""yfinance wrapper: lazy import (yfinance is heavy and network-dependent), retried
downloads, MultiIndex column flattening, and OHLCV_COLUMNS-shaped output.
"""

from __future__ import annotations

import pandas as pd

from utils.config_loader import get_config
from utils.constants import OHLCV_COLUMNS
from utils.logger import get_logger
from utils.retry import retry_from_config

logger = get_logger("market.yfinance_fetcher")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex (field, ticker) when downloading a single ticker
    via `Ticker.history()` in newer versions for some parameter combos; flatten to
    just the field name in that case.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df)
    df = df.rename(columns={"Adj Close": "Adj_Close"})
    if "Adj_Close" not in df.columns and "Close" in df.columns:
        df["Adj_Close"] = df["Close"]
    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OHLCV_COLUMNS]


@retry_from_config()
def get_history(ticker: str, lookback_days: int | None = None, interval: str | None = None) -> pd.DataFrame:
    """Download `lookback_days` of OHLCV history for `ticker` (e.g. "RELIANCE.NS")."""
    import yfinance as yf  # lazy: heavy + network dependency

    config = get_config()
    lookback_days = lookback_days or config.get("market.lookback_days", 200)
    interval = interval or config.get("market.history_interval", "1d")

    period = f"{lookback_days}d"
    history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if history is None or history.empty:
        logger.warning("Empty OHLCV history for %s (period=%s, interval=%s)", ticker, period, interval)
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    return _normalize_ohlcv(history)


def latest_ohlcv(ticker: str) -> dict:
    """Return the most recent OHLCV row for `ticker` as a plain dict, or empty dict
    if no history is available.
    """
    history = get_history(ticker, lookback_days=10)
    if history.empty:
        return {}
    last_row = history.iloc[-1]
    return {col: last_row[col] for col in OHLCV_COLUMNS}
