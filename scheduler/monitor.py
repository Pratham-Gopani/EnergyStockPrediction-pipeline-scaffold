"""Post-inference monitoring pass: join prediction_history.csv (which has
Predicted_Open/Close) against dataset_daily.csv (which has the realised OHLCV for
that date, once the market has closed) on Ticker + News_Date, back-fill
Actual_Open/Actual_Close, and hand each ticker's joined history to
prediction.evaluator.ModelMonitor to compute + persist realised accuracy metrics.
"""

from __future__ import annotations

import pandas as pd

from prediction.evaluator import ModelMonitor
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger("scheduler.monitor")


def _load_csv(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_monitoring(config=None) -> dict:
    config = config or get_config()
    predictions_path = config.output_file("predictions_file")
    daily_path = config.output_file("daily_file")

    predictions_df = _load_csv(predictions_path)
    daily_df = _load_csv(daily_path)

    if predictions_df.empty or daily_df.empty:
        logger.info("Skipping monitoring: predictions or daily dataset is empty")
        return {}

    daily_actuals = daily_df[["Ticker", "News_Date", "Open", "Close"]].rename(
        columns={"Open": "Actual_Open", "Close": "Actual_Close"}
    )

    merged = predictions_df.drop(columns=["Actual_Open", "Actual_Close"], errors="ignore").merge(
        daily_actuals, on=["Ticker", "News_Date"], how="inner"
    )

    if merged.empty:
        logger.info("Skipping monitoring: no predictions have realised actuals yet")
        return {}

    monitor = ModelMonitor(config)
    results: dict[str, dict] = {}
    for ticker, group in merged.groupby("Ticker"):
        try:
            results[ticker] = monitor.evaluate(ticker, group)
        except Exception:  # noqa: BLE001 - one ticker's evaluation failure shouldn't block others
            logger.exception("Monitoring evaluation failed for %s", ticker)

    logger.info("Monitoring complete for %d ticker(s)", len(results))
    return results
