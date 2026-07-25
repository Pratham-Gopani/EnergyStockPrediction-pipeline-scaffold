"""Post-inference monitoring pass: join prediction_history.csv (long format --
one row per Ticker/News_Date/Model_Name/Target) against dataset_daily.csv (which
has the realised Open/Close for that date, once the market has closed) on
Ticker + News_Date + Target, back-fill Actual_Value, and hand each
(ticker, model_name, target) group's joined history to
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


def _melt_actuals(daily_df: pd.DataFrame) -> pd.DataFrame:
    """dataset_daily.csv has one Open and one Close column per row; predictions
    are long-format (one Target per row), so melt Open/Close into matching
    Target="Open"/"Close" rows before joining.
    """
    open_actuals = daily_df[["Ticker", "News_Date", "Open"]].rename(columns={"Open": "Actual_Value"})
    open_actuals["Target"] = "Open"
    close_actuals = daily_df[["Ticker", "News_Date", "Close"]].rename(columns={"Close": "Actual_Value"})
    close_actuals["Target"] = "Close"
    return pd.concat([open_actuals, close_actuals], ignore_index=True)


def run_monitoring(config=None) -> dict:
    config = config or get_config()
    predictions_path = config.output_file("predictions_file")
    daily_path = config.output_file("daily_file")

    predictions_df = _load_csv(predictions_path)
    daily_df = _load_csv(daily_path)

    if predictions_df.empty or daily_df.empty:
        logger.info("Skipping monitoring: predictions or daily dataset is empty")
        return {}

    actuals = _melt_actuals(daily_df)
    merged = predictions_df.drop(columns=["Actual_Value"], errors="ignore").merge(
        actuals, on=["Ticker", "News_Date", "Target"], how="inner"
    )

    if merged.empty:
        logger.info("Skipping monitoring: no predictions have realised actuals yet")
        return {}

    monitor = ModelMonitor(config)
    results: dict[tuple, dict] = {}
    for (ticker, model_name, target), group in merged.groupby(["Ticker", "Model_Name", "Target"]):
        try:
            results[(ticker, model_name, target)] = monitor.evaluate(ticker, model_name, target, group)
        except Exception:  # noqa: BLE001 - one group's evaluation failure shouldn't block others
            logger.exception("Monitoring evaluation failed for %s/%s/%s", ticker, model_name, target)

    logger.info("Monitoring complete for %d (ticker, model, target) group(s)", len(results))
    return results
