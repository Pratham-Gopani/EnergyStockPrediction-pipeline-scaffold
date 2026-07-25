"""Post-inference monitoring: once actual OHLC for a previously predicted day is
known (joined in via scheduler.monitor), compute realised accuracy metrics
(MAE/RMSE/MAPE/R2/Direction Accuracy) PER (Ticker, Model_Name, Target) -- since
every uploaded model gets its own row in prediction_history.csv, each one is
scored independently rather than pooled into one "the model" number -- persist
them to performance_history.csv, and log an alert if MAPE or direction accuracy
breach the configured thresholds.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datasets.exporter import DatasetExporter
from utils.config_loader import get_config
from utils.constants import PERFORMANCE_COLUMNS
from utils.logger import get_logger

logger = get_logger("prediction.evaluator")


def compute_metrics(actual: pd.Series, predicted: pd.Series, prev_close: pd.Series) -> dict:
    """All three series must be aligned (same index/order) and non-empty."""
    actual_arr = actual.astype(float).to_numpy()
    predicted_arr = predicted.astype(float).to_numpy()
    prev_arr = prev_close.astype(float).to_numpy()

    mae = float(mean_absolute_error(actual_arr, predicted_arr))
    rmse = float(np.sqrt(mean_squared_error(actual_arr, predicted_arr)))

    nonzero_mask = actual_arr != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((actual_arr[nonzero_mask] - predicted_arr[nonzero_mask]) / actual_arr[nonzero_mask])) * 100
        )
    else:
        mape = float("nan")

    r2 = float(r2_score(actual_arr, predicted_arr)) if len(actual_arr) > 1 else float("nan")

    actual_direction = np.sign(actual_arr - prev_arr)
    predicted_direction = np.sign(predicted_arr - prev_arr)
    direction_accuracy = float(np.mean(actual_direction == predicted_direction))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Direction_Accuracy": direction_accuracy,
    }


class ModelMonitor:
    def __init__(self, config=None, exporter: DatasetExporter | None = None):
        self.config = config or get_config()
        self.exporter = exporter or DatasetExporter()
        self.mape_alert_threshold = self.config.get("monitoring.mape_alert_threshold", 5.0)
        self.direction_accuracy_alert_threshold = self.config.get(
            "monitoring.direction_accuracy_alert_threshold", 0.50
        )

    def evaluate(self, ticker: str, model_name: str, target: str, joined: pd.DataFrame) -> dict:
        """`joined` must have Actual_Value, Predicted_Value, Last_Close columns
        for one (ticker, model_name, target) group. Computes metrics, persists a
        row to performance_history.csv, and logs an alert if thresholds are
        breached.
        """
        metrics = compute_metrics(joined["Actual_Value"], joined["Predicted_Value"], joined["Last_Close"])

        row = {col: None for col in PERFORMANCE_COLUMNS}
        row.update(
            {
                "Ticker": ticker,
                "Model_Name": model_name,
                "Target": target,
                "Evaluated_At": datetime.now().isoformat(),
                "N_Samples": int(len(joined)),
                **metrics,
            }
        )

        self.exporter.export_performance(pd.DataFrame([row], columns=PERFORMANCE_COLUMNS))
        self._alert_if_needed(ticker, model_name, target, metrics)
        return row

    def _alert_if_needed(self, ticker: str, model_name: str, target: str, metrics: dict) -> None:
        label = f"{ticker}/{model_name}/{target}"
        mape = metrics.get("MAPE")
        direction_accuracy = metrics.get("Direction_Accuracy")

        if mape is not None and mape == mape and mape > self.mape_alert_threshold:
            logger.warning("ALERT: %s MAPE %.2f%% exceeds threshold %.2f%%", label, mape, self.mape_alert_threshold)
        if (
            direction_accuracy is not None
            and direction_accuracy == direction_accuracy
            and direction_accuracy < self.direction_accuracy_alert_threshold
        ):
            logger.warning(
                "ALERT: %s Direction Accuracy %.2f%% below threshold %.2f%%",
                label,
                direction_accuracy * 100,
                self.direction_accuracy_alert_threshold * 100,
            )
