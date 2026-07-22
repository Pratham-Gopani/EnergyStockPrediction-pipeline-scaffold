"""Thin wrapper around DatasetExporter.export_predictions so the scheduler layer
doesn't need to build a DataFrame by hand for every run.
"""

from __future__ import annotations

import pandas as pd

from datasets.exporter import DatasetExporter
from utils.constants import PREDICTION_COLUMNS


class PredictionHistory:
    def __init__(self, exporter: DatasetExporter | None = None):
        self._exporter = exporter or DatasetExporter()

    def append(self, prediction_rows: list[dict]) -> None:
        if not prediction_rows:
            return
        df = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
        self._exporter.export_predictions(df)
