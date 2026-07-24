"""Thin façade over datasets.updater.append_rows, one method per output file, so
callers never need to know the target filenames or column lists directly.
"""

from __future__ import annotations

import pandas as pd

from datasets.updater import append_rows
from utils.config_loader import get_config
from utils.constants import ARTICLE_COLUMNS, DAILY_COLUMNS, PERFORMANCE_COLUMNS, PREDICTION_COLUMNS


class DatasetExporter:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.config.ensure_dirs()
        self._backups_dir = self.config.path("backups_dir")

    def export_articles(self, articles_df: pd.DataFrame) -> None:
        target = self.config.output_file("articles_file")
        append_rows(target, articles_df, ARTICLE_COLUMNS, self._backups_dir)

    def export_daily(self, daily_df: pd.DataFrame) -> None:
        target = self.config.output_file("daily_file")
        append_rows(target, daily_df, DAILY_COLUMNS, self._backups_dir)

    def export_predictions(self, predictions_df: pd.DataFrame) -> None:
        target = self.config.output_file("predictions_file")
        append_rows(target, predictions_df, PREDICTION_COLUMNS, self._backups_dir)

    def export_performance(self, performance_df: pd.DataFrame) -> None:
        target = self.config.output_file("performance_file")
        append_rows(target, performance_df, PERFORMANCE_COLUMNS, self._backups_dir)
