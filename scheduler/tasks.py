"""The daily pipeline orchestration: calendar check -> news window -> a
cross-sectional reference-volatility pre-pass (needed by the adaptive threshold)
-> concurrent per-company processing (scrape -> dedup -> extract -> OHLCV ->
volatility -> build article dataset -> aggregate to a daily row), with per-company
try/except isolation so one company's failure never aborts the run -> validate ->
predict (lazy-loaded engine, sanitized) -> append-only persistence of
articles/daily/predictions -> a RunStats summary.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from datasets.aggregation import aggregate_company_day
from datasets.article_dataset import ArticleDatasetBuilder
from datasets.exporter import DatasetExporter
from datasets.validator import validate_daily_row
from market.indicators import average_true_range, rolling_return_volatility
from market.market_calendar import NewsWindow, TradingCalendar, now_ist
from market.yfinance_fetcher import get_history
from news.article_extractor import ArticleExtractor
from news.deduplicator import dedupe_articles
from news.google_news import GoogleNewsScraper
from nlp.adaptive_threshold import DEFAULT_V_REF, ThresholdParams
from prediction.feature_engineering import ticker_code_map
from prediction.history import PredictionHistory
from prediction.model_registry import ModelRunner
from prediction.postprocessing import sanitize_prediction, to_prediction_row
from utils.config_loader import get_config
from utils.constants import ARTICLE_COLUMNS, DAILY_COLUMNS, OHLCV_COLUMNS
from utils.logger import get_logger

logger = get_logger("scheduler.tasks")


@dataclass
class RunStats:
    reference_date: date
    ran: bool = True
    companies_processed: int = 0
    articles_collected: int = 0
    daily_rows_written: int = 0
    predictions_written: int = 0
    company_errors: dict[str, str] = field(default_factory=dict)


def _empty_ohlcv() -> dict:
    return {col: None for col in OHLCV_COLUMNS}


def _last_ohlcv_row(history: pd.DataFrame) -> dict:
    if history is None or history.empty:
        return _empty_ohlcv()
    last_row = history.iloc[-1]
    return {col: last_row.get(col) for col in OHLCV_COLUMNS}


def _latest_volatility(history: pd.DataFrame, window: int, measure: str) -> float | None:
    if history is None or history.empty or "Close" not in history:
        return None
    if measure == "atr" and {"High", "Low", "Close"}.issubset(history.columns):
        series = average_true_range(history["High"], history["Low"], history["Close"])
    else:
        series = rolling_return_volatility(history["Close"], window=window)
    series = series.dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


class Pipeline:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.config.ensure_dirs()
        self.calendar = TradingCalendar(self.config)
        self.scraper = GoogleNewsScraper(self.config)
        self.extractor = ArticleExtractor(self.config.get("news.request_timeout_seconds", 15))
        self.article_dataset_builder = ArticleDatasetBuilder(self.config)
        self.exporter = DatasetExporter(self.config)
        self.prediction_history = PredictionHistory(self.exporter)
        self.model_runner = ModelRunner(self.config)
        self._ticker_codes = ticker_code_map(self._companies())

    def _companies(self) -> list[dict]:
        return self.config.get("companies.companies") or []

    def _load_daily_history(self, ticker: str) -> pd.DataFrame:
        """Every previously-persisted daily row for this ticker (from
        dataset_daily.csv), sorted ascending by News_Date -- the history the
        multi-model registry needs for rolling/sequence features.
        """
        daily_path = self.config.output_file("daily_file")
        if not daily_path.exists():
            return pd.DataFrame(columns=DAILY_COLUMNS)
        full_history = pd.read_csv(daily_path)
        ticker_history = full_history[full_history["Ticker"] == ticker]
        return ticker_history.sort_values("News_Date").reset_index(drop=True)

    def _fetch_histories(self, companies: list[dict]) -> dict[str, pd.DataFrame]:
        histories: dict[str, pd.DataFrame] = {}
        for company in companies:
            ticker = company["ticker"]
            try:
                histories[ticker] = get_history(ticker)
            except Exception:  # noqa: BLE001 - isolate one ticker's market-data failure
                logger.exception("Failed to fetch OHLCV history for %s", ticker)
                histories[ticker] = pd.DataFrame(columns=OHLCV_COLUMNS)
        return histories

    def _reference_volatility(self, histories: dict[str, pd.DataFrame]) -> float:
        window = self.config.get("sentiment.rolling_window", 90)
        measure = self.config.get("sentiment.measure", "vol")
        values = [
            v
            for v in (_latest_volatility(history, window, measure) for history in histories.values())
            if v is not None
        ]
        if not values:
            return DEFAULT_V_REF
        return float(np.median(values))

    def _threshold_params(self, v_ref: float) -> ThresholdParams:
        return ThresholdParams(
            base=self.config.get("sentiment.base_threshold", 0.15),
            k=self.config.get("sentiment.k", 0.8),
            min_threshold=self.config.get("sentiment.min_threshold", 0.08),
            max_threshold=self.config.get("sentiment.max_threshold", 0.45),
            v_ref=v_ref if v_ref and v_ref > 0 else DEFAULT_V_REF,
        )

    def _process_company(self, company: dict, window: NewsWindow, history: pd.DataFrame) -> dict:
        ticker = company["ticker"]
        try:
            raw_articles = self.scraper.fetch_for_company(company, window)
            raw_articles = dedupe_articles(raw_articles)

            for article in raw_articles:
                extracted = self.extractor.extract(article.get("URL", ""), fallback_summary=article.get("Summary", ""))
                article["Summary"] = extracted.summary or extracted.text

            article_df = self.article_dataset_builder.build(raw_articles, company, window)
            ohlcv = _last_ohlcv_row(history)
            window_setting = self.config.get("sentiment.rolling_window", 90)
            measure = self.config.get("sentiment.measure", "vol")
            volatility = _latest_volatility(history, window_setting, measure)

            return {
                "ticker": ticker,
                "article_df": article_df,
                "ohlcv": ohlcv,
                "volatility": volatility,
                "company": company,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - per-company isolation, never abort the run
            logger.exception("Processing failed for %s", ticker)
            return {
                "ticker": ticker,
                "article_df": pd.DataFrame(columns=ARTICLE_COLUMNS),
                "ohlcv": _empty_ohlcv(),
                "volatility": None,
                "company": company,
                "error": str(exc),
            }

    def run(self, force_date: date | None = None) -> RunStats:
        reference_date = force_date or now_ist().date()

        if not self.calendar.should_run(reference_date):
            logger.info("Skipping run for %s: not an NSE trading day", reference_date)
            return RunStats(reference_date=reference_date, ran=False)

        window = self.calendar.news_window(reference_date)
        companies = self._companies()

        histories = self._fetch_histories(companies)
        v_ref = self._reference_volatility(histories)
        threshold_params = self._threshold_params(v_ref)

        concurrency = self.config.get("news.concurrency", 5)
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_company = {
                executor.submit(self._process_company, company, window, histories.get(company["ticker"])): company
                for company in companies
            }
            for future in as_completed(future_to_company):
                company = future_to_company[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - defensive: isolate unexpected thread errors
                    logger.exception("Unexpected top-level failure for %s", company["ticker"])
                    results.append(
                        {
                            "ticker": company["ticker"],
                            "article_df": pd.DataFrame(columns=ARTICLE_COLUMNS),
                            "ohlcv": _empty_ohlcv(),
                            "volatility": None,
                            "company": company,
                            "error": str(exc),
                        }
                    )

        article_frames: list[pd.DataFrame] = []
        daily_rows: list[dict] = []
        prediction_rows: list[dict] = []
        company_errors: dict[str, str] = {}

        for result in results:
            ticker = result["ticker"]
            if result["error"]:
                company_errors[ticker] = result["error"]

            article_df = result["article_df"]
            if article_df is not None and not article_df.empty:
                article_frames.append(article_df)

            daily_row = aggregate_company_day(
                article_df,
                ticker=ticker,
                company=result["company"].get("name"),
                news_date=window.reference_date.isoformat(),
                ohlcv=result["ohlcv"],
                volatility=result["volatility"],
                threshold_params=threshold_params,
            )

            problems = validate_daily_row(daily_row)
            if problems:
                logger.warning("Skipping daily row for %s (validation failed): %s", ticker, problems)
                continue
            daily_rows.append(daily_row)

            try:
                prior_history = self._load_daily_history(ticker)
                combined_history = pd.concat(
                    [prior_history, pd.DataFrame([daily_row])], ignore_index=True
                )
                model_predictions = self.model_runner.predict_all(combined_history, ticker, self._ticker_codes)
                for model_prediction in model_predictions:
                    sanitized = sanitize_prediction(model_prediction, daily_row.get("Close"))
                    prediction_rows.append(
                        to_prediction_row(
                            sanitized,
                            company=daily_row.get("Company"),
                            news_date=daily_row.get("News_Date"),
                            last_close=daily_row.get("Close"),
                            predicted_at=now_ist(),
                        )
                    )
                if not model_predictions:
                    logger.warning("No model produced a prediction for %s", ticker)
            except Exception as exc:  # noqa: BLE001 - a bad model/features shouldn't kill the run
                logger.exception("Prediction failed for %s", ticker)
                company_errors[ticker] = f"prediction failed: {exc}"

        if article_frames:
            self.exporter.export_articles(pd.concat(article_frames, ignore_index=True))
        if daily_rows:
            self.exporter.export_daily(pd.DataFrame(daily_rows, columns=DAILY_COLUMNS))
        if prediction_rows:
            self.prediction_history.append(prediction_rows)

        stats = RunStats(
            reference_date=reference_date,
            ran=True,
            companies_processed=len(companies),
            articles_collected=sum(len(df) for df in article_frames),
            daily_rows_written=len(daily_rows),
            predictions_written=len(prediction_rows),
            company_errors=company_errors,
        )
        logger.info(
            "Run complete for %s: %d companies, %d articles, %d daily rows, %d predictions, %d errors",
            reference_date,
            stats.companies_processed,
            stats.articles_collected,
            stats.daily_rows_written,
            stats.predictions_written,
            len(stats.company_errors),
        )
        return stats


def run_pipeline(force_date: date | None = None) -> RunStats:
    return Pipeline().run(force_date=force_date)
