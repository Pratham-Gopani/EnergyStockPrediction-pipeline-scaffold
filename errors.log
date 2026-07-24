"""Builds Dataset 1 (one row per article) from a batch of raw article dicts
(as produced by news.google_news + news.article_extractor) for a single
company/day. Orchestrates: FinBERT sentiment, 6-topic classification, article
relevance, headline strength, source reliability, recency weight, and the
composite news-impact weight -- then normalizes impact weights to sum to 1 across
the batch (per company/day, never a simple average).
"""

from __future__ import annotations

import pandas as pd

from market.market_calendar import NewsWindow
from nlp.article_relevance import RelevanceScorer
from nlp.headline_strength import headline_strength_blended
from nlp.sentiment import SentimentAnalyzer
from nlp.topic_classifier import classify as classify_topic
from utils.config_loader import get_config
from utils.constants import ARTICLE_COLUMNS
from utils.logger import get_logger
from weighting.company_weight import CompanyImportance
from weighting.impact_score import ImpactFactors
from weighting.normalize import normalize_weights
from weighting.recency_weight import recency_weight_from_timestamps
from weighting.source_weight import SourceReliability
from weighting.topic_weight import TopicImportance

logger = get_logger("datasets.article_dataset")


class ArticleDatasetBuilder:
    def __init__(self, config=None):
        self.config = config or get_config()
        self._sentiment = SentimentAnalyzer()
        self._relevance = RelevanceScorer()
        self._source_reliability = SourceReliability(self.config)
        self._company_importance = CompanyImportance(self.config)
        self._topic_importance = TopicImportance(self.config)
        self._use_zero_shot_headline = bool(self.config.get("nlp.headline_zero_shot_blend", False))

    @staticmethod
    def _scoring_text(article: dict) -> str:
        headline = article.get("Headline") or ""
        summary = article.get("Summary") or ""
        return f"{headline}. {summary}".strip(". ").strip()

    def build(self, raw_articles: list[dict], company: dict, window: NewsWindow) -> pd.DataFrame:
        if not raw_articles:
            return pd.DataFrame(columns=ARTICLE_COLUMNS)

        texts = [self._scoring_text(article) for article in raw_articles]
        sentiment_results = self._sentiment.analyze(texts)

        rows: list[dict] = []
        raw_weights: list[float] = []

        for article, text, sentiment in zip(raw_articles, texts, sentiment_results):
            topic_result = classify_topic(text)
            topic_probs = {k: v for k, v in topic_result.items() if k != "Predicted_Topic"}

            headline = article.get("Headline") or ""
            relevance = self._relevance.score(text, company)
            strength = headline_strength_blended(headline, use_zero_shot=self._use_zero_shot_headline)
            source_reliability = self._source_reliability.score(article.get("Source"))

            published_at = pd.to_datetime(article.get("Published_At"))
            recency = recency_weight_from_timestamps(published_at.to_pydatetime(), window.end)

            company_weight = self._company_importance.score(company.get("ticker"))
            topic_weight = self._topic_importance.score(topic_probs)

            factors = ImpactFactors(
                source_reliability=source_reliability,
                recency_weight=recency,
                article_relevance=relevance,
                headline_strength=strength,
                company_weight=company_weight,
                topic_weight=topic_weight,
            )
            raw_weight = factors.raw_weight()
            raw_weights.append(raw_weight)

            row = {col: article.get(col) for col in ARTICLE_COLUMNS}
            row.update(
                {
                    "Sentiment_Positive": sentiment.positive,
                    "Sentiment_Neutral": sentiment.neutral,
                    "Sentiment_Negative": sentiment.negative,
                    "Sentiment_Score": sentiment.score,
                    "Sentiment_Label": None,  # filled after adaptive threshold is known (aggregation stage)
                    **topic_probs,
                    "Predicted_Topic": topic_result["Predicted_Topic"],
                    "Article_Relevance": relevance,
                    "Headline_Strength": strength,
                    "Source_Reliability": source_reliability,
                    "Recency_Weight": recency,
                    "Company_Weight": company_weight,
                    "Topic_Weight": topic_weight,
                }
            )
            rows.append(row)

        normalized_weights = normalize_weights(raw_weights)
        for row, weight in zip(rows, normalized_weights):
            row["News_Impact_Weight"] = weight

        return pd.DataFrame(rows, columns=ARTICLE_COLUMNS)
