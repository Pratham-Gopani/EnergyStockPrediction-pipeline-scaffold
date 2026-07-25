"""High-level sentiment scoring on top of nlp.finbert: turns raw pos/neu/neg
probabilities into a single signed score (pos - neg) plus a SentimentResult per
article. Falls back to a flat neutral result if FinBERT scoring raises (e.g. model
download failure), so one bad article never aborts the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

from nlp.finbert import score_texts
from utils.logger import get_logger

logger = get_logger("nlp.sentiment")

NEUTRAL_FALLBACK = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}


@dataclass(frozen=True)
class SentimentResult:
    positive: float
    neutral: float
    negative: float
    score: float  # positive - negative, in [-1, 1]


def _to_result(probs: dict[str, float]) -> SentimentResult:
    positive = probs["positive"]
    neutral = probs["neutral"]
    negative = probs["negative"]
    return SentimentResult(positive=positive, neutral=neutral, negative=negative, score=positive - negative)


class SentimentAnalyzer:
    """Thin batch-scoring wrapper; instantiate once and reuse across a run."""

    def analyze(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []
        try:
            batch_probs = score_texts(texts)
        except Exception:  # noqa: BLE001 - any model/runtime failure -> neutral fallback
            logger.exception("FinBERT scoring failed for a batch of %d texts; falling back to neutral", len(texts))
            batch_probs = [NEUTRAL_FALLBACK] * len(texts)
        return [_to_result(probs) for probs in batch_probs]

    def analyze_one(self, text: str) -> SentimentResult:
        results = self.analyze([text])
        return results[0] if results else _to_result(NEUTRAL_FALLBACK)
