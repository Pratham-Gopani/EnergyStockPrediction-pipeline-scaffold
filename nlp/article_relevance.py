"""Scores how relevant an article's text is to a specific company. Primary signal
is cosine similarity between the article text embedding and a "company profile"
embedding (name + aliases + keywords), rescaled from [-1, 1] to [0, 1]. Falls back
to a lexical Jaccard-overlap score if embedding fails (e.g. model unavailable).
"""

from __future__ import annotations

from nlp.embedding_cache import EmbeddingProvider, cosine_similarity
from utils.helpers import normalize_headline
from utils.logger import get_logger

logger = get_logger("nlp.article_relevance")


def company_profile_text(company: dict) -> str:
    """Build a short descriptive text for a company from its config entry, used as
    the reference text for relevance embedding.
    """
    parts = [company.get("name", "")]
    parts.extend(company.get("aliases", []) or [])
    parts.extend(company.get("keywords", []) or [])
    sector = company.get("sector")
    if sector:
        parts.append(sector)
    return ". ".join(p for p in parts if p)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(normalize_headline(text_a).split())
    tokens_b = set(normalize_headline(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


class RelevanceScorer:
    def __init__(self, embedding_provider: EmbeddingProvider | None = None):
        self._embeddings = embedding_provider or EmbeddingProvider()

    def score(self, article_text: str, company: dict) -> float:
        """Return a relevance score in [0, 1]."""
        profile_text = company_profile_text(company)
        if not article_text or not profile_text:
            return 0.0
        try:
            article_vec = self._embeddings.embed(article_text)
            profile_vec = self._embeddings.embed(profile_text)
            if not article_vec or not profile_vec:
                raise ValueError("empty embedding")
            cosine = cosine_similarity(article_vec, profile_vec)
            return (cosine + 1.0) / 2.0
        except Exception:  # noqa: BLE001 - any embedding failure -> lexical fallback
            logger.warning("Embedding relevance failed; falling back to lexical Jaccard")
            return _jaccard_similarity(article_text, profile_text)
