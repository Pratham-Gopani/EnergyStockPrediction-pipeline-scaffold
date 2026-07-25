"""Full-text article extraction via newspaper3k (lazy-imported), with graceful
degradation: if download/parse fails, or the extracted body is too short to be
useful, we fall back to whatever summary/description text the news source
(gnews) already gave us.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from utils.logger import get_logger
from utils.retry import retry_from_config

logger = get_logger("news.article_extractor")

_WHITESPACE_RE = re.compile(r"\s+")
MIN_USEFUL_LENGTH = 40


@dataclass(frozen=True)
class ExtractedArticle:
    text: str
    summary: str
    used_fallback: bool


def clean_text(text: str) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text.strip())


class ArticleExtractor:
    def __init__(self, request_timeout_seconds: float = 15.0):
        self._timeout = request_timeout_seconds

    @retry_from_config()
    def _download_and_parse(self, url: str):
        from newspaper import Article  # lazy: heavy dependency

        article = Article(url)
        article.download()
        article.parse()
        return article

    def extract(self, url: str, fallback_summary: str = "") -> ExtractedArticle:
        """Try newspaper3k extraction; fall back to `fallback_summary` (typically
        the snippet/description gnews already returned) if extraction fails or
        yields too little text.
        """
        fallback_clean = clean_text(fallback_summary)
        try:
            article = self._download_and_parse(url)
            text = clean_text(article.text)
            if len(text) < MIN_USEFUL_LENGTH:
                raise ValueError("extracted text too short")
            summary = clean_text(getattr(article, "summary", "") or "") or fallback_clean
            return ExtractedArticle(text=text, summary=summary, used_fallback=False)
        except Exception as exc:  # noqa: BLE001 - any extraction failure -> fallback
            logger.info("Falling back to summary text for %s (%s)", url, exc)
            return ExtractedArticle(
                text=fallback_clean, summary=fallback_clean, used_fallback=True
            )
