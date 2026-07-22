"""Deduplicate articles by canonical URL and by normalized headline; the first
occurrence (in input order) of either key wins.
"""

from __future__ import annotations

from utils.helpers import canonical_url, normalize_headline


def dedupe_articles(articles: list[dict]) -> list[dict]:
    """`articles` are dicts with at least "url" and "headline" keys. Returns a new
    list preserving input order, dropping later duplicates by canonical URL or by
    normalized headline.
    """
    seen_urls: set[str] = set()
    seen_headlines: set[str] = set()
    result: list[dict] = []

    for article in articles:
        url_key = canonical_url(article.get("url", ""))
        headline_key = normalize_headline(article.get("headline", ""))

        if url_key and url_key in seen_urls:
            continue
        if headline_key and headline_key in seen_headlines:
            continue

        if url_key:
            seen_urls.add(url_key)
        if headline_key:
            seen_headlines.add(headline_key)
        result.append(article)

    return result
