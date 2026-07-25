"""Tests for news.deduplicator: dedupe by canonical URL and by normalized
headline, keeping the first occurrence.
"""

from __future__ import annotations

from news.deduplicator import dedupe_articles


def test_dedupes_by_canonical_url():
    articles = [
        {"url": "https://www.example.com/story?utm_source=twitter", "headline": "Story One"},
        {"url": "http://example.com/story/", "headline": "Story One Reshared"},
    ]
    result = dedupe_articles(articles)
    assert len(result) == 1
    assert result[0]["headline"] == "Story One"


def test_dedupes_by_normalized_headline():
    articles = [
        {"url": "https://a.example.com/1", "headline": "Company Wins Big Order!"},
        {"url": "https://b.example.com/2", "headline": "company wins big order"},
    ]
    result = dedupe_articles(articles)
    assert len(result) == 1


def test_distinct_articles_are_kept():
    articles = [
        {"url": "https://a.example.com/1", "headline": "Headline One"},
        {"url": "https://b.example.com/2", "headline": "Headline Two"},
    ]
    result = dedupe_articles(articles)
    assert len(result) == 2


def test_empty_input():
    assert dedupe_articles([]) == []
