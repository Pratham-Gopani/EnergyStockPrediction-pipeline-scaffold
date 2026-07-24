"""Google News fetching via gnews, one query per company (name + aliases combined
with OR), run concurrently across companies with a ThreadPoolExecutor.

gnews only filters results by calendar date, not by time-of-day, so a fetch for
"today" can return articles from any hour of the start/end dates. We therefore
enforce the exact hour-level NewsWindow ourselves after fetching, and separately
re-verify that each returned article actually mentions the company (word-boundary
match on name + aliases) since Google News' own relevance matching is fuzzy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.utils import parsedate_to_datetime

from market.market_calendar import IST, NewsWindow
from news.company_matcher import mentions_company
from news.url_validator import publisher_from_url
from utils.config_loader import get_config
from utils.constants import ARTICLE_COLUMNS
from utils.logger import get_logger
from utils.retry import retry_from_config

logger = get_logger("news.google_news")


def _build_query(company: dict) -> str:
    names = [company.get("name", "")] + list(company.get("aliases", []) or [])
    names = [n for n in names if n]
    quoted = [f'"{n}"' for n in names]
    return " OR ".join(quoted)


def _parse_published_at(raw_date: str) -> datetime | None:
    if not raw_date:
        return None
    try:
        parsed = parsedate_to_datetime(raw_date)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone(IST)
    return parsed.astimezone(IST)


def _within_window(published_at: datetime | None, window: NewsWindow) -> bool:
    if published_at is None:
        return False
    return window.start <= published_at <= window.end


def _to_article_row(raw_article: dict, company: dict, window: NewsWindow, published_at: datetime) -> dict:
    publisher = raw_article.get("publisher") or {}
    source = publisher.get("title") or publisher_from_url(raw_article.get("url", ""))
    row = {col: None for col in ARTICLE_COLUMNS}
    row.update(
        {
            "Ticker": company.get("ticker"),
            "Company": company.get("name"),
            "News_Date": window.reference_date.isoformat(),
            "Published_At": published_at.isoformat(),
            "Headline": raw_article.get("title", ""),
            "Summary": raw_article.get("description", ""),
            "URL": raw_article.get("url", ""),
            "Source": source,
        }
    )
    return row


class GoogleNewsScraper:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.max_results = self.config.get("news.max_articles_per_company", 40)
        self.concurrency = self.config.get("news.concurrency", 5)
        self.language = self.config.get("news.language", "en")
        self.country = self.config.get("news.country", "IN")

    @retry_from_config()
    def _fetch_raw(self, query: str, window: NewsWindow) -> list[dict]:
        from gnews import GNews  # lazy: heavy + network dependency

        client = GNews(
            language=self.language,
            country=self.country,
            max_results=self.max_results,
            start_date=window.start.date(),
            end_date=window.end.date(),
        )
        return client.get_news(query) or []

    def fetch_for_company(self, company: dict, window: NewsWindow) -> list[dict]:
        """Fetch, window-filter, and re-verify articles for a single company.
        Returns a list of ARTICLE_COLUMNS-shaped dicts (unscored -- sentiment/topic/
        weighting columns are filled in later by datasets.article_dataset).
        """
        query = _build_query(company)
        if not query:
            return []

        try:
            raw_articles = self._fetch_raw(query, window)
        except Exception:  # noqa: BLE001 - network/lib failure isolated to this company
            logger.exception("Google News fetch failed for %s", company.get("name"))
            return []

        rows: list[dict] = []
        for raw_article in raw_articles:
            published_at = _parse_published_at(raw_article.get("published date", ""))
            if not _within_window(published_at, window):
                continue

            text_to_check = f"{raw_article.get('title', '')} {raw_article.get('description', '')}"
            if not mentions_company(text_to_check, company):
                continue

            rows.append(_to_article_row(raw_article, company, window, published_at))

        return rows

    def fetch_all(self, companies: list[dict], window: NewsWindow) -> dict[str, list[dict]]:
        """Fetch for every company concurrently. Each company's failure is isolated
        (logged, empty result) so one bad query never blocks the others.
        """
        results: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_for_company, company, window): company["ticker"]
                for company in companies
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result()
                except Exception:  # noqa: BLE001 - defensive: isolate unexpected thread errors
                    logger.exception("Unexpected failure fetching news for %s", ticker)
                    results[ticker] = []
        return results
