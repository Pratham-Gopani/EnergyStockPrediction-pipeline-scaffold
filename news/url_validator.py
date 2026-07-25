"""URL validation and best-effort publisher-name inference from a URL's domain."""

from __future__ import annotations

from urllib.parse import urlparse

from utils.config_loader import get_config

_KNOWN_DOMAIN_TO_PUBLISHER = {
    "moneycontrol.com": "Moneycontrol",
    "economictimes.indiatimes.com": "The Economic Times",
    "business-standard.com": "Business Standard",
    "livemint.com": "Mint",
    "financialexpress.com": "Financial Express",
    "businesstoday.in": "Business Today",
    "cnbctv18.com": "CNBC-TV18",
    "thehindubusinessline.com": "The Hindu BusinessLine",
    "hindustantimes.com": "Hindustan Times",
    "timesofindia.indiatimes.com": "Times of India",
    "indiatoday.in": "India Today",
    "ndtv.com": "NDTV",
    "indianexpress.com": "The Indian Express",
    "zeebiz.com": "Zee Business",
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "ptinews.com": "Press Trust of India",
    "aninews.in": "ANI",
}


def is_valid_url(url: str) -> bool:
    if not url:
        return False
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def publisher_from_url(url: str) -> str:
    """Best-effort publisher name from the URL's domain: a known-domain lookup
    first, then the bare registrable-ish domain as a fallback, or "" if invalid.
    """
    if not is_valid_url(url):
        return ""
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    for domain, publisher in _KNOWN_DOMAIN_TO_PUBLISHER.items():
        if netloc == domain or netloc.endswith(f".{domain}"):
            return publisher

    config = get_config()
    verified_domains = (config.get("source_weights.verified_domains") or [])
    for domain in verified_domains:
        if netloc == domain or netloc.endswith(f".{domain}"):
            return netloc

    return netloc
