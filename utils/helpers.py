"""Small stateless helper functions shared across modules: text normalization,
URL canonicalization, hashing, numeric clamping/coercion, and softmax.
"""

from __future__ import annotations

import hashlib
import math
import re
from urllib.parse import urlsplit, urlunsplit

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

_TRACKING_PARAM_PREFIXES = ("utm_", "fbclid", "gclid", "ref", "ocid")


def normalize_headline(headline: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace so near-duplicate
    headlines (different case/punctuation) compare equal for dedup purposes.
    """
    if not headline:
        return ""
    text = headline.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def canonical_url(url: str) -> str:
    """Strip scheme, query-string tracking params, and trailing slash so the same
    article reached via different tracking links dedupes to one URL.
    """
    if not url:
        return ""
    parts = urlsplit(url.strip())
    query_pairs = [
        pair
        for pair in parts.query.split("&")
        if pair and not pair.split("=")[0].lower().startswith(_TRACKING_PARAM_PREFIXES)
    ]
    path = parts.path.rstrip("/")
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    rebuilt = urlunsplit(("", netloc, path, "&".join(query_pairs), ""))
    return rebuilt.lstrip("/")


def stable_hash(text: str) -> str:
    """Deterministic hash (sha256 hex digest) for cache keys, stable across processes
    and Python versions (unlike the builtin hash()).
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def clamp(value: float, lo: float, hi: float) -> float:
    """Clip value into [lo, hi]."""
    return max(lo, min(hi, value))


def safe_float(value, default: float = 0.0) -> float:
    """Coerce value to float, falling back to `default` on None/NaN/ValueError."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax over a list of raw scores."""
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    if total == 0:
        uniform = 1.0 / len(values)
        return [uniform] * len(values)
    return [e / total for e in exps]
