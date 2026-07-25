"""Word-boundary matching of a company's name/aliases against article text, used
both for building per-company search queries and for re-verifying (post-fetch)
that a returned article genuinely mentions the company, since gnews' own query
matching is not perfectly precise.
"""

from __future__ import annotations

import re


def _name_pattern(name: str) -> re.Pattern:
    escaped = re.escape(name.strip())
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def mentions_company(text: str, company: dict) -> bool:
    """True if `text` contains a word-boundary match of the company's name or any
    of its configured aliases.
    """
    if not text:
        return False
    candidates = [company.get("name", "")] + list(company.get("aliases", []) or [])
    for candidate in candidates:
        if not candidate:
            continue
        if _name_pattern(candidate).search(text):
            return True
    return False
