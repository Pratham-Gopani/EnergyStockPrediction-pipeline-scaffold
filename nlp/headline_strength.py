"""Headline "strength" scoring: how much intrinsic market-moving weight a headline
carries, independent of its sentiment polarity. Primarily a deterministic set of
regex anchors tuned to the spec's example headlines; optionally blended with a
zero-shot classifier score for headlines that don't match any anchor.

CRITICAL: anchor patterns must match inflected verb forms. `resign\\b` does NOT match
"resigns" or "resigned" (the word boundary sits right after "resign", so it only
matches the bare infinitive). Use `resign\\w*`, `quit\\w*`, `exits?`, etc.
"""

from __future__ import annotations

import re

from utils.helpers import clamp
from utils.logger import get_logger

logger = get_logger("nlp.headline_strength")

FLOOR_SCORE = 0.30

# Ordered by score descending purely for readability; matching takes the MAX score
# across all patterns that hit, not the first one.
_ANCHORS: list[tuple[re.Pattern, float]] = [
    # CEO / leadership resignation, ouster, or exit.
    (
        re.compile(
            r"\b(ceo|cfo|coo|cmd|chairman|md|managing director)\b[^.]{0,40}"
            r"\b(resign\w*|quit\w*|steps?\s+down|exits?|ousted|removed|sack\w*)\b",
            re.IGNORECASE,
        ),
        0.98,
    ),
    (
        re.compile(r"\b(resign\w*|quit\w*|steps?\s+down)\b[^.]{0,40}\b(ceo|cfo|coo|chairman|md)\b", re.IGNORECASE),
        0.98,
    ),
    # Major order / contract win.
    (
        re.compile(
            r"\b(wins?|bags?|secures?|clinches?|awarded)\b[^.]{0,40}"
            r"\b(order|contract|deal|project)\b",
            re.IGNORECASE,
        ),
        0.96,
    ),
    (re.compile(r"\bmega\s*deal\b", re.IGNORECASE), 0.96),
    # Quarterly / annual earnings.
    (
        re.compile(
            r"\b(q[1-4]\s*(fy)?\d{0,4}|quarterly)\b[^.]{0,40}\b(results?|earnings?|profit|revenue)\b",
            re.IGNORECASE,
        ),
        0.93,
    ),
    (re.compile(r"\bnet\s+profit\b", re.IGNORECASE), 0.93),
    (re.compile(r"\bannual\s+results?\b", re.IGNORECASE), 0.93),
    # Dividend announcements.
    (re.compile(r"\bdividend\w*\b", re.IGNORECASE), 0.75),
    (re.compile(r"\bbuyback\b", re.IGNORECASE), 0.72),
    # Regulatory / legal action.
    (re.compile(r"\b(probe|investigat\w*|penalt\w*|fined?|raid\w*)\b", re.IGNORECASE), 0.85),
    # Credit rating actions.
    (re.compile(r"\b(rating\s+(upgrad\w*|downgrad\w*)|rated?\s+(aa|a|bbb)\+?)\b", re.IGNORECASE), 0.80),
]


def headline_strength(headline: str) -> float:
    """Deterministic anchor score in [FLOOR_SCORE, 1.0]. Unmatched headlines get
    the FLOOR_SCORE (general commentary).
    """
    if not headline:
        return FLOOR_SCORE
    best_score = FLOOR_SCORE
    for pattern, score in _ANCHORS:
        if score > best_score and pattern.search(headline):
            best_score = score
    return best_score


def _zero_shot_strength(headline: str) -> float | None:
    """Optional zero-shot blend signal: classifies the headline against a
    high-impact vs. routine label pair using the shared zero-shot pipeline. Returns
    None (skip blending) if the classifier is unavailable.
    """
    try:
        from nlp.topic_classifier import run_zero_shot  # lazy: heavy dependency

        result = run_zero_shot(
            headline,
            ["major corporate event", "routine news commentary"],
        )
        labels = result["labels"]
        scores = result["scores"]
        major_idx = labels.index("major corporate event")
        return float(scores[major_idx])
    except Exception:  # noqa: BLE001 - classifier unavailable/misconfigured -> skip blend
        logger.debug("Zero-shot headline-strength blend unavailable; using anchors only")
        return None


def headline_strength_blended(headline: str, use_zero_shot: bool = False, blend_weight: float = 0.3) -> float:
    """Deterministic anchor score, optionally blended with a zero-shot signal.

    `blend_weight` is the weight given to the zero-shot score when it's available;
    the anchor score always gets `1 - blend_weight`. If the zero-shot pass fails or
    is disabled, the plain anchor score is returned unchanged.
    """
    anchor_score = headline_strength(headline)
    if not use_zero_shot:
        return anchor_score
    zero_shot_score = _zero_shot_strength(headline)
    if zero_shot_score is None:
        return anchor_score
    blended = (1 - blend_weight) * anchor_score + blend_weight * zero_shot_score
    return clamp(blended, FLOOR_SCORE, 1.0)
