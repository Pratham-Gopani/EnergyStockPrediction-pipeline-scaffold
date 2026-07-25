"""Adaptive sentiment-label threshold.

Rather than a single fixed cutoff, the label threshold scales with the stock's own
rolling volatility relative to the universe median, so a calm stock earns a
directional label at a smaller sentiment score than a volatile one.

Formula::

    r         = v / v_ref            # v = rolling 90d return-std (or ATR/Close);
                                      # v_ref = universe median, else 0.02 if unknown
    threshold = clamp( base * (1 + k * (r - 1)),  min_threshold,  max_threshold )

    label:  score >  +threshold -> Positive
            score < -threshold -> Negative
            else               -> Neutral

Defaults: base=0.15, k=0.8, min_threshold=0.08, max_threshold=0.45.

A more volatile stock (r > 1) needs a *stronger* sentiment score to earn a
directional label, since its price noise floor is higher; a calmer stock (r < 1)
earns a label at a lower score. The same threshold function is reused for both the
per-article sentiment score and the weighted daily-aggregated score -- there is only
one threshold formula in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
)
from utils.helpers import clamp

DEFAULT_V_REF = 0.02


@dataclass(frozen=True)
class ThresholdParams:
    base: float = 0.15
    k: float = 0.8
    min_threshold: float = 0.08
    max_threshold: float = 0.45
    v_ref: float = DEFAULT_V_REF


def adaptive_threshold(volatility: float | None, params: ThresholdParams) -> float:
    """Compute the volatility-scaled threshold for a single stock.

    `volatility` is the stock's own rolling return-volatility (or ATR/Close); if it
    is None/NaN/<=0 the ratio r defaults to 1 (i.e. the stock is treated as
    average-volatility and the plain `base` threshold applies).
    """
    v_ref = params.v_ref if params.v_ref and params.v_ref > 0 else DEFAULT_V_REF
    if volatility is None or volatility != volatility or volatility <= 0:  # NaN-safe check
        r = 1.0
    else:
        r = volatility / v_ref
    raw_threshold = params.base * (1 + params.k * (r - 1))
    return clamp(raw_threshold, params.min_threshold, params.max_threshold)


def label_from_score(score: float, threshold: float) -> str:
    """Map a sentiment score in [-1, 1] to Positive/Negative/Neutral given a
    (possibly volatility-adjusted) threshold.
    """
    if score > threshold:
        return SENTIMENT_LABEL_POSITIVE
    if score < -threshold:
        return SENTIMENT_LABEL_NEGATIVE
    return SENTIMENT_LABEL_NEUTRAL
