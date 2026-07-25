"""Recency weighting: exponential decay with a 24-hour half-life-ish scale, so an
article published right before the prediction cutoff outweighs one from the start
of the news window.
"""

from __future__ import annotations

import math
from datetime import datetime


def recency_weight(hours_before_prediction: float) -> float:
    """recency_weight = exp(-hours_before_prediction / 24). Negative input (an
    article timestamped after the cutoff, e.g. due to clock skew) is clamped to 0.
    """
    hours = max(0.0, hours_before_prediction)
    return math.exp(-hours / 24.0)


def recency_weight_from_timestamps(published_at: datetime, prediction_cutoff: datetime) -> float:
    delta = prediction_cutoff - published_at
    hours_before = delta.total_seconds() / 3600.0
    return recency_weight(hours_before)
