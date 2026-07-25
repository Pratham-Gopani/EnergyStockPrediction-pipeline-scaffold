"""Normalize a list of raw weights to sum to 1. An all-zero (or empty) input falls
back to a uniform distribution rather than dividing by zero.
"""

from __future__ import annotations


def normalize_weights(weights: list[float]) -> list[float]:
    if not weights:
        return []
    total = sum(weights)
    if total <= 0:
        uniform = 1.0 / len(weights)
        return [uniform] * len(weights)
    return [w / total for w in weights]
