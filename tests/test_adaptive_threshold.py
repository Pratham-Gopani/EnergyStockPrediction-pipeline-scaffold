"""Tests for nlp.adaptive_threshold: threshold must scale up with volatility, and
the same mid-range score should be labeled differently depending on how volatile
the stock is.
"""

from __future__ import annotations

from nlp.adaptive_threshold import ThresholdParams, adaptive_threshold, label_from_score

PARAMS = ThresholdParams(base=0.15, k=0.8, min_threshold=0.08, max_threshold=0.45, v_ref=0.02)


def test_higher_volatility_yields_higher_threshold():
    calm_threshold = adaptive_threshold(volatility=0.01, params=PARAMS)  # r < 1
    average_threshold = adaptive_threshold(volatility=0.02, params=PARAMS)  # r == 1
    volatile_threshold = adaptive_threshold(volatility=0.06, params=PARAMS)  # r > 1

    assert calm_threshold < average_threshold < volatile_threshold


def test_threshold_is_clamped_to_bounds():
    extremely_calm = adaptive_threshold(volatility=0.0001, params=PARAMS)
    extremely_volatile = adaptive_threshold(volatility=10.0, params=PARAMS)

    assert extremely_calm >= PARAMS.min_threshold
    assert extremely_volatile <= PARAMS.max_threshold


def test_missing_volatility_falls_back_to_base_threshold():
    threshold = adaptive_threshold(volatility=None, params=PARAMS)
    expected = PARAMS.base * (1 + PARAMS.k * (1.0 - 1))  # r defaults to 1
    assert threshold == expected


def test_mid_score_labels_calm_stock_positive_but_volatile_stock_neutral():
    mid_score = 0.20

    calm_threshold = adaptive_threshold(volatility=0.01, params=PARAMS)
    volatile_threshold = adaptive_threshold(volatility=0.08, params=PARAMS)

    assert label_from_score(mid_score, calm_threshold) == "Positive"
    assert label_from_score(mid_score, volatile_threshold) == "Neutral"


def test_label_from_score_negative_and_neutral_boundaries():
    threshold = 0.15
    assert label_from_score(0.20, threshold) == "Positive"
    assert label_from_score(-0.20, threshold) == "Negative"
    assert label_from_score(0.10, threshold) == "Neutral"
    assert label_from_score(0.15, threshold) == "Neutral"  # exactly at threshold -> not strictly greater
