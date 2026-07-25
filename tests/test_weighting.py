"""Tests for the weighting/ pure functions: source reliability longest-match
tiers, recency monotonic decay, weight normalization, and the impact-score
product.
"""

from __future__ import annotations

from weighting.impact_score import ImpactFactors
from weighting.normalize import normalize_weights
from weighting.recency_weight import recency_weight
from weighting.source_weight import SourceReliability


class FakeConfig:
    def __init__(self, data: dict):
        self._data = data

    def get(self, dotted_key: str, default=None):
        node = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def _source_reliability() -> SourceReliability:
    config = FakeConfig(
        {
            "source_weights": {
                "reliability": {
                    "Reuters": 1.00,
                    "The Economic Times": 0.95,
                },
                "default_verified": 0.80,
                "unknown": 0.65,
                "verified_domains": ["economictimes.indiatimes.com"],
            }
        }
    )
    return SourceReliability(config=config)


def test_source_reliability_exact_and_longest_match():
    reliability = _source_reliability()

    assert reliability.score("Reuters") == 1.00
    # Longer, more specific string still contains the shorter key -- longest match wins.
    assert reliability.score("The Economic Times Online") == 0.95


def test_source_reliability_named_but_unlisted_is_default_verified():
    reliability = _source_reliability()
    assert reliability.score("Some Random Financial Blog") == 0.80


def test_source_reliability_missing_is_unknown():
    reliability = _source_reliability()
    assert reliability.score(None) == 0.65
    assert reliability.score("") == 0.65


def test_recency_weight_monotonically_decreases_with_age():
    fresh = recency_weight(0)
    one_day_old = recency_weight(24)
    two_days_old = recency_weight(48)

    assert fresh == 1.0
    assert fresh > one_day_old > two_days_old


def test_recency_weight_clamps_negative_hours_to_full_weight():
    assert recency_weight(-5) == 1.0


def test_normalize_weights_sums_to_one():
    weights = [0.2, 0.3, 0.5]
    normalized = normalize_weights(weights)
    assert abs(sum(normalized) - 1.0) < 1e-9


def test_normalize_weights_all_zero_falls_back_to_uniform():
    normalized = normalize_weights([0.0, 0.0, 0.0, 0.0])
    assert abs(sum(normalized) - 1.0) < 1e-9
    assert all(abs(w - 0.25) < 1e-9 for w in normalized)


def test_normalize_weights_empty_list():
    assert normalize_weights([]) == []


def test_impact_score_is_product_of_factors():
    factors = ImpactFactors(
        source_reliability=0.9,
        recency_weight=0.5,
        article_relevance=0.8,
        headline_strength=0.7,
    )
    expected = 0.9 * 0.5 * 0.8 * 0.7
    assert abs(factors.raw_weight() - expected) < 1e-9


def test_impact_score_includes_optional_factors():
    factors = ImpactFactors(
        source_reliability=0.9,
        recency_weight=0.5,
        article_relevance=0.8,
        headline_strength=0.7,
        company_weight=1.2,
        topic_weight=0.9,
    )
    expected = 0.9 * 0.5 * 0.8 * 0.7 * 1.2 * 0.9
    assert abs(factors.raw_weight() - expected) < 1e-9
