"""Composite per-article news-impact weight:

    raw_weight = SourceReliability * RecencyWeight * ArticleRelevance * HeadlineStrength
                 [* CompanyImportance] [* TopicImportance]

The two bracketed factors are optional (default 1.0, i.e. no-ops) and controlled by
weighting.use_company_weight / weighting.use_topic_weight. The caller is responsible
for normalizing `raw_weight` across all of a company/day's articles via
weighting.normalize.normalize_weights -- this module only computes the per-article
raw product, it never normalizes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImpactFactors:
    source_reliability: float
    recency_weight: float
    article_relevance: float
    headline_strength: float
    company_weight: float = 1.0
    topic_weight: float = 1.0

    def raw_weight(self) -> float:
        return (
            self.source_reliability
            * self.recency_weight
            * self.article_relevance
            * self.headline_strength
            * self.company_weight
            * self.topic_weight
        )
