"""Optional per-topic importance multiplier for the news-impact weight, combined
across an article's topic-probability distribution (probability-weighted, since an
article is rarely 100% one topic). Defaults to uniform 1.0 (i.e. disabled) unless
weighting.use_topic_weight is enabled in settings.yaml.
"""

from __future__ import annotations

from utils.config_loader import get_config
from utils.constants import TOPIC_COLUMNS

DEFAULT_MULTIPLIER = 1.0


class TopicImportance:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.enabled = bool(self.config.get("weighting.use_topic_weight", False))
        multipliers = self.config.get("topic_weights.multipliers") or {}
        self._multipliers: dict[str, float] = {
            col: float(multipliers.get(col, DEFAULT_MULTIPLIER)) for col in TOPIC_COLUMNS
        }

    def score(self, topic_probs: dict[str, float]) -> float:
        """Probability-weighted blend of each topic's multiplier."""
        if not self.enabled or not topic_probs:
            return DEFAULT_MULTIPLIER
        return sum(topic_probs.get(col, 0.0) * self._multipliers[col] for col in TOPIC_COLUMNS)
