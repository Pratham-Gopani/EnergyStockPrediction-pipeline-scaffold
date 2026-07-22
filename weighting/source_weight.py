"""Source reliability lookup. Matching is by longest-substring match of the
publisher/domain string against the configured reliability table (case-insensitive):
the longest key that appears as a substring of the source string wins, so
"The Economic Times Online" still matches "The Economic Times". A named-but-unlisted
publisher falls back to `default_verified` (0.80); a missing/blank source falls back
to `unknown` (0.65).
"""

from __future__ import annotations

from utils.config_loader import get_config


class SourceReliability:
    def __init__(self, config=None):
        self.config = config or get_config()
        source_weights = self.config.get("source_weights") or {}
        self._table: dict[str, float] = {k.lower(): float(v) for k, v in (source_weights.get("reliability") or {}).items()}
        self._default_verified = float(source_weights.get("default_verified", 0.80))
        self._unknown = float(source_weights.get("unknown", 0.65))
    def score(self, source: str | None) -> float:
        if not source:
            return self._unknown
        source_lower = source.strip().lower()
        if not source_lower:
            return self._unknown

        best_match_key: str | None = None
        for key in self._table:
            if key in source_lower and (best_match_key is None or len(key) > len(best_match_key)):
                best_match_key = key
        if best_match_key is not None:
            return self._table[best_match_key]

        # Named but not in the reliability table -> treat as a verified-but-unrated source.
        return self._default_verified
