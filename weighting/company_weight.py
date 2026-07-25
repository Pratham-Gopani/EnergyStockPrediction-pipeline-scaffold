"""Optional per-company importance multiplier for the news-impact weight. Defaults
to 1.0 uniformly (i.e. disabled) unless a company config entry sets an explicit
`importance` value and weighting.use_company_weight is enabled in settings.yaml.
"""

from __future__ import annotations

from utils.config_loader import get_config

DEFAULT_IMPORTANCE = 1.0


class CompanyImportance:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.enabled = bool(self.config.get("weighting.use_company_weight", False))
        companies = self.config.get("companies.companies") or []
        self._importance: dict[str, float] = {
            company["ticker"]: float(company.get("importance", DEFAULT_IMPORTANCE)) for company in companies
        }

    def score(self, ticker: str) -> float:
        if not self.enabled:
            return DEFAULT_IMPORTANCE
        return self._importance.get(ticker, DEFAULT_IMPORTANCE)
