---
name: energy-stock-news-pipeline
description: Build (or extend) a production-grade, modular ML pipeline that predicts Indian energy-sector stock Open/Close prices from Google News sentiment + 6-topic classification + pretrained models. Use this skill whenever the user asks to scaffold, regenerate, or add to a "news-driven stock prediction" / "FinBERT sentiment + topic + OHLC" pipeline, mentions the EnergyStockPrediction project, or references news-impact weighting, adaptive volatility thresholds, article->daily aggregation, or an APScheduler-based daily prediction job. Follow this even if the user only describes the behavior without naming the project.
---

# Energy-Sector News -> OHLC Prediction Pipeline

A blueprint for a fault-tolerant pipeline that runs weekdays at 08:30 IST, scrapes
Google News for 20 Indian energy companies, scores each article (FinBERT sentiment +
6-topic classifier + relevance + headline strength), weights articles by a composite
news-impact weight, aggregates to one row per company/day, runs scaler->PCA->Open/Close
models, validates, and writes append-only CSVs. A monitoring layer scores realised
accuracy later.

## Non-negotiable requirements (from the spec)

- **Config-driven, no hard-coding.** The 20 companies, source weights, topic weights,
  holidays, schedule, and all thresholds live in `config/*.yaml`.
- **News window logic (IST, `Asia/Kolkata`):** normal day = previous trading day
  08:30 -> today 08:29:59; the Monday/post-holiday case is just "previous *trading* day",
  so implement one `previous_trading_day()` and derive the window from it — do NOT
  special-case Monday. No runs on weekends or NSE holidays.
- **Exactly six topic columns**, named verbatim:
  `Prob_Topic_1_Policy_Regulatory`, `Prob_Topic_2_Financial_Performance`,
  `Prob_Topic_3_Investment_FII_DII`, `Prob_Topic_4_Infrastructure_Expansion`,
  `Prob_Topic_5_Stock_Price_Action`, `Prob_Topic_6_Commodity_Macro`. Return all six + argmax.
- **News-impact weight** = SourceReliability x RecencyWeight x ArticleRelevance x
  HeadlineStrength, **normalized to sum to 1 per company/day**. Never simple-average.
  `RecencyWeight = exp(-hours_before_prediction / 24)`.
- **Adaptive sentiment threshold** — no fixed cutoff. Scale the label threshold with the
  stock's own rolling volatility (see formula below). Document it in the module docstring.
- **Aggregation to Dataset 2:** weighted average of sentiment score, 3 sentiment probs,
  and 6 topic probs; `first()` for News_Date + OHLCV; headlines joined with ` || `;
  `Predicted_Topic` = argmax of the **weighted** topic probs (NOT the mode); label from the
  weighted score via the adaptive threshold.
- **Append-only outputs**, never overwrite: `dataset_articles.csv`, `dataset_daily.csv`,
  `prediction_history.csv`. Back up the prior file before each append.
- **Reliability:** per-company try/except isolation (one failure never aborts the run);
  retry with exponential backoff + jitter on network calls; load models once and cache.
- **Two extra layers the user wants:** a pre-inference data-validation layer (missing
  features, ticker validity, prob sums ~= 1, skip+log if invalid) and a post-inference
  monitoring layer (MAE/RMSE/MAPE/R2/Direction Accuracy, persisted, with alerts).

## The adaptive threshold formula (must be documented in code)

```
r         = v / v_ref            # v = rolling 90d return-std (or ATR/Close); v_ref = universe median, else 0.02
threshold = clamp( base * (1 + k * (r - 1)),  min_threshold,  max_threshold )
label:  score >  +threshold -> Positive ;  score < -threshold -> Negative ; else Neutral
```
Defaults: `base=0.15, k=0.8, min=0.08, max=0.45`. More volatile stock (r>1) needs a
*stronger* score to earn a directional label. Reuse the same threshold on the weighted
aggregated score.

## Folder / module contract

Generate exactly this tree (packages each have `__init__.py`):

```
config/    companies.yaml source_weights.yaml topic_weights.yaml settings.yaml scheduler.yaml holidays.yaml
utils/     constants.py config_loader.py logger.py retry.py cache.py helpers.py
market/    market_calendar.py yfinance_fetcher.py indicators.py
nlp/       finbert.py sentiment.py adaptive_threshold.py topic_classifier.py article_relevance.py headline_strength.py embedding_cache.py
weighting/ source_weight.py recency_weight.py company_weight.py topic_weight.py impact_score.py normalize.py
news/      google_news.py article_extractor.py deduplicator.py company_matcher.py url_validator.py
datasets/  article_dataset.py aggregation.py validator.py updater.py exporter.py
prediction/ preprocessing.py inference.py postprocessing.py history.py evaluator.py
scheduler/ scheduler.py tasks.py monitor.py
tests/     conftest.py + one test per pure-logic module
scripts/   generate_dummy_models.py
models/    (empty; user drops scaler.pkl pca.pkl open_model.pkl close_model.pkl metadata.json topic_classifier/)
main.py requirements.txt README.md .env.example pyproject.toml
```

**Canonical schemas live in `utils/constants.py`** (`TOPIC_COLUMNS`, `TOPIC_LABELS`,
`SENTIMENT_COLUMNS`, `OHLCV_COLUMNS`, `ARTICLE_COLUMNS`, `DAILY_COLUMNS`,
`PREDICTION_COLUMNS`). Everything else imports these — never re-list columns inline.

## Build order (dependencies first)

1. `config/*.yaml` — data before code.
2. `utils/` — constants, then config_loader (lru_cache singleton, dotted `.get()`), logger
   (rotating file + console), retry (decorator, backoff+jitter), cache (thread-safe
   ModelRegistry `get_or_create` + JSON disk cache), helpers.
3. `market/market_calendar.py` — IST tz, `NewsWindow` dataclass, `previous_trading_day`,
   `news_window`, `should_run`. Plus `indicators.py` (rolling return-vol, ATR/close) and
   `yfinance_fetcher.py` (lazy import, retry, flatten MultiIndex, rename `Adj Close`).
4. `nlp/` — adaptive_threshold first (pure), then finbert/sentiment/embedding_cache/
   article_relevance/headline_strength/topic_classifier.
5. `weighting/` — all pure functions.
6. `news/` — deduplicator/company_matcher/url_validator (pure) then article_extractor and
   google_news (lazy imports).
7. `datasets/` — updater, validator, article_dataset, aggregation, exporter.
8. `prediction/` — preprocessing, inference, postprocessing, history, evaluator.
9. `scheduler/tasks.py` (the orchestration), scheduler.py, monitor.py.
10. `main.py` (argparse: `run [--force-date]`, `schedule`, `monitor`), root files, tests.

## Gotchas learned the hard way (bake these in from the start)

- **Lazy-import everything heavy.** `torch`, `transformers`, `sentence-transformers`,
  `yfinance`, `gnews`, `newspaper3k` must be imported *inside functions/factories*, never
  at module top level. Otherwise pytest can't even collect the pure-logic tests without the
  full heavy stack installed. Keep `nlp/__init__.py` empty so importing one nlp submodule
  doesn't drag in the others.
- **Headline-strength anchors must match inflected verbs.** A regex like `resign\b` will
  NOT match "resigns". Use `resign\w*`, `quit\w*`, `exits?` etc. Anchor scores must hit the
  spec examples: CEO resignation 0.98, major order 0.96, quarterly earnings 0.93,
  dividend 0.75, general commentary floor 0.30.
- **Scripts in `scripts/` need a sys.path bootstrap** (`sys.path.insert(0, project_root)`),
  because Python only puts the script's own dir on the path — `import utils` fails otherwise.
- **`pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`** so pytest and
  `python main.py` both resolve the top-level packages.
- **Source reliability tiers:** longest-substring match wins; a *named but unlisted*
  publisher -> `default_verified` (0.80), a *missing* source -> `unknown` (0.65).
- **Sanitize predictions** by clamping to +-20% around last close — dummy or broken models
  otherwise emit absurd numbers. This is a real safety net, not decoration.
- **gnews only filters by date**, so enforce the exact hour-level window yourself after
  fetching, and re-verify each article actually mentions the company (word-boundary match
  on name + aliases).

## Verification before declaring done

- `pip install pandas numpy PyYAML python-dateutil tzdata scikit-learn joblib pytest`
  then `python -m pytest -q` — the pure-logic suite must pass with NO heavy deps installed.
- `python scripts/generate_dummy_models.py` then a no-network smoke test:
  build synthetic article rows -> `aggregate_company_day` -> `PredictionEngine.predict` ->
  `sanitize_prediction`. Confirms the whole data path wires together.
- `python main.py --help` must succeed (proves no accidental heavy top-level imports).

## Honest caveats to always surface to the user

- The real pretrained models are the user's to supply; ship only
  `generate_dummy_models.py` (random-data artifacts) for plumbing tests.
- First real run needs network + disk (HF model downloads, Google News, Yahoo Finance);
  those integration paths can't be exercised in a sandboxed/offline environment.
- Google News + yfinance are unofficial; availability and fields vary.
- This is research infrastructure, not financial advice; outputs are model estimates.
