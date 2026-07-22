# EnergyStockPrediction

A modular, fault-tolerant pipeline that predicts next-session Open/Close prices for
20 Indian energy-sector stocks from Google News sentiment, 6-topic classification,
and a composite news-impact weighting scheme. Runs weekdays at 08:30 IST, scrapes
news for a precisely bounded IST window, scores every article, aggregates to one
row per company/day, runs a pretrained scaler -> PCA -> regression pipeline, and
writes append-only CSV outputs. A monitoring layer later scores realised accuracy.

## Architecture

```
config/      YAML: companies, source weights, topic weights, holidays, scheduler, settings
utils/       constants, config loader, logger, retry, cache, small helpers
market/      NSE trading calendar, OHLCV fetch (yfinance), volatility indicators
nlp/         FinBERT sentiment, 6-topic classifier, relevance, headline strength,
             adaptive sentiment threshold
weighting/   source reliability, recency decay, optional company/topic importance,
             normalization, composite impact score
news/        Google News scraping (gnews), dedup, company-mention matching,
             full-text extraction (newspaper3k)
datasets/    Dataset 1 (per-article) builder, Dataset 2 (per-company/day)
             aggregation, pre-inference validation, append-only CSV export
prediction/  preprocessing (scaler/PCA), inference, postprocessing (clamp +
             sanitize), prediction history, post-hoc monitoring/evaluation
scheduler/   the daily orchestration (tasks.py), APScheduler wiring
             (scheduler.py), and the monitoring job (monitor.py)
main.py      CLI: run / schedule / monitor
```

Data flows: `news.google_news` -> `news.article_extractor` -> `datasets.article_dataset`
(Dataset 1, one row per article) -> `datasets.aggregation` (Dataset 2, one row per
company/day) -> `datasets.validator` -> `prediction.preprocessing/inference/postprocessing`
-> append-only CSVs in `outputs/`.

## Install

```bash
pip install -r requirements.txt
```

For a lighter install that only exercises the pure-logic test suite (no
network/ML dependencies):

```bash
pip install pandas numpy PyYAML python-dateutil tzdata scikit-learn joblib pytest
```

## Configuration

Everything is config-driven under `config/*.yaml` -- no companies, weights, or
thresholds are hard-coded in Python:

- `companies.yaml` -- the 20-company universe (ticker, sector, aliases, keywords).
- `source_weights.yaml` -- publisher reliability tiers (longest-substring match),
  `default_verified` (named-but-unlisted publisher) and `unknown` (missing source)
  fallbacks.
- `topic_weights.yaml` -- optional per-topic importance multipliers (default 1.0)
  and the 6 topic descriptions used by zero-shot classification.
- `holidays.yaml` -- NSE trading holidays for the current + next year. **Verify
  the lunar/festival dates against the official NSE circular before relying on
  this for anything beyond dev/testing** -- only the fixed-date holidays
  (Republic Day, Independence Day, Gandhi Jayanti) are guaranteed correct as
  shipped.
- `scheduler.yaml` -- cron schedule (day/hour/minute/timezone) for the pipeline
  and monitoring jobs.
- `settings.yaml` -- paths, output filenames, NLP model names/device, sentiment
  threshold defaults, retry policy, logging, and monitoring alert thresholds.

`.env` (copy from `.env.example`) can override `NLP_DEVICE`, `LOG_LEVEL`, and
`HUGGINGFACE_TOKEN` without editing YAML.

## Running

```bash
python main.py run                       # run once, for today (IST)
python main.py run --force-date 2026-07-21  # run once, for a specific date
python main.py schedule                  # start the APScheduler daemon (blocking)
python main.py monitor                   # run the monitoring pass once
```

`main.py --help` succeeds with only the light dependency set installed -- every
heavy import (torch, transformers, sentence-transformers, yfinance, gnews,
newspaper3k, APScheduler) is lazy, loaded only inside the function that needs it.

## Model placement

Drop your trained artifacts into `models/`:

```
models/
  scaler.pkl              # fitted sklearn StandardScaler (or equivalent)
  pca.pkl                 # fitted sklearn PCA
  open_model.pkl          # regressor predicting next-session Open
  close_model.pkl         # regressor predicting next-session Close
  metadata.json           # {"model_version": "...", "feature_columns": [...]}
  topic_classifier/       # optional: a fine-tuned HF text-classification
                           # checkpoint (config.json, model weights, tokenizer)
                           # matching TOPIC_COLUMNS as label names. If absent,
                           # zero-shot NLI (bart-large-mnli) is used instead.
```

`metadata.json`'s `feature_columns` list must name every feature in the exact
order the scaler/PCA were fit on. If omitted, `prediction.preprocessing.DEFAULT_FEATURE_COLUMNS`
is used as a fallback (OHLCV + sentiment score/probs + 6 topic probs +
Article_Count + Volatility).

`scripts/generate_dummy_models.py` writes placeholder artifacts fit on random
data, purely so the preprocessing -> inference -> postprocessing path can be
exercised without a real model. **Replace these before using the pipeline for
anything beyond a plumbing test.**

## The adaptive sentiment threshold

No fixed cutoff: the label threshold scales with a stock's own rolling
volatility relative to the day's cross-sectional (universe) median.

```
r         = v / v_ref            # v = rolling 90d return-std (or ATR/Close);
                                  # v_ref = today's universe median, else 0.02
threshold = clamp( base * (1 + k * (r - 1)),  min_threshold,  max_threshold )

label:  score >  +threshold -> Positive
        score < -threshold -> Negative
        else               -> Neutral
```

Defaults: `base=0.15, k=0.8, min=0.08, max=0.45`. A more volatile stock (r > 1)
needs a stronger sentiment score to earn a directional label; a calmer stock
(r < 1) earns one at a lower score. The same threshold function is reused for
both per-article and weighted-daily-aggregate scores. See
`nlp/adaptive_threshold.py`'s module docstring for the authoritative version.

## Tests

```bash
python -m pytest -q
```

The full suite (46 tests as shipped) passes with only the light dependency set
installed -- every pure-logic module (time window, adaptive threshold,
weighting, aggregation, deduplication, indicators, headline strength, validation)
is exercised without touching torch/transformers/yfinance/gnews/newspaper3k.

## Caveats (read before relying on this)

- **The pretrained models are yours to supply.** This repo ships only
  `scripts/generate_dummy_models.py` (random-data artifacts) so the plumbing can
  be tested end-to-end. Predictions from the dummy artifacts are meaningless.
- **First real run needs network + disk**: HuggingFace model downloads
  (FinBERT, the embedding model, and the zero-shot classifier if no fine-tuned
  topic model is supplied), Google News scraping, and Yahoo Finance OHLCV --
  none of that can be exercised in a sandboxed/offline environment.
- **Google News and Yahoo Finance are unofficial, scraped data sources.**
  Field availability, rate limits, and result quality vary and can change
  without notice.
- **This is research infrastructure, not financial advice.** All outputs are
  model estimates from an experimental pipeline.
