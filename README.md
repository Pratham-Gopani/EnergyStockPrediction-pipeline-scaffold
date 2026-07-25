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
prediction/  feature_engineering (feature sets for the uploaded models),
             model_registry (multi-backend loader: sklearn/keras active today,
             xgboost/torch also supported for future models), postprocessing
             (clamp + sanitize, long-format rows), prediction
             history, post-hoc monitoring/evaluation. preprocessing.py/inference.py
             are a minimal single-model reference kept for scripts/generate_dummy_models.py
             -- NOT the active production path.
scheduler/   the daily orchestration (tasks.py), APScheduler wiring
             (scheduler.py), and the monitoring job (monitor.py)
main.py      CLI: run / schedule / monitor
scripts/     demo_predict_all_models.py (no-network demo), generate_dummy_models.py
```

Data flows: `news.google_news` -> `news.article_extractor` -> `datasets.article_dataset`
(Dataset 1, one row per article) -> `datasets.aggregation` (Dataset 2, one row per
company/day) -> `datasets.validator` -> `prediction.feature_engineering` (engineered
features) -> `prediction.model_registry` (every model in `config/models.yaml` runs a
prediction) -> `prediction.postprocessing` (sanitize) -> append-only CSVs in `outputs/`.

## Install

```bash
pip install -r requirements.txt
```

This now includes `tensorflow` (for the Keras LSTM models) alongside the
original `torch`/`transformers`/`sentence-transformers` stack (torch is still
needed for FinBERT sentiment and zero-shot topic classification, independent of
price prediction). For a lighter install that only exercises the pure-logic
test suite (no network/ML dependencies):

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

## Multi-model predictions

The pipeline runs **every** model declared in `config/models.yaml` for both Open
and Close, and stores every one of their predictions as its own row in
`prediction_history.csv` (long format: one row per
`Ticker + News_Date + Model_Name + Target`) -- there is no single "the"
prediction. As shipped, 2 model families x 2 targets = 4 rows per company/day:

| Model | Backend | File(s) | Feature order |
|---|---|---|---|
| `stock_news_rf` | sklearn | `models/pkl/stock_news_rf_model.pkl` (a `MultiOutputRegressor` predicting `[Open, Close]` in one call) + `models/pkl/stock_news_scaler.pkl` (input `StandardScaler`) + `models/pkl/stock_news_target_scaler_approx.pkl` (**approximate** output descaler, see below) | 14 named features, confirmed from the scaler's own `feature_names_in_`; output order and the target scaler are **assumed/approximated**, see caveat below |
| `lstm_keras` | keras (tensorflow) | `models/keras/stock_lstm(_close).keras` + `target_scaler(_close).gz` | 16/17 features x 15-day sequence -- **unverified**, see caveat below |

### Where to put new/updated model files

Everything lives under `models/`, already populated with what was uploaded:

```
models/
  pkl/     stock_news_rf_model.pkl, stock_news_scaler.pkl,
           stock_news_target_scaler_approx.pkl
  keras/   stock_lstm.keras, stock_lstm_close.keras, target_encoder.gz,
           target_scaler.gz, target_scaler_close.gz
  topic_classifier/   optional fine-tuned HF text-classification checkpoint (unrelated
                      to price prediction -- used by nlp/topic_classifier.py instead
                      of zero-shot NLI, if present)
```

To add or replace a model: drop the file under `models/pkl/` or `models/keras/`,
then add/edit its entry in `config/models.yaml` (name, target, backend, file path,
feature_set, and for sequence models, `sequence_length`). `feature_builder:
stock_news_row` picks the same-day-only 14-feature builder; omit it (or set
`engineered`) for the rolling-history 20-feature family the Keras models use.
`input_scaler_file` applies a scaler to a sklearn/xgboost model's features before
`.predict()`; `output_index` picks one column out of a multi-output model's
result. No code changes needed unless the new model needs a genuinely new
backend or feature set shape.

### `stock_news_rf`'s 14 same-day features

`prediction/feature_engineering.stock_news_feature_row()` builds
`News_Day_Open/High/Low/Close/Volume`, `Day_High_Low_Spread_Pct`,
`Day_Close_Open_Spread_Pct`, `Sentiment_Score`, and the 6 `Prob_Topic_*` columns
directly from the current day's already-aggregated row -- no rolling history
needed. Both the feature **names and order** are fully confirmed from
`stock_news_scaler.pkl`'s own `feature_names_in_`. The one thing that ISN'T
recoverable from the pickle: since the model predicts both Open and Close in a
single call (`MultiOutputRegressor`), nothing in the file records which output
column is which. `config/models.yaml` assumes `output_index: 0` = Open,
`output_index: 1` = Close (matching the Open-before-Close convention the
scaler's own feature list uses) -- swap the two values there if your training
script ordered them the other way.

### `stock_news_rf`'s target scaler is an APPROXIMATION, not a measured fact

The model's raw output isn't in real rupee units -- confirmed directly: even
realistic-looking inputs (Open/Close near ₹600-670, the scaler's own training
mean) produce raw outputs like `-2.6` or `0.03`, and a `RandomForestRegressor`
cannot extrapolate outside the range of the targets it was trained on, so the
*target* must have been scaled before training. No target scaler was uploaded,
and the original training data isn't available to fit one exactly, so
`scripts/generate_stock_news_target_scaler.py` builds an **approximate** one:
it borrows `News_Day_Open`/`News_Day_Close`'s mean+std from the *input* scaler
(fit on the same training data) as stand-ins for `Next_Day_Open`/`Next_Day_Close`'s
real mean+std, on the reasoning that "next day's price" and "today's price" are
the same time series shifted by one row.

**This approximation is only partially validated.** Testing it against
realistic inputs: the Close output consistently inverse-transforms to a
plausible price. The Open output, even for an input sitting right at the
scaler's own training-data mean, consistently inverse-transforms to something
strongly negative (e.g. -2175) -- which suggests output index 0 may not
actually be a simple z-scored absolute price the way this approximation
assumes (it could be a different quantity entirely, like a day-over-day
change/return that can legitimately go negative, or that half of the
`MultiOutputRegressor` may have been fit on something else). **If you can get
the original training script, that's the only way to fully resolve this** --
otherwise, treat `stock_news_rf`'s Open predictions specifically with extra
skepticism; the ±20% sanitize clamp (see Postprocessing) is what keeps a wrong
raw value like that from ever being stored as the final `Predicted_Value` --
check `Raw_Predicted_Value` in `prediction_history.csv` to see the pre-clamp
number.

### The rolling-history 20 engineered features (used by `lstm_keras`)

`prediction/feature_engineering.py` also computes an older 20-feature rolling
set (`Sentiment_Score`, the 6 `Prob_Topic_*` columns, `NewsDay_Intraday_Pct`,
`NewsDay_Range_Pct`, `Log_Volume`, `Month_sin/cos`, `DOW_sin/cos`,
`Sentiment_RollMean_3/7`, `Momentum_RollMean_3/7`, `Volatility_Roll7`,
`Ticker_Code`) from the accumulating `dataset_daily.csv` history for that
ticker, used only by the Keras LSTM models now (as 16/17-column size-matched
subsets). **Read that module's docstring carefully** -- these feature names
were confirmed from an earlier model generation's `feature_names_in_`, but the
exact formula behind each derived column and the `Ticker_Code` integer mapping
were not recoverable from any file and use standard, documented definitions.

### Unverified models: `lstm_keras` and `stock_news_rf`

Keras doesn't persist feature names in its saved artifacts (only shapes) --
there's no self-describing way to recover exactly which 16/17 features, in
which order, `lstm_keras` was trained on. The pipeline runs it anyway (sized to
match its declared input width) and logs a `WARNING` every time it produces a
prediction. Treat its numbers as a plumbing smoke test, not a trustworthy
forecast, until you can confirm the original feature order (ideally from the
training notebook/script) and update `prediction/feature_engineering.FEATURE_SETS`
to match exactly.

`stock_news_rf` is flagged `unverified` for a narrower but still real reason:
its 14 input features are fully confirmed, but (a) which of its 2 outputs is
Open vs. Close is assumed, and (b) its target scaler is an approximation that's
only partially validated -- see the section above. Both `lstm_keras` and
`stock_news_rf`'s predictions are protected by the same ±20% sanitize clamp, so
a bad raw value never becomes a stored `Predicted_Value`, but check
`Raw_Predicted_Value` in `prediction_history.csv` before trusting either one's
numbers for anything beyond a smoke test.

### Legacy single-model path

`prediction/inference.py` + `prediction/preprocessing.py` +
`scripts/generate_dummy_models.py` are a minimal single scaler->PCA->model
reference kept from the original scaffold -- they are **not** wired into
`scheduler/tasks.py` any more. The real prediction path is
`prediction.model_registry.ModelRunner`, driven entirely by `config/models.yaml`.

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

## Demo: test only the uploaded models (no network)

To check "do all 5 uploaded models load and produce a number" without waiting on
Google News / Yahoo Finance / HuggingFace downloads:

```bash
python scripts/demo_predict_all_models.py             # first configured company
python scripts/demo_predict_all_models.py RELIANCE.NS # a specific ticker
```

It builds a synthetic 20-day daily history in memory (enough for the LSTMs'
15-day sequence + 7-day rolling features), runs every model in
`config/models.yaml`, and prints a table of raw vs. sanitized (+-20% clamped)
predictions per model/target, flagging the unverified ones.

## Tests

```bash
python -m pytest -q
```

The full suite (56 tests as shipped) passes with only the light dependency set
installed -- every pure-logic module (time window, adaptive threshold,
weighting, aggregation, deduplication, indicators, headline strength, feature
engineering, validation) is exercised without touching
torch/transformers/tensorflow/yfinance/gnews/newspaper3k.

## Getting this running from a GitHub clone

```bash
git clone <your-repo-url>
cd EnergyStockPrediction   # or whatever you named the clone

python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env            # adjust NLP_DEVICE/LOG_LEVEL if needed

python -m pytest -q               # sanity check
python scripts/demo_predict_all_models.py   # confirm the 5 models load, no network needed

python main.py run                # full pipeline run for today (needs network)
python main.py schedule           # start the 08:30 IST daily scheduler (blocking)
python main.py monitor            # run the accuracy-monitoring pass once
```

The real model files under `models/pkl/` and `models/keras/` are committed to
this repo (see `.gitignore` -- only the *dummy* placeholder artifacts and the raw
upload zips are excluded), so a fresh clone has everything needed for the demo
script and for `main.py run` immediately -- no separate model upload step.

## Caveats (read before relying on this)

- **`lstm_keras` is unverified.** Its per-timestep feature order/composition is
  a documented best-effort reconstruction, not confirmed from the model files
  (see "Unverified models" above).
- **`stock_news_rf`'s 14 INPUT features ARE fully verified** -- confirmed from
  `stock_news_scaler.pkl`'s own `feature_names_in_`. Its 2-output ORDER (which is
  Open vs. Close) is assumed, and -- more importantly -- its TARGET SCALER is an
  approximation that testing shows is only partially correct: the Close output
  consistently converts to a plausible price, the Open output does not (see
  "target scaler is an APPROXIMATION" above). Treat its Open predictions with
  extra skepticism until the real training script can confirm or fix this.
- **`Ticker_Code`** (used only by `lstm_keras`'s feature family, not by
  `stock_news_rf`) **is a best-effort default** (alphabetical rank of the 20
  configured tickers) unless you know the exact `LabelEncoder` mapping used at
  training time -- see `prediction/feature_engineering.py`.
- **First real pipeline run needs network + disk**: HuggingFace model downloads
  (FinBERT, the embedding model, and the zero-shot classifier if no fine-tuned
  topic model is supplied), Google News scraping, and Yahoo Finance OHLCV --
  none of that can be exercised in a sandboxed/offline environment. The demo
  script (`scripts/demo_predict_all_models.py`) needs none of this.
- **Google News and Yahoo Finance are unofficial, scraped data sources.**
  Field availability, rate limits, and result quality vary and can change
  without notice.
- **This is research infrastructure, not financial advice.** All outputs are
  model estimates from an experimental pipeline.
