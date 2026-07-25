"""Demo/smoke test for the 5 uploaded models ONLY -- no news scraping, no
network calls, no yfinance. Builds a synthetic multi-day daily history for one
company (enough rows for the LSTM models' 15-day sequence and the 7-day rolling
features), runs every model declared in config/models.yaml through
prediction.model_registry.ModelRunner, sanitizes each prediction, and prints the
result table.

Run directly: `python scripts/demo_predict_all_models.py [TICKER]`
(TICKER defaults to the first company in config/companies.yaml.)

This is the fastest way to check "do all 5 uploaded models load and produce a
number" without waiting on Google News / Yahoo Finance / HuggingFace downloads.
It does NOT validate that the two unverified deep-learning models
(lstm_torch, lstm_keras) are numerically correct -- see
prediction/feature_engineering.py's module docstring for why that can't be
confirmed from the uploaded files alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

N_HISTORY_DAYS = 20
RANDOM_SEED = 7


def _build_synthetic_history(ticker: str, company_name: str):
    import numpy as np
    import pandas as pd

    from utils.constants import DAILY_COLUMNS, TOPIC_COLUMNS

    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=N_HISTORY_DAYS, freq="B")

    rows = []
    price = 550.0
    for day in dates:
        drift = rng.normal(0, 3.0)
        open_price = price
        close_price = max(1.0, price + drift)
        high_price = max(open_price, close_price) + abs(rng.normal(0, 1.5))
        low_price = min(open_price, close_price) - abs(rng.normal(0, 1.5))
        volume = rng.integers(500_000, 5_000_000)
        price = close_price

        topic_probs = rng.dirichlet(np.ones(len(TOPIC_COLUMNS)))
        sentiment_score = float(rng.uniform(-0.5, 0.5))
        sentiment_pos = max(0.0, (sentiment_score + 1) / 2 + rng.normal(0, 0.05))
        sentiment_neg = max(0.0, 1 - sentiment_pos - 0.2)
        sentiment_neu = max(0.0, 1 - sentiment_pos - sentiment_neg)

        row = {col: None for col in DAILY_COLUMNS}
        row.update(
            {
                "Ticker": ticker,
                "Company": company_name,
                "News_Date": day.date().isoformat(),
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Adj_Close": close_price,
                "Volume": volume,
                "Sentiment_Score": sentiment_score,
                "Sentiment_Positive": sentiment_pos,
                "Sentiment_Neutral": sentiment_neu,
                "Sentiment_Negative": sentiment_neg,
                "Sentiment_Label": "Neutral",
                "Headlines": f"Synthetic demo headline for {company_name} on {day.date()}",
                "Article_Count": int(rng.integers(1, 10)),
                "Volatility": float(abs(rng.normal(0.02, 0.01))),
            }
        )
        for col, prob in zip(TOPIC_COLUMNS, topic_probs):
            row[col] = float(prob)
        row["Predicted_Topic"] = TOPIC_COLUMNS[int(np.argmax(topic_probs))]
        rows.append(row)

    return pd.DataFrame(rows, columns=DAILY_COLUMNS)


def main() -> None:
    from prediction.feature_engineering import ticker_code_map
    from prediction.model_registry import ModelRunner
    from prediction.postprocessing import sanitize_prediction
    from utils.config_loader import get_config

    config = get_config()
    companies = config.get("companies.companies") or []
    if not companies:
        print("No companies configured in config/companies.yaml -- nothing to demo.")
        return

    requested_ticker = sys.argv[1] if len(sys.argv) > 1 else None
    company = next((c for c in companies if c["ticker"] == requested_ticker), companies[0])
    ticker = company["ticker"]

    print(f"Building synthetic {N_HISTORY_DAYS}-day history for {company['name']} ({ticker})...")
    history = _build_synthetic_history(ticker, company["name"])
    last_close = float(history.iloc[-1]["Close"])

    ticker_codes = ticker_code_map(companies)
    runner = ModelRunner(config)

    print(f"\nRunning {len(runner.specs)} configured model(s) (config/models.yaml)...\n")
    predictions = runner.predict_all(history, ticker, ticker_codes)

    if not predictions:
        print("No model produced a prediction -- check the errors logged above "
              "(commonly: a backend package like torch/tensorflow/xgboost isn't installed).")
        return

    header = f"{'Model':<20} {'Backend':<10} {'Target':<7} {'Raw':>12} {'Sanitized':>12} {'Unverified':>11}"
    print(header)
    print("-" * len(header))
    for prediction in predictions:
        sanitized = sanitize_prediction(prediction, last_close)
        print(
            f"{prediction.model_name:<20} {prediction.backend:<10} {prediction.target:<7} "
            f"{prediction.value:>12.2f} {sanitized.value:>12.2f} {str(prediction.unverified):>11}"
        )

    print(f"\n(Last synthetic close used for +-20% sanitization clamp: {last_close:.2f})")
    print(
        "\nNOTE: lstm_torch and lstm_keras are UNVERIFIED reconstructions -- see "
        "prediction/feature_engineering.py's module docstring before trusting their numbers."
    )


if __name__ == "__main__":
    main()
