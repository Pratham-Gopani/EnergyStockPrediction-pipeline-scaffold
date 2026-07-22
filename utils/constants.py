"""Canonical column/schema definitions shared across the whole pipeline.

Every module that produces or consumes a dataframe imports its column list from
here rather than re-listing column names inline, so the schema has exactly one
source of truth.
"""

# --- Topic classification -------------------------------------------------

TOPIC_COLUMNS = [
    "Prob_Topic_1_Policy_Regulatory",
    "Prob_Topic_2_Financial_Performance",
    "Prob_Topic_3_Investment_FII_DII",
    "Prob_Topic_4_Infrastructure_Expansion",
    "Prob_Topic_5_Stock_Price_Action",
    "Prob_Topic_6_Commodity_Macro",
]

TOPIC_LABELS = {
    "Prob_Topic_1_Policy_Regulatory": "Policy / Regulatory",
    "Prob_Topic_2_Financial_Performance": "Financial Performance",
    "Prob_Topic_3_Investment_FII_DII": "Investment / FII-DII",
    "Prob_Topic_4_Infrastructure_Expansion": "Infrastructure / Expansion",
    "Prob_Topic_5_Stock_Price_Action": "Stock Price Action",
    "Prob_Topic_6_Commodity_Macro": "Commodity / Macro",
}

# --- Sentiment -------------------------------------------------------------

SENTIMENT_LABEL_POSITIVE = "Positive"
SENTIMENT_LABEL_NEGATIVE = "Negative"
SENTIMENT_LABEL_NEUTRAL = "Neutral"

SENTIMENT_COLUMNS = [
    "Sentiment_Positive",
    "Sentiment_Neutral",
    "Sentiment_Negative",
    "Sentiment_Score",
    "Sentiment_Label",
]

# --- Market data -------------------------------------------------------------

OHLCV_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj_Close",
    "Volume",
]

# --- Dataset 1: per-article rows -------------------------------------------

ARTICLE_COLUMNS = [
    "Ticker",
    "Company",
    "News_Date",
    "Published_At",
    "Headline",
    "Summary",
    "URL",
    "Source",
    *SENTIMENT_COLUMNS,
    *TOPIC_COLUMNS,
    "Predicted_Topic",
    "Article_Relevance",
    "Headline_Strength",
    "Source_Reliability",
    "Recency_Weight",
    "Company_Weight",
    "Topic_Weight",
    "News_Impact_Weight",
]

# --- Dataset 2: per-company/day aggregated rows -----------------------------

DAILY_COLUMNS = [
    "Ticker",
    "Company",
    "News_Date",
    *OHLCV_COLUMNS,
    "Sentiment_Score",
    "Sentiment_Positive",
    "Sentiment_Neutral",
    "Sentiment_Negative",
    "Sentiment_Label",
    *TOPIC_COLUMNS,
    "Predicted_Topic",
    "Headlines",
    "Article_Count",
    "Volatility",
]

# --- Prediction outputs ------------------------------------------------------

PREDICTION_COLUMNS = [
    "Ticker",
    "Company",
    "News_Date",
    "Predicted_Open",
    "Predicted_Close",
    "Last_Close",
    "Actual_Open",
    "Actual_Close",
    "Model_Version",
    "Predicted_At",
]

PERFORMANCE_COLUMNS = [
    "Ticker",
    "Evaluated_At",
    "N_Samples",
    "MAE",
    "RMSE",
    "MAPE",
    "R2",
    "Direction_Accuracy",
]
