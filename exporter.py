app:
  name: EnergyStockPrediction
  version: "0.1.0"

paths:
  config_dir: "config"
  data_dir: "data"
  models_dir: "models"
  outputs_dir: "outputs"
  backups_dir: "outputs/backups"
  logs_dir: "logs"
  cache_dir: "data/cache"

outputs:
  articles_file: "dataset_articles.csv"
  daily_file: "dataset_daily.csv"
  predictions_file: "prediction_history.csv"
  performance_file: "performance_history.csv"

time_window:
  timezone: "Asia/Kolkata"
  daily_start_hour: 8
  daily_start_minute: 30

news:
  max_articles_per_company: 40
  concurrency: 5
  request_timeout_seconds: 15
  language: "en"
  country: "IN"

market:
  ticker_suffix: ".NS"
  lookback_days: 200
  history_interval: "1d"

nlp:
  finbert_model: "ProsusAI/finbert"
  embedding_model: "all-MiniLM-L6-v2"
  zero_shot_model: "facebook/bart-large-mnli"
  max_sequence_length: 512
  device: "auto"   # auto | cpu | cuda

sentiment:
  base_threshold: 0.15
  k: 0.8
  min_threshold: 0.08
  max_threshold: 0.45
  rolling_window: 90
  measure: "vol"   # vol (return std) | atr

weighting:
  # weights product order: source * recency * relevance * headline [* company * topic]
  use_company_weight: false
  use_topic_weight: false

prediction:
  scaler_file: "scaler.pkl"
  pca_file: "pca.pkl"
  open_model_file: "open_model.pkl"
  close_model_file: "close_model.pkl"
  metadata_file: "metadata.json"
  prob_sum_tolerance: 0.05
  clamp_pct: 0.20

retry:
  max_attempts: 4
  base_delay_seconds: 1.5
  backoff_multiplier: 2.0
  jitter_seconds: 0.5

logging:
  level: "INFO"
  max_bytes: 5242880
  backup_count: 5
  pipeline_log_file: "pipeline.log"
  error_log_file: "errors.log"

monitoring:
  mape_alert_threshold: 5.0
  direction_accuracy_alert_threshold: 0.50
