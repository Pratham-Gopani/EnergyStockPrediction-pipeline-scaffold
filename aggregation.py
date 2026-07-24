# Multi-model registry. Every entry is one trained artifact the pipeline will
# run a prediction through for a given target ("open" or "close"); every entry
# produces its own row in prediction_history.csv (see utils.constants
# PREDICTION_COLUMNS -- Model_Name is part of the row key). Paths are relative to
# paths.models_dir (see config/settings.yaml).
#
# `feature_set` names are resolved in prediction/feature_engineering.py's
# FEATURE_SETS dict. `unverified: true` marks models whose exact input feature
# composition could not be recovered from the artifact file itself (Keras/
# PyTorch don't persist feature names) -- see prediction/feature_engineering.py's
# module docstring for what that means in practice. A runtime warning is logged
# every time an `unverified` model produces a prediction.

models:
  - name: ridge_regression
    target: open
    backend: sklearn
    file: pkl/Open_Ridge_Regression_model.pkl
    feature_set: engineered_v1
    unverified: false

  - name: ridge_regression
    target: close
    backend: sklearn
    file: pkl/Close_Ridge_Regression_model.pkl
    feature_set: engineered_v1
    unverified: false

  - name: random_forest
    target: open
    backend: sklearn
    file: pkl/Open_Random_Forest_model.pkl
    feature_set: engineered_v1
    unverified: false

  - name: random_forest
    target: close
    backend: sklearn
    file: pkl/Close_Random_Forest_model.pkl
    feature_set: engineered_v1
    unverified: false

  - name: xgboost
    target: open
    backend: xgboost
    file: pkl/Open_XGBoost_model.pkl
    feature_set: engineered_v1
    unverified: false

  - name: xgboost
    target: close
    backend: xgboost
    file: pkl/Close_XGBoost_model.pkl
    feature_set: engineered_v1
    unverified: false

  - name: lstm_torch
    target: open
    backend: torch
    file: pkl/Open_LSTM_state_dict.pkl
    feature_set: engineered_v1_no_ticker
    sequence_length: 15
    architecture:
      input_size: 19
      hidden_size: 48
      head_hidden: 16
      head_activation: relu   # unverified -- state_dict has no activation params to confirm this
    unverified: true

  - name: lstm_torch
    target: close
    backend: torch
    file: pkl/Close_LSTM_state_dict.pkl
    feature_set: engineered_v1_no_ticker
    sequence_length: 15
    architecture:
      input_size: 19
      hidden_size: 48
      head_hidden: 16
      head_activation: relu
    unverified: true

  - name: lstm_keras
    target: open
    backend: keras
    file: keras/stock_lstm.keras
    feature_set: engineered_v1_keras_open
    sequence_length: 15
    target_scaler_file: keras/target_scaler.gz
    unverified: true

  - name: lstm_keras
    target: close
    backend: keras
    file: keras/stock_lstm_close.keras
    feature_set: engineered_v1_keras_close
    sequence_length: 15
    target_scaler_file: keras/target_scaler_close.gz
    unverified: true
