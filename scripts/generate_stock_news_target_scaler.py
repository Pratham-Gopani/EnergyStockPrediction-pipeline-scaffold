"""Builds an APPROXIMATE target (output) scaler for stock_news_rf_model.pkl.

*** WHY THIS EXISTS ***
stock_news_rf_model.pkl's raw output is not in real rupee units (verified: even
realistic-looking inputs produce outputs like -4.6, which no stock price can be --
a RandomForestRegressor cannot extrapolate outside the range of its training
targets, so the target itself must have been scaled before training). The
uploaded artifacts include an input-feature scaler but no target scaler, and the
original training data/script isn't available to fit one exactly.

*** THE APPROXIMATION ***
This script does NOT recover the true training-time target scaler -- that isn't
possible without the original target values. Instead it borrows the mean/std of
News_Day_Open and News_Day_Close from stock_news_scaler.pkl (the INPUT scaler,
fit on the same training dataset) as stand-ins for Next_Day_Open/Next_Day_Close's
mean/std. This is reasonable because "next day's price" and "today's price" are
the same underlying time series shifted by one row -- their population mean/std
should be nearly identical, especially aggregated across many tickers and dates --
but it is still an approximation, not a measured fact. If you ever get the real
training script/notebook, replace this file's contents with the actual fitted
target scaler and this approximation becomes unnecessary.

Run directly: `python scripts/generate_stock_news_target_scaler.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    import joblib
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from utils.config_loader import get_config

    config = get_config()
    input_scaler_path = config.path("models_dir") / "pkl" / "stock_news_scaler.pkl"
    output_path = config.path("models_dir") / "pkl" / "stock_news_target_scaler_approx.pkl"

    input_scaler = joblib.load(input_scaler_path)
    names = list(input_scaler.feature_names_in_)
    open_idx = names.index("News_Day_Open")
    close_idx = names.index("News_Day_Close")

    means = np.array([input_scaler.mean_[open_idx], input_scaler.mean_[close_idx]])
    stds = np.array([input_scaler.scale_[open_idx], input_scaler.scale_[close_idx]])

    target_scaler = StandardScaler()
    target_scaler.mean_ = means
    target_scaler.scale_ = stds
    target_scaler.var_ = stds**2
    target_scaler.n_features_in_ = 2
    target_scaler.n_samples_seen_ = input_scaler.n_samples_seen_
    target_scaler.feature_names_in_ = np.array(["Next_Day_Open", "Next_Day_Close"], dtype=object)

    joblib.dump(target_scaler, output_path)
    print(f"Wrote APPROXIMATE target scaler to {output_path}")
    print(f"  Open  -> mean={means[0]:.4f} std={stds[0]:.4f}")
    print(f"  Close -> mean={means[1]:.4f} std={stds[1]:.4f}")
    print("This is an approximation (see module docstring) -- replace with the real")
    print("training-time target scaler if you ever obtain the original training script.")


if __name__ == "__main__":
    main()
