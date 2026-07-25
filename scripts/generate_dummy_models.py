"""Generate placeholder scaler/PCA/Open/Close model artifacts for PLUMBING TESTS ONLY.

*** WARNING ***
These artifacts are fit on random synthetic data. Predictions produced from them
are meaningless noise -- they exist purely so prediction.preprocessing /
prediction.inference / prediction.postprocessing can be exercised end-to-end
(scaler.transform -> pca.transform -> model.predict -> sanitize) without a real
trained model. Before using this pipeline for anything beyond a smoke test, the
user MUST replace models/scaler.pkl, models/pca.pkl, models/open_model.pkl,
models/close_model.pkl, and models/metadata.json with real, properly trained
artifacts (and optionally models/topic_classifier/ for a fine-tuned topic model).

Run directly: `python scripts/generate_dummy_models.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Bootstrap: `python scripts/generate_dummy_models.py` only puts scripts/ on
# sys.path, not the project root, so `import utils` etc. would otherwise fail.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_VERSION = "dummy-0.1.0"
N_SAMPLES = 500
PCA_COMPONENTS = 8
RANDOM_SEED = 42


def main() -> None:
    import joblib
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    from prediction.preprocessing import DEFAULT_FEATURE_COLUMNS
    from utils.config_loader import get_config

    config = get_config()
    config.ensure_dirs()

    rng = np.random.default_rng(RANDOM_SEED)
    n_features = len(DEFAULT_FEATURE_COLUMNS)

    raw_features = rng.normal(loc=0.0, scale=1.0, size=(N_SAMPLES, n_features))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_features)

    n_components = min(PCA_COMPONENTS, n_features, N_SAMPLES)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reduced_features = pca.fit_transform(scaled_features)

    open_targets = rng.normal(loc=100.0, scale=10.0, size=N_SAMPLES)
    close_targets = open_targets + rng.normal(loc=0.0, scale=2.0, size=N_SAMPLES)

    open_model = LinearRegression().fit(reduced_features, open_targets)
    close_model = LinearRegression().fit(reduced_features, close_targets)

    joblib.dump(scaler, config.model_file("scaler_file"))
    joblib.dump(pca, config.model_file("pca_file"))
    joblib.dump(open_model, config.model_file("open_model_file"))
    joblib.dump(close_model, config.model_file("close_model_file"))

    metadata = {
        "model_version": MODEL_VERSION,
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "pca_components": n_components,
        "generated_by": "scripts/generate_dummy_models.py",
        "warning": "DUMMY artifacts fit on random data -- for plumbing tests only, not real predictions.",
    }
    metadata_path = config.model_file("metadata_file")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Wrote dummy model artifacts to {config.path('models_dir')}")
    for key in ("scaler_file", "pca_file", "open_model_file", "close_model_file", "metadata_file"):
        print(f"  - {config.model_file(key) if key != 'metadata_file' else metadata_path}")


if __name__ == "__main__":
    main()
