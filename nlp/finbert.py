"""FinBERT sentiment scoring. Loads ProsusAI/finbert once per process via
ModelRegistry and exposes a batch scoring function returning pos/neu/neg
probabilities per text. transformers/torch are imported lazily inside functions.
"""

from __future__ import annotations

from utils.cache import ModelRegistry
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger("nlp.finbert")

_REGISTRY_KEY = "finbert_pipeline"


def resolve_device(device_setting: str | None = None) -> int:
    """Return the transformers `device` index: -1 for CPU, 0 for the first CUDA GPU.
    "auto" (or None) picks CUDA if available, else CPU.
    """
    config = get_config()
    setting = device_setting or config.get("nlp.device", "auto")
    if setting == "cpu":
        return -1
    if setting == "cuda":
        return 0
    # auto
    try:
        import torch  # lazy: heavy dependency

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def _build_pipeline():
    from transformers import (  # lazy: heavy dependency
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    config = get_config()
    model_name = config.get("nlp.finbert_model", "ProsusAI/finbert")
    device = resolve_device()

    logger.info("Loading FinBERT model %s on device=%s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=device,
    )


def get_finbert_pipeline():
    registry = ModelRegistry()
    return registry.get_or_create(_REGISTRY_KEY, _build_pipeline)


def score_texts(texts: list[str]) -> list[dict[str, float]]:
    """Score a batch of texts, returning [{"positive": .., "neutral": .., "negative": ..}, ...]
    in the same order as `texts`. Empty input returns an empty list.
    """
    if not texts:
        return []

    config = get_config()
    max_length = config.get("nlp.max_sequence_length", 512)

    nlp_pipeline = get_finbert_pipeline()
    # HuggingFace's fast (Rust) tokenizers aren't safe to call from multiple
    # threads at once -- scheduler.tasks processes companies concurrently, so
    # without this lock two overlapping calls into this shared pipeline raise
    # RuntimeError: Already borrowed.
    with ModelRegistry().lock_for(_REGISTRY_KEY):
        raw_results = nlp_pipeline(list(texts), truncation=True, max_length=max_length)

    scored: list[dict[str, float]] = []
    for result in raw_results:
        label_scores = {entry["label"].lower(): float(entry["score"]) for entry in result}
        scored.append(
            {
                "positive": label_scores.get("positive", 0.0),
                "neutral": label_scores.get("neutral", 0.0),
                "negative": label_scores.get("negative", 0.0),
            }
        )
    return scored
