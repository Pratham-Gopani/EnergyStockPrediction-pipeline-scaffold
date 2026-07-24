"""6-topic classification for articles. Prefers a fine-tuned classifier checkpoint
at models/topic_classifier/ if the user has dropped one there; otherwise falls back
to zero-shot NLI (facebook/bart-large-mnli) scored against the topic descriptions
from config/topic_weights.yaml. Always returns all six TOPIC_COLUMNS probabilities
plus the argmax topic. On any classifier failure, returns a uniform distribution
rather than raising, so one bad article never aborts a batch.
"""

from __future__ import annotations

from pathlib import Path

from utils.cache import ModelRegistry
from utils.config_loader import get_config
from utils.constants import TOPIC_COLUMNS
from utils.logger import get_logger

logger = get_logger("nlp.topic_classifier")

_FINE_TUNED_REGISTRY_KEY = "topic_classifier_fine_tuned"
_ZERO_SHOT_REGISTRY_KEY = "zero_shot_pipeline"


def _uniform_distribution() -> dict[str, float]:
    n = len(TOPIC_COLUMNS)
    return {col: 1.0 / n for col in TOPIC_COLUMNS}


def _fine_tuned_model_dir() -> Path:
    config = get_config()
    return config.path("models_dir") / "topic_classifier"


def has_fine_tuned_classifier() -> bool:
    model_dir = _fine_tuned_model_dir()
    return model_dir.exists() and any(model_dir.iterdir())


def _build_fine_tuned_pipeline():
    from transformers import pipeline  # lazy: heavy dependency

    model_dir = _fine_tuned_model_dir()
    logger.info("Loading fine-tuned topic classifier from %s", model_dir)
    return pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir), top_k=None)


def _build_zero_shot_pipeline():
    from transformers import pipeline  # lazy: heavy dependency

    config = get_config()
    model_name = config.get("nlp.zero_shot_model", "facebook/bart-large-mnli")
    logger.info("Loading zero-shot classifier %s", model_name)
    return pipeline("zero-shot-classification", model=model_name)


def get_zero_shot_pipeline():
    registry = ModelRegistry()
    return registry.get_or_create(_ZERO_SHOT_REGISTRY_KEY, _build_zero_shot_pipeline)


def get_fine_tuned_pipeline():
    registry = ModelRegistry()
    return registry.get_or_create(_FINE_TUNED_REGISTRY_KEY, _build_fine_tuned_pipeline)


def _topic_descriptions() -> dict[str, str]:
    config = get_config()
    descriptions = config.get("topic_weights.descriptions") or {}
    return {col: descriptions.get(col, col) for col in TOPIC_COLUMNS}


def _classify_zero_shot(text: str) -> dict[str, float]:
    descriptions = _topic_descriptions()
    labels = list(descriptions.keys())
    hypothesis_texts = [descriptions[label] for label in labels]

    classifier = get_zero_shot_pipeline()
    result = classifier(text, candidate_labels=hypothesis_texts, multi_label=False)

    # result["labels"]/["scores"] are the hypothesis texts and their probabilities,
    # already sorted descending and summing to ~1 (multi_label=False -> softmax).
    text_to_column = {v: k for k, v in descriptions.items()}
    probs = {col: 0.0 for col in TOPIC_COLUMNS}
    for label_text, score in zip(result["labels"], result["scores"]):
        column = text_to_column.get(label_text)
        if column:
            probs[column] = float(score)
    return probs


def _classify_fine_tuned(text: str) -> dict[str, float]:
    classifier = get_fine_tuned_pipeline()
    result = classifier(text, truncation=True)[0]
    probs = {col: 0.0 for col in TOPIC_COLUMNS}
    for entry in result:
        label = entry["label"]
        if label in probs:
            probs[label] = float(entry["score"])
    return probs


def classify(text: str) -> dict:
    """Classify `text` into the 6 topics. Returns a dict with all TOPIC_COLUMNS keys
    plus "Predicted_Topic" (argmax column name). Falls back to a uniform
    distribution (argmax = first topic) on any classifier error.
    """
    if not text:
        probs = _uniform_distribution()
    else:
        try:
            if has_fine_tuned_classifier():
                probs = _classify_fine_tuned(text)
            else:
                probs = _classify_zero_shot(text)
        except Exception:  # noqa: BLE001 - any classifier failure -> uniform fallback
            logger.exception("Topic classification failed; falling back to uniform distribution")
            probs = _uniform_distribution()

    predicted_topic = max(probs, key=probs.get)
    return {**probs, "Predicted_Topic": predicted_topic}
