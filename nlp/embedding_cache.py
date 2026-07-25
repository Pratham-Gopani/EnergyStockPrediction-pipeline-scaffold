"""Sentence-embedding provider on top of sentence-transformers, with two layers of
caching: the model itself lives in ModelRegistry (loaded once per process), and
individual text embeddings are cached to disk keyed by a stable hash of the text so
repeated runs over the same headlines/summaries skip re-embedding entirely.
"""

from __future__ import annotations

import math

from utils.cache import DiskCache, ModelRegistry
from utils.config_loader import get_config
from utils.helpers import stable_hash
from utils.logger import get_logger

logger = get_logger("nlp.embedding_cache")

_REGISTRY_KEY = "sentence_embedding_model"


def _build_model():
    from sentence_transformers import SentenceTransformer  # lazy: heavy dependency

    config = get_config()
    model_name = config.get("nlp.embedding_model", "all-MiniLM-L6-v2")
    logger.info("Loading sentence-transformer model %s", model_name)
    return SentenceTransformer(model_name)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingProvider:
    """Get a sentence embedding for a piece of text, cached on disk by text hash."""

    def __init__(self, namespace: str = "embeddings"):
        self._disk_cache = DiskCache(namespace)

    def _model(self):
        registry = ModelRegistry()
        return registry.get_or_create(_REGISTRY_KEY, _build_model)

    def embed(self, text: str) -> list[float]:
        if not text:
            return []
        key = stable_hash(text)
        cached = self._disk_cache.get(key)
        if cached is not None:
            return cached
        # Locked: sentence-transformers' fast tokenizer isn't safe to call from
        # multiple threads at once (scheduler.tasks processes companies
        # concurrently) -- see utils.cache.ModelRegistry.lock_for's docstring.
        with ModelRegistry().lock_for(_REGISTRY_KEY):
            vector = self._model().encode(text, convert_to_numpy=False)
        vector_list = [float(v) for v in vector]
        self._disk_cache.set(key, vector_list)
        return vector_list

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
