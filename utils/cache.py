"""Two caching primitives used across the pipeline:

- ModelRegistry: a thread-safe in-process singleton for expensive-to-load objects
  (FinBERT pipeline, embedding model, zero-shot classifier). `get_or_create` loads a
  named factory at most once per process, no matter how many callers ask for it
  concurrently.
- DiskCache: a small JSON-backed key/value cache under paths.cache_dir, used e.g. to
  persist sentence-embeddings keyed by a text hash across runs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from utils.config_loader import get_config


class ModelRegistry:
    """Process-wide singleton cache for expensive model objects."""

    _instance: "ModelRegistry | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._store = {}
                instance._store_lock = threading.Lock()
                cls._instance = instance
            return cls._instance

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> Any:
        """Return the cached object for `key`, calling `factory()` at most once."""
        if key in self._store:
            return self._store[key]
        with self._store_lock:
            if key not in self._store:
                self._store[key] = factory()
            return self._store[key]

    def clear(self) -> None:
        with self._store_lock:
            self._store.clear()


class DiskCache:
    """Flat JSON-file-per-key disk cache rooted at paths.cache_dir/<namespace>/."""

    def __init__(self, namespace: str):
        config = get_config()
        self._dir: Path = config.path("cache_dir") / namespace
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str, default: Any = None) -> Any:
        path = self._path_for(key)
        if not path.exists():
            return default
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return default

    def set(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        with self._lock:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(value, fh)

    def has(self, key: str) -> bool:
        return self._path_for(key).exists()
