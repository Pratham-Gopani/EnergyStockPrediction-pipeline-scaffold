"""Loads and merges the YAML files under config/ into one dotted-access Config object.

`settings.yaml`'s top-level sections (app, paths, outputs, time_window, news, market,
nlp, sentiment, weighting, prediction, retry, logging, monitoring) are promoted to the
config root. The other files are nested under a key derived from their filename:
`companies.yaml` -> "companies", `source_weights.yaml` -> "source_weights",
`topic_weights.yaml` -> "topic_weights", `holidays.yaml` -> "holidays",
`scheduler.yaml` -> "scheduler".

Use `get_config()` everywhere; it is a process-wide cached singleton so YAML is
parsed once. Environment variables listed in `_ENV_OVERRIDES` take precedence over
YAML values, letting deployments override e.g. NLP_DEVICE or LOG_LEVEL without
editing files.
"""

from __future__ import annotations

import copy
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files whose top-level keys are promoted directly into the config root.
_PROMOTED_FILES = {"settings.yaml"}

# env var name -> dotted config key it overrides.
_ENV_OVERRIDES = {
    "NLP_DEVICE": "nlp.device",
    "LOG_LEVEL": "logging.level",
    "HUGGINGFACE_TOKEN": "nlp.huggingface_token",
}


def _load_dotenv_if_present(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # lazy: optional dependency

        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        # python-dotenv not installed: silently skip, os.environ is still honored.
        pass


class Config:
    """Dotted-key read-only view over the merged YAML config tree."""

    def __init__(self, config_dir: Path | None = None, project_root: Path | None = None):
        self.project_root = project_root or PROJECT_ROOT
        self.config_dir = config_dir or (self.project_root / "config")
        _load_dotenv_if_present(self.project_root)
        self._data: dict[str, Any] = self._load()
        self._apply_env_overrides()

    def _load(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if not self.config_dir.exists():
            return merged
        for yaml_path in sorted(self.config_dir.glob("*.yaml")):
            with open(yaml_path, "r", encoding="utf-8") as fh:
                content = yaml.safe_load(fh) or {}
            if yaml_path.name in _PROMOTED_FILES:
                merged.update(content)
            else:
                key = yaml_path.stem
                merged[key] = content
        return merged

    def _apply_env_overrides(self) -> None:
        for env_name, dotted_key in _ENV_OVERRIDES.items():
            value = os.environ.get(env_name)
            if value is not None:
                self._set(dotted_key, value)

    def _set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up a dotted key, e.g. config.get("nlp.finbert_model")."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # --- path helpers --------------------------------------------------

    def path(self, key: str) -> Path:
        """Resolve a paths.* entry (e.g. "data_dir") to an absolute Path."""
        relative = self.get(f"paths.{key}")
        if relative is None:
            raise KeyError(f"No such path config: paths.{key}")
        return self.project_root / relative

    def output_file(self, key: str) -> Path:
        """Resolve an outputs.* filename (e.g. "daily_file") under outputs_dir."""
        filename = self.get(f"outputs.{key}")
        if filename is None:
            raise KeyError(f"No such output config: outputs.{key}")
        return self.path("outputs_dir") / filename

    def model_file(self, key: str) -> Path:
        """Resolve a prediction.*_file entry (e.g. "scaler_file") under models_dir."""
        filename = self.get(f"prediction.{key}")
        if filename is None:
            raise KeyError(f"No such model file config: prediction.{key}")
        return self.path("models_dir") / filename

    def ensure_dirs(self) -> None:
        """Create every configured directory (data/models/outputs/backups/logs/cache)."""
        for key in ("data_dir", "models_dir", "outputs_dir", "backups_dir", "logs_dir", "cache_dir"):
            candidate = self.get(f"paths.{key}")
            if candidate:
                (self.project_root / candidate).mkdir(parents=True, exist_ok=True)


_lock = threading.Lock()


@lru_cache(maxsize=None)
def get_config(config_dir: str | None = None, project_root: str | None = None) -> Config:
    """Process-wide cached Config singleton. Args are strings so lru_cache can hash them."""
    with _lock:
        return Config(
            config_dir=Path(config_dir) if config_dir else None,
            project_root=Path(project_root) if project_root else None,
        )
