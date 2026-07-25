"""Namespaced logging setup: a console handler plus two rotating file handlers
(pipeline.log for everything, errors.log for WARNING+). All loggers returned by
`get_logger` live under the "energy." namespace so a single `logging.getLogger("energy")`
level change (or handler) affects the whole pipeline.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from utils.config_loader import get_config

_CONFIGURED = False
_NAMESPACE = "energy"


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    config = get_config()
    config.ensure_dirs()

    level_name = config.get("logging.level", "INFO")
    max_bytes = config.get("logging.max_bytes", 5 * 1024 * 1024)
    backup_count = config.get("logging.backup_count", 5)
    logs_dir = config.path("logs_dir")

    pipeline_log_path = logs_dir / config.get("logging.pipeline_log_file", "pipeline.log")
    error_log_path = logs_dir / config.get("logging.error_log_file", "errors.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger(_NAMESPACE)
    root.setLevel(getattr(logging, str(level_name).upper(), logging.INFO))
    root.propagate = False

    if not root.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(root.level)
        root.addHandler(console_handler)

        pipeline_handler = RotatingFileHandler(
            pipeline_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        pipeline_handler.setFormatter(formatter)
        pipeline_handler.setLevel(root.level)
        root.addHandler(pipeline_handler)

        error_handler = RotatingFileHandler(
            error_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.WARNING)
        root.addHandler(error_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the "energy." namespace, e.g. get_logger("news.google_news")."""
    _configure_root()
    qualified = name if name.startswith(f"{_NAMESPACE}.") or name == _NAMESPACE else f"{_NAMESPACE}.{name}"
    return logging.getLogger(qualified)
