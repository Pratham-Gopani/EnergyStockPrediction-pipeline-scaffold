"""Retry decorator with exponential backoff + jitter, used to wrap flaky network calls
(Google News, yfinance, HuggingFace downloads) so a single transient failure never
aborts the whole pipeline run.
"""

from __future__ import annotations

import random
import time
from functools import wraps
from typing import Any, Callable, Type

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger("utils.retry")


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    jitter: float = 0.5,
    exceptions: tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    """Retry the wrapped function on `exceptions`, sleeping
    base_delay * backoff_multiplier**attempt + random_jitter between attempts.
    Re-raises the last exception once max_attempts is exhausted.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - intentionally broad, caller-scoped
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        logger.error(
                            "%s failed after %d attempts: %s", func.__qualname__, max_attempts, exc
                        )
                        raise
                    delay = base_delay * (backoff_multiplier**attempt) + random.uniform(0, jitter)
                    logger.warning(
                        "%s attempt %d/%d failed (%s); retrying in %.2fs",
                        func.__qualname__,
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise last_exc  # pragma: no cover - unreachable, defensive

        return wrapper

    return decorator


def retry_from_config(exceptions: tuple[Type[BaseException], ...] = (Exception,)) -> Callable:
    """Same as `retry`, but reads max_attempts/base_delay/backoff/jitter from
    config/settings.yaml's `retry` section instead of hard-coded arguments.
    """
    config = get_config()
    return retry(
        max_attempts=config.get("retry.max_attempts", 4),
        base_delay=config.get("retry.base_delay_seconds", 1.5),
        backoff_multiplier=config.get("retry.backoff_multiplier", 2.0),
        jitter=config.get("retry.jitter_seconds", 0.5),
        exceptions=exceptions,
    )
