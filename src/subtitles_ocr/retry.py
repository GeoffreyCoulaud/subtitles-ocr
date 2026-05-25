import functools
import logging
import time
from collections.abc import Callable
from typing import Self

from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class RetryConfig(BaseModel):
    max_retries: int
    backoff_seconds: tuple[float, ...]

    @model_validator(mode="after")
    def _lengths_match(self) -> Self:
        if len(self.backoff_seconds) != self.max_retries:
            raise ValueError("backoff_seconds must have exactly max_retries entries")
        return self


def retry_method(is_retryable: Callable[[Exception], bool]):
    """Decorator for methods on classes with a `retry: RetryConfig` attribute.

    Propagates non-retryable exceptions immediately. On exhaustion, re-raises
    the last retryable exception so the caller can wrap it with stage context.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            config: RetryConfig = self.retry
            for attempt in range(config.max_retries + 1):
                try:
                    return fn(self, *args, **kwargs)
                except Exception as e:
                    if not is_retryable(e):
                        raise
                    if attempt == config.max_retries:
                        raise
                    delay = config.backoff_seconds[attempt]
                    logger.warning(
                        "%s attempt %d/%d failed (%s: %s). Retrying in %.1fs.",
                        fn.__qualname__,
                        attempt + 1,
                        config.max_retries + 1,
                        type(e).__name__,
                        e,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator
