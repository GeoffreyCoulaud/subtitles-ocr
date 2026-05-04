# src/subtitles_ocr/pipeline/retry.py
import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from openai import (
    APIConnectionError, APITimeoutError, RateLimitError, InternalServerError,
    AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError,
)

log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class RetryConfig:
    max_attempts: int = 10
    base_delay: float = 1.0
    max_delay: float = 30.0


class RetryExhausted(Exception):
    """All retry attempts exhausted."""


class NonRetryable(Exception):
    """Error that must not be retried."""


_NON_RETRYABLE_TYPES = (
    OSError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
)

_RETRYABLE_TYPES = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    ValueError,
    RuntimeError,
)


def with_retry(
    fn: Callable[[], T],
    config: RetryConfig,
    logger: logging.Logger = log,
) -> T:
    last_error: Exception | None = None
    for attempt in range(config.max_attempts):
        try:
            return fn()
        except _NON_RETRYABLE_TYPES as e:
            raise NonRetryable(str(e)) from e
        except _RETRYABLE_TYPES as e:
            last_error = e
            if attempt < config.max_attempts - 1:
                delay = min(config.base_delay * (2 ** attempt), config.max_delay)
                logger.warning(
                    "Attempt %d/%d failed (%s): %s — retrying in %.1fs",
                    attempt + 1, config.max_attempts, type(e).__name__, e, delay,
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "Attempt %d/%d failed (%s): %s — retries exhausted",
                    attempt + 1, config.max_attempts, type(e).__name__, e,
                )
    raise RetryExhausted(f"All {config.max_attempts} attempts failed") from last_error
