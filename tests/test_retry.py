import time
import logging

import pytest
from pydantic import ValidationError

from subtitles_ocr.retry import RetryConfig, retry_method


# ---------------------------------------------------------------------------
# RetryConfig validation
# ---------------------------------------------------------------------------


def test_retry_config_length_mismatch_raises():
    with pytest.raises(ValidationError):
        RetryConfig(max_retries=2, backoff_seconds=(1.0,))


def test_retry_config_valid_construction():
    cfg = RetryConfig(max_retries=2, backoff_seconds=(1.0, 2.0))
    assert cfg.max_retries == 2
    assert cfg.backoff_seconds == (1.0, 2.0)


def test_retry_config_zero_retries_valid():
    cfg = RetryConfig(max_retries=0, backoff_seconds=())
    assert cfg.max_retries == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _always_retryable(e: Exception) -> bool:
    return True


def _never_retryable(e: Exception) -> bool:
    return False


class _Host:
    """Minimal class with a self.retry attribute for decorator testing."""

    def __init__(self, max_retries: int = 2, backoff_seconds=(0.0, 0.0)) -> None:
        self.retry = RetryConfig(
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        self.call_count = 0

    @retry_method(is_retryable=_always_retryable)
    def always_fails(self):
        self.call_count += 1
        raise RuntimeError("boom")

    @retry_method(is_retryable=_always_retryable)
    def succeeds_after(self, n: int):
        self.call_count += 1
        if self.call_count < n:
            raise RuntimeError("not yet")
        return "ok"

    @retry_method(is_retryable=_always_retryable)
    def always_succeeds(self):
        self.call_count += 1
        return "ok"

    @retry_method(is_retryable=_never_retryable)
    def raises_non_retryable(self):
        self.call_count += 1
        raise ValueError("fatal")


# ---------------------------------------------------------------------------
# Happy path: succeeds on first call
# ---------------------------------------------------------------------------


def test_happy_path_called_once():
    host = _Host()
    result = host.always_succeeds()
    assert result == "ok"
    assert host.call_count == 1


# ---------------------------------------------------------------------------
# Retry-then-success: fails on first attempt, succeeds on second
# ---------------------------------------------------------------------------


def test_retry_then_success():
    host = _Host(max_retries=2, backoff_seconds=(0.0, 0.0))
    result = host.succeeds_after(2)
    assert result == "ok"
    assert host.call_count == 2


# ---------------------------------------------------------------------------
# Exhaustion: fails 3 times (1 initial + 2 retries), last exception propagated
# ---------------------------------------------------------------------------


def test_exhaustion_propagates_last_exception():
    host = _Host(max_retries=2, backoff_seconds=(0.0, 0.0))
    with pytest.raises(RuntimeError, match="boom"):
        host.always_fails()
    # 1 initial attempt + 2 retries = 3 total
    assert host.call_count == 3


# ---------------------------------------------------------------------------
# Non-retryable: propagates immediately after 1 call
# ---------------------------------------------------------------------------


def test_non_retryable_propagates_immediately():
    host = _Host(max_retries=2, backoff_seconds=(0.0, 0.0))
    with pytest.raises(ValueError, match="fatal"):
        host.raises_non_retryable()
    assert host.call_count == 1


# ---------------------------------------------------------------------------
# Backoff: time.sleep called with correct values
# ---------------------------------------------------------------------------


def test_sleep_called_with_backoff_values(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    host = _Host(max_retries=2, backoff_seconds=(1.5, 3.0))
    with pytest.raises(RuntimeError):
        host.always_fails()

    # Two retries → two sleeps with the configured backoff values
    assert sleep_calls == [1.5, 3.0]


# ---------------------------------------------------------------------------
# Logging: warning emitted on each retry
# ---------------------------------------------------------------------------


def test_warning_logged_on_each_retry(caplog):
    host = _Host(max_retries=2, backoff_seconds=(0.0, 0.0))
    with caplog.at_level(logging.WARNING, logger="subtitles_ocr.retry"):
        with pytest.raises(RuntimeError):
            host.always_fails()
    # One warning per retry (2 retries)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
