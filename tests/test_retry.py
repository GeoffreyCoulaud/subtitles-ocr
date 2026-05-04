# tests/test_retry.py
import pytest
import logging
from unittest.mock import patch, call
import openai
from subtitles_ocr.pipeline.retry import (
    RetryConfig, RetryExhausted, NonRetryable, with_retry,
    _RETRYABLE_TYPES, _NON_RETRYABLE_TYPES,
)


def _raise(exc_type, msg="test"):
    def fn():
        raise exc_type(msg)
    return fn


def test_retry_config_defaults():
    cfg = RetryConfig()
    assert cfg.max_attempts == 10
    assert cfg.base_delay == 1.0
    assert cfg.max_delay == 30.0


def test_with_retry_returns_result_on_first_success():
    assert with_retry(lambda: 42, RetryConfig()) == 42


def test_with_retry_retries_on_retryable_error():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("transient")
        return "ok"
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = with_retry(fn, RetryConfig(max_attempts=5))
    assert result == "ok"
    assert len(calls) == 3


def test_with_retry_raises_retry_exhausted_after_max_attempts():
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        with pytest.raises(RetryExhausted):
            with_retry(_raise(ValueError), RetryConfig(max_attempts=3))


def test_with_retry_calls_fn_exactly_max_attempts_times():
    calls = []
    def fn():
        calls.append(1)
        raise ValueError("always")
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        with pytest.raises(RetryExhausted):
            with_retry(fn, RetryConfig(max_attempts=4))
    assert len(calls) == 4


def test_with_retry_raises_non_retryable_immediately():
    calls = []
    def fn():
        calls.append(1)
        raise OSError("disk error")
    with pytest.raises(NonRetryable):
        with_retry(fn, RetryConfig(max_attempts=10))
    assert len(calls) == 1


def test_with_retry_exponential_backoff():
    with patch("subtitles_ocr.pipeline.retry.time.sleep") as mock_sleep:
        with pytest.raises(RetryExhausted):
            with_retry(
                _raise(ValueError),
                RetryConfig(max_attempts=4, base_delay=1.0, max_delay=100.0),
            )
    # Sleeps before attempts 2, 3, 4 — not before attempt 1, not after last
    assert mock_sleep.call_args_list == [call(1.0), call(2.0), call(4.0)]


def test_with_retry_delay_capped_at_max_delay():
    with patch("subtitles_ocr.pipeline.retry.time.sleep") as mock_sleep:
        with pytest.raises(RetryExhausted):
            with_retry(
                _raise(RuntimeError),
                RetryConfig(max_attempts=5, base_delay=1.0, max_delay=3.0),
            )
    delays = [c.args[0] for c in mock_sleep.call_args_list]
    assert delays == [1.0, 2.0, 3.0, 3.0]


def test_with_retry_no_sleep_with_max_attempts_one():
    with patch("subtitles_ocr.pipeline.retry.time.sleep") as mock_sleep:
        with pytest.raises(RetryExhausted):
            with_retry(_raise(ValueError), RetryConfig(max_attempts=1))
    mock_sleep.assert_not_called()


def test_with_retry_logs_warning_on_retry(caplog):
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("transient")
        return "ok"
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        with caplog.at_level(logging.WARNING, logger="subtitles_ocr.pipeline.retry"):
            with_retry(fn, RetryConfig(max_attempts=3))
    assert any("retrying" in r.message.lower() for r in caplog.records)


def test_with_retry_logs_warning_on_exhaustion(caplog):
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        with caplog.at_level(logging.WARNING, logger="subtitles_ocr.pipeline.retry"):
            with pytest.raises(RetryExhausted):
                with_retry(_raise(ValueError), RetryConfig(max_attempts=2))
    assert any("exhausted" in r.message.lower() for r in caplog.records)


def test_non_retryable_types_tuple():
    assert OSError in _NON_RETRYABLE_TYPES
    assert openai.AuthenticationError in _NON_RETRYABLE_TYPES
    assert openai.PermissionDeniedError in _NON_RETRYABLE_TYPES
    assert openai.NotFoundError in _NON_RETRYABLE_TYPES
    assert openai.BadRequestError in _NON_RETRYABLE_TYPES


def test_retryable_types_tuple():
    assert openai.APIConnectionError in _RETRYABLE_TYPES
    assert openai.APITimeoutError in _RETRYABLE_TYPES
    assert openai.RateLimitError in _RETRYABLE_TYPES
    assert openai.InternalServerError in _RETRYABLE_TYPES
    assert ValueError in _RETRYABLE_TYPES
    assert RuntimeError in _RETRYABLE_TYPES
