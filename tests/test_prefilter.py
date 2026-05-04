# tests/test_prefilter.py
import pytest
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
from subtitles_ocr.models import FrameGroup
from subtitles_ocr.pipeline.prefilter import prefilter_groups
from subtitles_ocr.pipeline.retry import RetryConfig
from subtitles_ocr.vlm.prompt import PREFILTER_PROMPT


def _no_retry() -> RetryConfig:
    return RetryConfig(max_attempts=1)


def _group(name: str = "a") -> FrameGroup:
    return FrameGroup(start_time=0.0, end_time=1.0, frame=Path(f"frames/{name}.jpg"))


def test_prefilter_prompt_is_defined():
    assert isinstance(PREFILTER_PROMPT, str)
    assert "has_text" in PREFILTER_PROMPT


def test_has_text_true_returns_true():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": true}'
    result = list(prefilter_groups([_group()], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [True]


def test_has_text_false_returns_false():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": false}'
    result = list(prefilter_groups([_group()], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [False]


def test_has_text_string_true_returns_true():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": "true"}'
    result = list(prefilter_groups([_group()], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [True]


def test_has_text_string_false_returns_false():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": "false"}'
    result = list(prefilter_groups([_group()], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [False]


def test_has_text_string_mixed_case_returns_false():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": "False"}'
    result = list(prefilter_groups([_group()], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [False]


def test_invalid_json_yields_none_after_exhausting_retries():
    client = MagicMock()
    client.analyze.return_value = "not json"
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(prefilter_groups([_group()], client, "p", workers=1, retry_config=RetryConfig(max_attempts=2)))
    assert result == [None]


def test_missing_field_yields_none():
    client = MagicMock()
    client.analyze.return_value = '{"result": "yes"}'
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(prefilter_groups([_group()], client, "p", workers=1, retry_config=RetryConfig(max_attempts=2)))
    assert result == [None]


def test_unrecognised_string_yields_none():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": "yes"}'
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(prefilter_groups([_group()], client, "p", workers=1, retry_config=RetryConfig(max_attempts=2)))
    assert result == [None]


def test_runtime_error_from_client_yields_none():
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("network error")
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(prefilter_groups([_group("a"), _group("b")], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [None, None]


def test_error_on_one_element_does_not_block_others():
    client = MagicMock()
    client.analyze.side_effect = [RuntimeError("fail"), '{"has_text": true}']
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(prefilter_groups([_group("a"), _group("b")], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [None, True]


def test_order_preserved_with_multiple_workers():
    client = MagicMock()
    client.analyze.side_effect = [
        '{"has_text": false}',
        '{"has_text": true}',
        '{"has_text": false}',
        '{"has_text": true}',
    ]
    groups = [_group("a"), _group("b"), _group("c"), _group("d")]
    result = list(prefilter_groups(groups, client, "p", workers=4, retry_config=_no_retry()))
    assert result == [False, True, False, True]


def test_empty_groups_returns_empty():
    client = MagicMock()
    result = list(prefilter_groups([], client, "p", workers=4, retry_config=_no_retry()))
    assert result == []


def test_analyze_called_with_json_mode():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": true}'
    group = _group()
    list(prefilter_groups([group], client, "prompt text", workers=1, retry_config=_no_retry()))
    client.analyze.assert_called_once_with(group.frame, "prompt text", json_mode=True)


def test_non_retryable_oserror_yields_none_without_retries():
    client = MagicMock()
    client.analyze.side_effect = OSError("no such file")
    result = list(prefilter_groups([_group()], client, "p", workers=1, retry_config=RetryConfig(max_attempts=10)))
    assert result == [None]
    assert client.analyze.call_count == 1
