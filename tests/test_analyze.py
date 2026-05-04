# tests/test_analyze.py
import json
import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from subtitles_ocr.models import FrameGroup, SubtitleElement
from subtitles_ocr.pipeline.analyze import analyze_group, analyze_groups, parse_elements
from subtitles_ocr.pipeline.retry import RetryConfig

VALID_ELEMENT = {
    "text": "Bonjour",
    "style": "regular",
    "color": "white",
    "border_color": "black",
    "position": "bottom",
    "alignment": "center",
}


def _no_retry() -> RetryConfig:
    return RetryConfig(max_attempts=1)


def _group() -> FrameGroup:
    return FrameGroup(start_time=1.0, end_time=2.5, frame=Path("frames/000024.jpg"))


# --- parse_elements ---

def test_parse_elements_valid_response():
    elements = parse_elements(json.dumps([VALID_ELEMENT]))
    assert len(elements) == 1
    assert elements[0].text == "Bonjour"
    assert elements[0].color == "#FFFFFF"


def test_parse_elements_empty_array():
    assert parse_elements("[]") == []


def test_parse_elements_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_elements("not json at all")


def test_parse_elements_multiple():
    raw = json.dumps([VALID_ELEMENT, {**VALID_ELEMENT, "text": "Au revoir"}])
    elements = parse_elements(raw)
    assert len(elements) == 2
    assert elements[1].text == "Au revoir"


def test_parse_elements_single_object():
    elements = parse_elements(json.dumps(VALID_ELEMENT))
    assert len(elements) == 1
    assert elements[0].text == "Bonjour"


def test_parse_elements_unexpected_type_raises():
    with pytest.raises(ValueError, match="expected JSON"):
        parse_elements('"just a string"')


def test_parse_elements_partial_invalid_keeps_valid():
    raw = json.dumps([VALID_ELEMENT, {"text": "Bad"}])
    elements = parse_elements(raw)
    assert len(elements) == 1
    assert elements[0].text == "Bonjour"


def test_parse_elements_all_invalid_raises():
    """Garbage output (valid JSON but no item passes schema) must raise so retry can trigger."""
    garbage = {"]0E@#@$&,FB8$.-B2=A3F766=E+*9*)?2AC@3#1F9B.0": "#)8#-.:"}
    with pytest.raises(ValueError, match="all.*failed"):
        parse_elements(json.dumps(garbage))


def test_parse_elements_empty_array_does_not_raise():
    """True empty response (no subtitles) must stay silent — don't confuse with garbage."""
    assert parse_elements("[]") == []


def test_parse_elements_empty_object_returns_empty():
    """Empty dict {} is the model saying 'nothing here' — treat as [] without retrying."""
    assert parse_elements("{}") == []


def test_analyze_groups_does_not_retry_empty_object():
    """When model returns {}, accept it as no-subtitle frame on the first attempt; do not retry."""
    client = MagicMock()
    client.analyze.return_value = "{}"
    result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert len(result) == 1
    assert result[0] is not None
    assert result[0].elements == []
    assert client.analyze.call_count == 1


# --- analyze_group ---

def test_analyze_group_returns_correct_timing():
    client = MagicMock()
    client.analyze.return_value = "[]"
    analysis = analyze_group(_group(), client, prompt="p")
    assert analysis.start_time == 1.0
    assert analysis.end_time == 2.5


def test_analyze_group_parses_elements():
    client = MagicMock()
    client.analyze.return_value = json.dumps([VALID_ELEMENT])
    analysis = analyze_group(_group(), client, prompt="p")
    assert len(analysis.elements) == 1
    assert analysis.elements[0].text == "Bonjour"


def test_analyze_group_logs_raw_at_debug(caplog):
    client = MagicMock()
    client.analyze.return_value = "[]"
    with caplog.at_level(logging.DEBUG, logger="subtitles_ocr.pipeline.analyze"):
        analyze_group(_group(), client, prompt="p")
    assert any("raw →" in r.message and r.levelno == logging.DEBUG for r in caplog.records)


def test_analyze_group_logs_no_elements(caplog):
    client = MagicMock()
    client.analyze.return_value = "[]"
    with caplog.at_level(logging.INFO, logger="subtitles_ocr.pipeline.analyze"):
        analyze_group(_group(), client, prompt="p")
    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_msgs) == 1
    assert "(no elements)" in info_msgs[0]


def test_analyze_group_propagates_client_error():
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        analyze_group(_group(), client, prompt="p")


def test_analyze_group_propagates_parse_error():
    client = MagicMock()
    client.analyze.return_value = "not json"
    with pytest.raises(json.JSONDecodeError):
        analyze_group(_group(), client, prompt="p")


# --- analyze_groups ---

def test_analyze_groups_skips_vlm_when_no_text():
    client = MagicMock()
    group = _group()
    result = list(analyze_groups([group], [False], client, "p", workers=1, retry_config=_no_retry()))
    assert len(result) == 1
    assert result[0].elements == []
    assert result[0].start_time == group.start_time
    client.analyze.assert_not_called()


def test_analyze_groups_calls_vlm_when_has_text():
    client = MagicMock()
    client.analyze.return_value = json.dumps([VALID_ELEMENT])
    result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert len(result) == 1
    assert len(result[0].elements) == 1
    client.analyze.assert_called_once()


def test_analyze_groups_yields_none_on_exhausted_retries():
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("model failed")
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [None]


def test_analyze_groups_logs_warning_on_failure(caplog):
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("model failed")
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        with caplog.at_level(logging.WARNING, logger="subtitles_ocr.pipeline.analyze"):
            list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_analyze_groups_error_on_one_does_not_block_others():
    client = MagicMock()
    client.analyze.side_effect = [RuntimeError("fail"), json.dumps([VALID_ELEMENT])]
    groups = [
        FrameGroup(start_time=1.0, end_time=2.0, frame=Path("frames/a.jpg")),
        FrameGroup(start_time=3.0, end_time=4.0, frame=Path("frames/b.jpg")),
    ]
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(analyze_groups(groups, [True, True], client, "p", workers=1, retry_config=_no_retry()))
    assert result[0] is None
    assert result[1] is not None
    assert len(result[1].elements) == 1


def test_analyze_groups_preserves_order():
    client = MagicMock()
    client.analyze.return_value = "[]"
    groups = [
        FrameGroup(start_time=1.0, end_time=2.0, frame=Path("frames/a.jpg")),
        FrameGroup(start_time=3.0, end_time=4.0, frame=Path("frames/b.jpg")),
        FrameGroup(start_time=5.0, end_time=6.0, frame=Path("frames/c.jpg")),
    ]
    result = list(analyze_groups(groups, [True, True, True], client, "p", workers=3, retry_config=_no_retry()))
    assert [r.start_time for r in result] == [1.0, 3.0, 5.0]


def test_analyze_groups_mixed_filter():
    client = MagicMock()
    client.analyze.return_value = json.dumps([VALID_ELEMENT])
    groups = [_group(), _group(), _group()]
    result = list(analyze_groups(groups, [False, True, False], client, "p", workers=1, retry_config=_no_retry()))
    assert len(result) == 3
    assert result[0].elements == []
    assert len(result[1].elements) == 1
    assert result[2].elements == []
    client.analyze.assert_called_once()


def test_analyze_groups_empty_returns_empty():
    client = MagicMock()
    result = list(analyze_groups([], [], client, "p", workers=1, retry_config=_no_retry()))
    assert result == []


def test_analyze_groups_retries_on_all_invalid_items():
    """When the model returns garbage JSON, analyze_groups must retry rather than silently yield empty."""
    client = MagicMock()
    garbage = json.dumps({"]0E@#@$&,FB8$.-B2=A3F766=E+*9*)?2AC@3#1F9B.0": "#)8#-.:"})
    valid = json.dumps([VALID_ELEMENT])
    client.analyze.side_effect = [garbage, valid]
    retry_config = RetryConfig(max_attempts=2)
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=retry_config))
    assert result[0] is not None
    assert len(result[0].elements) == 1
    assert client.analyze.call_count == 2


def test_analyze_groups_yields_none_on_non_retryable():
    client = MagicMock()
    client.analyze.side_effect = OSError("disk error")
    result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [None]
    assert client.analyze.call_count == 1
